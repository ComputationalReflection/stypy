
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Functions
3: ---------
4: .. autosummary::
5:    :toctree: generated/
6: 
7:     line_search_armijo
8:     line_search_wolfe1
9:     line_search_wolfe2
10:     scalar_search_wolfe1
11:     scalar_search_wolfe2
12: 
13: '''
14: from __future__ import division, print_function, absolute_import
15: 
16: from warnings import warn
17: 
18: from scipy.optimize import minpack2
19: import numpy as np
20: from scipy._lib.six import xrange
21: 
22: __all__ = ['LineSearchWarning', 'line_search_wolfe1', 'line_search_wolfe2',
23:            'scalar_search_wolfe1', 'scalar_search_wolfe2',
24:            'line_search_armijo']
25: 
26: class LineSearchWarning(RuntimeWarning):
27:     pass
28: 
29: 
30: #------------------------------------------------------------------------------
31: # Minpack's Wolfe line and scalar searches
32: #------------------------------------------------------------------------------
33: 
34: def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
35:                        old_fval=None, old_old_fval=None,
36:                        args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
37:                        xtol=1e-14):
38:     '''
39:     As `scalar_search_wolfe1` but do a line search to direction `pk`
40: 
41:     Parameters
42:     ----------
43:     f : callable
44:         Function `f(x)`
45:     fprime : callable
46:         Gradient of `f`
47:     xk : array_like
48:         Current point
49:     pk : array_like
50:         Search direction
51: 
52:     gfk : array_like, optional
53:         Gradient of `f` at point `xk`
54:     old_fval : float, optional
55:         Value of `f` at point `xk`
56:     old_old_fval : float, optional
57:         Value of `f` at point preceding `xk`
58: 
59:     The rest of the parameters are the same as for `scalar_search_wolfe1`.
60: 
61:     Returns
62:     -------
63:     stp, f_count, g_count, fval, old_fval
64:         As in `line_search_wolfe1`
65:     gval : array
66:         Gradient of `f` at the final point
67: 
68:     '''
69:     if gfk is None:
70:         gfk = fprime(xk)
71: 
72:     if isinstance(fprime, tuple):
73:         eps = fprime[1]
74:         fprime = fprime[0]
75:         newargs = (f, eps) + args
76:         gradient = False
77:     else:
78:         newargs = args
79:         gradient = True
80: 
81:     gval = [gfk]
82:     gc = [0]
83:     fc = [0]
84: 
85:     def phi(s):
86:         fc[0] += 1
87:         return f(xk + s*pk, *args)
88: 
89:     def derphi(s):
90:         gval[0] = fprime(xk + s*pk, *newargs)
91:         if gradient:
92:             gc[0] += 1
93:         else:
94:             fc[0] += len(xk) + 1
95:         return np.dot(gval[0], pk)
96: 
97:     derphi0 = np.dot(gfk, pk)
98: 
99:     stp, fval, old_fval = scalar_search_wolfe1(
100:             phi, derphi, old_fval, old_old_fval, derphi0,
101:             c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)
102: 
103:     return stp, fc[0], gc[0], fval, old_fval, gval[0]
104: 
105: 
106: def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
107:                          c1=1e-4, c2=0.9,
108:                          amax=50, amin=1e-8, xtol=1e-14):
109:     '''
110:     Scalar function search for alpha that satisfies strong Wolfe conditions
111: 
112:     alpha > 0 is assumed to be a descent direction.
113: 
114:     Parameters
115:     ----------
116:     phi : callable phi(alpha)
117:         Function at point `alpha`
118:     derphi : callable dphi(alpha)
119:         Derivative `d phi(alpha)/ds`. Returns a scalar.
120: 
121:     phi0 : float, optional
122:         Value of `f` at 0
123:     old_phi0 : float, optional
124:         Value of `f` at the previous point
125:     derphi0 : float, optional
126:         Value `derphi` at 0
127:     c1, c2 : float, optional
128:         Wolfe parameters
129:     amax, amin : float, optional
130:         Maximum and minimum step size
131:     xtol : float, optional
132:         Relative tolerance for an acceptable step.
133: 
134:     Returns
135:     -------
136:     alpha : float
137:         Step size, or None if no suitable step was found
138:     phi : float
139:         Value of `phi` at the new point `alpha`
140:     phi0 : float
141:         Value of `phi` at `alpha=0`
142: 
143:     Notes
144:     -----
145:     Uses routine DCSRCH from MINPACK.
146: 
147:     '''
148: 
149:     if phi0 is None:
150:         phi0 = phi(0.)
151:     if derphi0 is None:
152:         derphi0 = derphi(0.)
153: 
154:     if old_phi0 is not None and derphi0 != 0:
155:         alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
156:         if alpha1 < 0:
157:             alpha1 = 1.0
158:     else:
159:         alpha1 = 1.0
160: 
161:     phi1 = phi0
162:     derphi1 = derphi0
163:     isave = np.zeros((2,), np.intc)
164:     dsave = np.zeros((13,), float)
165:     task = b'START'
166: 
167:     maxiter = 100
168:     for i in xrange(maxiter):
169:         stp, phi1, derphi1, task = minpack2.dcsrch(alpha1, phi1, derphi1,
170:                                                    c1, c2, xtol, task,
171:                                                    amin, amax, isave, dsave)
172:         if task[:2] == b'FG':
173:             alpha1 = stp
174:             phi1 = phi(stp)
175:             derphi1 = derphi(stp)
176:         else:
177:             break
178:     else:
179:         # maxiter reached, the line search did not converge
180:         stp = None
181: 
182:     if task[:5] == b'ERROR' or task[:4] == b'WARN':
183:         stp = None  # failed
184: 
185:     return stp, phi1, phi0
186: 
187: line_search = line_search_wolfe1
188: 
189: 
190: #------------------------------------------------------------------------------
191: # Pure-Python Wolfe line and scalar searches
192: #------------------------------------------------------------------------------
193: 
194: def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
195:                        old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
196:                        extra_condition=None, maxiter=10):
197:     '''Find alpha that satisfies strong Wolfe conditions.
198: 
199:     Parameters
200:     ----------
201:     f : callable f(x,*args)
202:         Objective function.
203:     myfprime : callable f'(x,*args)
204:         Objective function gradient.
205:     xk : ndarray
206:         Starting point.
207:     pk : ndarray
208:         Search direction.
209:     gfk : ndarray, optional
210:         Gradient value for x=xk (xk being the current parameter
211:         estimate). Will be recomputed if omitted.
212:     old_fval : float, optional
213:         Function value for x=xk. Will be recomputed if omitted.
214:     old_old_fval : float, optional
215:         Function value for the point preceding x=xk
216:     args : tuple, optional
217:         Additional arguments passed to objective function.
218:     c1 : float, optional
219:         Parameter for Armijo condition rule.
220:     c2 : float, optional
221:         Parameter for curvature condition rule.
222:     amax : float, optional
223:         Maximum step size
224:     extra_condition : callable, optional
225:         A callable of the form ``extra_condition(alpha, x, f, g)``
226:         returning a boolean. Arguments are the proposed step ``alpha``
227:         and the corresponding ``x``, ``f`` and ``g`` values. The line search 
228:         accepts the value of ``alpha`` only if this 
229:         callable returns ``True``. If the callable returns ``False`` 
230:         for the step length, the algorithm will continue with 
231:         new iterates. The callable is only called for iterates 
232:         satisfying the strong Wolfe conditions.
233:     maxiter : int, optional
234:         Maximum number of iterations to perform
235: 
236:     Returns
237:     -------
238:     alpha : float or None
239:         Alpha for which ``x_new = x0 + alpha * pk``,
240:         or None if the line search algorithm did not converge.
241:     fc : int
242:         Number of function evaluations made.
243:     gc : int
244:         Number of gradient evaluations made.
245:     new_fval : float or None
246:         New function value ``f(x_new)=f(x0+alpha*pk)``,
247:         or None if the line search algorithm did not converge.
248:     old_fval : float
249:         Old function value ``f(x0)``.
250:     new_slope : float or None
251:         The local slope along the search direction at the
252:         new value ``<myfprime(x_new), pk>``,
253:         or None if the line search algorithm did not converge.
254: 
255: 
256:     Notes
257:     -----
258:     Uses the line search algorithm to enforce strong Wolfe
259:     conditions.  See Wright and Nocedal, 'Numerical Optimization',
260:     1999, pg. 59-60.
261: 
262:     For the zoom phase it uses an algorithm by [...].
263: 
264:     '''
265:     fc = [0]
266:     gc = [0]
267:     gval = [None]
268:     gval_alpha = [None]
269: 
270:     def phi(alpha):
271:         fc[0] += 1
272:         return f(xk + alpha * pk, *args)
273: 
274:     if isinstance(myfprime, tuple):
275:         def derphi(alpha):
276:             fc[0] += len(xk) + 1
277:             eps = myfprime[1]
278:             fprime = myfprime[0]
279:             newargs = (f, eps) + args
280:             gval[0] = fprime(xk + alpha * pk, *newargs)  # store for later use
281:             gval_alpha[0] = alpha
282:             return np.dot(gval[0], pk)
283:     else:
284:         fprime = myfprime
285: 
286:         def derphi(alpha):
287:             gc[0] += 1
288:             gval[0] = fprime(xk + alpha * pk, *args)  # store for later use
289:             gval_alpha[0] = alpha
290:             return np.dot(gval[0], pk)
291: 
292:     if gfk is None:
293:         gfk = fprime(xk, *args)
294:     derphi0 = np.dot(gfk, pk)
295: 
296:     if extra_condition is not None:
297:         # Add the current gradient as argument, to avoid needless
298:         # re-evaluation
299:         def extra_condition2(alpha, phi):
300:             if gval_alpha[0] != alpha:
301:                 derphi(alpha)
302:             x = xk + alpha * pk
303:             return extra_condition(alpha, x, phi, gval[0])
304:     else:
305:         extra_condition2 = None
306: 
307:     alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
308:             phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
309:             extra_condition2, maxiter=maxiter)
310: 
311:     if derphi_star is None:
312:         warn('The line search algorithm did not converge', LineSearchWarning)
313:     else:
314:         # derphi_star is a number (derphi) -- so use the most recently
315:         # calculated gradient used in computing it derphi = gfk*pk
316:         # this is the gradient at the next step no need to compute it
317:         # again in the outer loop.
318:         derphi_star = gval[0]
319: 
320:     return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star
321: 
322: 
323: def scalar_search_wolfe2(phi, derphi=None, phi0=None,
324:                          old_phi0=None, derphi0=None,
325:                          c1=1e-4, c2=0.9, amax=None,
326:                          extra_condition=None, maxiter=10):
327:     '''Find alpha that satisfies strong Wolfe conditions.
328: 
329:     alpha > 0 is assumed to be a descent direction.
330: 
331:     Parameters
332:     ----------
333:     phi : callable f(x)
334:         Objective scalar function.
335:     derphi : callable f'(x), optional
336:         Objective function derivative (can be None)
337:     phi0 : float, optional
338:         Value of phi at s=0
339:     old_phi0 : float, optional
340:         Value of phi at previous point
341:     derphi0 : float, optional
342:         Value of derphi at s=0
343:     c1 : float, optional
344:         Parameter for Armijo condition rule.
345:     c2 : float, optional
346:         Parameter for curvature condition rule.
347:     amax : float, optional
348:         Maximum step size
349:     extra_condition : callable, optional
350:         A callable of the form ``extra_condition(alpha, phi_value)``
351:         returning a boolean. The line search accepts the value
352:         of ``alpha`` only if this callable returns ``True``.
353:         If the callable returns ``False`` for the step length,
354:         the algorithm will continue with new iterates.
355:         The callable is only called for iterates satisfying
356:         the strong Wolfe conditions.
357:     maxiter : int, optional
358:         Maximum number of iterations to perform
359: 
360:     Returns
361:     -------
362:     alpha_star : float or None
363:         Best alpha, or None if the line search algorithm did not converge.
364:     phi_star : float
365:         phi at alpha_star
366:     phi0 : float
367:         phi at 0
368:     derphi_star : float or None
369:         derphi at alpha_star, or None if the line search algorithm
370:         did not converge.
371: 
372:     Notes
373:     -----
374:     Uses the line search algorithm to enforce strong Wolfe
375:     conditions.  See Wright and Nocedal, 'Numerical Optimization',
376:     1999, pg. 59-60.
377: 
378:     For the zoom phase it uses an algorithm by [...].
379: 
380:     '''
381: 
382:     if phi0 is None:
383:         phi0 = phi(0.)
384: 
385:     if derphi0 is None and derphi is not None:
386:         derphi0 = derphi(0.)
387: 
388:     alpha0 = 0
389:     if old_phi0 is not None and derphi0 != 0:
390:         alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
391:     else:
392:         alpha1 = 1.0
393: 
394:     if alpha1 < 0:
395:         alpha1 = 1.0
396: 
397:     phi_a1 = phi(alpha1)
398:     #derphi_a1 = derphi(alpha1)  evaluated below
399: 
400:     phi_a0 = phi0
401:     derphi_a0 = derphi0
402: 
403:     if extra_condition is None:
404:         extra_condition = lambda alpha, phi: True
405: 
406:     for i in xrange(maxiter):
407:         if alpha1 == 0 or (amax is not None and alpha0 == amax):
408:             # alpha1 == 0: This shouldn't happen. Perhaps the increment has
409:             # slipped below machine precision?
410:             alpha_star = None
411:             phi_star = phi0
412:             phi0 = old_phi0
413:             derphi_star = None
414: 
415:             if alpha1 == 0:
416:                 msg = 'Rounding errors prevent the line search from converging'
417:             else:
418:                 msg = "The line search algorithm could not find a solution " + \
419:                       "less than or equal to amax: %s" % amax
420: 
421:             warn(msg, LineSearchWarning)
422:             break
423: 
424:         if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
425:            ((phi_a1 >= phi_a0) and (i > 1)):
426:             alpha_star, phi_star, derphi_star = \
427:                         _zoom(alpha0, alpha1, phi_a0,
428:                               phi_a1, derphi_a0, phi, derphi,
429:                               phi0, derphi0, c1, c2, extra_condition)
430:             break
431: 
432:         derphi_a1 = derphi(alpha1)
433:         if (abs(derphi_a1) <= -c2*derphi0):
434:             if extra_condition(alpha1, phi_a1):
435:                 alpha_star = alpha1
436:                 phi_star = phi_a1
437:                 derphi_star = derphi_a1
438:                 break
439: 
440:         if (derphi_a1 >= 0):
441:             alpha_star, phi_star, derphi_star = \
442:                         _zoom(alpha1, alpha0, phi_a1,
443:                               phi_a0, derphi_a1, phi, derphi,
444:                               phi0, derphi0, c1, c2, extra_condition)
445:             break
446: 
447:         alpha2 = 2 * alpha1  # increase by factor of two on each iteration
448:         if amax is not None:
449:             alpha2 = min(alpha2, amax)
450:         alpha0 = alpha1
451:         alpha1 = alpha2
452:         phi_a0 = phi_a1
453:         phi_a1 = phi(alpha1)
454:         derphi_a0 = derphi_a1
455: 
456:     else:
457:         # stopping test maxiter reached
458:         alpha_star = alpha1
459:         phi_star = phi_a1
460:         derphi_star = None
461:         warn('The line search algorithm did not converge', LineSearchWarning)
462: 
463:     return alpha_star, phi_star, phi0, derphi_star
464: 
465: 
466: def _cubicmin(a, fa, fpa, b, fb, c, fc):
467:     '''
468:     Finds the minimizer for a cubic polynomial that goes through the
469:     points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
470: 
471:     If no minimizer can be found return None
472: 
473:     '''
474:     # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
475: 
476:     with np.errstate(divide='raise', over='raise', invalid='raise'):
477:         try:
478:             C = fpa
479:             db = b - a
480:             dc = c - a
481:             denom = (db * dc) ** 2 * (db - dc)
482:             d1 = np.empty((2, 2))
483:             d1[0, 0] = dc ** 2
484:             d1[0, 1] = -db ** 2
485:             d1[1, 0] = -dc ** 3
486:             d1[1, 1] = db ** 3
487:             [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
488:                                             fc - fa - C * dc]).flatten())
489:             A /= denom
490:             B /= denom
491:             radical = B * B - 3 * A * C
492:             xmin = a + (-B + np.sqrt(radical)) / (3 * A)
493:         except ArithmeticError:
494:             return None
495:     if not np.isfinite(xmin):
496:         return None
497:     return xmin
498: 
499: 
500: def _quadmin(a, fa, fpa, b, fb):
501:     '''
502:     Finds the minimizer for a quadratic polynomial that goes through
503:     the points (a,fa), (b,fb) with derivative at a of fpa,
504: 
505:     '''
506:     # f(x) = B*(x-a)^2 + C*(x-a) + D
507:     with np.errstate(divide='raise', over='raise', invalid='raise'):
508:         try:
509:             D = fa
510:             C = fpa
511:             db = b - a * 1.0
512:             B = (fb - D - C * db) / (db * db)
513:             xmin = a - C / (2.0 * B)
514:         except ArithmeticError:
515:             return None
516:     if not np.isfinite(xmin):
517:         return None
518:     return xmin
519: 
520: 
521: def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
522:           phi, derphi, phi0, derphi0, c1, c2, extra_condition):
523:     '''
524:     Part of the optimization algorithm in `scalar_search_wolfe2`.
525:     '''
526: 
527:     maxiter = 10
528:     i = 0
529:     delta1 = 0.2  # cubic interpolant check
530:     delta2 = 0.1  # quadratic interpolant check
531:     phi_rec = phi0
532:     a_rec = 0
533:     while True:
534:         # interpolate to find a trial step length between a_lo and
535:         # a_hi Need to choose interpolation here.  Use cubic
536:         # interpolation and then if the result is within delta *
537:         # dalpha or outside of the interval bounded by a_lo or a_hi
538:         # then use quadratic interpolation, if the result is still too
539:         # close, then use bisection
540: 
541:         dalpha = a_hi - a_lo
542:         if dalpha < 0:
543:             a, b = a_hi, a_lo
544:         else:
545:             a, b = a_lo, a_hi
546: 
547:         # minimizer of cubic interpolant
548:         # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
549:         #
550:         # if the result is too close to the end points (or out of the
551:         # interval) then use quadratic interpolation with phi_lo,
552:         # derphi_lo and phi_hi if the result is stil too close to the
553:         # end points (or out of the interval) then use bisection
554: 
555:         if (i > 0):
556:             cchk = delta1 * dalpha
557:             a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
558:                             a_rec, phi_rec)
559:         if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
560:             qchk = delta2 * dalpha
561:             a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
562:             if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
563:                 a_j = a_lo + 0.5*dalpha
564: 
565:         # Check new value of a_j
566: 
567:         phi_aj = phi(a_j)
568:         if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
569:             phi_rec = phi_hi
570:             a_rec = a_hi
571:             a_hi = a_j
572:             phi_hi = phi_aj
573:         else:
574:             derphi_aj = derphi(a_j)
575:             if abs(derphi_aj) <= -c2*derphi0 and extra_condition(a_j, phi_aj):
576:                 a_star = a_j
577:                 val_star = phi_aj
578:                 valprime_star = derphi_aj
579:                 break
580:             if derphi_aj*(a_hi - a_lo) >= 0:
581:                 phi_rec = phi_hi
582:                 a_rec = a_hi
583:                 a_hi = a_lo
584:                 phi_hi = phi_lo
585:             else:
586:                 phi_rec = phi_lo
587:                 a_rec = a_lo
588:             a_lo = a_j
589:             phi_lo = phi_aj
590:             derphi_lo = derphi_aj
591:         i += 1
592:         if (i > maxiter):
593:             # Failed to find a conforming step size
594:             a_star = None
595:             val_star = None
596:             valprime_star = None
597:             break
598:     return a_star, val_star, valprime_star
599: 
600: 
601: #------------------------------------------------------------------------------
602: # Armijo line and scalar searches
603: #------------------------------------------------------------------------------
604: 
605: def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
606:     '''Minimize over alpha, the function ``f(xk+alpha pk)``.
607: 
608:     Parameters
609:     ----------
610:     f : callable
611:         Function to be minimized.
612:     xk : array_like
613:         Current point.
614:     pk : array_like
615:         Search direction.
616:     gfk : array_like
617:         Gradient of `f` at point `xk`.
618:     old_fval : float
619:         Value of `f` at point `xk`.
620:     args : tuple, optional
621:         Optional arguments.
622:     c1 : float, optional
623:         Value to control stopping criterion.
624:     alpha0 : scalar, optional
625:         Value of `alpha` at start of the optimization.
626: 
627:     Returns
628:     -------
629:     alpha
630:     f_count
631:     f_val_at_alpha
632: 
633:     Notes
634:     -----
635:     Uses the interpolation algorithm (Armijo backtracking) as suggested by
636:     Wright and Nocedal in 'Numerical Optimization', 1999, pg. 56-57
637: 
638:     '''
639:     xk = np.atleast_1d(xk)
640:     fc = [0]
641: 
642:     def phi(alpha1):
643:         fc[0] += 1
644:         return f(xk + alpha1*pk, *args)
645: 
646:     if old_fval is None:
647:         phi0 = phi(0.)
648:     else:
649:         phi0 = old_fval  # compute f(xk) -- done in past loop
650: 
651:     derphi0 = np.dot(gfk, pk)
652:     alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
653:                                        alpha0=alpha0)
654:     return alpha, fc[0], phi1
655: 
656: 
657: def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
658:     '''
659:     Compatibility wrapper for `line_search_armijo`
660:     '''
661:     r = line_search_armijo(f, xk, pk, gfk, old_fval, args=args, c1=c1,
662:                            alpha0=alpha0)
663:     return r[0], r[1], 0, r[2]
664: 
665: 
666: def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
667:     '''Minimize over alpha, the function ``phi(alpha)``.
668: 
669:     Uses the interpolation algorithm (Armijo backtracking) as suggested by
670:     Wright and Nocedal in 'Numerical Optimization', 1999, pg. 56-57
671: 
672:     alpha > 0 is assumed to be a descent direction.
673: 
674:     Returns
675:     -------
676:     alpha
677:     phi1
678: 
679:     '''
680:     phi_a0 = phi(alpha0)
681:     if phi_a0 <= phi0 + c1*alpha0*derphi0:
682:         return alpha0, phi_a0
683: 
684:     # Otherwise compute the minimizer of a quadratic interpolant:
685: 
686:     alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
687:     phi_a1 = phi(alpha1)
688: 
689:     if (phi_a1 <= phi0 + c1*alpha1*derphi0):
690:         return alpha1, phi_a1
691: 
692:     # Otherwise loop with cubic interpolation until we find an alpha which
693:     # satifies the first Wolfe condition (since we are backtracking, we will
694:     # assume that the value of alpha is not too small and satisfies the second
695:     # condition.
696: 
697:     while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
698:         factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
699:         a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
700:             alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
701:         a = a / factor
702:         b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
703:             alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
704:         b = b / factor
705: 
706:         alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
707:         phi_a2 = phi(alpha2)
708: 
709:         if (phi_a2 <= phi0 + c1*alpha2*derphi0):
710:             return alpha2, phi_a2
711: 
712:         if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
713:             alpha2 = alpha1 / 2.0
714: 
715:         alpha0 = alpha1
716:         alpha1 = alpha2
717:         phi_a0 = phi_a1
718:         phi_a1 = phi_a2
719: 
720:     # Failed to find a suitable step length
721:     return None, phi_a1
722: 
723: 
724: #------------------------------------------------------------------------------
725: # Non-monotone line search for DF-SANE
726: #------------------------------------------------------------------------------
727: 
728: def _nonmonotone_line_search_cruz(f, x_k, d, prev_fs, eta,
729:                                   gamma=1e-4, tau_min=0.1, tau_max=0.5):
730:     '''
731:     Nonmonotone backtracking line search as described in [1]_
732: 
733:     Parameters
734:     ----------
735:     f : callable
736:         Function returning a tuple ``(f, F)`` where ``f`` is the value
737:         of a merit function and ``F`` the residual.
738:     x_k : ndarray
739:         Initial position
740:     d : ndarray
741:         Search direction
742:     prev_fs : float
743:         List of previous merit function values. Should have ``len(prev_fs) <= M``
744:         where ``M`` is the nonmonotonicity window parameter.
745:     eta : float
746:         Allowed merit function increase, see [1]_
747:     gamma, tau_min, tau_max : float, optional
748:         Search parameters, see [1]_
749: 
750:     Returns
751:     -------
752:     alpha : float
753:         Step length
754:     xp : ndarray
755:         Next position
756:     fp : float
757:         Merit function value at next position
758:     Fp : ndarray
759:         Residual at next position
760: 
761:     References
762:     ----------
763:     [1] "Spectral residual method without gradient information for solving
764:         large-scale nonlinear systems of equations." W. La Cruz,
765:         J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).
766: 
767:     '''
768:     f_k = prev_fs[-1]
769:     f_bar = max(prev_fs)
770: 
771:     alpha_p = 1
772:     alpha_m = 1
773:     alpha = 1
774: 
775:     while True:
776:         xp = x_k + alpha_p * d
777:         fp, Fp = f(xp)
778: 
779:         if fp <= f_bar + eta - gamma * alpha_p**2 * f_k:
780:             alpha = alpha_p
781:             break
782: 
783:         alpha_tp = alpha_p**2 * f_k / (fp + (2*alpha_p - 1)*f_k)
784: 
785:         xp = x_k - alpha_m * d
786:         fp, Fp = f(xp)
787: 
788:         if fp <= f_bar + eta - gamma * alpha_m**2 * f_k:
789:             alpha = -alpha_m
790:             break
791: 
792:         alpha_tm = alpha_m**2 * f_k / (fp + (2*alpha_m - 1)*f_k)
793: 
794:         alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
795:         alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)
796: 
797:     return alpha, xp, fp, Fp
798: 
799: 
800: def _nonmonotone_line_search_cheng(f, x_k, d, f_k, C, Q, eta,
801:                                    gamma=1e-4, tau_min=0.1, tau_max=0.5,
802:                                    nu=0.85):
803:     '''
804:     Nonmonotone line search from [1]
805: 
806:     Parameters
807:     ----------
808:     f : callable
809:         Function returning a tuple ``(f, F)`` where ``f`` is the value
810:         of a merit function and ``F`` the residual.
811:     x_k : ndarray
812:         Initial position
813:     d : ndarray
814:         Search direction
815:     f_k : float
816:         Initial merit function value
817:     C, Q : float
818:         Control parameters. On the first iteration, give values
819:         Q=1.0, C=f_k
820:     eta : float
821:         Allowed merit function increase, see [1]_
822:     nu, gamma, tau_min, tau_max : float, optional
823:         Search parameters, see [1]_
824: 
825:     Returns
826:     -------
827:     alpha : float
828:         Step length
829:     xp : ndarray
830:         Next position
831:     fp : float
832:         Merit function value at next position
833:     Fp : ndarray
834:         Residual at next position
835:     C : float
836:         New value for the control parameter C
837:     Q : float
838:         New value for the control parameter Q
839: 
840:     References
841:     ----------
842:     .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line
843:            search and its application to the spectral residual
844:            method'', IMA J. Numer. Anal. 29, 814 (2009).
845: 
846:     '''
847:     alpha_p = 1
848:     alpha_m = 1
849:     alpha = 1
850: 
851:     while True:
852:         xp = x_k + alpha_p * d
853:         fp, Fp = f(xp)
854: 
855:         if fp <= C + eta - gamma * alpha_p**2 * f_k:
856:             alpha = alpha_p
857:             break
858: 
859:         alpha_tp = alpha_p**2 * f_k / (fp + (2*alpha_p - 1)*f_k)
860: 
861:         xp = x_k - alpha_m * d
862:         fp, Fp = f(xp)
863: 
864:         if fp <= C + eta - gamma * alpha_m**2 * f_k:
865:             alpha = -alpha_m
866:             break
867: 
868:         alpha_tm = alpha_m**2 * f_k / (fp + (2*alpha_m - 1)*f_k)
869: 
870:         alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
871:         alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)
872: 
873:     # Update C and Q
874:     Q_next = nu * Q + 1
875:     C = (nu * Q * (C + eta) + fp) / Q_next
876:     Q = Q_next
877: 
878:     return alpha, xp, fp, Fp, C, Q
879: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_169068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\nFunctions\n---------\n.. autosummary::\n   :toctree: generated/\n\n    line_search_armijo\n    line_search_wolfe1\n    line_search_wolfe2\n    scalar_search_wolfe1\n    scalar_search_wolfe2\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from warnings import warn' statement (line 16)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.optimize import minpack2' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_169069 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize')

if (type(import_169069) is not StypyTypeError):

    if (import_169069 != 'pyd_module'):
        __import__(import_169069)
        sys_modules_169070 = sys.modules[import_169069]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize', sys_modules_169070.module_type_store, module_type_store, ['minpack2'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_169070, sys_modules_169070.module_type_store, module_type_store)
    else:
        from scipy.optimize import minpack2

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize', None, module_type_store, ['minpack2'], [minpack2])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize', import_169069)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import numpy' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_169071 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy')

if (type(import_169071) is not StypyTypeError):

    if (import_169071 != 'pyd_module'):
        __import__(import_169071)
        sys_modules_169072 = sys.modules[import_169071]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'np', sys_modules_169072.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy', import_169071)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy._lib.six import xrange' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_169073 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy._lib.six')

if (type(import_169073) is not StypyTypeError):

    if (import_169073 != 'pyd_module'):
        __import__(import_169073)
        sys_modules_169074 = sys.modules[import_169073]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy._lib.six', sys_modules_169074.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_169074, sys_modules_169074.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy._lib.six', import_169073)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 22):

# Assigning a List to a Name (line 22):
__all__ = ['LineSearchWarning', 'line_search_wolfe1', 'line_search_wolfe2', 'scalar_search_wolfe1', 'scalar_search_wolfe2', 'line_search_armijo']
module_type_store.set_exportable_members(['LineSearchWarning', 'line_search_wolfe1', 'line_search_wolfe2', 'scalar_search_wolfe1', 'scalar_search_wolfe2', 'line_search_armijo'])

# Obtaining an instance of the builtin type 'list' (line 22)
list_169075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_169076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'str', 'LineSearchWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_169075, str_169076)
# Adding element type (line 22)
str_169077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', 'line_search_wolfe1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_169075, str_169077)
# Adding element type (line 22)
str_169078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 54), 'str', 'line_search_wolfe2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_169075, str_169078)
# Adding element type (line 22)
str_169079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'str', 'scalar_search_wolfe1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_169075, str_169079)
# Adding element type (line 22)
str_169080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'str', 'scalar_search_wolfe2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_169075, str_169080)
# Adding element type (line 22)
str_169081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'str', 'line_search_armijo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_169075, str_169081)

# Assigning a type to the variable '__all__' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '__all__', list_169075)
# Declaration of the 'LineSearchWarning' class
# Getting the type of 'RuntimeWarning' (line 26)
RuntimeWarning_169082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'RuntimeWarning')

class LineSearchWarning(RuntimeWarning_169082, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 0, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LineSearchWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LineSearchWarning' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'LineSearchWarning', LineSearchWarning)

@norecursion
def line_search_wolfe1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 34)
    None_169083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 46), 'None')
    # Getting the type of 'None' (line 35)
    None_169084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'None')
    # Getting the type of 'None' (line 35)
    None_169085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 51), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_169086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    
    float_169087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'float')
    float_169088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 44), 'float')
    int_169089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 54), 'int')
    float_169090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 63), 'float')
    float_169091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'float')
    defaults = [None_169083, None_169084, None_169085, tuple_169086, float_169087, float_169088, int_169089, float_169090, float_169091]
    # Create a new context for function 'line_search_wolfe1'
    module_type_store = module_type_store.open_function_context('line_search_wolfe1', 34, 0, False)
    
    # Passed parameters checking function
    line_search_wolfe1.stypy_localization = localization
    line_search_wolfe1.stypy_type_of_self = None
    line_search_wolfe1.stypy_type_store = module_type_store
    line_search_wolfe1.stypy_function_name = 'line_search_wolfe1'
    line_search_wolfe1.stypy_param_names_list = ['f', 'fprime', 'xk', 'pk', 'gfk', 'old_fval', 'old_old_fval', 'args', 'c1', 'c2', 'amax', 'amin', 'xtol']
    line_search_wolfe1.stypy_varargs_param_name = None
    line_search_wolfe1.stypy_kwargs_param_name = None
    line_search_wolfe1.stypy_call_defaults = defaults
    line_search_wolfe1.stypy_call_varargs = varargs
    line_search_wolfe1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'line_search_wolfe1', ['f', 'fprime', 'xk', 'pk', 'gfk', 'old_fval', 'old_old_fval', 'args', 'c1', 'c2', 'amax', 'amin', 'xtol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'line_search_wolfe1', localization, ['f', 'fprime', 'xk', 'pk', 'gfk', 'old_fval', 'old_old_fval', 'args', 'c1', 'c2', 'amax', 'amin', 'xtol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'line_search_wolfe1(...)' code ##################

    str_169092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n    As `scalar_search_wolfe1` but do a line search to direction `pk`\n\n    Parameters\n    ----------\n    f : callable\n        Function `f(x)`\n    fprime : callable\n        Gradient of `f`\n    xk : array_like\n        Current point\n    pk : array_like\n        Search direction\n\n    gfk : array_like, optional\n        Gradient of `f` at point `xk`\n    old_fval : float, optional\n        Value of `f` at point `xk`\n    old_old_fval : float, optional\n        Value of `f` at point preceding `xk`\n\n    The rest of the parameters are the same as for `scalar_search_wolfe1`.\n\n    Returns\n    -------\n    stp, f_count, g_count, fval, old_fval\n        As in `line_search_wolfe1`\n    gval : array\n        Gradient of `f` at the final point\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 69)
    # Getting the type of 'gfk' (line 69)
    gfk_169093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 7), 'gfk')
    # Getting the type of 'None' (line 69)
    None_169094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'None')
    
    (may_be_169095, more_types_in_union_169096) = may_be_none(gfk_169093, None_169094)

    if may_be_169095:

        if more_types_in_union_169096:
            # Runtime conditional SSA (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to fprime(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'xk' (line 70)
        xk_169098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'xk', False)
        # Processing the call keyword arguments (line 70)
        kwargs_169099 = {}
        # Getting the type of 'fprime' (line 70)
        fprime_169097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'fprime', False)
        # Calling fprime(args, kwargs) (line 70)
        fprime_call_result_169100 = invoke(stypy.reporting.localization.Localization(__file__, 70, 14), fprime_169097, *[xk_169098], **kwargs_169099)
        
        # Assigning a type to the variable 'gfk' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'gfk', fprime_call_result_169100)

        if more_types_in_union_169096:
            # SSA join for if statement (line 69)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 72)
    # Getting the type of 'tuple' (line 72)
    tuple_169101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'tuple')
    # Getting the type of 'fprime' (line 72)
    fprime_169102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'fprime')
    
    (may_be_169103, more_types_in_union_169104) = may_be_subtype(tuple_169101, fprime_169102)

    if may_be_169103:

        if more_types_in_union_169104:
            # Runtime conditional SSA (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'fprime' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'fprime', remove_not_subtype_from_union(fprime_169102, tuple))
        
        # Assigning a Subscript to a Name (line 73):
        
        # Assigning a Subscript to a Name (line 73):
        
        # Obtaining the type of the subscript
        int_169105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'int')
        # Getting the type of 'fprime' (line 73)
        fprime_169106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'fprime')
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___169107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 14), fprime_169106, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_169108 = invoke(stypy.reporting.localization.Localization(__file__, 73, 14), getitem___169107, int_169105)
        
        # Assigning a type to the variable 'eps' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'eps', subscript_call_result_169108)
        
        # Assigning a Subscript to a Name (line 74):
        
        # Assigning a Subscript to a Name (line 74):
        
        # Obtaining the type of the subscript
        int_169109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'int')
        # Getting the type of 'fprime' (line 74)
        fprime_169110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'fprime')
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___169111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), fprime_169110, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_169112 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), getitem___169111, int_169109)
        
        # Assigning a type to the variable 'fprime' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'fprime', subscript_call_result_169112)
        
        # Assigning a BinOp to a Name (line 75):
        
        # Assigning a BinOp to a Name (line 75):
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_169113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        # Getting the type of 'f' (line 75)
        f_169114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), tuple_169113, f_169114)
        # Adding element type (line 75)
        # Getting the type of 'eps' (line 75)
        eps_169115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'eps')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 19), tuple_169113, eps_169115)
        
        # Getting the type of 'args' (line 75)
        args_169116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'args')
        # Applying the binary operator '+' (line 75)
        result_add_169117 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 18), '+', tuple_169113, args_169116)
        
        # Assigning a type to the variable 'newargs' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'newargs', result_add_169117)
        
        # Assigning a Name to a Name (line 76):
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'False' (line 76)
        False_169118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'False')
        # Assigning a type to the variable 'gradient' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'gradient', False_169118)

        if more_types_in_union_169104:
            # Runtime conditional SSA for else branch (line 72)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_169103) or more_types_in_union_169104):
        # Assigning a type to the variable 'fprime' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'fprime', remove_subtype_from_union(fprime_169102, tuple))
        
        # Assigning a Name to a Name (line 78):
        
        # Assigning a Name to a Name (line 78):
        # Getting the type of 'args' (line 78)
        args_169119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), 'args')
        # Assigning a type to the variable 'newargs' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'newargs', args_169119)
        
        # Assigning a Name to a Name (line 79):
        
        # Assigning a Name to a Name (line 79):
        # Getting the type of 'True' (line 79)
        True_169120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'True')
        # Assigning a type to the variable 'gradient' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'gradient', True_169120)

        if (may_be_169103 and more_types_in_union_169104):
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 81):
    
    # Assigning a List to a Name (line 81):
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_169121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    # Getting the type of 'gfk' (line 81)
    gfk_169122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'gfk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_169121, gfk_169122)
    
    # Assigning a type to the variable 'gval' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'gval', list_169121)
    
    # Assigning a List to a Name (line 82):
    
    # Assigning a List to a Name (line 82):
    
    # Obtaining an instance of the builtin type 'list' (line 82)
    list_169123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 82)
    # Adding element type (line 82)
    int_169124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 9), list_169123, int_169124)
    
    # Assigning a type to the variable 'gc' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'gc', list_169123)
    
    # Assigning a List to a Name (line 83):
    
    # Assigning a List to a Name (line 83):
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_169125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    int_169126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 9), list_169125, int_169126)
    
    # Assigning a type to the variable 'fc' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'fc', list_169125)

    @norecursion
    def phi(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'phi'
        module_type_store = module_type_store.open_function_context('phi', 85, 4, False)
        
        # Passed parameters checking function
        phi.stypy_localization = localization
        phi.stypy_type_of_self = None
        phi.stypy_type_store = module_type_store
        phi.stypy_function_name = 'phi'
        phi.stypy_param_names_list = ['s']
        phi.stypy_varargs_param_name = None
        phi.stypy_kwargs_param_name = None
        phi.stypy_call_defaults = defaults
        phi.stypy_call_varargs = varargs
        phi.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'phi', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'phi', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'phi(...)' code ##################

        
        # Getting the type of 'fc' (line 86)
        fc_169127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'fc')
        
        # Obtaining the type of the subscript
        int_169128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'int')
        # Getting the type of 'fc' (line 86)
        fc_169129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'fc')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___169130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), fc_169129, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_169131 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___169130, int_169128)
        
        int_169132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 17), 'int')
        # Applying the binary operator '+=' (line 86)
        result_iadd_169133 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 8), '+=', subscript_call_result_169131, int_169132)
        # Getting the type of 'fc' (line 86)
        fc_169134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'fc')
        int_169135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'int')
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), fc_169134, (int_169135, result_iadd_169133))
        
        
        # Call to f(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'xk' (line 87)
        xk_169137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'xk', False)
        # Getting the type of 's' (line 87)
        s_169138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 's', False)
        # Getting the type of 'pk' (line 87)
        pk_169139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'pk', False)
        # Applying the binary operator '*' (line 87)
        result_mul_169140 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 22), '*', s_169138, pk_169139)
        
        # Applying the binary operator '+' (line 87)
        result_add_169141 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 17), '+', xk_169137, result_mul_169140)
        
        # Getting the type of 'args' (line 87)
        args_169142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'args', False)
        # Processing the call keyword arguments (line 87)
        kwargs_169143 = {}
        # Getting the type of 'f' (line 87)
        f_169136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'f', False)
        # Calling f(args, kwargs) (line 87)
        f_call_result_169144 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), f_169136, *[result_add_169141, args_169142], **kwargs_169143)
        
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', f_call_result_169144)
        
        # ################# End of 'phi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'phi' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_169145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_169145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'phi'
        return stypy_return_type_169145

    # Assigning a type to the variable 'phi' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'phi', phi)

    @norecursion
    def derphi(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'derphi'
        module_type_store = module_type_store.open_function_context('derphi', 89, 4, False)
        
        # Passed parameters checking function
        derphi.stypy_localization = localization
        derphi.stypy_type_of_self = None
        derphi.stypy_type_store = module_type_store
        derphi.stypy_function_name = 'derphi'
        derphi.stypy_param_names_list = ['s']
        derphi.stypy_varargs_param_name = None
        derphi.stypy_kwargs_param_name = None
        derphi.stypy_call_defaults = defaults
        derphi.stypy_call_varargs = varargs
        derphi.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'derphi', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'derphi', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'derphi(...)' code ##################

        
        # Assigning a Call to a Subscript (line 90):
        
        # Assigning a Call to a Subscript (line 90):
        
        # Call to fprime(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'xk' (line 90)
        xk_169147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'xk', False)
        # Getting the type of 's' (line 90)
        s_169148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 's', False)
        # Getting the type of 'pk' (line 90)
        pk_169149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'pk', False)
        # Applying the binary operator '*' (line 90)
        result_mul_169150 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 30), '*', s_169148, pk_169149)
        
        # Applying the binary operator '+' (line 90)
        result_add_169151 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 25), '+', xk_169147, result_mul_169150)
        
        # Getting the type of 'newargs' (line 90)
        newargs_169152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 37), 'newargs', False)
        # Processing the call keyword arguments (line 90)
        kwargs_169153 = {}
        # Getting the type of 'fprime' (line 90)
        fprime_169146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'fprime', False)
        # Calling fprime(args, kwargs) (line 90)
        fprime_call_result_169154 = invoke(stypy.reporting.localization.Localization(__file__, 90, 18), fprime_169146, *[result_add_169151, newargs_169152], **kwargs_169153)
        
        # Getting the type of 'gval' (line 90)
        gval_169155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'gval')
        int_169156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'int')
        # Storing an element on a container (line 90)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), gval_169155, (int_169156, fprime_call_result_169154))
        
        # Getting the type of 'gradient' (line 91)
        gradient_169157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'gradient')
        # Testing the type of an if condition (line 91)
        if_condition_169158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), gradient_169157)
        # Assigning a type to the variable 'if_condition_169158' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_169158', if_condition_169158)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'gc' (line 92)
        gc_169159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'gc')
        
        # Obtaining the type of the subscript
        int_169160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 15), 'int')
        # Getting the type of 'gc' (line 92)
        gc_169161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'gc')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___169162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), gc_169161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_169163 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), getitem___169162, int_169160)
        
        int_169164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 21), 'int')
        # Applying the binary operator '+=' (line 92)
        result_iadd_169165 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 12), '+=', subscript_call_result_169163, int_169164)
        # Getting the type of 'gc' (line 92)
        gc_169166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'gc')
        int_169167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 15), 'int')
        # Storing an element on a container (line 92)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 12), gc_169166, (int_169167, result_iadd_169165))
        
        # SSA branch for the else part of an if statement (line 91)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'fc' (line 94)
        fc_169168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'fc')
        
        # Obtaining the type of the subscript
        int_169169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 15), 'int')
        # Getting the type of 'fc' (line 94)
        fc_169170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'fc')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___169171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), fc_169170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_169172 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), getitem___169171, int_169169)
        
        
        # Call to len(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'xk' (line 94)
        xk_169174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'xk', False)
        # Processing the call keyword arguments (line 94)
        kwargs_169175 = {}
        # Getting the type of 'len' (line 94)
        len_169173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'len', False)
        # Calling len(args, kwargs) (line 94)
        len_call_result_169176 = invoke(stypy.reporting.localization.Localization(__file__, 94, 21), len_169173, *[xk_169174], **kwargs_169175)
        
        int_169177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 31), 'int')
        # Applying the binary operator '+' (line 94)
        result_add_169178 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 21), '+', len_call_result_169176, int_169177)
        
        # Applying the binary operator '+=' (line 94)
        result_iadd_169179 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 12), '+=', subscript_call_result_169172, result_add_169178)
        # Getting the type of 'fc' (line 94)
        fc_169180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'fc')
        int_169181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 15), 'int')
        # Storing an element on a container (line 94)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 12), fc_169180, (int_169181, result_iadd_169179))
        
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to dot(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining the type of the subscript
        int_169184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 27), 'int')
        # Getting the type of 'gval' (line 95)
        gval_169185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'gval', False)
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___169186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 22), gval_169185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_169187 = invoke(stypy.reporting.localization.Localization(__file__, 95, 22), getitem___169186, int_169184)
        
        # Getting the type of 'pk' (line 95)
        pk_169188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 31), 'pk', False)
        # Processing the call keyword arguments (line 95)
        kwargs_169189 = {}
        # Getting the type of 'np' (line 95)
        np_169182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 95)
        dot_169183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), np_169182, 'dot')
        # Calling dot(args, kwargs) (line 95)
        dot_call_result_169190 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), dot_169183, *[subscript_call_result_169187, pk_169188], **kwargs_169189)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', dot_call_result_169190)
        
        # ################# End of 'derphi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'derphi' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_169191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_169191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'derphi'
        return stypy_return_type_169191

    # Assigning a type to the variable 'derphi' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'derphi', derphi)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to dot(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'gfk' (line 97)
    gfk_169194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'gfk', False)
    # Getting the type of 'pk' (line 97)
    pk_169195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'pk', False)
    # Processing the call keyword arguments (line 97)
    kwargs_169196 = {}
    # Getting the type of 'np' (line 97)
    np_169192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 97)
    dot_169193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 14), np_169192, 'dot')
    # Calling dot(args, kwargs) (line 97)
    dot_call_result_169197 = invoke(stypy.reporting.localization.Localization(__file__, 97, 14), dot_169193, *[gfk_169194, pk_169195], **kwargs_169196)
    
    # Assigning a type to the variable 'derphi0' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'derphi0', dot_call_result_169197)
    
    # Assigning a Call to a Tuple (line 99):
    
    # Assigning a Subscript to a Name (line 99):
    
    # Obtaining the type of the subscript
    int_169198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'int')
    
    # Call to scalar_search_wolfe1(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'phi' (line 100)
    phi_169200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'phi', False)
    # Getting the type of 'derphi' (line 100)
    derphi_169201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'derphi', False)
    # Getting the type of 'old_fval' (line 100)
    old_fval_169202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'old_fval', False)
    # Getting the type of 'old_old_fval' (line 100)
    old_old_fval_169203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 'old_old_fval', False)
    # Getting the type of 'derphi0' (line 100)
    derphi0_169204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 49), 'derphi0', False)
    # Processing the call keyword arguments (line 99)
    # Getting the type of 'c1' (line 101)
    c1_169205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'c1', False)
    keyword_169206 = c1_169205
    # Getting the type of 'c2' (line 101)
    c2_169207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'c2', False)
    keyword_169208 = c2_169207
    # Getting the type of 'amax' (line 101)
    amax_169209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'amax', False)
    keyword_169210 = amax_169209
    # Getting the type of 'amin' (line 101)
    amin_169211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'amin', False)
    keyword_169212 = amin_169211
    # Getting the type of 'xtol' (line 101)
    xtol_169213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 53), 'xtol', False)
    keyword_169214 = xtol_169213
    kwargs_169215 = {'c2': keyword_169208, 'c1': keyword_169206, 'amin': keyword_169212, 'amax': keyword_169210, 'xtol': keyword_169214}
    # Getting the type of 'scalar_search_wolfe1' (line 99)
    scalar_search_wolfe1_169199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'scalar_search_wolfe1', False)
    # Calling scalar_search_wolfe1(args, kwargs) (line 99)
    scalar_search_wolfe1_call_result_169216 = invoke(stypy.reporting.localization.Localization(__file__, 99, 26), scalar_search_wolfe1_169199, *[phi_169200, derphi_169201, old_fval_169202, old_old_fval_169203, derphi0_169204], **kwargs_169215)
    
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___169217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 4), scalar_search_wolfe1_call_result_169216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_169218 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), getitem___169217, int_169198)
    
    # Assigning a type to the variable 'tuple_var_assignment_169035' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_169035', subscript_call_result_169218)
    
    # Assigning a Subscript to a Name (line 99):
    
    # Obtaining the type of the subscript
    int_169219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'int')
    
    # Call to scalar_search_wolfe1(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'phi' (line 100)
    phi_169221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'phi', False)
    # Getting the type of 'derphi' (line 100)
    derphi_169222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'derphi', False)
    # Getting the type of 'old_fval' (line 100)
    old_fval_169223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'old_fval', False)
    # Getting the type of 'old_old_fval' (line 100)
    old_old_fval_169224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 'old_old_fval', False)
    # Getting the type of 'derphi0' (line 100)
    derphi0_169225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 49), 'derphi0', False)
    # Processing the call keyword arguments (line 99)
    # Getting the type of 'c1' (line 101)
    c1_169226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'c1', False)
    keyword_169227 = c1_169226
    # Getting the type of 'c2' (line 101)
    c2_169228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'c2', False)
    keyword_169229 = c2_169228
    # Getting the type of 'amax' (line 101)
    amax_169230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'amax', False)
    keyword_169231 = amax_169230
    # Getting the type of 'amin' (line 101)
    amin_169232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'amin', False)
    keyword_169233 = amin_169232
    # Getting the type of 'xtol' (line 101)
    xtol_169234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 53), 'xtol', False)
    keyword_169235 = xtol_169234
    kwargs_169236 = {'c2': keyword_169229, 'c1': keyword_169227, 'amin': keyword_169233, 'amax': keyword_169231, 'xtol': keyword_169235}
    # Getting the type of 'scalar_search_wolfe1' (line 99)
    scalar_search_wolfe1_169220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'scalar_search_wolfe1', False)
    # Calling scalar_search_wolfe1(args, kwargs) (line 99)
    scalar_search_wolfe1_call_result_169237 = invoke(stypy.reporting.localization.Localization(__file__, 99, 26), scalar_search_wolfe1_169220, *[phi_169221, derphi_169222, old_fval_169223, old_old_fval_169224, derphi0_169225], **kwargs_169236)
    
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___169238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 4), scalar_search_wolfe1_call_result_169237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_169239 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), getitem___169238, int_169219)
    
    # Assigning a type to the variable 'tuple_var_assignment_169036' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_169036', subscript_call_result_169239)
    
    # Assigning a Subscript to a Name (line 99):
    
    # Obtaining the type of the subscript
    int_169240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'int')
    
    # Call to scalar_search_wolfe1(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'phi' (line 100)
    phi_169242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'phi', False)
    # Getting the type of 'derphi' (line 100)
    derphi_169243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'derphi', False)
    # Getting the type of 'old_fval' (line 100)
    old_fval_169244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'old_fval', False)
    # Getting the type of 'old_old_fval' (line 100)
    old_old_fval_169245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 'old_old_fval', False)
    # Getting the type of 'derphi0' (line 100)
    derphi0_169246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 49), 'derphi0', False)
    # Processing the call keyword arguments (line 99)
    # Getting the type of 'c1' (line 101)
    c1_169247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'c1', False)
    keyword_169248 = c1_169247
    # Getting the type of 'c2' (line 101)
    c2_169249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'c2', False)
    keyword_169250 = c2_169249
    # Getting the type of 'amax' (line 101)
    amax_169251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'amax', False)
    keyword_169252 = amax_169251
    # Getting the type of 'amin' (line 101)
    amin_169253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'amin', False)
    keyword_169254 = amin_169253
    # Getting the type of 'xtol' (line 101)
    xtol_169255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 53), 'xtol', False)
    keyword_169256 = xtol_169255
    kwargs_169257 = {'c2': keyword_169250, 'c1': keyword_169248, 'amin': keyword_169254, 'amax': keyword_169252, 'xtol': keyword_169256}
    # Getting the type of 'scalar_search_wolfe1' (line 99)
    scalar_search_wolfe1_169241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'scalar_search_wolfe1', False)
    # Calling scalar_search_wolfe1(args, kwargs) (line 99)
    scalar_search_wolfe1_call_result_169258 = invoke(stypy.reporting.localization.Localization(__file__, 99, 26), scalar_search_wolfe1_169241, *[phi_169242, derphi_169243, old_fval_169244, old_old_fval_169245, derphi0_169246], **kwargs_169257)
    
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___169259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 4), scalar_search_wolfe1_call_result_169258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_169260 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), getitem___169259, int_169240)
    
    # Assigning a type to the variable 'tuple_var_assignment_169037' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_169037', subscript_call_result_169260)
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'tuple_var_assignment_169035' (line 99)
    tuple_var_assignment_169035_169261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_169035')
    # Assigning a type to the variable 'stp' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stp', tuple_var_assignment_169035_169261)
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'tuple_var_assignment_169036' (line 99)
    tuple_var_assignment_169036_169262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_169036')
    # Assigning a type to the variable 'fval' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 9), 'fval', tuple_var_assignment_169036_169262)
    
    # Assigning a Name to a Name (line 99):
    # Getting the type of 'tuple_var_assignment_169037' (line 99)
    tuple_var_assignment_169037_169263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tuple_var_assignment_169037')
    # Assigning a type to the variable 'old_fval' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'old_fval', tuple_var_assignment_169037_169263)
    
    # Obtaining an instance of the builtin type 'tuple' (line 103)
    tuple_169264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 103)
    # Adding element type (line 103)
    # Getting the type of 'stp' (line 103)
    stp_169265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'stp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_169264, stp_169265)
    # Adding element type (line 103)
    
    # Obtaining the type of the subscript
    int_169266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'int')
    # Getting the type of 'fc' (line 103)
    fc_169267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'fc')
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___169268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), fc_169267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_169269 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), getitem___169268, int_169266)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_169264, subscript_call_result_169269)
    # Adding element type (line 103)
    
    # Obtaining the type of the subscript
    int_169270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'int')
    # Getting the type of 'gc' (line 103)
    gc_169271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'gc')
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___169272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 23), gc_169271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_169273 = invoke(stypy.reporting.localization.Localization(__file__, 103, 23), getitem___169272, int_169270)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_169264, subscript_call_result_169273)
    # Adding element type (line 103)
    # Getting the type of 'fval' (line 103)
    fval_169274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'fval')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_169264, fval_169274)
    # Adding element type (line 103)
    # Getting the type of 'old_fval' (line 103)
    old_fval_169275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'old_fval')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_169264, old_fval_169275)
    # Adding element type (line 103)
    
    # Obtaining the type of the subscript
    int_169276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 51), 'int')
    # Getting the type of 'gval' (line 103)
    gval_169277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'gval')
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___169278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 46), gval_169277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_169279 = invoke(stypy.reporting.localization.Localization(__file__, 103, 46), getitem___169278, int_169276)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_169264, subscript_call_result_169279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type', tuple_169264)
    
    # ################# End of 'line_search_wolfe1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'line_search_wolfe1' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_169280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169280)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'line_search_wolfe1'
    return stypy_return_type_169280

# Assigning a type to the variable 'line_search_wolfe1' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'line_search_wolfe1', line_search_wolfe1)

@norecursion
def scalar_search_wolfe1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 106)
    None_169281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 43), 'None')
    # Getting the type of 'None' (line 106)
    None_169282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 58), 'None')
    # Getting the type of 'None' (line 106)
    None_169283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 72), 'None')
    float_169284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 28), 'float')
    float_169285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 37), 'float')
    int_169286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 30), 'int')
    float_169287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'float')
    float_169288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 50), 'float')
    defaults = [None_169281, None_169282, None_169283, float_169284, float_169285, int_169286, float_169287, float_169288]
    # Create a new context for function 'scalar_search_wolfe1'
    module_type_store = module_type_store.open_function_context('scalar_search_wolfe1', 106, 0, False)
    
    # Passed parameters checking function
    scalar_search_wolfe1.stypy_localization = localization
    scalar_search_wolfe1.stypy_type_of_self = None
    scalar_search_wolfe1.stypy_type_store = module_type_store
    scalar_search_wolfe1.stypy_function_name = 'scalar_search_wolfe1'
    scalar_search_wolfe1.stypy_param_names_list = ['phi', 'derphi', 'phi0', 'old_phi0', 'derphi0', 'c1', 'c2', 'amax', 'amin', 'xtol']
    scalar_search_wolfe1.stypy_varargs_param_name = None
    scalar_search_wolfe1.stypy_kwargs_param_name = None
    scalar_search_wolfe1.stypy_call_defaults = defaults
    scalar_search_wolfe1.stypy_call_varargs = varargs
    scalar_search_wolfe1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'scalar_search_wolfe1', ['phi', 'derphi', 'phi0', 'old_phi0', 'derphi0', 'c1', 'c2', 'amax', 'amin', 'xtol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'scalar_search_wolfe1', localization, ['phi', 'derphi', 'phi0', 'old_phi0', 'derphi0', 'c1', 'c2', 'amax', 'amin', 'xtol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'scalar_search_wolfe1(...)' code ##################

    str_169289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'str', '\n    Scalar function search for alpha that satisfies strong Wolfe conditions\n\n    alpha > 0 is assumed to be a descent direction.\n\n    Parameters\n    ----------\n    phi : callable phi(alpha)\n        Function at point `alpha`\n    derphi : callable dphi(alpha)\n        Derivative `d phi(alpha)/ds`. Returns a scalar.\n\n    phi0 : float, optional\n        Value of `f` at 0\n    old_phi0 : float, optional\n        Value of `f` at the previous point\n    derphi0 : float, optional\n        Value `derphi` at 0\n    c1, c2 : float, optional\n        Wolfe parameters\n    amax, amin : float, optional\n        Maximum and minimum step size\n    xtol : float, optional\n        Relative tolerance for an acceptable step.\n\n    Returns\n    -------\n    alpha : float\n        Step size, or None if no suitable step was found\n    phi : float\n        Value of `phi` at the new point `alpha`\n    phi0 : float\n        Value of `phi` at `alpha=0`\n\n    Notes\n    -----\n    Uses routine DCSRCH from MINPACK.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 149)
    # Getting the type of 'phi0' (line 149)
    phi0_169290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 'phi0')
    # Getting the type of 'None' (line 149)
    None_169291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'None')
    
    (may_be_169292, more_types_in_union_169293) = may_be_none(phi0_169290, None_169291)

    if may_be_169292:

        if more_types_in_union_169293:
            # Runtime conditional SSA (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 150):
        
        # Assigning a Call to a Name (line 150):
        
        # Call to phi(...): (line 150)
        # Processing the call arguments (line 150)
        float_169295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 19), 'float')
        # Processing the call keyword arguments (line 150)
        kwargs_169296 = {}
        # Getting the type of 'phi' (line 150)
        phi_169294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'phi', False)
        # Calling phi(args, kwargs) (line 150)
        phi_call_result_169297 = invoke(stypy.reporting.localization.Localization(__file__, 150, 15), phi_169294, *[float_169295], **kwargs_169296)
        
        # Assigning a type to the variable 'phi0' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'phi0', phi_call_result_169297)

        if more_types_in_union_169293:
            # SSA join for if statement (line 149)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 151)
    # Getting the type of 'derphi0' (line 151)
    derphi0_169298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'derphi0')
    # Getting the type of 'None' (line 151)
    None_169299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'None')
    
    (may_be_169300, more_types_in_union_169301) = may_be_none(derphi0_169298, None_169299)

    if may_be_169300:

        if more_types_in_union_169301:
            # Runtime conditional SSA (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to derphi(...): (line 152)
        # Processing the call arguments (line 152)
        float_169303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'float')
        # Processing the call keyword arguments (line 152)
        kwargs_169304 = {}
        # Getting the type of 'derphi' (line 152)
        derphi_169302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'derphi', False)
        # Calling derphi(args, kwargs) (line 152)
        derphi_call_result_169305 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), derphi_169302, *[float_169303], **kwargs_169304)
        
        # Assigning a type to the variable 'derphi0' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'derphi0', derphi_call_result_169305)

        if more_types_in_union_169301:
            # SSA join for if statement (line 151)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'old_phi0' (line 154)
    old_phi0_169306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 7), 'old_phi0')
    # Getting the type of 'None' (line 154)
    None_169307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'None')
    # Applying the binary operator 'isnot' (line 154)
    result_is_not_169308 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 7), 'isnot', old_phi0_169306, None_169307)
    
    
    # Getting the type of 'derphi0' (line 154)
    derphi0_169309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 32), 'derphi0')
    int_169310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 43), 'int')
    # Applying the binary operator '!=' (line 154)
    result_ne_169311 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 32), '!=', derphi0_169309, int_169310)
    
    # Applying the binary operator 'and' (line 154)
    result_and_keyword_169312 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 7), 'and', result_is_not_169308, result_ne_169311)
    
    # Testing the type of an if condition (line 154)
    if_condition_169313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 4), result_and_keyword_169312)
    # Assigning a type to the variable 'if_condition_169313' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'if_condition_169313', if_condition_169313)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 155):
    
    # Assigning a Call to a Name (line 155):
    
    # Call to min(...): (line 155)
    # Processing the call arguments (line 155)
    float_169315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'float')
    float_169316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 26), 'float')
    int_169317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 31), 'int')
    # Applying the binary operator '*' (line 155)
    result_mul_169318 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 26), '*', float_169316, int_169317)
    
    # Getting the type of 'phi0' (line 155)
    phi0_169319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'phi0', False)
    # Getting the type of 'old_phi0' (line 155)
    old_phi0_169320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'old_phi0', False)
    # Applying the binary operator '-' (line 155)
    result_sub_169321 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 34), '-', phi0_169319, old_phi0_169320)
    
    # Applying the binary operator '*' (line 155)
    result_mul_169322 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 32), '*', result_mul_169318, result_sub_169321)
    
    # Getting the type of 'derphi0' (line 155)
    derphi0_169323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 51), 'derphi0', False)
    # Applying the binary operator 'div' (line 155)
    result_div_169324 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 50), 'div', result_mul_169322, derphi0_169323)
    
    # Processing the call keyword arguments (line 155)
    kwargs_169325 = {}
    # Getting the type of 'min' (line 155)
    min_169314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'min', False)
    # Calling min(args, kwargs) (line 155)
    min_call_result_169326 = invoke(stypy.reporting.localization.Localization(__file__, 155, 17), min_169314, *[float_169315, result_div_169324], **kwargs_169325)
    
    # Assigning a type to the variable 'alpha1' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'alpha1', min_call_result_169326)
    
    
    # Getting the type of 'alpha1' (line 156)
    alpha1_169327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'alpha1')
    int_169328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'int')
    # Applying the binary operator '<' (line 156)
    result_lt_169329 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), '<', alpha1_169327, int_169328)
    
    # Testing the type of an if condition (line 156)
    if_condition_169330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_lt_169329)
    # Assigning a type to the variable 'if_condition_169330' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_169330', if_condition_169330)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 157):
    
    # Assigning a Num to a Name (line 157):
    float_169331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 21), 'float')
    # Assigning a type to the variable 'alpha1' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'alpha1', float_169331)
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 154)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 159):
    
    # Assigning a Num to a Name (line 159):
    float_169332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 17), 'float')
    # Assigning a type to the variable 'alpha1' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'alpha1', float_169332)
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 161):
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'phi0' (line 161)
    phi0_169333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'phi0')
    # Assigning a type to the variable 'phi1' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'phi1', phi0_169333)
    
    # Assigning a Name to a Name (line 162):
    
    # Assigning a Name to a Name (line 162):
    # Getting the type of 'derphi0' (line 162)
    derphi0_169334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 14), 'derphi0')
    # Assigning a type to the variable 'derphi1' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'derphi1', derphi0_169334)
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to zeros(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_169337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    int_169338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 22), tuple_169337, int_169338)
    
    # Getting the type of 'np' (line 163)
    np_169339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'np', False)
    # Obtaining the member 'intc' of a type (line 163)
    intc_169340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 27), np_169339, 'intc')
    # Processing the call keyword arguments (line 163)
    kwargs_169341 = {}
    # Getting the type of 'np' (line 163)
    np_169335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 163)
    zeros_169336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), np_169335, 'zeros')
    # Calling zeros(args, kwargs) (line 163)
    zeros_call_result_169342 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), zeros_169336, *[tuple_169337, intc_169340], **kwargs_169341)
    
    # Assigning a type to the variable 'isave' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'isave', zeros_call_result_169342)
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to zeros(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_169345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    int_169346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 22), tuple_169345, int_169346)
    
    # Getting the type of 'float' (line 164)
    float_169347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'float', False)
    # Processing the call keyword arguments (line 164)
    kwargs_169348 = {}
    # Getting the type of 'np' (line 164)
    np_169343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 164)
    zeros_169344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), np_169343, 'zeros')
    # Calling zeros(args, kwargs) (line 164)
    zeros_call_result_169349 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), zeros_169344, *[tuple_169345, float_169347], **kwargs_169348)
    
    # Assigning a type to the variable 'dsave' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'dsave', zeros_call_result_169349)
    
    # Assigning a Str to a Name (line 165):
    
    # Assigning a Str to a Name (line 165):
    str_169350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 11), 'str', 'START')
    # Assigning a type to the variable 'task' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'task', str_169350)
    
    # Assigning a Num to a Name (line 167):
    
    # Assigning a Num to a Name (line 167):
    int_169351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 14), 'int')
    # Assigning a type to the variable 'maxiter' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'maxiter', int_169351)
    
    
    # Call to xrange(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'maxiter' (line 168)
    maxiter_169353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'maxiter', False)
    # Processing the call keyword arguments (line 168)
    kwargs_169354 = {}
    # Getting the type of 'xrange' (line 168)
    xrange_169352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 168)
    xrange_call_result_169355 = invoke(stypy.reporting.localization.Localization(__file__, 168, 13), xrange_169352, *[maxiter_169353], **kwargs_169354)
    
    # Testing the type of a for loop iterable (line 168)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 4), xrange_call_result_169355)
    # Getting the type of the for loop variable (line 168)
    for_loop_var_169356 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 4), xrange_call_result_169355)
    # Assigning a type to the variable 'i' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'i', for_loop_var_169356)
    # SSA begins for a for statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 169):
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_169357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
    
    # Call to dcsrch(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'alpha1' (line 169)
    alpha1_169360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 51), 'alpha1', False)
    # Getting the type of 'phi1' (line 169)
    phi1_169361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 59), 'phi1', False)
    # Getting the type of 'derphi1' (line 169)
    derphi1_169362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 65), 'derphi1', False)
    # Getting the type of 'c1' (line 170)
    c1_169363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 51), 'c1', False)
    # Getting the type of 'c2' (line 170)
    c2_169364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 55), 'c2', False)
    # Getting the type of 'xtol' (line 170)
    xtol_169365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 59), 'xtol', False)
    # Getting the type of 'task' (line 170)
    task_169366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 65), 'task', False)
    # Getting the type of 'amin' (line 171)
    amin_169367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 51), 'amin', False)
    # Getting the type of 'amax' (line 171)
    amax_169368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 57), 'amax', False)
    # Getting the type of 'isave' (line 171)
    isave_169369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 63), 'isave', False)
    # Getting the type of 'dsave' (line 171)
    dsave_169370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 70), 'dsave', False)
    # Processing the call keyword arguments (line 169)
    kwargs_169371 = {}
    # Getting the type of 'minpack2' (line 169)
    minpack2_169358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'minpack2', False)
    # Obtaining the member 'dcsrch' of a type (line 169)
    dcsrch_169359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 35), minpack2_169358, 'dcsrch')
    # Calling dcsrch(args, kwargs) (line 169)
    dcsrch_call_result_169372 = invoke(stypy.reporting.localization.Localization(__file__, 169, 35), dcsrch_169359, *[alpha1_169360, phi1_169361, derphi1_169362, c1_169363, c2_169364, xtol_169365, task_169366, amin_169367, amax_169368, isave_169369, dsave_169370], **kwargs_169371)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___169373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), dcsrch_call_result_169372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_169374 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___169373, int_169357)
    
    # Assigning a type to the variable 'tuple_var_assignment_169038' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_169038', subscript_call_result_169374)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_169375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
    
    # Call to dcsrch(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'alpha1' (line 169)
    alpha1_169378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 51), 'alpha1', False)
    # Getting the type of 'phi1' (line 169)
    phi1_169379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 59), 'phi1', False)
    # Getting the type of 'derphi1' (line 169)
    derphi1_169380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 65), 'derphi1', False)
    # Getting the type of 'c1' (line 170)
    c1_169381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 51), 'c1', False)
    # Getting the type of 'c2' (line 170)
    c2_169382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 55), 'c2', False)
    # Getting the type of 'xtol' (line 170)
    xtol_169383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 59), 'xtol', False)
    # Getting the type of 'task' (line 170)
    task_169384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 65), 'task', False)
    # Getting the type of 'amin' (line 171)
    amin_169385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 51), 'amin', False)
    # Getting the type of 'amax' (line 171)
    amax_169386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 57), 'amax', False)
    # Getting the type of 'isave' (line 171)
    isave_169387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 63), 'isave', False)
    # Getting the type of 'dsave' (line 171)
    dsave_169388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 70), 'dsave', False)
    # Processing the call keyword arguments (line 169)
    kwargs_169389 = {}
    # Getting the type of 'minpack2' (line 169)
    minpack2_169376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'minpack2', False)
    # Obtaining the member 'dcsrch' of a type (line 169)
    dcsrch_169377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 35), minpack2_169376, 'dcsrch')
    # Calling dcsrch(args, kwargs) (line 169)
    dcsrch_call_result_169390 = invoke(stypy.reporting.localization.Localization(__file__, 169, 35), dcsrch_169377, *[alpha1_169378, phi1_169379, derphi1_169380, c1_169381, c2_169382, xtol_169383, task_169384, amin_169385, amax_169386, isave_169387, dsave_169388], **kwargs_169389)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___169391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), dcsrch_call_result_169390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_169392 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___169391, int_169375)
    
    # Assigning a type to the variable 'tuple_var_assignment_169039' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_169039', subscript_call_result_169392)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_169393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
    
    # Call to dcsrch(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'alpha1' (line 169)
    alpha1_169396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 51), 'alpha1', False)
    # Getting the type of 'phi1' (line 169)
    phi1_169397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 59), 'phi1', False)
    # Getting the type of 'derphi1' (line 169)
    derphi1_169398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 65), 'derphi1', False)
    # Getting the type of 'c1' (line 170)
    c1_169399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 51), 'c1', False)
    # Getting the type of 'c2' (line 170)
    c2_169400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 55), 'c2', False)
    # Getting the type of 'xtol' (line 170)
    xtol_169401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 59), 'xtol', False)
    # Getting the type of 'task' (line 170)
    task_169402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 65), 'task', False)
    # Getting the type of 'amin' (line 171)
    amin_169403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 51), 'amin', False)
    # Getting the type of 'amax' (line 171)
    amax_169404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 57), 'amax', False)
    # Getting the type of 'isave' (line 171)
    isave_169405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 63), 'isave', False)
    # Getting the type of 'dsave' (line 171)
    dsave_169406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 70), 'dsave', False)
    # Processing the call keyword arguments (line 169)
    kwargs_169407 = {}
    # Getting the type of 'minpack2' (line 169)
    minpack2_169394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'minpack2', False)
    # Obtaining the member 'dcsrch' of a type (line 169)
    dcsrch_169395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 35), minpack2_169394, 'dcsrch')
    # Calling dcsrch(args, kwargs) (line 169)
    dcsrch_call_result_169408 = invoke(stypy.reporting.localization.Localization(__file__, 169, 35), dcsrch_169395, *[alpha1_169396, phi1_169397, derphi1_169398, c1_169399, c2_169400, xtol_169401, task_169402, amin_169403, amax_169404, isave_169405, dsave_169406], **kwargs_169407)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___169409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), dcsrch_call_result_169408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_169410 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___169409, int_169393)
    
    # Assigning a type to the variable 'tuple_var_assignment_169040' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_169040', subscript_call_result_169410)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_169411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
    
    # Call to dcsrch(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'alpha1' (line 169)
    alpha1_169414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 51), 'alpha1', False)
    # Getting the type of 'phi1' (line 169)
    phi1_169415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 59), 'phi1', False)
    # Getting the type of 'derphi1' (line 169)
    derphi1_169416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 65), 'derphi1', False)
    # Getting the type of 'c1' (line 170)
    c1_169417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 51), 'c1', False)
    # Getting the type of 'c2' (line 170)
    c2_169418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 55), 'c2', False)
    # Getting the type of 'xtol' (line 170)
    xtol_169419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 59), 'xtol', False)
    # Getting the type of 'task' (line 170)
    task_169420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 65), 'task', False)
    # Getting the type of 'amin' (line 171)
    amin_169421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 51), 'amin', False)
    # Getting the type of 'amax' (line 171)
    amax_169422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 57), 'amax', False)
    # Getting the type of 'isave' (line 171)
    isave_169423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 63), 'isave', False)
    # Getting the type of 'dsave' (line 171)
    dsave_169424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 70), 'dsave', False)
    # Processing the call keyword arguments (line 169)
    kwargs_169425 = {}
    # Getting the type of 'minpack2' (line 169)
    minpack2_169412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'minpack2', False)
    # Obtaining the member 'dcsrch' of a type (line 169)
    dcsrch_169413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 35), minpack2_169412, 'dcsrch')
    # Calling dcsrch(args, kwargs) (line 169)
    dcsrch_call_result_169426 = invoke(stypy.reporting.localization.Localization(__file__, 169, 35), dcsrch_169413, *[alpha1_169414, phi1_169415, derphi1_169416, c1_169417, c2_169418, xtol_169419, task_169420, amin_169421, amax_169422, isave_169423, dsave_169424], **kwargs_169425)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___169427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), dcsrch_call_result_169426, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_169428 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___169427, int_169411)
    
    # Assigning a type to the variable 'tuple_var_assignment_169041' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_169041', subscript_call_result_169428)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_169038' (line 169)
    tuple_var_assignment_169038_169429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_169038')
    # Assigning a type to the variable 'stp' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stp', tuple_var_assignment_169038_169429)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_169039' (line 169)
    tuple_var_assignment_169039_169430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_169039')
    # Assigning a type to the variable 'phi1' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'phi1', tuple_var_assignment_169039_169430)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_169040' (line 169)
    tuple_var_assignment_169040_169431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_169040')
    # Assigning a type to the variable 'derphi1' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'derphi1', tuple_var_assignment_169040_169431)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_169041' (line 169)
    tuple_var_assignment_169041_169432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_169041')
    # Assigning a type to the variable 'task' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'task', tuple_var_assignment_169041_169432)
    
    
    
    # Obtaining the type of the subscript
    int_169433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 17), 'int')
    slice_169434 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 172, 11), None, int_169433, None)
    # Getting the type of 'task' (line 172)
    task_169435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'task')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___169436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 11), task_169435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_169437 = invoke(stypy.reporting.localization.Localization(__file__, 172, 11), getitem___169436, slice_169434)
    
    str_169438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 23), 'str', 'FG')
    # Applying the binary operator '==' (line 172)
    result_eq_169439 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), '==', subscript_call_result_169437, str_169438)
    
    # Testing the type of an if condition (line 172)
    if_condition_169440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), result_eq_169439)
    # Assigning a type to the variable 'if_condition_169440' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_169440', if_condition_169440)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 173):
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'stp' (line 173)
    stp_169441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'stp')
    # Assigning a type to the variable 'alpha1' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'alpha1', stp_169441)
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to phi(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'stp' (line 174)
    stp_169443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'stp', False)
    # Processing the call keyword arguments (line 174)
    kwargs_169444 = {}
    # Getting the type of 'phi' (line 174)
    phi_169442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 19), 'phi', False)
    # Calling phi(args, kwargs) (line 174)
    phi_call_result_169445 = invoke(stypy.reporting.localization.Localization(__file__, 174, 19), phi_169442, *[stp_169443], **kwargs_169444)
    
    # Assigning a type to the variable 'phi1' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'phi1', phi_call_result_169445)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to derphi(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'stp' (line 175)
    stp_169447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 29), 'stp', False)
    # Processing the call keyword arguments (line 175)
    kwargs_169448 = {}
    # Getting the type of 'derphi' (line 175)
    derphi_169446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 22), 'derphi', False)
    # Calling derphi(args, kwargs) (line 175)
    derphi_call_result_169449 = invoke(stypy.reporting.localization.Localization(__file__, 175, 22), derphi_169446, *[stp_169447], **kwargs_169448)
    
    # Assigning a type to the variable 'derphi1' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'derphi1', derphi_call_result_169449)
    # SSA branch for the else part of an if statement (line 172)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of a for statement (line 168)
    module_type_store.open_ssa_branch('for loop else')
    
    # Assigning a Name to a Name (line 180):
    
    # Assigning a Name to a Name (line 180):
    # Getting the type of 'None' (line 180)
    None_169450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 14), 'None')
    # Assigning a type to the variable 'stp' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'stp', None_169450)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_169451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 13), 'int')
    slice_169452 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 7), None, int_169451, None)
    # Getting the type of 'task' (line 182)
    task_169453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 7), 'task')
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___169454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 7), task_169453, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_169455 = invoke(stypy.reporting.localization.Localization(__file__, 182, 7), getitem___169454, slice_169452)
    
    str_169456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 19), 'str', 'ERROR')
    # Applying the binary operator '==' (line 182)
    result_eq_169457 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 7), '==', subscript_call_result_169455, str_169456)
    
    
    
    # Obtaining the type of the subscript
    int_169458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'int')
    slice_169459 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 31), None, int_169458, None)
    # Getting the type of 'task' (line 182)
    task_169460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'task')
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___169461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 31), task_169460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_169462 = invoke(stypy.reporting.localization.Localization(__file__, 182, 31), getitem___169461, slice_169459)
    
    str_169463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 43), 'str', 'WARN')
    # Applying the binary operator '==' (line 182)
    result_eq_169464 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 31), '==', subscript_call_result_169462, str_169463)
    
    # Applying the binary operator 'or' (line 182)
    result_or_keyword_169465 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 7), 'or', result_eq_169457, result_eq_169464)
    
    # Testing the type of an if condition (line 182)
    if_condition_169466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 4), result_or_keyword_169465)
    # Assigning a type to the variable 'if_condition_169466' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'if_condition_169466', if_condition_169466)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 183):
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'None' (line 183)
    None_169467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 14), 'None')
    # Assigning a type to the variable 'stp' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stp', None_169467)
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 185)
    tuple_169468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 185)
    # Adding element type (line 185)
    # Getting the type of 'stp' (line 185)
    stp_169469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'stp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 11), tuple_169468, stp_169469)
    # Adding element type (line 185)
    # Getting the type of 'phi1' (line 185)
    phi1_169470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'phi1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 11), tuple_169468, phi1_169470)
    # Adding element type (line 185)
    # Getting the type of 'phi0' (line 185)
    phi0_169471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'phi0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 11), tuple_169468, phi0_169471)
    
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type', tuple_169468)
    
    # ################# End of 'scalar_search_wolfe1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'scalar_search_wolfe1' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_169472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'scalar_search_wolfe1'
    return stypy_return_type_169472

# Assigning a type to the variable 'scalar_search_wolfe1' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'scalar_search_wolfe1', scalar_search_wolfe1)

# Assigning a Name to a Name (line 187):

# Assigning a Name to a Name (line 187):
# Getting the type of 'line_search_wolfe1' (line 187)
line_search_wolfe1_169473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'line_search_wolfe1')
# Assigning a type to the variable 'line_search' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'line_search', line_search_wolfe1_169473)

@norecursion
def line_search_wolfe2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 194)
    None_169474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 48), 'None')
    # Getting the type of 'None' (line 194)
    None_169475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 63), 'None')
    # Getting the type of 'None' (line 195)
    None_169476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 36), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_169477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    
    float_169478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 54), 'float')
    float_169479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 63), 'float')
    # Getting the type of 'None' (line 195)
    None_169480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 73), 'None')
    # Getting the type of 'None' (line 196)
    None_169481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 'None')
    int_169482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 53), 'int')
    defaults = [None_169474, None_169475, None_169476, tuple_169477, float_169478, float_169479, None_169480, None_169481, int_169482]
    # Create a new context for function 'line_search_wolfe2'
    module_type_store = module_type_store.open_function_context('line_search_wolfe2', 194, 0, False)
    
    # Passed parameters checking function
    line_search_wolfe2.stypy_localization = localization
    line_search_wolfe2.stypy_type_of_self = None
    line_search_wolfe2.stypy_type_store = module_type_store
    line_search_wolfe2.stypy_function_name = 'line_search_wolfe2'
    line_search_wolfe2.stypy_param_names_list = ['f', 'myfprime', 'xk', 'pk', 'gfk', 'old_fval', 'old_old_fval', 'args', 'c1', 'c2', 'amax', 'extra_condition', 'maxiter']
    line_search_wolfe2.stypy_varargs_param_name = None
    line_search_wolfe2.stypy_kwargs_param_name = None
    line_search_wolfe2.stypy_call_defaults = defaults
    line_search_wolfe2.stypy_call_varargs = varargs
    line_search_wolfe2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'line_search_wolfe2', ['f', 'myfprime', 'xk', 'pk', 'gfk', 'old_fval', 'old_old_fval', 'args', 'c1', 'c2', 'amax', 'extra_condition', 'maxiter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'line_search_wolfe2', localization, ['f', 'myfprime', 'xk', 'pk', 'gfk', 'old_fval', 'old_old_fval', 'args', 'c1', 'c2', 'amax', 'extra_condition', 'maxiter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'line_search_wolfe2(...)' code ##################

    str_169483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, (-1)), 'str', "Find alpha that satisfies strong Wolfe conditions.\n\n    Parameters\n    ----------\n    f : callable f(x,*args)\n        Objective function.\n    myfprime : callable f'(x,*args)\n        Objective function gradient.\n    xk : ndarray\n        Starting point.\n    pk : ndarray\n        Search direction.\n    gfk : ndarray, optional\n        Gradient value for x=xk (xk being the current parameter\n        estimate). Will be recomputed if omitted.\n    old_fval : float, optional\n        Function value for x=xk. Will be recomputed if omitted.\n    old_old_fval : float, optional\n        Function value for the point preceding x=xk\n    args : tuple, optional\n        Additional arguments passed to objective function.\n    c1 : float, optional\n        Parameter for Armijo condition rule.\n    c2 : float, optional\n        Parameter for curvature condition rule.\n    amax : float, optional\n        Maximum step size\n    extra_condition : callable, optional\n        A callable of the form ``extra_condition(alpha, x, f, g)``\n        returning a boolean. Arguments are the proposed step ``alpha``\n        and the corresponding ``x``, ``f`` and ``g`` values. The line search \n        accepts the value of ``alpha`` only if this \n        callable returns ``True``. If the callable returns ``False`` \n        for the step length, the algorithm will continue with \n        new iterates. The callable is only called for iterates \n        satisfying the strong Wolfe conditions.\n    maxiter : int, optional\n        Maximum number of iterations to perform\n\n    Returns\n    -------\n    alpha : float or None\n        Alpha for which ``x_new = x0 + alpha * pk``,\n        or None if the line search algorithm did not converge.\n    fc : int\n        Number of function evaluations made.\n    gc : int\n        Number of gradient evaluations made.\n    new_fval : float or None\n        New function value ``f(x_new)=f(x0+alpha*pk)``,\n        or None if the line search algorithm did not converge.\n    old_fval : float\n        Old function value ``f(x0)``.\n    new_slope : float or None\n        The local slope along the search direction at the\n        new value ``<myfprime(x_new), pk>``,\n        or None if the line search algorithm did not converge.\n\n\n    Notes\n    -----\n    Uses the line search algorithm to enforce strong Wolfe\n    conditions.  See Wright and Nocedal, 'Numerical Optimization',\n    1999, pg. 59-60.\n\n    For the zoom phase it uses an algorithm by [...].\n\n    ")
    
    # Assigning a List to a Name (line 265):
    
    # Assigning a List to a Name (line 265):
    
    # Obtaining an instance of the builtin type 'list' (line 265)
    list_169484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 265)
    # Adding element type (line 265)
    int_169485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 9), list_169484, int_169485)
    
    # Assigning a type to the variable 'fc' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'fc', list_169484)
    
    # Assigning a List to a Name (line 266):
    
    # Assigning a List to a Name (line 266):
    
    # Obtaining an instance of the builtin type 'list' (line 266)
    list_169486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 266)
    # Adding element type (line 266)
    int_169487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 9), list_169486, int_169487)
    
    # Assigning a type to the variable 'gc' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'gc', list_169486)
    
    # Assigning a List to a Name (line 267):
    
    # Assigning a List to a Name (line 267):
    
    # Obtaining an instance of the builtin type 'list' (line 267)
    list_169488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 267)
    # Adding element type (line 267)
    # Getting the type of 'None' (line 267)
    None_169489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 11), list_169488, None_169489)
    
    # Assigning a type to the variable 'gval' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'gval', list_169488)
    
    # Assigning a List to a Name (line 268):
    
    # Assigning a List to a Name (line 268):
    
    # Obtaining an instance of the builtin type 'list' (line 268)
    list_169490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 268)
    # Adding element type (line 268)
    # Getting the type of 'None' (line 268)
    None_169491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 18), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 17), list_169490, None_169491)
    
    # Assigning a type to the variable 'gval_alpha' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'gval_alpha', list_169490)

    @norecursion
    def phi(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'phi'
        module_type_store = module_type_store.open_function_context('phi', 270, 4, False)
        
        # Passed parameters checking function
        phi.stypy_localization = localization
        phi.stypy_type_of_self = None
        phi.stypy_type_store = module_type_store
        phi.stypy_function_name = 'phi'
        phi.stypy_param_names_list = ['alpha']
        phi.stypy_varargs_param_name = None
        phi.stypy_kwargs_param_name = None
        phi.stypy_call_defaults = defaults
        phi.stypy_call_varargs = varargs
        phi.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'phi', ['alpha'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'phi', localization, ['alpha'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'phi(...)' code ##################

        
        # Getting the type of 'fc' (line 271)
        fc_169492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'fc')
        
        # Obtaining the type of the subscript
        int_169493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 11), 'int')
        # Getting the type of 'fc' (line 271)
        fc_169494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'fc')
        # Obtaining the member '__getitem__' of a type (line 271)
        getitem___169495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), fc_169494, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 271)
        subscript_call_result_169496 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), getitem___169495, int_169493)
        
        int_169497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 17), 'int')
        # Applying the binary operator '+=' (line 271)
        result_iadd_169498 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 8), '+=', subscript_call_result_169496, int_169497)
        # Getting the type of 'fc' (line 271)
        fc_169499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'fc')
        int_169500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 11), 'int')
        # Storing an element on a container (line 271)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 8), fc_169499, (int_169500, result_iadd_169498))
        
        
        # Call to f(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'xk' (line 272)
        xk_169502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'xk', False)
        # Getting the type of 'alpha' (line 272)
        alpha_169503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'alpha', False)
        # Getting the type of 'pk' (line 272)
        pk_169504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 30), 'pk', False)
        # Applying the binary operator '*' (line 272)
        result_mul_169505 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 22), '*', alpha_169503, pk_169504)
        
        # Applying the binary operator '+' (line 272)
        result_add_169506 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 17), '+', xk_169502, result_mul_169505)
        
        # Getting the type of 'args' (line 272)
        args_169507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), 'args', False)
        # Processing the call keyword arguments (line 272)
        kwargs_169508 = {}
        # Getting the type of 'f' (line 272)
        f_169501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'f', False)
        # Calling f(args, kwargs) (line 272)
        f_call_result_169509 = invoke(stypy.reporting.localization.Localization(__file__, 272, 15), f_169501, *[result_add_169506, args_169507], **kwargs_169508)
        
        # Assigning a type to the variable 'stypy_return_type' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'stypy_return_type', f_call_result_169509)
        
        # ################# End of 'phi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'phi' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_169510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_169510)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'phi'
        return stypy_return_type_169510

    # Assigning a type to the variable 'phi' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'phi', phi)
    
    # Type idiom detected: calculating its left and rigth part (line 274)
    # Getting the type of 'tuple' (line 274)
    tuple_169511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 28), 'tuple')
    # Getting the type of 'myfprime' (line 274)
    myfprime_169512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 18), 'myfprime')
    
    (may_be_169513, more_types_in_union_169514) = may_be_subtype(tuple_169511, myfprime_169512)

    if may_be_169513:

        if more_types_in_union_169514:
            # Runtime conditional SSA (line 274)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'myfprime' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'myfprime', remove_not_subtype_from_union(myfprime_169512, tuple))

        @norecursion
        def derphi(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'derphi'
            module_type_store = module_type_store.open_function_context('derphi', 275, 8, False)
            
            # Passed parameters checking function
            derphi.stypy_localization = localization
            derphi.stypy_type_of_self = None
            derphi.stypy_type_store = module_type_store
            derphi.stypy_function_name = 'derphi'
            derphi.stypy_param_names_list = ['alpha']
            derphi.stypy_varargs_param_name = None
            derphi.stypy_kwargs_param_name = None
            derphi.stypy_call_defaults = defaults
            derphi.stypy_call_varargs = varargs
            derphi.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'derphi', ['alpha'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'derphi', localization, ['alpha'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'derphi(...)' code ##################

            
            # Getting the type of 'fc' (line 276)
            fc_169515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'fc')
            
            # Obtaining the type of the subscript
            int_169516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 15), 'int')
            # Getting the type of 'fc' (line 276)
            fc_169517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'fc')
            # Obtaining the member '__getitem__' of a type (line 276)
            getitem___169518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), fc_169517, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 276)
            subscript_call_result_169519 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), getitem___169518, int_169516)
            
            
            # Call to len(...): (line 276)
            # Processing the call arguments (line 276)
            # Getting the type of 'xk' (line 276)
            xk_169521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 25), 'xk', False)
            # Processing the call keyword arguments (line 276)
            kwargs_169522 = {}
            # Getting the type of 'len' (line 276)
            len_169520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 21), 'len', False)
            # Calling len(args, kwargs) (line 276)
            len_call_result_169523 = invoke(stypy.reporting.localization.Localization(__file__, 276, 21), len_169520, *[xk_169521], **kwargs_169522)
            
            int_169524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 31), 'int')
            # Applying the binary operator '+' (line 276)
            result_add_169525 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 21), '+', len_call_result_169523, int_169524)
            
            # Applying the binary operator '+=' (line 276)
            result_iadd_169526 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 12), '+=', subscript_call_result_169519, result_add_169525)
            # Getting the type of 'fc' (line 276)
            fc_169527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'fc')
            int_169528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 15), 'int')
            # Storing an element on a container (line 276)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 12), fc_169527, (int_169528, result_iadd_169526))
            
            
            # Assigning a Subscript to a Name (line 277):
            
            # Assigning a Subscript to a Name (line 277):
            
            # Obtaining the type of the subscript
            int_169529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 27), 'int')
            # Getting the type of 'myfprime' (line 277)
            myfprime_169530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'myfprime')
            # Obtaining the member '__getitem__' of a type (line 277)
            getitem___169531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 18), myfprime_169530, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 277)
            subscript_call_result_169532 = invoke(stypy.reporting.localization.Localization(__file__, 277, 18), getitem___169531, int_169529)
            
            # Assigning a type to the variable 'eps' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'eps', subscript_call_result_169532)
            
            # Assigning a Subscript to a Name (line 278):
            
            # Assigning a Subscript to a Name (line 278):
            
            # Obtaining the type of the subscript
            int_169533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 30), 'int')
            # Getting the type of 'myfprime' (line 278)
            myfprime_169534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'myfprime')
            # Obtaining the member '__getitem__' of a type (line 278)
            getitem___169535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 21), myfprime_169534, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 278)
            subscript_call_result_169536 = invoke(stypy.reporting.localization.Localization(__file__, 278, 21), getitem___169535, int_169533)
            
            # Assigning a type to the variable 'fprime' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'fprime', subscript_call_result_169536)
            
            # Assigning a BinOp to a Name (line 279):
            
            # Assigning a BinOp to a Name (line 279):
            
            # Obtaining an instance of the builtin type 'tuple' (line 279)
            tuple_169537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 279)
            # Adding element type (line 279)
            # Getting the type of 'f' (line 279)
            f_169538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 23), 'f')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 23), tuple_169537, f_169538)
            # Adding element type (line 279)
            # Getting the type of 'eps' (line 279)
            eps_169539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 26), 'eps')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 23), tuple_169537, eps_169539)
            
            # Getting the type of 'args' (line 279)
            args_169540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 33), 'args')
            # Applying the binary operator '+' (line 279)
            result_add_169541 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 22), '+', tuple_169537, args_169540)
            
            # Assigning a type to the variable 'newargs' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'newargs', result_add_169541)
            
            # Assigning a Call to a Subscript (line 280):
            
            # Assigning a Call to a Subscript (line 280):
            
            # Call to fprime(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'xk' (line 280)
            xk_169543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'xk', False)
            # Getting the type of 'alpha' (line 280)
            alpha_169544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 34), 'alpha', False)
            # Getting the type of 'pk' (line 280)
            pk_169545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 42), 'pk', False)
            # Applying the binary operator '*' (line 280)
            result_mul_169546 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 34), '*', alpha_169544, pk_169545)
            
            # Applying the binary operator '+' (line 280)
            result_add_169547 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 29), '+', xk_169543, result_mul_169546)
            
            # Getting the type of 'newargs' (line 280)
            newargs_169548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 47), 'newargs', False)
            # Processing the call keyword arguments (line 280)
            kwargs_169549 = {}
            # Getting the type of 'fprime' (line 280)
            fprime_169542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 22), 'fprime', False)
            # Calling fprime(args, kwargs) (line 280)
            fprime_call_result_169550 = invoke(stypy.reporting.localization.Localization(__file__, 280, 22), fprime_169542, *[result_add_169547, newargs_169548], **kwargs_169549)
            
            # Getting the type of 'gval' (line 280)
            gval_169551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'gval')
            int_169552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 17), 'int')
            # Storing an element on a container (line 280)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 12), gval_169551, (int_169552, fprime_call_result_169550))
            
            # Assigning a Name to a Subscript (line 281):
            
            # Assigning a Name to a Subscript (line 281):
            # Getting the type of 'alpha' (line 281)
            alpha_169553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'alpha')
            # Getting the type of 'gval_alpha' (line 281)
            gval_alpha_169554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'gval_alpha')
            int_169555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 23), 'int')
            # Storing an element on a container (line 281)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 12), gval_alpha_169554, (int_169555, alpha_169553))
            
            # Call to dot(...): (line 282)
            # Processing the call arguments (line 282)
            
            # Obtaining the type of the subscript
            int_169558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 31), 'int')
            # Getting the type of 'gval' (line 282)
            gval_169559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'gval', False)
            # Obtaining the member '__getitem__' of a type (line 282)
            getitem___169560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 26), gval_169559, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 282)
            subscript_call_result_169561 = invoke(stypy.reporting.localization.Localization(__file__, 282, 26), getitem___169560, int_169558)
            
            # Getting the type of 'pk' (line 282)
            pk_169562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 35), 'pk', False)
            # Processing the call keyword arguments (line 282)
            kwargs_169563 = {}
            # Getting the type of 'np' (line 282)
            np_169556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'np', False)
            # Obtaining the member 'dot' of a type (line 282)
            dot_169557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 19), np_169556, 'dot')
            # Calling dot(args, kwargs) (line 282)
            dot_call_result_169564 = invoke(stypy.reporting.localization.Localization(__file__, 282, 19), dot_169557, *[subscript_call_result_169561, pk_169562], **kwargs_169563)
            
            # Assigning a type to the variable 'stypy_return_type' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', dot_call_result_169564)
            
            # ################# End of 'derphi(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'derphi' in the type store
            # Getting the type of 'stypy_return_type' (line 275)
            stypy_return_type_169565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_169565)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'derphi'
            return stypy_return_type_169565

        # Assigning a type to the variable 'derphi' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'derphi', derphi)

        if more_types_in_union_169514:
            # Runtime conditional SSA for else branch (line 274)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_169513) or more_types_in_union_169514):
        # Assigning a type to the variable 'myfprime' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'myfprime', remove_subtype_from_union(myfprime_169512, tuple))
        
        # Assigning a Name to a Name (line 284):
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'myfprime' (line 284)
        myfprime_169566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 17), 'myfprime')
        # Assigning a type to the variable 'fprime' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'fprime', myfprime_169566)

        @norecursion
        def derphi(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'derphi'
            module_type_store = module_type_store.open_function_context('derphi', 286, 8, False)
            
            # Passed parameters checking function
            derphi.stypy_localization = localization
            derphi.stypy_type_of_self = None
            derphi.stypy_type_store = module_type_store
            derphi.stypy_function_name = 'derphi'
            derphi.stypy_param_names_list = ['alpha']
            derphi.stypy_varargs_param_name = None
            derphi.stypy_kwargs_param_name = None
            derphi.stypy_call_defaults = defaults
            derphi.stypy_call_varargs = varargs
            derphi.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'derphi', ['alpha'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'derphi', localization, ['alpha'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'derphi(...)' code ##################

            
            # Getting the type of 'gc' (line 287)
            gc_169567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'gc')
            
            # Obtaining the type of the subscript
            int_169568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 15), 'int')
            # Getting the type of 'gc' (line 287)
            gc_169569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'gc')
            # Obtaining the member '__getitem__' of a type (line 287)
            getitem___169570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), gc_169569, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 287)
            subscript_call_result_169571 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), getitem___169570, int_169568)
            
            int_169572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 21), 'int')
            # Applying the binary operator '+=' (line 287)
            result_iadd_169573 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 12), '+=', subscript_call_result_169571, int_169572)
            # Getting the type of 'gc' (line 287)
            gc_169574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'gc')
            int_169575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 15), 'int')
            # Storing an element on a container (line 287)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 12), gc_169574, (int_169575, result_iadd_169573))
            
            
            # Assigning a Call to a Subscript (line 288):
            
            # Assigning a Call to a Subscript (line 288):
            
            # Call to fprime(...): (line 288)
            # Processing the call arguments (line 288)
            # Getting the type of 'xk' (line 288)
            xk_169577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 29), 'xk', False)
            # Getting the type of 'alpha' (line 288)
            alpha_169578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 34), 'alpha', False)
            # Getting the type of 'pk' (line 288)
            pk_169579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 42), 'pk', False)
            # Applying the binary operator '*' (line 288)
            result_mul_169580 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 34), '*', alpha_169578, pk_169579)
            
            # Applying the binary operator '+' (line 288)
            result_add_169581 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 29), '+', xk_169577, result_mul_169580)
            
            # Getting the type of 'args' (line 288)
            args_169582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 47), 'args', False)
            # Processing the call keyword arguments (line 288)
            kwargs_169583 = {}
            # Getting the type of 'fprime' (line 288)
            fprime_169576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'fprime', False)
            # Calling fprime(args, kwargs) (line 288)
            fprime_call_result_169584 = invoke(stypy.reporting.localization.Localization(__file__, 288, 22), fprime_169576, *[result_add_169581, args_169582], **kwargs_169583)
            
            # Getting the type of 'gval' (line 288)
            gval_169585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'gval')
            int_169586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 17), 'int')
            # Storing an element on a container (line 288)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 12), gval_169585, (int_169586, fprime_call_result_169584))
            
            # Assigning a Name to a Subscript (line 289):
            
            # Assigning a Name to a Subscript (line 289):
            # Getting the type of 'alpha' (line 289)
            alpha_169587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 28), 'alpha')
            # Getting the type of 'gval_alpha' (line 289)
            gval_alpha_169588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'gval_alpha')
            int_169589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 23), 'int')
            # Storing an element on a container (line 289)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 12), gval_alpha_169588, (int_169589, alpha_169587))
            
            # Call to dot(...): (line 290)
            # Processing the call arguments (line 290)
            
            # Obtaining the type of the subscript
            int_169592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 31), 'int')
            # Getting the type of 'gval' (line 290)
            gval_169593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 26), 'gval', False)
            # Obtaining the member '__getitem__' of a type (line 290)
            getitem___169594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 26), gval_169593, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 290)
            subscript_call_result_169595 = invoke(stypy.reporting.localization.Localization(__file__, 290, 26), getitem___169594, int_169592)
            
            # Getting the type of 'pk' (line 290)
            pk_169596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 35), 'pk', False)
            # Processing the call keyword arguments (line 290)
            kwargs_169597 = {}
            # Getting the type of 'np' (line 290)
            np_169590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'np', False)
            # Obtaining the member 'dot' of a type (line 290)
            dot_169591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 19), np_169590, 'dot')
            # Calling dot(args, kwargs) (line 290)
            dot_call_result_169598 = invoke(stypy.reporting.localization.Localization(__file__, 290, 19), dot_169591, *[subscript_call_result_169595, pk_169596], **kwargs_169597)
            
            # Assigning a type to the variable 'stypy_return_type' (line 290)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'stypy_return_type', dot_call_result_169598)
            
            # ################# End of 'derphi(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'derphi' in the type store
            # Getting the type of 'stypy_return_type' (line 286)
            stypy_return_type_169599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_169599)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'derphi'
            return stypy_return_type_169599

        # Assigning a type to the variable 'derphi' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'derphi', derphi)

        if (may_be_169513 and more_types_in_union_169514):
            # SSA join for if statement (line 274)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 292)
    # Getting the type of 'gfk' (line 292)
    gfk_169600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 7), 'gfk')
    # Getting the type of 'None' (line 292)
    None_169601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'None')
    
    (may_be_169602, more_types_in_union_169603) = may_be_none(gfk_169600, None_169601)

    if may_be_169602:

        if more_types_in_union_169603:
            # Runtime conditional SSA (line 292)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to fprime(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'xk' (line 293)
        xk_169605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 21), 'xk', False)
        # Getting the type of 'args' (line 293)
        args_169606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'args', False)
        # Processing the call keyword arguments (line 293)
        kwargs_169607 = {}
        # Getting the type of 'fprime' (line 293)
        fprime_169604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 14), 'fprime', False)
        # Calling fprime(args, kwargs) (line 293)
        fprime_call_result_169608 = invoke(stypy.reporting.localization.Localization(__file__, 293, 14), fprime_169604, *[xk_169605, args_169606], **kwargs_169607)
        
        # Assigning a type to the variable 'gfk' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'gfk', fprime_call_result_169608)

        if more_types_in_union_169603:
            # SSA join for if statement (line 292)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 294):
    
    # Assigning a Call to a Name (line 294):
    
    # Call to dot(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'gfk' (line 294)
    gfk_169611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'gfk', False)
    # Getting the type of 'pk' (line 294)
    pk_169612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'pk', False)
    # Processing the call keyword arguments (line 294)
    kwargs_169613 = {}
    # Getting the type of 'np' (line 294)
    np_169609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 294)
    dot_169610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 14), np_169609, 'dot')
    # Calling dot(args, kwargs) (line 294)
    dot_call_result_169614 = invoke(stypy.reporting.localization.Localization(__file__, 294, 14), dot_169610, *[gfk_169611, pk_169612], **kwargs_169613)
    
    # Assigning a type to the variable 'derphi0' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'derphi0', dot_call_result_169614)
    
    # Type idiom detected: calculating its left and rigth part (line 296)
    # Getting the type of 'extra_condition' (line 296)
    extra_condition_169615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'extra_condition')
    # Getting the type of 'None' (line 296)
    None_169616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 30), 'None')
    
    (may_be_169617, more_types_in_union_169618) = may_not_be_none(extra_condition_169615, None_169616)

    if may_be_169617:

        if more_types_in_union_169618:
            # Runtime conditional SSA (line 296)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def extra_condition2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'extra_condition2'
            module_type_store = module_type_store.open_function_context('extra_condition2', 299, 8, False)
            
            # Passed parameters checking function
            extra_condition2.stypy_localization = localization
            extra_condition2.stypy_type_of_self = None
            extra_condition2.stypy_type_store = module_type_store
            extra_condition2.stypy_function_name = 'extra_condition2'
            extra_condition2.stypy_param_names_list = ['alpha', 'phi']
            extra_condition2.stypy_varargs_param_name = None
            extra_condition2.stypy_kwargs_param_name = None
            extra_condition2.stypy_call_defaults = defaults
            extra_condition2.stypy_call_varargs = varargs
            extra_condition2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'extra_condition2', ['alpha', 'phi'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'extra_condition2', localization, ['alpha', 'phi'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'extra_condition2(...)' code ##################

            
            
            
            # Obtaining the type of the subscript
            int_169619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 26), 'int')
            # Getting the type of 'gval_alpha' (line 300)
            gval_alpha_169620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'gval_alpha')
            # Obtaining the member '__getitem__' of a type (line 300)
            getitem___169621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 15), gval_alpha_169620, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 300)
            subscript_call_result_169622 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), getitem___169621, int_169619)
            
            # Getting the type of 'alpha' (line 300)
            alpha_169623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 32), 'alpha')
            # Applying the binary operator '!=' (line 300)
            result_ne_169624 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 15), '!=', subscript_call_result_169622, alpha_169623)
            
            # Testing the type of an if condition (line 300)
            if_condition_169625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 12), result_ne_169624)
            # Assigning a type to the variable 'if_condition_169625' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'if_condition_169625', if_condition_169625)
            # SSA begins for if statement (line 300)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to derphi(...): (line 301)
            # Processing the call arguments (line 301)
            # Getting the type of 'alpha' (line 301)
            alpha_169627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'alpha', False)
            # Processing the call keyword arguments (line 301)
            kwargs_169628 = {}
            # Getting the type of 'derphi' (line 301)
            derphi_169626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'derphi', False)
            # Calling derphi(args, kwargs) (line 301)
            derphi_call_result_169629 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), derphi_169626, *[alpha_169627], **kwargs_169628)
            
            # SSA join for if statement (line 300)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 302):
            
            # Assigning a BinOp to a Name (line 302):
            # Getting the type of 'xk' (line 302)
            xk_169630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'xk')
            # Getting the type of 'alpha' (line 302)
            alpha_169631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 21), 'alpha')
            # Getting the type of 'pk' (line 302)
            pk_169632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 29), 'pk')
            # Applying the binary operator '*' (line 302)
            result_mul_169633 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 21), '*', alpha_169631, pk_169632)
            
            # Applying the binary operator '+' (line 302)
            result_add_169634 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 16), '+', xk_169630, result_mul_169633)
            
            # Assigning a type to the variable 'x' (line 302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'x', result_add_169634)
            
            # Call to extra_condition(...): (line 303)
            # Processing the call arguments (line 303)
            # Getting the type of 'alpha' (line 303)
            alpha_169636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 35), 'alpha', False)
            # Getting the type of 'x' (line 303)
            x_169637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 42), 'x', False)
            # Getting the type of 'phi' (line 303)
            phi_169638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 45), 'phi', False)
            
            # Obtaining the type of the subscript
            int_169639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 55), 'int')
            # Getting the type of 'gval' (line 303)
            gval_169640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 50), 'gval', False)
            # Obtaining the member '__getitem__' of a type (line 303)
            getitem___169641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 50), gval_169640, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 303)
            subscript_call_result_169642 = invoke(stypy.reporting.localization.Localization(__file__, 303, 50), getitem___169641, int_169639)
            
            # Processing the call keyword arguments (line 303)
            kwargs_169643 = {}
            # Getting the type of 'extra_condition' (line 303)
            extra_condition_169635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'extra_condition', False)
            # Calling extra_condition(args, kwargs) (line 303)
            extra_condition_call_result_169644 = invoke(stypy.reporting.localization.Localization(__file__, 303, 19), extra_condition_169635, *[alpha_169636, x_169637, phi_169638, subscript_call_result_169642], **kwargs_169643)
            
            # Assigning a type to the variable 'stypy_return_type' (line 303)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'stypy_return_type', extra_condition_call_result_169644)
            
            # ################# End of 'extra_condition2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'extra_condition2' in the type store
            # Getting the type of 'stypy_return_type' (line 299)
            stypy_return_type_169645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_169645)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'extra_condition2'
            return stypy_return_type_169645

        # Assigning a type to the variable 'extra_condition2' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'extra_condition2', extra_condition2)

        if more_types_in_union_169618:
            # Runtime conditional SSA for else branch (line 296)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_169617) or more_types_in_union_169618):
        
        # Assigning a Name to a Name (line 305):
        
        # Assigning a Name to a Name (line 305):
        # Getting the type of 'None' (line 305)
        None_169646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 27), 'None')
        # Assigning a type to the variable 'extra_condition2' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'extra_condition2', None_169646)

        if (may_be_169617 and more_types_in_union_169618):
            # SSA join for if statement (line 296)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 307):
    
    # Assigning a Subscript to a Name (line 307):
    
    # Obtaining the type of the subscript
    int_169647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 4), 'int')
    
    # Call to scalar_search_wolfe2(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'phi' (line 308)
    phi_169649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'phi', False)
    # Getting the type of 'derphi' (line 308)
    derphi_169650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 'derphi', False)
    # Getting the type of 'old_fval' (line 308)
    old_fval_169651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'old_fval', False)
    # Getting the type of 'old_old_fval' (line 308)
    old_old_fval_169652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 35), 'old_old_fval', False)
    # Getting the type of 'derphi0' (line 308)
    derphi0_169653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 49), 'derphi0', False)
    # Getting the type of 'c1' (line 308)
    c1_169654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 58), 'c1', False)
    # Getting the type of 'c2' (line 308)
    c2_169655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 62), 'c2', False)
    # Getting the type of 'amax' (line 308)
    amax_169656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 66), 'amax', False)
    # Getting the type of 'extra_condition2' (line 309)
    extra_condition2_169657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'extra_condition2', False)
    # Processing the call keyword arguments (line 307)
    # Getting the type of 'maxiter' (line 309)
    maxiter_169658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 38), 'maxiter', False)
    keyword_169659 = maxiter_169658
    kwargs_169660 = {'maxiter': keyword_169659}
    # Getting the type of 'scalar_search_wolfe2' (line 307)
    scalar_search_wolfe2_169648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 50), 'scalar_search_wolfe2', False)
    # Calling scalar_search_wolfe2(args, kwargs) (line 307)
    scalar_search_wolfe2_call_result_169661 = invoke(stypy.reporting.localization.Localization(__file__, 307, 50), scalar_search_wolfe2_169648, *[phi_169649, derphi_169650, old_fval_169651, old_old_fval_169652, derphi0_169653, c1_169654, c2_169655, amax_169656, extra_condition2_169657], **kwargs_169660)
    
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___169662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 4), scalar_search_wolfe2_call_result_169661, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_169663 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), getitem___169662, int_169647)
    
    # Assigning a type to the variable 'tuple_var_assignment_169042' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'tuple_var_assignment_169042', subscript_call_result_169663)
    
    # Assigning a Subscript to a Name (line 307):
    
    # Obtaining the type of the subscript
    int_169664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 4), 'int')
    
    # Call to scalar_search_wolfe2(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'phi' (line 308)
    phi_169666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'phi', False)
    # Getting the type of 'derphi' (line 308)
    derphi_169667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 'derphi', False)
    # Getting the type of 'old_fval' (line 308)
    old_fval_169668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'old_fval', False)
    # Getting the type of 'old_old_fval' (line 308)
    old_old_fval_169669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 35), 'old_old_fval', False)
    # Getting the type of 'derphi0' (line 308)
    derphi0_169670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 49), 'derphi0', False)
    # Getting the type of 'c1' (line 308)
    c1_169671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 58), 'c1', False)
    # Getting the type of 'c2' (line 308)
    c2_169672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 62), 'c2', False)
    # Getting the type of 'amax' (line 308)
    amax_169673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 66), 'amax', False)
    # Getting the type of 'extra_condition2' (line 309)
    extra_condition2_169674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'extra_condition2', False)
    # Processing the call keyword arguments (line 307)
    # Getting the type of 'maxiter' (line 309)
    maxiter_169675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 38), 'maxiter', False)
    keyword_169676 = maxiter_169675
    kwargs_169677 = {'maxiter': keyword_169676}
    # Getting the type of 'scalar_search_wolfe2' (line 307)
    scalar_search_wolfe2_169665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 50), 'scalar_search_wolfe2', False)
    # Calling scalar_search_wolfe2(args, kwargs) (line 307)
    scalar_search_wolfe2_call_result_169678 = invoke(stypy.reporting.localization.Localization(__file__, 307, 50), scalar_search_wolfe2_169665, *[phi_169666, derphi_169667, old_fval_169668, old_old_fval_169669, derphi0_169670, c1_169671, c2_169672, amax_169673, extra_condition2_169674], **kwargs_169677)
    
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___169679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 4), scalar_search_wolfe2_call_result_169678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_169680 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), getitem___169679, int_169664)
    
    # Assigning a type to the variable 'tuple_var_assignment_169043' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'tuple_var_assignment_169043', subscript_call_result_169680)
    
    # Assigning a Subscript to a Name (line 307):
    
    # Obtaining the type of the subscript
    int_169681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 4), 'int')
    
    # Call to scalar_search_wolfe2(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'phi' (line 308)
    phi_169683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'phi', False)
    # Getting the type of 'derphi' (line 308)
    derphi_169684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 'derphi', False)
    # Getting the type of 'old_fval' (line 308)
    old_fval_169685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'old_fval', False)
    # Getting the type of 'old_old_fval' (line 308)
    old_old_fval_169686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 35), 'old_old_fval', False)
    # Getting the type of 'derphi0' (line 308)
    derphi0_169687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 49), 'derphi0', False)
    # Getting the type of 'c1' (line 308)
    c1_169688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 58), 'c1', False)
    # Getting the type of 'c2' (line 308)
    c2_169689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 62), 'c2', False)
    # Getting the type of 'amax' (line 308)
    amax_169690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 66), 'amax', False)
    # Getting the type of 'extra_condition2' (line 309)
    extra_condition2_169691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'extra_condition2', False)
    # Processing the call keyword arguments (line 307)
    # Getting the type of 'maxiter' (line 309)
    maxiter_169692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 38), 'maxiter', False)
    keyword_169693 = maxiter_169692
    kwargs_169694 = {'maxiter': keyword_169693}
    # Getting the type of 'scalar_search_wolfe2' (line 307)
    scalar_search_wolfe2_169682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 50), 'scalar_search_wolfe2', False)
    # Calling scalar_search_wolfe2(args, kwargs) (line 307)
    scalar_search_wolfe2_call_result_169695 = invoke(stypy.reporting.localization.Localization(__file__, 307, 50), scalar_search_wolfe2_169682, *[phi_169683, derphi_169684, old_fval_169685, old_old_fval_169686, derphi0_169687, c1_169688, c2_169689, amax_169690, extra_condition2_169691], **kwargs_169694)
    
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___169696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 4), scalar_search_wolfe2_call_result_169695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_169697 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), getitem___169696, int_169681)
    
    # Assigning a type to the variable 'tuple_var_assignment_169044' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'tuple_var_assignment_169044', subscript_call_result_169697)
    
    # Assigning a Subscript to a Name (line 307):
    
    # Obtaining the type of the subscript
    int_169698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 4), 'int')
    
    # Call to scalar_search_wolfe2(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'phi' (line 308)
    phi_169700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'phi', False)
    # Getting the type of 'derphi' (line 308)
    derphi_169701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 'derphi', False)
    # Getting the type of 'old_fval' (line 308)
    old_fval_169702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'old_fval', False)
    # Getting the type of 'old_old_fval' (line 308)
    old_old_fval_169703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 35), 'old_old_fval', False)
    # Getting the type of 'derphi0' (line 308)
    derphi0_169704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 49), 'derphi0', False)
    # Getting the type of 'c1' (line 308)
    c1_169705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 58), 'c1', False)
    # Getting the type of 'c2' (line 308)
    c2_169706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 62), 'c2', False)
    # Getting the type of 'amax' (line 308)
    amax_169707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 66), 'amax', False)
    # Getting the type of 'extra_condition2' (line 309)
    extra_condition2_169708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'extra_condition2', False)
    # Processing the call keyword arguments (line 307)
    # Getting the type of 'maxiter' (line 309)
    maxiter_169709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 38), 'maxiter', False)
    keyword_169710 = maxiter_169709
    kwargs_169711 = {'maxiter': keyword_169710}
    # Getting the type of 'scalar_search_wolfe2' (line 307)
    scalar_search_wolfe2_169699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 50), 'scalar_search_wolfe2', False)
    # Calling scalar_search_wolfe2(args, kwargs) (line 307)
    scalar_search_wolfe2_call_result_169712 = invoke(stypy.reporting.localization.Localization(__file__, 307, 50), scalar_search_wolfe2_169699, *[phi_169700, derphi_169701, old_fval_169702, old_old_fval_169703, derphi0_169704, c1_169705, c2_169706, amax_169707, extra_condition2_169708], **kwargs_169711)
    
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___169713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 4), scalar_search_wolfe2_call_result_169712, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_169714 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), getitem___169713, int_169698)
    
    # Assigning a type to the variable 'tuple_var_assignment_169045' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'tuple_var_assignment_169045', subscript_call_result_169714)
    
    # Assigning a Name to a Name (line 307):
    # Getting the type of 'tuple_var_assignment_169042' (line 307)
    tuple_var_assignment_169042_169715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'tuple_var_assignment_169042')
    # Assigning a type to the variable 'alpha_star' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'alpha_star', tuple_var_assignment_169042_169715)
    
    # Assigning a Name to a Name (line 307):
    # Getting the type of 'tuple_var_assignment_169043' (line 307)
    tuple_var_assignment_169043_169716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'tuple_var_assignment_169043')
    # Assigning a type to the variable 'phi_star' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'phi_star', tuple_var_assignment_169043_169716)
    
    # Assigning a Name to a Name (line 307):
    # Getting the type of 'tuple_var_assignment_169044' (line 307)
    tuple_var_assignment_169044_169717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'tuple_var_assignment_169044')
    # Assigning a type to the variable 'old_fval' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 26), 'old_fval', tuple_var_assignment_169044_169717)
    
    # Assigning a Name to a Name (line 307):
    # Getting the type of 'tuple_var_assignment_169045' (line 307)
    tuple_var_assignment_169045_169718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'tuple_var_assignment_169045')
    # Assigning a type to the variable 'derphi_star' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 36), 'derphi_star', tuple_var_assignment_169045_169718)
    
    # Type idiom detected: calculating its left and rigth part (line 311)
    # Getting the type of 'derphi_star' (line 311)
    derphi_star_169719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 7), 'derphi_star')
    # Getting the type of 'None' (line 311)
    None_169720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 22), 'None')
    
    (may_be_169721, more_types_in_union_169722) = may_be_none(derphi_star_169719, None_169720)

    if may_be_169721:

        if more_types_in_union_169722:
            # Runtime conditional SSA (line 311)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to warn(...): (line 312)
        # Processing the call arguments (line 312)
        str_169724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 13), 'str', 'The line search algorithm did not converge')
        # Getting the type of 'LineSearchWarning' (line 312)
        LineSearchWarning_169725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 59), 'LineSearchWarning', False)
        # Processing the call keyword arguments (line 312)
        kwargs_169726 = {}
        # Getting the type of 'warn' (line 312)
        warn_169723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'warn', False)
        # Calling warn(args, kwargs) (line 312)
        warn_call_result_169727 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), warn_169723, *[str_169724, LineSearchWarning_169725], **kwargs_169726)
        

        if more_types_in_union_169722:
            # Runtime conditional SSA for else branch (line 311)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_169721) or more_types_in_union_169722):
        
        # Assigning a Subscript to a Name (line 318):
        
        # Assigning a Subscript to a Name (line 318):
        
        # Obtaining the type of the subscript
        int_169728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'int')
        # Getting the type of 'gval' (line 318)
        gval_169729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 22), 'gval')
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___169730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 22), gval_169729, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_169731 = invoke(stypy.reporting.localization.Localization(__file__, 318, 22), getitem___169730, int_169728)
        
        # Assigning a type to the variable 'derphi_star' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'derphi_star', subscript_call_result_169731)

        if (may_be_169721 and more_types_in_union_169722):
            # SSA join for if statement (line 311)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 320)
    tuple_169732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 320)
    # Adding element type (line 320)
    # Getting the type of 'alpha_star' (line 320)
    alpha_star_169733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), 'alpha_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 11), tuple_169732, alpha_star_169733)
    # Adding element type (line 320)
    
    # Obtaining the type of the subscript
    int_169734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 26), 'int')
    # Getting the type of 'fc' (line 320)
    fc_169735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'fc')
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___169736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 23), fc_169735, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 320)
    subscript_call_result_169737 = invoke(stypy.reporting.localization.Localization(__file__, 320, 23), getitem___169736, int_169734)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 11), tuple_169732, subscript_call_result_169737)
    # Adding element type (line 320)
    
    # Obtaining the type of the subscript
    int_169738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 33), 'int')
    # Getting the type of 'gc' (line 320)
    gc_169739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 30), 'gc')
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___169740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 30), gc_169739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 320)
    subscript_call_result_169741 = invoke(stypy.reporting.localization.Localization(__file__, 320, 30), getitem___169740, int_169738)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 11), tuple_169732, subscript_call_result_169741)
    # Adding element type (line 320)
    # Getting the type of 'phi_star' (line 320)
    phi_star_169742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 37), 'phi_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 11), tuple_169732, phi_star_169742)
    # Adding element type (line 320)
    # Getting the type of 'old_fval' (line 320)
    old_fval_169743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 47), 'old_fval')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 11), tuple_169732, old_fval_169743)
    # Adding element type (line 320)
    # Getting the type of 'derphi_star' (line 320)
    derphi_star_169744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 57), 'derphi_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 11), tuple_169732, derphi_star_169744)
    
    # Assigning a type to the variable 'stypy_return_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type', tuple_169732)
    
    # ################# End of 'line_search_wolfe2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'line_search_wolfe2' in the type store
    # Getting the type of 'stypy_return_type' (line 194)
    stypy_return_type_169745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169745)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'line_search_wolfe2'
    return stypy_return_type_169745

# Assigning a type to the variable 'line_search_wolfe2' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'line_search_wolfe2', line_search_wolfe2)

@norecursion
def scalar_search_wolfe2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 323)
    None_169746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 37), 'None')
    # Getting the type of 'None' (line 323)
    None_169747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 48), 'None')
    # Getting the type of 'None' (line 324)
    None_169748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'None')
    # Getting the type of 'None' (line 324)
    None_169749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 48), 'None')
    float_169750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'float')
    float_169751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 37), 'float')
    # Getting the type of 'None' (line 325)
    None_169752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 47), 'None')
    # Getting the type of 'None' (line 326)
    None_169753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 41), 'None')
    int_169754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 55), 'int')
    defaults = [None_169746, None_169747, None_169748, None_169749, float_169750, float_169751, None_169752, None_169753, int_169754]
    # Create a new context for function 'scalar_search_wolfe2'
    module_type_store = module_type_store.open_function_context('scalar_search_wolfe2', 323, 0, False)
    
    # Passed parameters checking function
    scalar_search_wolfe2.stypy_localization = localization
    scalar_search_wolfe2.stypy_type_of_self = None
    scalar_search_wolfe2.stypy_type_store = module_type_store
    scalar_search_wolfe2.stypy_function_name = 'scalar_search_wolfe2'
    scalar_search_wolfe2.stypy_param_names_list = ['phi', 'derphi', 'phi0', 'old_phi0', 'derphi0', 'c1', 'c2', 'amax', 'extra_condition', 'maxiter']
    scalar_search_wolfe2.stypy_varargs_param_name = None
    scalar_search_wolfe2.stypy_kwargs_param_name = None
    scalar_search_wolfe2.stypy_call_defaults = defaults
    scalar_search_wolfe2.stypy_call_varargs = varargs
    scalar_search_wolfe2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'scalar_search_wolfe2', ['phi', 'derphi', 'phi0', 'old_phi0', 'derphi0', 'c1', 'c2', 'amax', 'extra_condition', 'maxiter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'scalar_search_wolfe2', localization, ['phi', 'derphi', 'phi0', 'old_phi0', 'derphi0', 'c1', 'c2', 'amax', 'extra_condition', 'maxiter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'scalar_search_wolfe2(...)' code ##################

    str_169755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, (-1)), 'str', "Find alpha that satisfies strong Wolfe conditions.\n\n    alpha > 0 is assumed to be a descent direction.\n\n    Parameters\n    ----------\n    phi : callable f(x)\n        Objective scalar function.\n    derphi : callable f'(x), optional\n        Objective function derivative (can be None)\n    phi0 : float, optional\n        Value of phi at s=0\n    old_phi0 : float, optional\n        Value of phi at previous point\n    derphi0 : float, optional\n        Value of derphi at s=0\n    c1 : float, optional\n        Parameter for Armijo condition rule.\n    c2 : float, optional\n        Parameter for curvature condition rule.\n    amax : float, optional\n        Maximum step size\n    extra_condition : callable, optional\n        A callable of the form ``extra_condition(alpha, phi_value)``\n        returning a boolean. The line search accepts the value\n        of ``alpha`` only if this callable returns ``True``.\n        If the callable returns ``False`` for the step length,\n        the algorithm will continue with new iterates.\n        The callable is only called for iterates satisfying\n        the strong Wolfe conditions.\n    maxiter : int, optional\n        Maximum number of iterations to perform\n\n    Returns\n    -------\n    alpha_star : float or None\n        Best alpha, or None if the line search algorithm did not converge.\n    phi_star : float\n        phi at alpha_star\n    phi0 : float\n        phi at 0\n    derphi_star : float or None\n        derphi at alpha_star, or None if the line search algorithm\n        did not converge.\n\n    Notes\n    -----\n    Uses the line search algorithm to enforce strong Wolfe\n    conditions.  See Wright and Nocedal, 'Numerical Optimization',\n    1999, pg. 59-60.\n\n    For the zoom phase it uses an algorithm by [...].\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 382)
    # Getting the type of 'phi0' (line 382)
    phi0_169756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 7), 'phi0')
    # Getting the type of 'None' (line 382)
    None_169757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'None')
    
    (may_be_169758, more_types_in_union_169759) = may_be_none(phi0_169756, None_169757)

    if may_be_169758:

        if more_types_in_union_169759:
            # Runtime conditional SSA (line 382)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to phi(...): (line 383)
        # Processing the call arguments (line 383)
        float_169761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 19), 'float')
        # Processing the call keyword arguments (line 383)
        kwargs_169762 = {}
        # Getting the type of 'phi' (line 383)
        phi_169760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'phi', False)
        # Calling phi(args, kwargs) (line 383)
        phi_call_result_169763 = invoke(stypy.reporting.localization.Localization(__file__, 383, 15), phi_169760, *[float_169761], **kwargs_169762)
        
        # Assigning a type to the variable 'phi0' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'phi0', phi_call_result_169763)

        if more_types_in_union_169759:
            # SSA join for if statement (line 382)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'derphi0' (line 385)
    derphi0_169764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 7), 'derphi0')
    # Getting the type of 'None' (line 385)
    None_169765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'None')
    # Applying the binary operator 'is' (line 385)
    result_is__169766 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 7), 'is', derphi0_169764, None_169765)
    
    
    # Getting the type of 'derphi' (line 385)
    derphi_169767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 27), 'derphi')
    # Getting the type of 'None' (line 385)
    None_169768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 41), 'None')
    # Applying the binary operator 'isnot' (line 385)
    result_is_not_169769 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 27), 'isnot', derphi_169767, None_169768)
    
    # Applying the binary operator 'and' (line 385)
    result_and_keyword_169770 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 7), 'and', result_is__169766, result_is_not_169769)
    
    # Testing the type of an if condition (line 385)
    if_condition_169771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 4), result_and_keyword_169770)
    # Assigning a type to the variable 'if_condition_169771' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'if_condition_169771', if_condition_169771)
    # SSA begins for if statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 386):
    
    # Assigning a Call to a Name (line 386):
    
    # Call to derphi(...): (line 386)
    # Processing the call arguments (line 386)
    float_169773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 25), 'float')
    # Processing the call keyword arguments (line 386)
    kwargs_169774 = {}
    # Getting the type of 'derphi' (line 386)
    derphi_169772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 18), 'derphi', False)
    # Calling derphi(args, kwargs) (line 386)
    derphi_call_result_169775 = invoke(stypy.reporting.localization.Localization(__file__, 386, 18), derphi_169772, *[float_169773], **kwargs_169774)
    
    # Assigning a type to the variable 'derphi0' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'derphi0', derphi_call_result_169775)
    # SSA join for if statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 388):
    
    # Assigning a Num to a Name (line 388):
    int_169776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 13), 'int')
    # Assigning a type to the variable 'alpha0' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'alpha0', int_169776)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'old_phi0' (line 389)
    old_phi0_169777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 7), 'old_phi0')
    # Getting the type of 'None' (line 389)
    None_169778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 23), 'None')
    # Applying the binary operator 'isnot' (line 389)
    result_is_not_169779 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 7), 'isnot', old_phi0_169777, None_169778)
    
    
    # Getting the type of 'derphi0' (line 389)
    derphi0_169780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'derphi0')
    int_169781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 43), 'int')
    # Applying the binary operator '!=' (line 389)
    result_ne_169782 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 32), '!=', derphi0_169780, int_169781)
    
    # Applying the binary operator 'and' (line 389)
    result_and_keyword_169783 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 7), 'and', result_is_not_169779, result_ne_169782)
    
    # Testing the type of an if condition (line 389)
    if_condition_169784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 4), result_and_keyword_169783)
    # Assigning a type to the variable 'if_condition_169784' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'if_condition_169784', if_condition_169784)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 390):
    
    # Assigning a Call to a Name (line 390):
    
    # Call to min(...): (line 390)
    # Processing the call arguments (line 390)
    float_169786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 21), 'float')
    float_169787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 26), 'float')
    int_169788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 31), 'int')
    # Applying the binary operator '*' (line 390)
    result_mul_169789 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 26), '*', float_169787, int_169788)
    
    # Getting the type of 'phi0' (line 390)
    phi0_169790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 34), 'phi0', False)
    # Getting the type of 'old_phi0' (line 390)
    old_phi0_169791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 41), 'old_phi0', False)
    # Applying the binary operator '-' (line 390)
    result_sub_169792 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 34), '-', phi0_169790, old_phi0_169791)
    
    # Applying the binary operator '*' (line 390)
    result_mul_169793 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 32), '*', result_mul_169789, result_sub_169792)
    
    # Getting the type of 'derphi0' (line 390)
    derphi0_169794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 51), 'derphi0', False)
    # Applying the binary operator 'div' (line 390)
    result_div_169795 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 50), 'div', result_mul_169793, derphi0_169794)
    
    # Processing the call keyword arguments (line 390)
    kwargs_169796 = {}
    # Getting the type of 'min' (line 390)
    min_169785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 17), 'min', False)
    # Calling min(args, kwargs) (line 390)
    min_call_result_169797 = invoke(stypy.reporting.localization.Localization(__file__, 390, 17), min_169785, *[float_169786, result_div_169795], **kwargs_169796)
    
    # Assigning a type to the variable 'alpha1' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'alpha1', min_call_result_169797)
    # SSA branch for the else part of an if statement (line 389)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 392):
    
    # Assigning a Num to a Name (line 392):
    float_169798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 17), 'float')
    # Assigning a type to the variable 'alpha1' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'alpha1', float_169798)
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'alpha1' (line 394)
    alpha1_169799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 7), 'alpha1')
    int_169800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 16), 'int')
    # Applying the binary operator '<' (line 394)
    result_lt_169801 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 7), '<', alpha1_169799, int_169800)
    
    # Testing the type of an if condition (line 394)
    if_condition_169802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 4), result_lt_169801)
    # Assigning a type to the variable 'if_condition_169802' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'if_condition_169802', if_condition_169802)
    # SSA begins for if statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 395):
    
    # Assigning a Num to a Name (line 395):
    float_169803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 17), 'float')
    # Assigning a type to the variable 'alpha1' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'alpha1', float_169803)
    # SSA join for if statement (line 394)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 397):
    
    # Assigning a Call to a Name (line 397):
    
    # Call to phi(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'alpha1' (line 397)
    alpha1_169805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 17), 'alpha1', False)
    # Processing the call keyword arguments (line 397)
    kwargs_169806 = {}
    # Getting the type of 'phi' (line 397)
    phi_169804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 13), 'phi', False)
    # Calling phi(args, kwargs) (line 397)
    phi_call_result_169807 = invoke(stypy.reporting.localization.Localization(__file__, 397, 13), phi_169804, *[alpha1_169805], **kwargs_169806)
    
    # Assigning a type to the variable 'phi_a1' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'phi_a1', phi_call_result_169807)
    
    # Assigning a Name to a Name (line 400):
    
    # Assigning a Name to a Name (line 400):
    # Getting the type of 'phi0' (line 400)
    phi0_169808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'phi0')
    # Assigning a type to the variable 'phi_a0' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'phi_a0', phi0_169808)
    
    # Assigning a Name to a Name (line 401):
    
    # Assigning a Name to a Name (line 401):
    # Getting the type of 'derphi0' (line 401)
    derphi0_169809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'derphi0')
    # Assigning a type to the variable 'derphi_a0' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'derphi_a0', derphi0_169809)
    
    # Type idiom detected: calculating its left and rigth part (line 403)
    # Getting the type of 'extra_condition' (line 403)
    extra_condition_169810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 7), 'extra_condition')
    # Getting the type of 'None' (line 403)
    None_169811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 26), 'None')
    
    (may_be_169812, more_types_in_union_169813) = may_be_none(extra_condition_169810, None_169811)

    if may_be_169812:

        if more_types_in_union_169813:
            # Runtime conditional SSA (line 403)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Lambda to a Name (line 404):
        
        # Assigning a Lambda to a Name (line 404):

        @norecursion
        def _stypy_temp_lambda_48(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_48'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_48', 404, 26, True)
            # Passed parameters checking function
            _stypy_temp_lambda_48.stypy_localization = localization
            _stypy_temp_lambda_48.stypy_type_of_self = None
            _stypy_temp_lambda_48.stypy_type_store = module_type_store
            _stypy_temp_lambda_48.stypy_function_name = '_stypy_temp_lambda_48'
            _stypy_temp_lambda_48.stypy_param_names_list = ['alpha', 'phi']
            _stypy_temp_lambda_48.stypy_varargs_param_name = None
            _stypy_temp_lambda_48.stypy_kwargs_param_name = None
            _stypy_temp_lambda_48.stypy_call_defaults = defaults
            _stypy_temp_lambda_48.stypy_call_varargs = varargs
            _stypy_temp_lambda_48.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_48', ['alpha', 'phi'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_48', ['alpha', 'phi'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'True' (line 404)
            True_169814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 45), 'True')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 404)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 26), 'stypy_return_type', True_169814)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_48' in the type store
            # Getting the type of 'stypy_return_type' (line 404)
            stypy_return_type_169815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 26), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_169815)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_48'
            return stypy_return_type_169815

        # Assigning a type to the variable '_stypy_temp_lambda_48' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 26), '_stypy_temp_lambda_48', _stypy_temp_lambda_48)
        # Getting the type of '_stypy_temp_lambda_48' (line 404)
        _stypy_temp_lambda_48_169816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 26), '_stypy_temp_lambda_48')
        # Assigning a type to the variable 'extra_condition' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'extra_condition', _stypy_temp_lambda_48_169816)

        if more_types_in_union_169813:
            # SSA join for if statement (line 403)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to xrange(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'maxiter' (line 406)
    maxiter_169818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'maxiter', False)
    # Processing the call keyword arguments (line 406)
    kwargs_169819 = {}
    # Getting the type of 'xrange' (line 406)
    xrange_169817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 406)
    xrange_call_result_169820 = invoke(stypy.reporting.localization.Localization(__file__, 406, 13), xrange_169817, *[maxiter_169818], **kwargs_169819)
    
    # Testing the type of a for loop iterable (line 406)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 406, 4), xrange_call_result_169820)
    # Getting the type of the for loop variable (line 406)
    for_loop_var_169821 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 406, 4), xrange_call_result_169820)
    # Assigning a type to the variable 'i' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'i', for_loop_var_169821)
    # SSA begins for a for statement (line 406)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'alpha1' (line 407)
    alpha1_169822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'alpha1')
    int_169823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 21), 'int')
    # Applying the binary operator '==' (line 407)
    result_eq_169824 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 11), '==', alpha1_169822, int_169823)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'amax' (line 407)
    amax_169825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 27), 'amax')
    # Getting the type of 'None' (line 407)
    None_169826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 39), 'None')
    # Applying the binary operator 'isnot' (line 407)
    result_is_not_169827 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 27), 'isnot', amax_169825, None_169826)
    
    
    # Getting the type of 'alpha0' (line 407)
    alpha0_169828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'alpha0')
    # Getting the type of 'amax' (line 407)
    amax_169829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 58), 'amax')
    # Applying the binary operator '==' (line 407)
    result_eq_169830 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 48), '==', alpha0_169828, amax_169829)
    
    # Applying the binary operator 'and' (line 407)
    result_and_keyword_169831 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 27), 'and', result_is_not_169827, result_eq_169830)
    
    # Applying the binary operator 'or' (line 407)
    result_or_keyword_169832 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 11), 'or', result_eq_169824, result_and_keyword_169831)
    
    # Testing the type of an if condition (line 407)
    if_condition_169833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 8), result_or_keyword_169832)
    # Assigning a type to the variable 'if_condition_169833' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'if_condition_169833', if_condition_169833)
    # SSA begins for if statement (line 407)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 410):
    
    # Assigning a Name to a Name (line 410):
    # Getting the type of 'None' (line 410)
    None_169834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'None')
    # Assigning a type to the variable 'alpha_star' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'alpha_star', None_169834)
    
    # Assigning a Name to a Name (line 411):
    
    # Assigning a Name to a Name (line 411):
    # Getting the type of 'phi0' (line 411)
    phi0_169835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 23), 'phi0')
    # Assigning a type to the variable 'phi_star' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'phi_star', phi0_169835)
    
    # Assigning a Name to a Name (line 412):
    
    # Assigning a Name to a Name (line 412):
    # Getting the type of 'old_phi0' (line 412)
    old_phi0_169836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 19), 'old_phi0')
    # Assigning a type to the variable 'phi0' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'phi0', old_phi0_169836)
    
    # Assigning a Name to a Name (line 413):
    
    # Assigning a Name to a Name (line 413):
    # Getting the type of 'None' (line 413)
    None_169837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 26), 'None')
    # Assigning a type to the variable 'derphi_star' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'derphi_star', None_169837)
    
    
    # Getting the type of 'alpha1' (line 415)
    alpha1_169838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'alpha1')
    int_169839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 25), 'int')
    # Applying the binary operator '==' (line 415)
    result_eq_169840 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 15), '==', alpha1_169838, int_169839)
    
    # Testing the type of an if condition (line 415)
    if_condition_169841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 12), result_eq_169840)
    # Assigning a type to the variable 'if_condition_169841' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'if_condition_169841', if_condition_169841)
    # SSA begins for if statement (line 415)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 416):
    
    # Assigning a Str to a Name (line 416):
    str_169842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 22), 'str', 'Rounding errors prevent the line search from converging')
    # Assigning a type to the variable 'msg' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'msg', str_169842)
    # SSA branch for the else part of an if statement (line 415)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 418):
    
    # Assigning a BinOp to a Name (line 418):
    str_169843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 22), 'str', 'The line search algorithm could not find a solution ')
    str_169844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 22), 'str', 'less than or equal to amax: %s')
    # Getting the type of 'amax' (line 419)
    amax_169845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 57), 'amax')
    # Applying the binary operator '%' (line 419)
    result_mod_169846 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 22), '%', str_169844, amax_169845)
    
    # Applying the binary operator '+' (line 418)
    result_add_169847 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 22), '+', str_169843, result_mod_169846)
    
    # Assigning a type to the variable 'msg' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'msg', result_add_169847)
    # SSA join for if statement (line 415)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to warn(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'msg' (line 421)
    msg_169849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 17), 'msg', False)
    # Getting the type of 'LineSearchWarning' (line 421)
    LineSearchWarning_169850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 22), 'LineSearchWarning', False)
    # Processing the call keyword arguments (line 421)
    kwargs_169851 = {}
    # Getting the type of 'warn' (line 421)
    warn_169848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'warn', False)
    # Calling warn(args, kwargs) (line 421)
    warn_call_result_169852 = invoke(stypy.reporting.localization.Localization(__file__, 421, 12), warn_169848, *[msg_169849, LineSearchWarning_169850], **kwargs_169851)
    
    # SSA join for if statement (line 407)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'phi_a1' (line 424)
    phi_a1_169853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'phi_a1')
    # Getting the type of 'phi0' (line 424)
    phi0_169854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 'phi0')
    # Getting the type of 'c1' (line 424)
    c1_169855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 28), 'c1')
    # Getting the type of 'alpha1' (line 424)
    alpha1_169856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 33), 'alpha1')
    # Applying the binary operator '*' (line 424)
    result_mul_169857 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 28), '*', c1_169855, alpha1_169856)
    
    # Getting the type of 'derphi0' (line 424)
    derphi0_169858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 42), 'derphi0')
    # Applying the binary operator '*' (line 424)
    result_mul_169859 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 40), '*', result_mul_169857, derphi0_169858)
    
    # Applying the binary operator '+' (line 424)
    result_add_169860 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 21), '+', phi0_169854, result_mul_169859)
    
    # Applying the binary operator '>' (line 424)
    result_gt_169861 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 12), '>', phi_a1_169853, result_add_169860)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'phi_a1' (line 425)
    phi_a1_169862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 13), 'phi_a1')
    # Getting the type of 'phi_a0' (line 425)
    phi_a0_169863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'phi_a0')
    # Applying the binary operator '>=' (line 425)
    result_ge_169864 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 13), '>=', phi_a1_169862, phi_a0_169863)
    
    
    # Getting the type of 'i' (line 425)
    i_169865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 36), 'i')
    int_169866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 40), 'int')
    # Applying the binary operator '>' (line 425)
    result_gt_169867 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 36), '>', i_169865, int_169866)
    
    # Applying the binary operator 'and' (line 425)
    result_and_keyword_169868 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 12), 'and', result_ge_169864, result_gt_169867)
    
    # Applying the binary operator 'or' (line 424)
    result_or_keyword_169869 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 11), 'or', result_gt_169861, result_and_keyword_169868)
    
    # Testing the type of an if condition (line 424)
    if_condition_169870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 8), result_or_keyword_169869)
    # Assigning a type to the variable 'if_condition_169870' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'if_condition_169870', if_condition_169870)
    # SSA begins for if statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 426):
    
    # Assigning a Subscript to a Name (line 426):
    
    # Obtaining the type of the subscript
    int_169871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 12), 'int')
    
    # Call to _zoom(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'alpha0' (line 427)
    alpha0_169873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 30), 'alpha0', False)
    # Getting the type of 'alpha1' (line 427)
    alpha1_169874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 38), 'alpha1', False)
    # Getting the type of 'phi_a0' (line 427)
    phi_a0_169875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 46), 'phi_a0', False)
    # Getting the type of 'phi_a1' (line 428)
    phi_a1_169876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 30), 'phi_a1', False)
    # Getting the type of 'derphi_a0' (line 428)
    derphi_a0_169877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 38), 'derphi_a0', False)
    # Getting the type of 'phi' (line 428)
    phi_169878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 49), 'phi', False)
    # Getting the type of 'derphi' (line 428)
    derphi_169879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 54), 'derphi', False)
    # Getting the type of 'phi0' (line 429)
    phi0_169880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 30), 'phi0', False)
    # Getting the type of 'derphi0' (line 429)
    derphi0_169881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 36), 'derphi0', False)
    # Getting the type of 'c1' (line 429)
    c1_169882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 45), 'c1', False)
    # Getting the type of 'c2' (line 429)
    c2_169883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 49), 'c2', False)
    # Getting the type of 'extra_condition' (line 429)
    extra_condition_169884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 53), 'extra_condition', False)
    # Processing the call keyword arguments (line 427)
    kwargs_169885 = {}
    # Getting the type of '_zoom' (line 427)
    _zoom_169872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 24), '_zoom', False)
    # Calling _zoom(args, kwargs) (line 427)
    _zoom_call_result_169886 = invoke(stypy.reporting.localization.Localization(__file__, 427, 24), _zoom_169872, *[alpha0_169873, alpha1_169874, phi_a0_169875, phi_a1_169876, derphi_a0_169877, phi_169878, derphi_169879, phi0_169880, derphi0_169881, c1_169882, c2_169883, extra_condition_169884], **kwargs_169885)
    
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___169887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), _zoom_call_result_169886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_169888 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), getitem___169887, int_169871)
    
    # Assigning a type to the variable 'tuple_var_assignment_169046' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'tuple_var_assignment_169046', subscript_call_result_169888)
    
    # Assigning a Subscript to a Name (line 426):
    
    # Obtaining the type of the subscript
    int_169889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 12), 'int')
    
    # Call to _zoom(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'alpha0' (line 427)
    alpha0_169891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 30), 'alpha0', False)
    # Getting the type of 'alpha1' (line 427)
    alpha1_169892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 38), 'alpha1', False)
    # Getting the type of 'phi_a0' (line 427)
    phi_a0_169893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 46), 'phi_a0', False)
    # Getting the type of 'phi_a1' (line 428)
    phi_a1_169894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 30), 'phi_a1', False)
    # Getting the type of 'derphi_a0' (line 428)
    derphi_a0_169895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 38), 'derphi_a0', False)
    # Getting the type of 'phi' (line 428)
    phi_169896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 49), 'phi', False)
    # Getting the type of 'derphi' (line 428)
    derphi_169897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 54), 'derphi', False)
    # Getting the type of 'phi0' (line 429)
    phi0_169898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 30), 'phi0', False)
    # Getting the type of 'derphi0' (line 429)
    derphi0_169899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 36), 'derphi0', False)
    # Getting the type of 'c1' (line 429)
    c1_169900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 45), 'c1', False)
    # Getting the type of 'c2' (line 429)
    c2_169901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 49), 'c2', False)
    # Getting the type of 'extra_condition' (line 429)
    extra_condition_169902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 53), 'extra_condition', False)
    # Processing the call keyword arguments (line 427)
    kwargs_169903 = {}
    # Getting the type of '_zoom' (line 427)
    _zoom_169890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 24), '_zoom', False)
    # Calling _zoom(args, kwargs) (line 427)
    _zoom_call_result_169904 = invoke(stypy.reporting.localization.Localization(__file__, 427, 24), _zoom_169890, *[alpha0_169891, alpha1_169892, phi_a0_169893, phi_a1_169894, derphi_a0_169895, phi_169896, derphi_169897, phi0_169898, derphi0_169899, c1_169900, c2_169901, extra_condition_169902], **kwargs_169903)
    
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___169905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), _zoom_call_result_169904, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_169906 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), getitem___169905, int_169889)
    
    # Assigning a type to the variable 'tuple_var_assignment_169047' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'tuple_var_assignment_169047', subscript_call_result_169906)
    
    # Assigning a Subscript to a Name (line 426):
    
    # Obtaining the type of the subscript
    int_169907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 12), 'int')
    
    # Call to _zoom(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'alpha0' (line 427)
    alpha0_169909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 30), 'alpha0', False)
    # Getting the type of 'alpha1' (line 427)
    alpha1_169910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 38), 'alpha1', False)
    # Getting the type of 'phi_a0' (line 427)
    phi_a0_169911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 46), 'phi_a0', False)
    # Getting the type of 'phi_a1' (line 428)
    phi_a1_169912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 30), 'phi_a1', False)
    # Getting the type of 'derphi_a0' (line 428)
    derphi_a0_169913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 38), 'derphi_a0', False)
    # Getting the type of 'phi' (line 428)
    phi_169914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 49), 'phi', False)
    # Getting the type of 'derphi' (line 428)
    derphi_169915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 54), 'derphi', False)
    # Getting the type of 'phi0' (line 429)
    phi0_169916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 30), 'phi0', False)
    # Getting the type of 'derphi0' (line 429)
    derphi0_169917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 36), 'derphi0', False)
    # Getting the type of 'c1' (line 429)
    c1_169918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 45), 'c1', False)
    # Getting the type of 'c2' (line 429)
    c2_169919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 49), 'c2', False)
    # Getting the type of 'extra_condition' (line 429)
    extra_condition_169920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 53), 'extra_condition', False)
    # Processing the call keyword arguments (line 427)
    kwargs_169921 = {}
    # Getting the type of '_zoom' (line 427)
    _zoom_169908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 24), '_zoom', False)
    # Calling _zoom(args, kwargs) (line 427)
    _zoom_call_result_169922 = invoke(stypy.reporting.localization.Localization(__file__, 427, 24), _zoom_169908, *[alpha0_169909, alpha1_169910, phi_a0_169911, phi_a1_169912, derphi_a0_169913, phi_169914, derphi_169915, phi0_169916, derphi0_169917, c1_169918, c2_169919, extra_condition_169920], **kwargs_169921)
    
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___169923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), _zoom_call_result_169922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_169924 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), getitem___169923, int_169907)
    
    # Assigning a type to the variable 'tuple_var_assignment_169048' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'tuple_var_assignment_169048', subscript_call_result_169924)
    
    # Assigning a Name to a Name (line 426):
    # Getting the type of 'tuple_var_assignment_169046' (line 426)
    tuple_var_assignment_169046_169925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'tuple_var_assignment_169046')
    # Assigning a type to the variable 'alpha_star' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'alpha_star', tuple_var_assignment_169046_169925)
    
    # Assigning a Name to a Name (line 426):
    # Getting the type of 'tuple_var_assignment_169047' (line 426)
    tuple_var_assignment_169047_169926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'tuple_var_assignment_169047')
    # Assigning a type to the variable 'phi_star' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 24), 'phi_star', tuple_var_assignment_169047_169926)
    
    # Assigning a Name to a Name (line 426):
    # Getting the type of 'tuple_var_assignment_169048' (line 426)
    tuple_var_assignment_169048_169927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'tuple_var_assignment_169048')
    # Assigning a type to the variable 'derphi_star' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 34), 'derphi_star', tuple_var_assignment_169048_169927)
    # SSA join for if statement (line 424)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 432):
    
    # Assigning a Call to a Name (line 432):
    
    # Call to derphi(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'alpha1' (line 432)
    alpha1_169929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 27), 'alpha1', False)
    # Processing the call keyword arguments (line 432)
    kwargs_169930 = {}
    # Getting the type of 'derphi' (line 432)
    derphi_169928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 20), 'derphi', False)
    # Calling derphi(args, kwargs) (line 432)
    derphi_call_result_169931 = invoke(stypy.reporting.localization.Localization(__file__, 432, 20), derphi_169928, *[alpha1_169929], **kwargs_169930)
    
    # Assigning a type to the variable 'derphi_a1' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'derphi_a1', derphi_call_result_169931)
    
    
    
    # Call to abs(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'derphi_a1' (line 433)
    derphi_a1_169933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 16), 'derphi_a1', False)
    # Processing the call keyword arguments (line 433)
    kwargs_169934 = {}
    # Getting the type of 'abs' (line 433)
    abs_169932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'abs', False)
    # Calling abs(args, kwargs) (line 433)
    abs_call_result_169935 = invoke(stypy.reporting.localization.Localization(__file__, 433, 12), abs_169932, *[derphi_a1_169933], **kwargs_169934)
    
    
    # Getting the type of 'c2' (line 433)
    c2_169936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 31), 'c2')
    # Applying the 'usub' unary operator (line 433)
    result___neg___169937 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 30), 'usub', c2_169936)
    
    # Getting the type of 'derphi0' (line 433)
    derphi0_169938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 34), 'derphi0')
    # Applying the binary operator '*' (line 433)
    result_mul_169939 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 30), '*', result___neg___169937, derphi0_169938)
    
    # Applying the binary operator '<=' (line 433)
    result_le_169940 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 12), '<=', abs_call_result_169935, result_mul_169939)
    
    # Testing the type of an if condition (line 433)
    if_condition_169941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 8), result_le_169940)
    # Assigning a type to the variable 'if_condition_169941' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'if_condition_169941', if_condition_169941)
    # SSA begins for if statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to extra_condition(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'alpha1' (line 434)
    alpha1_169943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 31), 'alpha1', False)
    # Getting the type of 'phi_a1' (line 434)
    phi_a1_169944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 39), 'phi_a1', False)
    # Processing the call keyword arguments (line 434)
    kwargs_169945 = {}
    # Getting the type of 'extra_condition' (line 434)
    extra_condition_169942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'extra_condition', False)
    # Calling extra_condition(args, kwargs) (line 434)
    extra_condition_call_result_169946 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), extra_condition_169942, *[alpha1_169943, phi_a1_169944], **kwargs_169945)
    
    # Testing the type of an if condition (line 434)
    if_condition_169947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 12), extra_condition_call_result_169946)
    # Assigning a type to the variable 'if_condition_169947' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'if_condition_169947', if_condition_169947)
    # SSA begins for if statement (line 434)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 435):
    
    # Assigning a Name to a Name (line 435):
    # Getting the type of 'alpha1' (line 435)
    alpha1_169948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 29), 'alpha1')
    # Assigning a type to the variable 'alpha_star' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 16), 'alpha_star', alpha1_169948)
    
    # Assigning a Name to a Name (line 436):
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'phi_a1' (line 436)
    phi_a1_169949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 27), 'phi_a1')
    # Assigning a type to the variable 'phi_star' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'phi_star', phi_a1_169949)
    
    # Assigning a Name to a Name (line 437):
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'derphi_a1' (line 437)
    derphi_a1_169950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 30), 'derphi_a1')
    # Assigning a type to the variable 'derphi_star' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'derphi_star', derphi_a1_169950)
    # SSA join for if statement (line 434)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'derphi_a1' (line 440)
    derphi_a1_169951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'derphi_a1')
    int_169952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 25), 'int')
    # Applying the binary operator '>=' (line 440)
    result_ge_169953 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 12), '>=', derphi_a1_169951, int_169952)
    
    # Testing the type of an if condition (line 440)
    if_condition_169954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 440, 8), result_ge_169953)
    # Assigning a type to the variable 'if_condition_169954' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'if_condition_169954', if_condition_169954)
    # SSA begins for if statement (line 440)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 441):
    
    # Assigning a Subscript to a Name (line 441):
    
    # Obtaining the type of the subscript
    int_169955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 12), 'int')
    
    # Call to _zoom(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'alpha1' (line 442)
    alpha1_169957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 30), 'alpha1', False)
    # Getting the type of 'alpha0' (line 442)
    alpha0_169958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 38), 'alpha0', False)
    # Getting the type of 'phi_a1' (line 442)
    phi_a1_169959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 46), 'phi_a1', False)
    # Getting the type of 'phi_a0' (line 443)
    phi_a0_169960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 30), 'phi_a0', False)
    # Getting the type of 'derphi_a1' (line 443)
    derphi_a1_169961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 38), 'derphi_a1', False)
    # Getting the type of 'phi' (line 443)
    phi_169962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 49), 'phi', False)
    # Getting the type of 'derphi' (line 443)
    derphi_169963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 54), 'derphi', False)
    # Getting the type of 'phi0' (line 444)
    phi0_169964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 30), 'phi0', False)
    # Getting the type of 'derphi0' (line 444)
    derphi0_169965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 36), 'derphi0', False)
    # Getting the type of 'c1' (line 444)
    c1_169966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 45), 'c1', False)
    # Getting the type of 'c2' (line 444)
    c2_169967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 49), 'c2', False)
    # Getting the type of 'extra_condition' (line 444)
    extra_condition_169968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 53), 'extra_condition', False)
    # Processing the call keyword arguments (line 442)
    kwargs_169969 = {}
    # Getting the type of '_zoom' (line 442)
    _zoom_169956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 24), '_zoom', False)
    # Calling _zoom(args, kwargs) (line 442)
    _zoom_call_result_169970 = invoke(stypy.reporting.localization.Localization(__file__, 442, 24), _zoom_169956, *[alpha1_169957, alpha0_169958, phi_a1_169959, phi_a0_169960, derphi_a1_169961, phi_169962, derphi_169963, phi0_169964, derphi0_169965, c1_169966, c2_169967, extra_condition_169968], **kwargs_169969)
    
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___169971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), _zoom_call_result_169970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_169972 = invoke(stypy.reporting.localization.Localization(__file__, 441, 12), getitem___169971, int_169955)
    
    # Assigning a type to the variable 'tuple_var_assignment_169049' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'tuple_var_assignment_169049', subscript_call_result_169972)
    
    # Assigning a Subscript to a Name (line 441):
    
    # Obtaining the type of the subscript
    int_169973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 12), 'int')
    
    # Call to _zoom(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'alpha1' (line 442)
    alpha1_169975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 30), 'alpha1', False)
    # Getting the type of 'alpha0' (line 442)
    alpha0_169976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 38), 'alpha0', False)
    # Getting the type of 'phi_a1' (line 442)
    phi_a1_169977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 46), 'phi_a1', False)
    # Getting the type of 'phi_a0' (line 443)
    phi_a0_169978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 30), 'phi_a0', False)
    # Getting the type of 'derphi_a1' (line 443)
    derphi_a1_169979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 38), 'derphi_a1', False)
    # Getting the type of 'phi' (line 443)
    phi_169980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 49), 'phi', False)
    # Getting the type of 'derphi' (line 443)
    derphi_169981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 54), 'derphi', False)
    # Getting the type of 'phi0' (line 444)
    phi0_169982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 30), 'phi0', False)
    # Getting the type of 'derphi0' (line 444)
    derphi0_169983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 36), 'derphi0', False)
    # Getting the type of 'c1' (line 444)
    c1_169984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 45), 'c1', False)
    # Getting the type of 'c2' (line 444)
    c2_169985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 49), 'c2', False)
    # Getting the type of 'extra_condition' (line 444)
    extra_condition_169986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 53), 'extra_condition', False)
    # Processing the call keyword arguments (line 442)
    kwargs_169987 = {}
    # Getting the type of '_zoom' (line 442)
    _zoom_169974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 24), '_zoom', False)
    # Calling _zoom(args, kwargs) (line 442)
    _zoom_call_result_169988 = invoke(stypy.reporting.localization.Localization(__file__, 442, 24), _zoom_169974, *[alpha1_169975, alpha0_169976, phi_a1_169977, phi_a0_169978, derphi_a1_169979, phi_169980, derphi_169981, phi0_169982, derphi0_169983, c1_169984, c2_169985, extra_condition_169986], **kwargs_169987)
    
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___169989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), _zoom_call_result_169988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_169990 = invoke(stypy.reporting.localization.Localization(__file__, 441, 12), getitem___169989, int_169973)
    
    # Assigning a type to the variable 'tuple_var_assignment_169050' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'tuple_var_assignment_169050', subscript_call_result_169990)
    
    # Assigning a Subscript to a Name (line 441):
    
    # Obtaining the type of the subscript
    int_169991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 12), 'int')
    
    # Call to _zoom(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'alpha1' (line 442)
    alpha1_169993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 30), 'alpha1', False)
    # Getting the type of 'alpha0' (line 442)
    alpha0_169994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 38), 'alpha0', False)
    # Getting the type of 'phi_a1' (line 442)
    phi_a1_169995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 46), 'phi_a1', False)
    # Getting the type of 'phi_a0' (line 443)
    phi_a0_169996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 30), 'phi_a0', False)
    # Getting the type of 'derphi_a1' (line 443)
    derphi_a1_169997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 38), 'derphi_a1', False)
    # Getting the type of 'phi' (line 443)
    phi_169998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 49), 'phi', False)
    # Getting the type of 'derphi' (line 443)
    derphi_169999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 54), 'derphi', False)
    # Getting the type of 'phi0' (line 444)
    phi0_170000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 30), 'phi0', False)
    # Getting the type of 'derphi0' (line 444)
    derphi0_170001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 36), 'derphi0', False)
    # Getting the type of 'c1' (line 444)
    c1_170002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 45), 'c1', False)
    # Getting the type of 'c2' (line 444)
    c2_170003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 49), 'c2', False)
    # Getting the type of 'extra_condition' (line 444)
    extra_condition_170004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 53), 'extra_condition', False)
    # Processing the call keyword arguments (line 442)
    kwargs_170005 = {}
    # Getting the type of '_zoom' (line 442)
    _zoom_169992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 24), '_zoom', False)
    # Calling _zoom(args, kwargs) (line 442)
    _zoom_call_result_170006 = invoke(stypy.reporting.localization.Localization(__file__, 442, 24), _zoom_169992, *[alpha1_169993, alpha0_169994, phi_a1_169995, phi_a0_169996, derphi_a1_169997, phi_169998, derphi_169999, phi0_170000, derphi0_170001, c1_170002, c2_170003, extra_condition_170004], **kwargs_170005)
    
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___170007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), _zoom_call_result_170006, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_170008 = invoke(stypy.reporting.localization.Localization(__file__, 441, 12), getitem___170007, int_169991)
    
    # Assigning a type to the variable 'tuple_var_assignment_169051' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'tuple_var_assignment_169051', subscript_call_result_170008)
    
    # Assigning a Name to a Name (line 441):
    # Getting the type of 'tuple_var_assignment_169049' (line 441)
    tuple_var_assignment_169049_170009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'tuple_var_assignment_169049')
    # Assigning a type to the variable 'alpha_star' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'alpha_star', tuple_var_assignment_169049_170009)
    
    # Assigning a Name to a Name (line 441):
    # Getting the type of 'tuple_var_assignment_169050' (line 441)
    tuple_var_assignment_169050_170010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'tuple_var_assignment_169050')
    # Assigning a type to the variable 'phi_star' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 24), 'phi_star', tuple_var_assignment_169050_170010)
    
    # Assigning a Name to a Name (line 441):
    # Getting the type of 'tuple_var_assignment_169051' (line 441)
    tuple_var_assignment_169051_170011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'tuple_var_assignment_169051')
    # Assigning a type to the variable 'derphi_star' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 34), 'derphi_star', tuple_var_assignment_169051_170011)
    # SSA join for if statement (line 440)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 447):
    
    # Assigning a BinOp to a Name (line 447):
    int_170012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 17), 'int')
    # Getting the type of 'alpha1' (line 447)
    alpha1_170013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 21), 'alpha1')
    # Applying the binary operator '*' (line 447)
    result_mul_170014 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 17), '*', int_170012, alpha1_170013)
    
    # Assigning a type to the variable 'alpha2' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'alpha2', result_mul_170014)
    
    # Type idiom detected: calculating its left and rigth part (line 448)
    # Getting the type of 'amax' (line 448)
    amax_170015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'amax')
    # Getting the type of 'None' (line 448)
    None_170016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 23), 'None')
    
    (may_be_170017, more_types_in_union_170018) = may_not_be_none(amax_170015, None_170016)

    if may_be_170017:

        if more_types_in_union_170018:
            # Runtime conditional SSA (line 448)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 449):
        
        # Assigning a Call to a Name (line 449):
        
        # Call to min(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'alpha2' (line 449)
        alpha2_170020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 25), 'alpha2', False)
        # Getting the type of 'amax' (line 449)
        amax_170021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 33), 'amax', False)
        # Processing the call keyword arguments (line 449)
        kwargs_170022 = {}
        # Getting the type of 'min' (line 449)
        min_170019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 21), 'min', False)
        # Calling min(args, kwargs) (line 449)
        min_call_result_170023 = invoke(stypy.reporting.localization.Localization(__file__, 449, 21), min_170019, *[alpha2_170020, amax_170021], **kwargs_170022)
        
        # Assigning a type to the variable 'alpha2' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'alpha2', min_call_result_170023)

        if more_types_in_union_170018:
            # SSA join for if statement (line 448)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 450):
    
    # Assigning a Name to a Name (line 450):
    # Getting the type of 'alpha1' (line 450)
    alpha1_170024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 17), 'alpha1')
    # Assigning a type to the variable 'alpha0' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'alpha0', alpha1_170024)
    
    # Assigning a Name to a Name (line 451):
    
    # Assigning a Name to a Name (line 451):
    # Getting the type of 'alpha2' (line 451)
    alpha2_170025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 17), 'alpha2')
    # Assigning a type to the variable 'alpha1' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'alpha1', alpha2_170025)
    
    # Assigning a Name to a Name (line 452):
    
    # Assigning a Name to a Name (line 452):
    # Getting the type of 'phi_a1' (line 452)
    phi_a1_170026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 17), 'phi_a1')
    # Assigning a type to the variable 'phi_a0' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'phi_a0', phi_a1_170026)
    
    # Assigning a Call to a Name (line 453):
    
    # Assigning a Call to a Name (line 453):
    
    # Call to phi(...): (line 453)
    # Processing the call arguments (line 453)
    # Getting the type of 'alpha1' (line 453)
    alpha1_170028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 21), 'alpha1', False)
    # Processing the call keyword arguments (line 453)
    kwargs_170029 = {}
    # Getting the type of 'phi' (line 453)
    phi_170027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 17), 'phi', False)
    # Calling phi(args, kwargs) (line 453)
    phi_call_result_170030 = invoke(stypy.reporting.localization.Localization(__file__, 453, 17), phi_170027, *[alpha1_170028], **kwargs_170029)
    
    # Assigning a type to the variable 'phi_a1' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'phi_a1', phi_call_result_170030)
    
    # Assigning a Name to a Name (line 454):
    
    # Assigning a Name to a Name (line 454):
    # Getting the type of 'derphi_a1' (line 454)
    derphi_a1_170031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 20), 'derphi_a1')
    # Assigning a type to the variable 'derphi_a0' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'derphi_a0', derphi_a1_170031)
    # SSA branch for the else part of a for statement (line 406)
    module_type_store.open_ssa_branch('for loop else')
    
    # Assigning a Name to a Name (line 458):
    
    # Assigning a Name to a Name (line 458):
    # Getting the type of 'alpha1' (line 458)
    alpha1_170032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 21), 'alpha1')
    # Assigning a type to the variable 'alpha_star' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'alpha_star', alpha1_170032)
    
    # Assigning a Name to a Name (line 459):
    
    # Assigning a Name to a Name (line 459):
    # Getting the type of 'phi_a1' (line 459)
    phi_a1_170033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 19), 'phi_a1')
    # Assigning a type to the variable 'phi_star' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'phi_star', phi_a1_170033)
    
    # Assigning a Name to a Name (line 460):
    
    # Assigning a Name to a Name (line 460):
    # Getting the type of 'None' (line 460)
    None_170034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 22), 'None')
    # Assigning a type to the variable 'derphi_star' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'derphi_star', None_170034)
    
    # Call to warn(...): (line 461)
    # Processing the call arguments (line 461)
    str_170036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 13), 'str', 'The line search algorithm did not converge')
    # Getting the type of 'LineSearchWarning' (line 461)
    LineSearchWarning_170037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 59), 'LineSearchWarning', False)
    # Processing the call keyword arguments (line 461)
    kwargs_170038 = {}
    # Getting the type of 'warn' (line 461)
    warn_170035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 461)
    warn_call_result_170039 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), warn_170035, *[str_170036, LineSearchWarning_170037], **kwargs_170038)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 463)
    tuple_170040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 463)
    # Adding element type (line 463)
    # Getting the type of 'alpha_star' (line 463)
    alpha_star_170041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'alpha_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), tuple_170040, alpha_star_170041)
    # Adding element type (line 463)
    # Getting the type of 'phi_star' (line 463)
    phi_star_170042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 23), 'phi_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), tuple_170040, phi_star_170042)
    # Adding element type (line 463)
    # Getting the type of 'phi0' (line 463)
    phi0_170043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 33), 'phi0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), tuple_170040, phi0_170043)
    # Adding element type (line 463)
    # Getting the type of 'derphi_star' (line 463)
    derphi_star_170044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 39), 'derphi_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), tuple_170040, derphi_star_170044)
    
    # Assigning a type to the variable 'stypy_return_type' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type', tuple_170040)
    
    # ################# End of 'scalar_search_wolfe2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'scalar_search_wolfe2' in the type store
    # Getting the type of 'stypy_return_type' (line 323)
    stypy_return_type_170045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170045)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'scalar_search_wolfe2'
    return stypy_return_type_170045

# Assigning a type to the variable 'scalar_search_wolfe2' (line 323)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'scalar_search_wolfe2', scalar_search_wolfe2)

@norecursion
def _cubicmin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_cubicmin'
    module_type_store = module_type_store.open_function_context('_cubicmin', 466, 0, False)
    
    # Passed parameters checking function
    _cubicmin.stypy_localization = localization
    _cubicmin.stypy_type_of_self = None
    _cubicmin.stypy_type_store = module_type_store
    _cubicmin.stypy_function_name = '_cubicmin'
    _cubicmin.stypy_param_names_list = ['a', 'fa', 'fpa', 'b', 'fb', 'c', 'fc']
    _cubicmin.stypy_varargs_param_name = None
    _cubicmin.stypy_kwargs_param_name = None
    _cubicmin.stypy_call_defaults = defaults
    _cubicmin.stypy_call_varargs = varargs
    _cubicmin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_cubicmin', ['a', 'fa', 'fpa', 'b', 'fb', 'c', 'fc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_cubicmin', localization, ['a', 'fa', 'fpa', 'b', 'fb', 'c', 'fc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_cubicmin(...)' code ##################

    str_170046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, (-1)), 'str', '\n    Finds the minimizer for a cubic polynomial that goes through the\n    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.\n\n    If no minimizer can be found return None\n\n    ')
    
    # Call to errstate(...): (line 476)
    # Processing the call keyword arguments (line 476)
    str_170049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 28), 'str', 'raise')
    keyword_170050 = str_170049
    str_170051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 42), 'str', 'raise')
    keyword_170052 = str_170051
    str_170053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 59), 'str', 'raise')
    keyword_170054 = str_170053
    kwargs_170055 = {'over': keyword_170052, 'divide': keyword_170050, 'invalid': keyword_170054}
    # Getting the type of 'np' (line 476)
    np_170047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 476)
    errstate_170048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 9), np_170047, 'errstate')
    # Calling errstate(args, kwargs) (line 476)
    errstate_call_result_170056 = invoke(stypy.reporting.localization.Localization(__file__, 476, 9), errstate_170048, *[], **kwargs_170055)
    
    with_170057 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 476, 9), errstate_call_result_170056, 'with parameter', '__enter__', '__exit__')

    if with_170057:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 476)
        enter___170058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 9), errstate_call_result_170056, '__enter__')
        with_enter_170059 = invoke(stypy.reporting.localization.Localization(__file__, 476, 9), enter___170058)
        
        
        # SSA begins for try-except statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Name (line 478):
        
        # Assigning a Name to a Name (line 478):
        # Getting the type of 'fpa' (line 478)
        fpa_170060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 16), 'fpa')
        # Assigning a type to the variable 'C' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'C', fpa_170060)
        
        # Assigning a BinOp to a Name (line 479):
        
        # Assigning a BinOp to a Name (line 479):
        # Getting the type of 'b' (line 479)
        b_170061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 17), 'b')
        # Getting the type of 'a' (line 479)
        a_170062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 21), 'a')
        # Applying the binary operator '-' (line 479)
        result_sub_170063 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 17), '-', b_170061, a_170062)
        
        # Assigning a type to the variable 'db' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'db', result_sub_170063)
        
        # Assigning a BinOp to a Name (line 480):
        
        # Assigning a BinOp to a Name (line 480):
        # Getting the type of 'c' (line 480)
        c_170064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 17), 'c')
        # Getting the type of 'a' (line 480)
        a_170065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 21), 'a')
        # Applying the binary operator '-' (line 480)
        result_sub_170066 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 17), '-', c_170064, a_170065)
        
        # Assigning a type to the variable 'dc' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'dc', result_sub_170066)
        
        # Assigning a BinOp to a Name (line 481):
        
        # Assigning a BinOp to a Name (line 481):
        # Getting the type of 'db' (line 481)
        db_170067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 21), 'db')
        # Getting the type of 'dc' (line 481)
        dc_170068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 26), 'dc')
        # Applying the binary operator '*' (line 481)
        result_mul_170069 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 21), '*', db_170067, dc_170068)
        
        int_170070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 33), 'int')
        # Applying the binary operator '**' (line 481)
        result_pow_170071 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 20), '**', result_mul_170069, int_170070)
        
        # Getting the type of 'db' (line 481)
        db_170072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 38), 'db')
        # Getting the type of 'dc' (line 481)
        dc_170073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 43), 'dc')
        # Applying the binary operator '-' (line 481)
        result_sub_170074 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 38), '-', db_170072, dc_170073)
        
        # Applying the binary operator '*' (line 481)
        result_mul_170075 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 20), '*', result_pow_170071, result_sub_170074)
        
        # Assigning a type to the variable 'denom' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'denom', result_mul_170075)
        
        # Assigning a Call to a Name (line 482):
        
        # Assigning a Call to a Name (line 482):
        
        # Call to empty(...): (line 482)
        # Processing the call arguments (line 482)
        
        # Obtaining an instance of the builtin type 'tuple' (line 482)
        tuple_170078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 482)
        # Adding element type (line 482)
        int_170079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 27), tuple_170078, int_170079)
        # Adding element type (line 482)
        int_170080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 27), tuple_170078, int_170080)
        
        # Processing the call keyword arguments (line 482)
        kwargs_170081 = {}
        # Getting the type of 'np' (line 482)
        np_170076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 482)
        empty_170077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 17), np_170076, 'empty')
        # Calling empty(args, kwargs) (line 482)
        empty_call_result_170082 = invoke(stypy.reporting.localization.Localization(__file__, 482, 17), empty_170077, *[tuple_170078], **kwargs_170081)
        
        # Assigning a type to the variable 'd1' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'd1', empty_call_result_170082)
        
        # Assigning a BinOp to a Subscript (line 483):
        
        # Assigning a BinOp to a Subscript (line 483):
        # Getting the type of 'dc' (line 483)
        dc_170083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 23), 'dc')
        int_170084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 29), 'int')
        # Applying the binary operator '**' (line 483)
        result_pow_170085 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 23), '**', dc_170083, int_170084)
        
        # Getting the type of 'd1' (line 483)
        d1_170086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'd1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 483)
        tuple_170087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 483)
        # Adding element type (line 483)
        int_170088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 15), tuple_170087, int_170088)
        # Adding element type (line 483)
        int_170089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 15), tuple_170087, int_170089)
        
        # Storing an element on a container (line 483)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 12), d1_170086, (tuple_170087, result_pow_170085))
        
        # Assigning a UnaryOp to a Subscript (line 484):
        
        # Assigning a UnaryOp to a Subscript (line 484):
        
        # Getting the type of 'db' (line 484)
        db_170090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 24), 'db')
        int_170091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 30), 'int')
        # Applying the binary operator '**' (line 484)
        result_pow_170092 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 24), '**', db_170090, int_170091)
        
        # Applying the 'usub' unary operator (line 484)
        result___neg___170093 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 23), 'usub', result_pow_170092)
        
        # Getting the type of 'd1' (line 484)
        d1_170094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'd1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 484)
        tuple_170095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 484)
        # Adding element type (line 484)
        int_170096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 15), tuple_170095, int_170096)
        # Adding element type (line 484)
        int_170097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 15), tuple_170095, int_170097)
        
        # Storing an element on a container (line 484)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), d1_170094, (tuple_170095, result___neg___170093))
        
        # Assigning a UnaryOp to a Subscript (line 485):
        
        # Assigning a UnaryOp to a Subscript (line 485):
        
        # Getting the type of 'dc' (line 485)
        dc_170098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 24), 'dc')
        int_170099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 30), 'int')
        # Applying the binary operator '**' (line 485)
        result_pow_170100 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 24), '**', dc_170098, int_170099)
        
        # Applying the 'usub' unary operator (line 485)
        result___neg___170101 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 23), 'usub', result_pow_170100)
        
        # Getting the type of 'd1' (line 485)
        d1_170102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'd1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 485)
        tuple_170103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 485)
        # Adding element type (line 485)
        int_170104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 15), tuple_170103, int_170104)
        # Adding element type (line 485)
        int_170105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 15), tuple_170103, int_170105)
        
        # Storing an element on a container (line 485)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 12), d1_170102, (tuple_170103, result___neg___170101))
        
        # Assigning a BinOp to a Subscript (line 486):
        
        # Assigning a BinOp to a Subscript (line 486):
        # Getting the type of 'db' (line 486)
        db_170106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 23), 'db')
        int_170107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 29), 'int')
        # Applying the binary operator '**' (line 486)
        result_pow_170108 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 23), '**', db_170106, int_170107)
        
        # Getting the type of 'd1' (line 486)
        d1_170109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'd1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 486)
        tuple_170110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 486)
        # Adding element type (line 486)
        int_170111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 15), tuple_170110, int_170111)
        # Adding element type (line 486)
        int_170112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 15), tuple_170110, int_170112)
        
        # Storing an element on a container (line 486)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 12), d1_170109, (tuple_170110, result_pow_170108))
        
        # Assigning a Call to a List (line 487):
        
        # Assigning a Subscript to a Name (line 487):
        
        # Obtaining the type of the subscript
        int_170113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 12), 'int')
        
        # Call to dot(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'd1' (line 487)
        d1_170116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 28), 'd1', False)
        
        # Call to flatten(...): (line 487)
        # Processing the call keyword arguments (line 487)
        kwargs_170137 = {}
        
        # Call to asarray(...): (line 487)
        # Processing the call arguments (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 487)
        list_170119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 487)
        # Adding element type (line 487)
        # Getting the type of 'fb' (line 487)
        fb_170120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 44), 'fb', False)
        # Getting the type of 'fa' (line 487)
        fa_170121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 49), 'fa', False)
        # Applying the binary operator '-' (line 487)
        result_sub_170122 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 44), '-', fb_170120, fa_170121)
        
        # Getting the type of 'C' (line 487)
        C_170123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 54), 'C', False)
        # Getting the type of 'db' (line 487)
        db_170124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 58), 'db', False)
        # Applying the binary operator '*' (line 487)
        result_mul_170125 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 54), '*', C_170123, db_170124)
        
        # Applying the binary operator '-' (line 487)
        result_sub_170126 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 52), '-', result_sub_170122, result_mul_170125)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 43), list_170119, result_sub_170126)
        # Adding element type (line 487)
        # Getting the type of 'fc' (line 488)
        fc_170127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 44), 'fc', False)
        # Getting the type of 'fa' (line 488)
        fa_170128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 49), 'fa', False)
        # Applying the binary operator '-' (line 488)
        result_sub_170129 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 44), '-', fc_170127, fa_170128)
        
        # Getting the type of 'C' (line 488)
        C_170130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 54), 'C', False)
        # Getting the type of 'dc' (line 488)
        dc_170131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 58), 'dc', False)
        # Applying the binary operator '*' (line 488)
        result_mul_170132 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 54), '*', C_170130, dc_170131)
        
        # Applying the binary operator '-' (line 488)
        result_sub_170133 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 52), '-', result_sub_170129, result_mul_170132)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 43), list_170119, result_sub_170133)
        
        # Processing the call keyword arguments (line 487)
        kwargs_170134 = {}
        # Getting the type of 'np' (line 487)
        np_170117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 32), 'np', False)
        # Obtaining the member 'asarray' of a type (line 487)
        asarray_170118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 32), np_170117, 'asarray')
        # Calling asarray(args, kwargs) (line 487)
        asarray_call_result_170135 = invoke(stypy.reporting.localization.Localization(__file__, 487, 32), asarray_170118, *[list_170119], **kwargs_170134)
        
        # Obtaining the member 'flatten' of a type (line 487)
        flatten_170136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 32), asarray_call_result_170135, 'flatten')
        # Calling flatten(args, kwargs) (line 487)
        flatten_call_result_170138 = invoke(stypy.reporting.localization.Localization(__file__, 487, 32), flatten_170136, *[], **kwargs_170137)
        
        # Processing the call keyword arguments (line 487)
        kwargs_170139 = {}
        # Getting the type of 'np' (line 487)
        np_170114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 21), 'np', False)
        # Obtaining the member 'dot' of a type (line 487)
        dot_170115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 21), np_170114, 'dot')
        # Calling dot(args, kwargs) (line 487)
        dot_call_result_170140 = invoke(stypy.reporting.localization.Localization(__file__, 487, 21), dot_170115, *[d1_170116, flatten_call_result_170138], **kwargs_170139)
        
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___170141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), dot_call_result_170140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_170142 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), getitem___170141, int_170113)
        
        # Assigning a type to the variable 'list_var_assignment_169052' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'list_var_assignment_169052', subscript_call_result_170142)
        
        # Assigning a Subscript to a Name (line 487):
        
        # Obtaining the type of the subscript
        int_170143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 12), 'int')
        
        # Call to dot(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'd1' (line 487)
        d1_170146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 28), 'd1', False)
        
        # Call to flatten(...): (line 487)
        # Processing the call keyword arguments (line 487)
        kwargs_170167 = {}
        
        # Call to asarray(...): (line 487)
        # Processing the call arguments (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 487)
        list_170149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 487)
        # Adding element type (line 487)
        # Getting the type of 'fb' (line 487)
        fb_170150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 44), 'fb', False)
        # Getting the type of 'fa' (line 487)
        fa_170151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 49), 'fa', False)
        # Applying the binary operator '-' (line 487)
        result_sub_170152 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 44), '-', fb_170150, fa_170151)
        
        # Getting the type of 'C' (line 487)
        C_170153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 54), 'C', False)
        # Getting the type of 'db' (line 487)
        db_170154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 58), 'db', False)
        # Applying the binary operator '*' (line 487)
        result_mul_170155 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 54), '*', C_170153, db_170154)
        
        # Applying the binary operator '-' (line 487)
        result_sub_170156 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 52), '-', result_sub_170152, result_mul_170155)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 43), list_170149, result_sub_170156)
        # Adding element type (line 487)
        # Getting the type of 'fc' (line 488)
        fc_170157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 44), 'fc', False)
        # Getting the type of 'fa' (line 488)
        fa_170158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 49), 'fa', False)
        # Applying the binary operator '-' (line 488)
        result_sub_170159 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 44), '-', fc_170157, fa_170158)
        
        # Getting the type of 'C' (line 488)
        C_170160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 54), 'C', False)
        # Getting the type of 'dc' (line 488)
        dc_170161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 58), 'dc', False)
        # Applying the binary operator '*' (line 488)
        result_mul_170162 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 54), '*', C_170160, dc_170161)
        
        # Applying the binary operator '-' (line 488)
        result_sub_170163 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 52), '-', result_sub_170159, result_mul_170162)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 43), list_170149, result_sub_170163)
        
        # Processing the call keyword arguments (line 487)
        kwargs_170164 = {}
        # Getting the type of 'np' (line 487)
        np_170147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 32), 'np', False)
        # Obtaining the member 'asarray' of a type (line 487)
        asarray_170148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 32), np_170147, 'asarray')
        # Calling asarray(args, kwargs) (line 487)
        asarray_call_result_170165 = invoke(stypy.reporting.localization.Localization(__file__, 487, 32), asarray_170148, *[list_170149], **kwargs_170164)
        
        # Obtaining the member 'flatten' of a type (line 487)
        flatten_170166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 32), asarray_call_result_170165, 'flatten')
        # Calling flatten(args, kwargs) (line 487)
        flatten_call_result_170168 = invoke(stypy.reporting.localization.Localization(__file__, 487, 32), flatten_170166, *[], **kwargs_170167)
        
        # Processing the call keyword arguments (line 487)
        kwargs_170169 = {}
        # Getting the type of 'np' (line 487)
        np_170144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 21), 'np', False)
        # Obtaining the member 'dot' of a type (line 487)
        dot_170145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 21), np_170144, 'dot')
        # Calling dot(args, kwargs) (line 487)
        dot_call_result_170170 = invoke(stypy.reporting.localization.Localization(__file__, 487, 21), dot_170145, *[d1_170146, flatten_call_result_170168], **kwargs_170169)
        
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___170171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), dot_call_result_170170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_170172 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), getitem___170171, int_170143)
        
        # Assigning a type to the variable 'list_var_assignment_169053' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'list_var_assignment_169053', subscript_call_result_170172)
        
        # Assigning a Name to a Name (line 487):
        # Getting the type of 'list_var_assignment_169052' (line 487)
        list_var_assignment_169052_170173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'list_var_assignment_169052')
        # Assigning a type to the variable 'A' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 13), 'A', list_var_assignment_169052_170173)
        
        # Assigning a Name to a Name (line 487):
        # Getting the type of 'list_var_assignment_169053' (line 487)
        list_var_assignment_169053_170174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'list_var_assignment_169053')
        # Assigning a type to the variable 'B' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'B', list_var_assignment_169053_170174)
        
        # Getting the type of 'A' (line 489)
        A_170175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'A')
        # Getting the type of 'denom' (line 489)
        denom_170176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'denom')
        # Applying the binary operator 'div=' (line 489)
        result_div_170177 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 12), 'div=', A_170175, denom_170176)
        # Assigning a type to the variable 'A' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'A', result_div_170177)
        
        
        # Getting the type of 'B' (line 490)
        B_170178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'B')
        # Getting the type of 'denom' (line 490)
        denom_170179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 17), 'denom')
        # Applying the binary operator 'div=' (line 490)
        result_div_170180 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 12), 'div=', B_170178, denom_170179)
        # Assigning a type to the variable 'B' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'B', result_div_170180)
        
        
        # Assigning a BinOp to a Name (line 491):
        
        # Assigning a BinOp to a Name (line 491):
        # Getting the type of 'B' (line 491)
        B_170181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 22), 'B')
        # Getting the type of 'B' (line 491)
        B_170182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 26), 'B')
        # Applying the binary operator '*' (line 491)
        result_mul_170183 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 22), '*', B_170181, B_170182)
        
        int_170184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 30), 'int')
        # Getting the type of 'A' (line 491)
        A_170185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 34), 'A')
        # Applying the binary operator '*' (line 491)
        result_mul_170186 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 30), '*', int_170184, A_170185)
        
        # Getting the type of 'C' (line 491)
        C_170187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 38), 'C')
        # Applying the binary operator '*' (line 491)
        result_mul_170188 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 36), '*', result_mul_170186, C_170187)
        
        # Applying the binary operator '-' (line 491)
        result_sub_170189 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 22), '-', result_mul_170183, result_mul_170188)
        
        # Assigning a type to the variable 'radical' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'radical', result_sub_170189)
        
        # Assigning a BinOp to a Name (line 492):
        
        # Assigning a BinOp to a Name (line 492):
        # Getting the type of 'a' (line 492)
        a_170190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 19), 'a')
        
        # Getting the type of 'B' (line 492)
        B_170191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 25), 'B')
        # Applying the 'usub' unary operator (line 492)
        result___neg___170192 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 24), 'usub', B_170191)
        
        
        # Call to sqrt(...): (line 492)
        # Processing the call arguments (line 492)
        # Getting the type of 'radical' (line 492)
        radical_170195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 37), 'radical', False)
        # Processing the call keyword arguments (line 492)
        kwargs_170196 = {}
        # Getting the type of 'np' (line 492)
        np_170193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 29), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 492)
        sqrt_170194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 29), np_170193, 'sqrt')
        # Calling sqrt(args, kwargs) (line 492)
        sqrt_call_result_170197 = invoke(stypy.reporting.localization.Localization(__file__, 492, 29), sqrt_170194, *[radical_170195], **kwargs_170196)
        
        # Applying the binary operator '+' (line 492)
        result_add_170198 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 24), '+', result___neg___170192, sqrt_call_result_170197)
        
        int_170199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 50), 'int')
        # Getting the type of 'A' (line 492)
        A_170200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 54), 'A')
        # Applying the binary operator '*' (line 492)
        result_mul_170201 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 50), '*', int_170199, A_170200)
        
        # Applying the binary operator 'div' (line 492)
        result_div_170202 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 23), 'div', result_add_170198, result_mul_170201)
        
        # Applying the binary operator '+' (line 492)
        result_add_170203 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 19), '+', a_170190, result_div_170202)
        
        # Assigning a type to the variable 'xmin' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'xmin', result_add_170203)
        # SSA branch for the except part of a try statement (line 477)
        # SSA branch for the except 'ArithmeticError' branch of a try statement (line 477)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 494)
        None_170204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'stypy_return_type', None_170204)
        # SSA join for try-except statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 476)
        exit___170205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 9), errstate_call_result_170056, '__exit__')
        with_exit_170206 = invoke(stypy.reporting.localization.Localization(__file__, 476, 9), exit___170205, None, None, None)

    
    
    
    # Call to isfinite(...): (line 495)
    # Processing the call arguments (line 495)
    # Getting the type of 'xmin' (line 495)
    xmin_170209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 23), 'xmin', False)
    # Processing the call keyword arguments (line 495)
    kwargs_170210 = {}
    # Getting the type of 'np' (line 495)
    np_170207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 11), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 495)
    isfinite_170208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 11), np_170207, 'isfinite')
    # Calling isfinite(args, kwargs) (line 495)
    isfinite_call_result_170211 = invoke(stypy.reporting.localization.Localization(__file__, 495, 11), isfinite_170208, *[xmin_170209], **kwargs_170210)
    
    # Applying the 'not' unary operator (line 495)
    result_not__170212 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 7), 'not', isfinite_call_result_170211)
    
    # Testing the type of an if condition (line 495)
    if_condition_170213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 495, 4), result_not__170212)
    # Assigning a type to the variable 'if_condition_170213' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'if_condition_170213', if_condition_170213)
    # SSA begins for if statement (line 495)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 496)
    None_170214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', None_170214)
    # SSA join for if statement (line 495)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'xmin' (line 497)
    xmin_170215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 11), 'xmin')
    # Assigning a type to the variable 'stypy_return_type' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'stypy_return_type', xmin_170215)
    
    # ################# End of '_cubicmin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_cubicmin' in the type store
    # Getting the type of 'stypy_return_type' (line 466)
    stypy_return_type_170216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170216)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_cubicmin'
    return stypy_return_type_170216

# Assigning a type to the variable '_cubicmin' (line 466)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), '_cubicmin', _cubicmin)

@norecursion
def _quadmin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_quadmin'
    module_type_store = module_type_store.open_function_context('_quadmin', 500, 0, False)
    
    # Passed parameters checking function
    _quadmin.stypy_localization = localization
    _quadmin.stypy_type_of_self = None
    _quadmin.stypy_type_store = module_type_store
    _quadmin.stypy_function_name = '_quadmin'
    _quadmin.stypy_param_names_list = ['a', 'fa', 'fpa', 'b', 'fb']
    _quadmin.stypy_varargs_param_name = None
    _quadmin.stypy_kwargs_param_name = None
    _quadmin.stypy_call_defaults = defaults
    _quadmin.stypy_call_varargs = varargs
    _quadmin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quadmin', ['a', 'fa', 'fpa', 'b', 'fb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quadmin', localization, ['a', 'fa', 'fpa', 'b', 'fb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quadmin(...)' code ##################

    str_170217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, (-1)), 'str', '\n    Finds the minimizer for a quadratic polynomial that goes through\n    the points (a,fa), (b,fb) with derivative at a of fpa,\n\n    ')
    
    # Call to errstate(...): (line 507)
    # Processing the call keyword arguments (line 507)
    str_170220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 28), 'str', 'raise')
    keyword_170221 = str_170220
    str_170222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 42), 'str', 'raise')
    keyword_170223 = str_170222
    str_170224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 59), 'str', 'raise')
    keyword_170225 = str_170224
    kwargs_170226 = {'over': keyword_170223, 'divide': keyword_170221, 'invalid': keyword_170225}
    # Getting the type of 'np' (line 507)
    np_170218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 507)
    errstate_170219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 9), np_170218, 'errstate')
    # Calling errstate(args, kwargs) (line 507)
    errstate_call_result_170227 = invoke(stypy.reporting.localization.Localization(__file__, 507, 9), errstate_170219, *[], **kwargs_170226)
    
    with_170228 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 507, 9), errstate_call_result_170227, 'with parameter', '__enter__', '__exit__')

    if with_170228:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 507)
        enter___170229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 9), errstate_call_result_170227, '__enter__')
        with_enter_170230 = invoke(stypy.reporting.localization.Localization(__file__, 507, 9), enter___170229)
        
        
        # SSA begins for try-except statement (line 508)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Name (line 509):
        
        # Assigning a Name to a Name (line 509):
        # Getting the type of 'fa' (line 509)
        fa_170231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'fa')
        # Assigning a type to the variable 'D' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'D', fa_170231)
        
        # Assigning a Name to a Name (line 510):
        
        # Assigning a Name to a Name (line 510):
        # Getting the type of 'fpa' (line 510)
        fpa_170232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'fpa')
        # Assigning a type to the variable 'C' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'C', fpa_170232)
        
        # Assigning a BinOp to a Name (line 511):
        
        # Assigning a BinOp to a Name (line 511):
        # Getting the type of 'b' (line 511)
        b_170233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 17), 'b')
        # Getting the type of 'a' (line 511)
        a_170234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 21), 'a')
        float_170235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 25), 'float')
        # Applying the binary operator '*' (line 511)
        result_mul_170236 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 21), '*', a_170234, float_170235)
        
        # Applying the binary operator '-' (line 511)
        result_sub_170237 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 17), '-', b_170233, result_mul_170236)
        
        # Assigning a type to the variable 'db' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'db', result_sub_170237)
        
        # Assigning a BinOp to a Name (line 512):
        
        # Assigning a BinOp to a Name (line 512):
        # Getting the type of 'fb' (line 512)
        fb_170238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 17), 'fb')
        # Getting the type of 'D' (line 512)
        D_170239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 22), 'D')
        # Applying the binary operator '-' (line 512)
        result_sub_170240 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 17), '-', fb_170238, D_170239)
        
        # Getting the type of 'C' (line 512)
        C_170241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 26), 'C')
        # Getting the type of 'db' (line 512)
        db_170242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 30), 'db')
        # Applying the binary operator '*' (line 512)
        result_mul_170243 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 26), '*', C_170241, db_170242)
        
        # Applying the binary operator '-' (line 512)
        result_sub_170244 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 24), '-', result_sub_170240, result_mul_170243)
        
        # Getting the type of 'db' (line 512)
        db_170245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 37), 'db')
        # Getting the type of 'db' (line 512)
        db_170246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 42), 'db')
        # Applying the binary operator '*' (line 512)
        result_mul_170247 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 37), '*', db_170245, db_170246)
        
        # Applying the binary operator 'div' (line 512)
        result_div_170248 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 16), 'div', result_sub_170244, result_mul_170247)
        
        # Assigning a type to the variable 'B' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'B', result_div_170248)
        
        # Assigning a BinOp to a Name (line 513):
        
        # Assigning a BinOp to a Name (line 513):
        # Getting the type of 'a' (line 513)
        a_170249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 19), 'a')
        # Getting the type of 'C' (line 513)
        C_170250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 23), 'C')
        float_170251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 28), 'float')
        # Getting the type of 'B' (line 513)
        B_170252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 34), 'B')
        # Applying the binary operator '*' (line 513)
        result_mul_170253 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 28), '*', float_170251, B_170252)
        
        # Applying the binary operator 'div' (line 513)
        result_div_170254 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 23), 'div', C_170250, result_mul_170253)
        
        # Applying the binary operator '-' (line 513)
        result_sub_170255 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 19), '-', a_170249, result_div_170254)
        
        # Assigning a type to the variable 'xmin' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'xmin', result_sub_170255)
        # SSA branch for the except part of a try statement (line 508)
        # SSA branch for the except 'ArithmeticError' branch of a try statement (line 508)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 515)
        None_170256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'stypy_return_type', None_170256)
        # SSA join for try-except statement (line 508)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 507)
        exit___170257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 9), errstate_call_result_170227, '__exit__')
        with_exit_170258 = invoke(stypy.reporting.localization.Localization(__file__, 507, 9), exit___170257, None, None, None)

    
    
    
    # Call to isfinite(...): (line 516)
    # Processing the call arguments (line 516)
    # Getting the type of 'xmin' (line 516)
    xmin_170261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 23), 'xmin', False)
    # Processing the call keyword arguments (line 516)
    kwargs_170262 = {}
    # Getting the type of 'np' (line 516)
    np_170259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 516)
    isfinite_170260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 11), np_170259, 'isfinite')
    # Calling isfinite(args, kwargs) (line 516)
    isfinite_call_result_170263 = invoke(stypy.reporting.localization.Localization(__file__, 516, 11), isfinite_170260, *[xmin_170261], **kwargs_170262)
    
    # Applying the 'not' unary operator (line 516)
    result_not__170264 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 7), 'not', isfinite_call_result_170263)
    
    # Testing the type of an if condition (line 516)
    if_condition_170265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 516, 4), result_not__170264)
    # Assigning a type to the variable 'if_condition_170265' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'if_condition_170265', if_condition_170265)
    # SSA begins for if statement (line 516)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 517)
    None_170266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'stypy_return_type', None_170266)
    # SSA join for if statement (line 516)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'xmin' (line 518)
    xmin_170267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 11), 'xmin')
    # Assigning a type to the variable 'stypy_return_type' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'stypy_return_type', xmin_170267)
    
    # ################# End of '_quadmin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quadmin' in the type store
    # Getting the type of 'stypy_return_type' (line 500)
    stypy_return_type_170268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170268)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quadmin'
    return stypy_return_type_170268

# Assigning a type to the variable '_quadmin' (line 500)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 0), '_quadmin', _quadmin)

@norecursion
def _zoom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_zoom'
    module_type_store = module_type_store.open_function_context('_zoom', 521, 0, False)
    
    # Passed parameters checking function
    _zoom.stypy_localization = localization
    _zoom.stypy_type_of_self = None
    _zoom.stypy_type_store = module_type_store
    _zoom.stypy_function_name = '_zoom'
    _zoom.stypy_param_names_list = ['a_lo', 'a_hi', 'phi_lo', 'phi_hi', 'derphi_lo', 'phi', 'derphi', 'phi0', 'derphi0', 'c1', 'c2', 'extra_condition']
    _zoom.stypy_varargs_param_name = None
    _zoom.stypy_kwargs_param_name = None
    _zoom.stypy_call_defaults = defaults
    _zoom.stypy_call_varargs = varargs
    _zoom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_zoom', ['a_lo', 'a_hi', 'phi_lo', 'phi_hi', 'derphi_lo', 'phi', 'derphi', 'phi0', 'derphi0', 'c1', 'c2', 'extra_condition'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_zoom', localization, ['a_lo', 'a_hi', 'phi_lo', 'phi_hi', 'derphi_lo', 'phi', 'derphi', 'phi0', 'derphi0', 'c1', 'c2', 'extra_condition'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_zoom(...)' code ##################

    str_170269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, (-1)), 'str', '\n    Part of the optimization algorithm in `scalar_search_wolfe2`.\n    ')
    
    # Assigning a Num to a Name (line 527):
    
    # Assigning a Num to a Name (line 527):
    int_170270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 14), 'int')
    # Assigning a type to the variable 'maxiter' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'maxiter', int_170270)
    
    # Assigning a Num to a Name (line 528):
    
    # Assigning a Num to a Name (line 528):
    int_170271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 8), 'int')
    # Assigning a type to the variable 'i' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'i', int_170271)
    
    # Assigning a Num to a Name (line 529):
    
    # Assigning a Num to a Name (line 529):
    float_170272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 13), 'float')
    # Assigning a type to the variable 'delta1' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'delta1', float_170272)
    
    # Assigning a Num to a Name (line 530):
    
    # Assigning a Num to a Name (line 530):
    float_170273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 13), 'float')
    # Assigning a type to the variable 'delta2' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'delta2', float_170273)
    
    # Assigning a Name to a Name (line 531):
    
    # Assigning a Name to a Name (line 531):
    # Getting the type of 'phi0' (line 531)
    phi0_170274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 14), 'phi0')
    # Assigning a type to the variable 'phi_rec' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'phi_rec', phi0_170274)
    
    # Assigning a Num to a Name (line 532):
    
    # Assigning a Num to a Name (line 532):
    int_170275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 12), 'int')
    # Assigning a type to the variable 'a_rec' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'a_rec', int_170275)
    
    # Getting the type of 'True' (line 533)
    True_170276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 10), 'True')
    # Testing the type of an if condition (line 533)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 533, 4), True_170276)
    # SSA begins for while statement (line 533)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 541):
    
    # Assigning a BinOp to a Name (line 541):
    # Getting the type of 'a_hi' (line 541)
    a_hi_170277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 17), 'a_hi')
    # Getting the type of 'a_lo' (line 541)
    a_lo_170278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 24), 'a_lo')
    # Applying the binary operator '-' (line 541)
    result_sub_170279 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 17), '-', a_hi_170277, a_lo_170278)
    
    # Assigning a type to the variable 'dalpha' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'dalpha', result_sub_170279)
    
    
    # Getting the type of 'dalpha' (line 542)
    dalpha_170280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 11), 'dalpha')
    int_170281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 20), 'int')
    # Applying the binary operator '<' (line 542)
    result_lt_170282 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 11), '<', dalpha_170280, int_170281)
    
    # Testing the type of an if condition (line 542)
    if_condition_170283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 542, 8), result_lt_170282)
    # Assigning a type to the variable 'if_condition_170283' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'if_condition_170283', if_condition_170283)
    # SSA begins for if statement (line 542)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 543):
    
    # Assigning a Name to a Name (line 543):
    # Getting the type of 'a_hi' (line 543)
    a_hi_170284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 19), 'a_hi')
    # Assigning a type to the variable 'tuple_assignment_169054' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'tuple_assignment_169054', a_hi_170284)
    
    # Assigning a Name to a Name (line 543):
    # Getting the type of 'a_lo' (line 543)
    a_lo_170285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 25), 'a_lo')
    # Assigning a type to the variable 'tuple_assignment_169055' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'tuple_assignment_169055', a_lo_170285)
    
    # Assigning a Name to a Name (line 543):
    # Getting the type of 'tuple_assignment_169054' (line 543)
    tuple_assignment_169054_170286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'tuple_assignment_169054')
    # Assigning a type to the variable 'a' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'a', tuple_assignment_169054_170286)
    
    # Assigning a Name to a Name (line 543):
    # Getting the type of 'tuple_assignment_169055' (line 543)
    tuple_assignment_169055_170287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'tuple_assignment_169055')
    # Assigning a type to the variable 'b' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 15), 'b', tuple_assignment_169055_170287)
    # SSA branch for the else part of an if statement (line 542)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 545):
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'a_lo' (line 545)
    a_lo_170288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 19), 'a_lo')
    # Assigning a type to the variable 'tuple_assignment_169056' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'tuple_assignment_169056', a_lo_170288)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'a_hi' (line 545)
    a_hi_170289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 25), 'a_hi')
    # Assigning a type to the variable 'tuple_assignment_169057' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'tuple_assignment_169057', a_hi_170289)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_assignment_169056' (line 545)
    tuple_assignment_169056_170290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'tuple_assignment_169056')
    # Assigning a type to the variable 'a' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'a', tuple_assignment_169056_170290)
    
    # Assigning a Name to a Name (line 545):
    # Getting the type of 'tuple_assignment_169057' (line 545)
    tuple_assignment_169057_170291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'tuple_assignment_169057')
    # Assigning a type to the variable 'b' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 15), 'b', tuple_assignment_169057_170291)
    # SSA join for if statement (line 542)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'i' (line 555)
    i_170292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'i')
    int_170293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 16), 'int')
    # Applying the binary operator '>' (line 555)
    result_gt_170294 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 12), '>', i_170292, int_170293)
    
    # Testing the type of an if condition (line 555)
    if_condition_170295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 8), result_gt_170294)
    # Assigning a type to the variable 'if_condition_170295' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'if_condition_170295', if_condition_170295)
    # SSA begins for if statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 556):
    
    # Assigning a BinOp to a Name (line 556):
    # Getting the type of 'delta1' (line 556)
    delta1_170296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 19), 'delta1')
    # Getting the type of 'dalpha' (line 556)
    dalpha_170297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 28), 'dalpha')
    # Applying the binary operator '*' (line 556)
    result_mul_170298 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 19), '*', delta1_170296, dalpha_170297)
    
    # Assigning a type to the variable 'cchk' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'cchk', result_mul_170298)
    
    # Assigning a Call to a Name (line 557):
    
    # Assigning a Call to a Name (line 557):
    
    # Call to _cubicmin(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'a_lo' (line 557)
    a_lo_170300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 28), 'a_lo', False)
    # Getting the type of 'phi_lo' (line 557)
    phi_lo_170301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 34), 'phi_lo', False)
    # Getting the type of 'derphi_lo' (line 557)
    derphi_lo_170302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 42), 'derphi_lo', False)
    # Getting the type of 'a_hi' (line 557)
    a_hi_170303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 53), 'a_hi', False)
    # Getting the type of 'phi_hi' (line 557)
    phi_hi_170304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 59), 'phi_hi', False)
    # Getting the type of 'a_rec' (line 558)
    a_rec_170305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 28), 'a_rec', False)
    # Getting the type of 'phi_rec' (line 558)
    phi_rec_170306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 35), 'phi_rec', False)
    # Processing the call keyword arguments (line 557)
    kwargs_170307 = {}
    # Getting the type of '_cubicmin' (line 557)
    _cubicmin_170299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 18), '_cubicmin', False)
    # Calling _cubicmin(args, kwargs) (line 557)
    _cubicmin_call_result_170308 = invoke(stypy.reporting.localization.Localization(__file__, 557, 18), _cubicmin_170299, *[a_lo_170300, phi_lo_170301, derphi_lo_170302, a_hi_170303, phi_hi_170304, a_rec_170305, phi_rec_170306], **kwargs_170307)
    
    # Assigning a type to the variable 'a_j' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'a_j', _cubicmin_call_result_170308)
    # SSA join for if statement (line 555)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'i' (line 559)
    i_170309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'i')
    int_170310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 17), 'int')
    # Applying the binary operator '==' (line 559)
    result_eq_170311 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 12), '==', i_170309, int_170310)
    
    
    # Getting the type of 'a_j' (line 559)
    a_j_170312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 24), 'a_j')
    # Getting the type of 'None' (line 559)
    None_170313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 31), 'None')
    # Applying the binary operator 'is' (line 559)
    result_is__170314 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 24), 'is', a_j_170312, None_170313)
    
    # Applying the binary operator 'or' (line 559)
    result_or_keyword_170315 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 11), 'or', result_eq_170311, result_is__170314)
    
    # Getting the type of 'a_j' (line 559)
    a_j_170316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 41), 'a_j')
    # Getting the type of 'b' (line 559)
    b_170317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 47), 'b')
    # Getting the type of 'cchk' (line 559)
    cchk_170318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 51), 'cchk')
    # Applying the binary operator '-' (line 559)
    result_sub_170319 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 47), '-', b_170317, cchk_170318)
    
    # Applying the binary operator '>' (line 559)
    result_gt_170320 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 41), '>', a_j_170316, result_sub_170319)
    
    # Applying the binary operator 'or' (line 559)
    result_or_keyword_170321 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 11), 'or', result_or_keyword_170315, result_gt_170320)
    
    # Getting the type of 'a_j' (line 559)
    a_j_170322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 61), 'a_j')
    # Getting the type of 'a' (line 559)
    a_170323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 67), 'a')
    # Getting the type of 'cchk' (line 559)
    cchk_170324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 71), 'cchk')
    # Applying the binary operator '+' (line 559)
    result_add_170325 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 67), '+', a_170323, cchk_170324)
    
    # Applying the binary operator '<' (line 559)
    result_lt_170326 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 61), '<', a_j_170322, result_add_170325)
    
    # Applying the binary operator 'or' (line 559)
    result_or_keyword_170327 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 11), 'or', result_or_keyword_170321, result_lt_170326)
    
    # Testing the type of an if condition (line 559)
    if_condition_170328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 8), result_or_keyword_170327)
    # Assigning a type to the variable 'if_condition_170328' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'if_condition_170328', if_condition_170328)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 560):
    
    # Assigning a BinOp to a Name (line 560):
    # Getting the type of 'delta2' (line 560)
    delta2_170329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 19), 'delta2')
    # Getting the type of 'dalpha' (line 560)
    dalpha_170330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 28), 'dalpha')
    # Applying the binary operator '*' (line 560)
    result_mul_170331 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 19), '*', delta2_170329, dalpha_170330)
    
    # Assigning a type to the variable 'qchk' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'qchk', result_mul_170331)
    
    # Assigning a Call to a Name (line 561):
    
    # Assigning a Call to a Name (line 561):
    
    # Call to _quadmin(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'a_lo' (line 561)
    a_lo_170333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 27), 'a_lo', False)
    # Getting the type of 'phi_lo' (line 561)
    phi_lo_170334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 33), 'phi_lo', False)
    # Getting the type of 'derphi_lo' (line 561)
    derphi_lo_170335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 41), 'derphi_lo', False)
    # Getting the type of 'a_hi' (line 561)
    a_hi_170336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 52), 'a_hi', False)
    # Getting the type of 'phi_hi' (line 561)
    phi_hi_170337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 58), 'phi_hi', False)
    # Processing the call keyword arguments (line 561)
    kwargs_170338 = {}
    # Getting the type of '_quadmin' (line 561)
    _quadmin_170332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 18), '_quadmin', False)
    # Calling _quadmin(args, kwargs) (line 561)
    _quadmin_call_result_170339 = invoke(stypy.reporting.localization.Localization(__file__, 561, 18), _quadmin_170332, *[a_lo_170333, phi_lo_170334, derphi_lo_170335, a_hi_170336, phi_hi_170337], **kwargs_170338)
    
    # Assigning a type to the variable 'a_j' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'a_j', _quadmin_call_result_170339)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a_j' (line 562)
    a_j_170340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'a_j')
    # Getting the type of 'None' (line 562)
    None_170341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 23), 'None')
    # Applying the binary operator 'is' (line 562)
    result_is__170342 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 16), 'is', a_j_170340, None_170341)
    
    
    # Getting the type of 'a_j' (line 562)
    a_j_170343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 33), 'a_j')
    # Getting the type of 'b' (line 562)
    b_170344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 39), 'b')
    # Getting the type of 'qchk' (line 562)
    qchk_170345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 41), 'qchk')
    # Applying the binary operator '-' (line 562)
    result_sub_170346 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 39), '-', b_170344, qchk_170345)
    
    # Applying the binary operator '>' (line 562)
    result_gt_170347 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 33), '>', a_j_170343, result_sub_170346)
    
    # Applying the binary operator 'or' (line 562)
    result_or_keyword_170348 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 15), 'or', result_is__170342, result_gt_170347)
    
    # Getting the type of 'a_j' (line 562)
    a_j_170349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 51), 'a_j')
    # Getting the type of 'a' (line 562)
    a_170350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 57), 'a')
    # Getting the type of 'qchk' (line 562)
    qchk_170351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 59), 'qchk')
    # Applying the binary operator '+' (line 562)
    result_add_170352 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 57), '+', a_170350, qchk_170351)
    
    # Applying the binary operator '<' (line 562)
    result_lt_170353 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 51), '<', a_j_170349, result_add_170352)
    
    # Applying the binary operator 'or' (line 562)
    result_or_keyword_170354 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 15), 'or', result_or_keyword_170348, result_lt_170353)
    
    # Testing the type of an if condition (line 562)
    if_condition_170355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 12), result_or_keyword_170354)
    # Assigning a type to the variable 'if_condition_170355' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'if_condition_170355', if_condition_170355)
    # SSA begins for if statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 563):
    
    # Assigning a BinOp to a Name (line 563):
    # Getting the type of 'a_lo' (line 563)
    a_lo_170356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 22), 'a_lo')
    float_170357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 29), 'float')
    # Getting the type of 'dalpha' (line 563)
    dalpha_170358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 33), 'dalpha')
    # Applying the binary operator '*' (line 563)
    result_mul_170359 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 29), '*', float_170357, dalpha_170358)
    
    # Applying the binary operator '+' (line 563)
    result_add_170360 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 22), '+', a_lo_170356, result_mul_170359)
    
    # Assigning a type to the variable 'a_j' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'a_j', result_add_170360)
    # SSA join for if statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to phi(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'a_j' (line 567)
    a_j_170362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 'a_j', False)
    # Processing the call keyword arguments (line 567)
    kwargs_170363 = {}
    # Getting the type of 'phi' (line 567)
    phi_170361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 17), 'phi', False)
    # Calling phi(args, kwargs) (line 567)
    phi_call_result_170364 = invoke(stypy.reporting.localization.Localization(__file__, 567, 17), phi_170361, *[a_j_170362], **kwargs_170363)
    
    # Assigning a type to the variable 'phi_aj' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'phi_aj', phi_call_result_170364)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'phi_aj' (line 568)
    phi_aj_170365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'phi_aj')
    # Getting the type of 'phi0' (line 568)
    phi0_170366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 21), 'phi0')
    # Getting the type of 'c1' (line 568)
    c1_170367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 28), 'c1')
    # Getting the type of 'a_j' (line 568)
    a_j_170368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 31), 'a_j')
    # Applying the binary operator '*' (line 568)
    result_mul_170369 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 28), '*', c1_170367, a_j_170368)
    
    # Getting the type of 'derphi0' (line 568)
    derphi0_170370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 35), 'derphi0')
    # Applying the binary operator '*' (line 568)
    result_mul_170371 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 34), '*', result_mul_170369, derphi0_170370)
    
    # Applying the binary operator '+' (line 568)
    result_add_170372 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 21), '+', phi0_170366, result_mul_170371)
    
    # Applying the binary operator '>' (line 568)
    result_gt_170373 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 12), '>', phi_aj_170365, result_add_170372)
    
    
    # Getting the type of 'phi_aj' (line 568)
    phi_aj_170374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 48), 'phi_aj')
    # Getting the type of 'phi_lo' (line 568)
    phi_lo_170375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 58), 'phi_lo')
    # Applying the binary operator '>=' (line 568)
    result_ge_170376 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 48), '>=', phi_aj_170374, phi_lo_170375)
    
    # Applying the binary operator 'or' (line 568)
    result_or_keyword_170377 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 11), 'or', result_gt_170373, result_ge_170376)
    
    # Testing the type of an if condition (line 568)
    if_condition_170378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 8), result_or_keyword_170377)
    # Assigning a type to the variable 'if_condition_170378' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'if_condition_170378', if_condition_170378)
    # SSA begins for if statement (line 568)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 569):
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'phi_hi' (line 569)
    phi_hi_170379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 22), 'phi_hi')
    # Assigning a type to the variable 'phi_rec' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'phi_rec', phi_hi_170379)
    
    # Assigning a Name to a Name (line 570):
    
    # Assigning a Name to a Name (line 570):
    # Getting the type of 'a_hi' (line 570)
    a_hi_170380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 20), 'a_hi')
    # Assigning a type to the variable 'a_rec' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'a_rec', a_hi_170380)
    
    # Assigning a Name to a Name (line 571):
    
    # Assigning a Name to a Name (line 571):
    # Getting the type of 'a_j' (line 571)
    a_j_170381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 19), 'a_j')
    # Assigning a type to the variable 'a_hi' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'a_hi', a_j_170381)
    
    # Assigning a Name to a Name (line 572):
    
    # Assigning a Name to a Name (line 572):
    # Getting the type of 'phi_aj' (line 572)
    phi_aj_170382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 21), 'phi_aj')
    # Assigning a type to the variable 'phi_hi' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'phi_hi', phi_aj_170382)
    # SSA branch for the else part of an if statement (line 568)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 574):
    
    # Assigning a Call to a Name (line 574):
    
    # Call to derphi(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'a_j' (line 574)
    a_j_170384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 31), 'a_j', False)
    # Processing the call keyword arguments (line 574)
    kwargs_170385 = {}
    # Getting the type of 'derphi' (line 574)
    derphi_170383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 24), 'derphi', False)
    # Calling derphi(args, kwargs) (line 574)
    derphi_call_result_170386 = invoke(stypy.reporting.localization.Localization(__file__, 574, 24), derphi_170383, *[a_j_170384], **kwargs_170385)
    
    # Assigning a type to the variable 'derphi_aj' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'derphi_aj', derphi_call_result_170386)
    
    
    # Evaluating a boolean operation
    
    
    # Call to abs(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'derphi_aj' (line 575)
    derphi_aj_170388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 19), 'derphi_aj', False)
    # Processing the call keyword arguments (line 575)
    kwargs_170389 = {}
    # Getting the type of 'abs' (line 575)
    abs_170387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 15), 'abs', False)
    # Calling abs(args, kwargs) (line 575)
    abs_call_result_170390 = invoke(stypy.reporting.localization.Localization(__file__, 575, 15), abs_170387, *[derphi_aj_170388], **kwargs_170389)
    
    
    # Getting the type of 'c2' (line 575)
    c2_170391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 34), 'c2')
    # Applying the 'usub' unary operator (line 575)
    result___neg___170392 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 33), 'usub', c2_170391)
    
    # Getting the type of 'derphi0' (line 575)
    derphi0_170393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 37), 'derphi0')
    # Applying the binary operator '*' (line 575)
    result_mul_170394 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 33), '*', result___neg___170392, derphi0_170393)
    
    # Applying the binary operator '<=' (line 575)
    result_le_170395 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 15), '<=', abs_call_result_170390, result_mul_170394)
    
    
    # Call to extra_condition(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'a_j' (line 575)
    a_j_170397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 65), 'a_j', False)
    # Getting the type of 'phi_aj' (line 575)
    phi_aj_170398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 70), 'phi_aj', False)
    # Processing the call keyword arguments (line 575)
    kwargs_170399 = {}
    # Getting the type of 'extra_condition' (line 575)
    extra_condition_170396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 49), 'extra_condition', False)
    # Calling extra_condition(args, kwargs) (line 575)
    extra_condition_call_result_170400 = invoke(stypy.reporting.localization.Localization(__file__, 575, 49), extra_condition_170396, *[a_j_170397, phi_aj_170398], **kwargs_170399)
    
    # Applying the binary operator 'and' (line 575)
    result_and_keyword_170401 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 15), 'and', result_le_170395, extra_condition_call_result_170400)
    
    # Testing the type of an if condition (line 575)
    if_condition_170402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 12), result_and_keyword_170401)
    # Assigning a type to the variable 'if_condition_170402' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'if_condition_170402', if_condition_170402)
    # SSA begins for if statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 576):
    
    # Assigning a Name to a Name (line 576):
    # Getting the type of 'a_j' (line 576)
    a_j_170403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 25), 'a_j')
    # Assigning a type to the variable 'a_star' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'a_star', a_j_170403)
    
    # Assigning a Name to a Name (line 577):
    
    # Assigning a Name to a Name (line 577):
    # Getting the type of 'phi_aj' (line 577)
    phi_aj_170404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 27), 'phi_aj')
    # Assigning a type to the variable 'val_star' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'val_star', phi_aj_170404)
    
    # Assigning a Name to a Name (line 578):
    
    # Assigning a Name to a Name (line 578):
    # Getting the type of 'derphi_aj' (line 578)
    derphi_aj_170405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 32), 'derphi_aj')
    # Assigning a type to the variable 'valprime_star' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'valprime_star', derphi_aj_170405)
    # SSA join for if statement (line 575)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'derphi_aj' (line 580)
    derphi_aj_170406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'derphi_aj')
    # Getting the type of 'a_hi' (line 580)
    a_hi_170407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 26), 'a_hi')
    # Getting the type of 'a_lo' (line 580)
    a_lo_170408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 33), 'a_lo')
    # Applying the binary operator '-' (line 580)
    result_sub_170409 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 26), '-', a_hi_170407, a_lo_170408)
    
    # Applying the binary operator '*' (line 580)
    result_mul_170410 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 15), '*', derphi_aj_170406, result_sub_170409)
    
    int_170411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 42), 'int')
    # Applying the binary operator '>=' (line 580)
    result_ge_170412 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 15), '>=', result_mul_170410, int_170411)
    
    # Testing the type of an if condition (line 580)
    if_condition_170413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 12), result_ge_170412)
    # Assigning a type to the variable 'if_condition_170413' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'if_condition_170413', if_condition_170413)
    # SSA begins for if statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 581):
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'phi_hi' (line 581)
    phi_hi_170414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 26), 'phi_hi')
    # Assigning a type to the variable 'phi_rec' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'phi_rec', phi_hi_170414)
    
    # Assigning a Name to a Name (line 582):
    
    # Assigning a Name to a Name (line 582):
    # Getting the type of 'a_hi' (line 582)
    a_hi_170415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 24), 'a_hi')
    # Assigning a type to the variable 'a_rec' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'a_rec', a_hi_170415)
    
    # Assigning a Name to a Name (line 583):
    
    # Assigning a Name to a Name (line 583):
    # Getting the type of 'a_lo' (line 583)
    a_lo_170416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 23), 'a_lo')
    # Assigning a type to the variable 'a_hi' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 16), 'a_hi', a_lo_170416)
    
    # Assigning a Name to a Name (line 584):
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'phi_lo' (line 584)
    phi_lo_170417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 25), 'phi_lo')
    # Assigning a type to the variable 'phi_hi' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'phi_hi', phi_lo_170417)
    # SSA branch for the else part of an if statement (line 580)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 586):
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'phi_lo' (line 586)
    phi_lo_170418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 26), 'phi_lo')
    # Assigning a type to the variable 'phi_rec' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'phi_rec', phi_lo_170418)
    
    # Assigning a Name to a Name (line 587):
    
    # Assigning a Name to a Name (line 587):
    # Getting the type of 'a_lo' (line 587)
    a_lo_170419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 24), 'a_lo')
    # Assigning a type to the variable 'a_rec' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'a_rec', a_lo_170419)
    # SSA join for if statement (line 580)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 588):
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'a_j' (line 588)
    a_j_170420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 19), 'a_j')
    # Assigning a type to the variable 'a_lo' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'a_lo', a_j_170420)
    
    # Assigning a Name to a Name (line 589):
    
    # Assigning a Name to a Name (line 589):
    # Getting the type of 'phi_aj' (line 589)
    phi_aj_170421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'phi_aj')
    # Assigning a type to the variable 'phi_lo' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'phi_lo', phi_aj_170421)
    
    # Assigning a Name to a Name (line 590):
    
    # Assigning a Name to a Name (line 590):
    # Getting the type of 'derphi_aj' (line 590)
    derphi_aj_170422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 24), 'derphi_aj')
    # Assigning a type to the variable 'derphi_lo' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'derphi_lo', derphi_aj_170422)
    # SSA join for if statement (line 568)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'i' (line 591)
    i_170423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'i')
    int_170424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 13), 'int')
    # Applying the binary operator '+=' (line 591)
    result_iadd_170425 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 8), '+=', i_170423, int_170424)
    # Assigning a type to the variable 'i' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'i', result_iadd_170425)
    
    
    
    # Getting the type of 'i' (line 592)
    i_170426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'i')
    # Getting the type of 'maxiter' (line 592)
    maxiter_170427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'maxiter')
    # Applying the binary operator '>' (line 592)
    result_gt_170428 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 12), '>', i_170426, maxiter_170427)
    
    # Testing the type of an if condition (line 592)
    if_condition_170429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 8), result_gt_170428)
    # Assigning a type to the variable 'if_condition_170429' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'if_condition_170429', if_condition_170429)
    # SSA begins for if statement (line 592)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 594):
    
    # Assigning a Name to a Name (line 594):
    # Getting the type of 'None' (line 594)
    None_170430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 21), 'None')
    # Assigning a type to the variable 'a_star' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'a_star', None_170430)
    
    # Assigning a Name to a Name (line 595):
    
    # Assigning a Name to a Name (line 595):
    # Getting the type of 'None' (line 595)
    None_170431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 23), 'None')
    # Assigning a type to the variable 'val_star' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'val_star', None_170431)
    
    # Assigning a Name to a Name (line 596):
    
    # Assigning a Name to a Name (line 596):
    # Getting the type of 'None' (line 596)
    None_170432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 28), 'None')
    # Assigning a type to the variable 'valprime_star' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 12), 'valprime_star', None_170432)
    # SSA join for if statement (line 592)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 533)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 598)
    tuple_170433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 598)
    # Adding element type (line 598)
    # Getting the type of 'a_star' (line 598)
    a_star_170434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 11), 'a_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 11), tuple_170433, a_star_170434)
    # Adding element type (line 598)
    # Getting the type of 'val_star' (line 598)
    val_star_170435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 19), 'val_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 11), tuple_170433, val_star_170435)
    # Adding element type (line 598)
    # Getting the type of 'valprime_star' (line 598)
    valprime_star_170436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 29), 'valprime_star')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 11), tuple_170433, valprime_star_170436)
    
    # Assigning a type to the variable 'stypy_return_type' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'stypy_return_type', tuple_170433)
    
    # ################# End of '_zoom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_zoom' in the type store
    # Getting the type of 'stypy_return_type' (line 521)
    stypy_return_type_170437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170437)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_zoom'
    return stypy_return_type_170437

# Assigning a type to the variable '_zoom' (line 521)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 0), '_zoom', _zoom)

@norecursion
def line_search_armijo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 605)
    tuple_170438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 605)
    
    float_170439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 61), 'float')
    int_170440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 74), 'int')
    defaults = [tuple_170438, float_170439, int_170440]
    # Create a new context for function 'line_search_armijo'
    module_type_store = module_type_store.open_function_context('line_search_armijo', 605, 0, False)
    
    # Passed parameters checking function
    line_search_armijo.stypy_localization = localization
    line_search_armijo.stypy_type_of_self = None
    line_search_armijo.stypy_type_store = module_type_store
    line_search_armijo.stypy_function_name = 'line_search_armijo'
    line_search_armijo.stypy_param_names_list = ['f', 'xk', 'pk', 'gfk', 'old_fval', 'args', 'c1', 'alpha0']
    line_search_armijo.stypy_varargs_param_name = None
    line_search_armijo.stypy_kwargs_param_name = None
    line_search_armijo.stypy_call_defaults = defaults
    line_search_armijo.stypy_call_varargs = varargs
    line_search_armijo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'line_search_armijo', ['f', 'xk', 'pk', 'gfk', 'old_fval', 'args', 'c1', 'alpha0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'line_search_armijo', localization, ['f', 'xk', 'pk', 'gfk', 'old_fval', 'args', 'c1', 'alpha0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'line_search_armijo(...)' code ##################

    str_170441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, (-1)), 'str', "Minimize over alpha, the function ``f(xk+alpha pk)``.\n\n    Parameters\n    ----------\n    f : callable\n        Function to be minimized.\n    xk : array_like\n        Current point.\n    pk : array_like\n        Search direction.\n    gfk : array_like\n        Gradient of `f` at point `xk`.\n    old_fval : float\n        Value of `f` at point `xk`.\n    args : tuple, optional\n        Optional arguments.\n    c1 : float, optional\n        Value to control stopping criterion.\n    alpha0 : scalar, optional\n        Value of `alpha` at start of the optimization.\n\n    Returns\n    -------\n    alpha\n    f_count\n    f_val_at_alpha\n\n    Notes\n    -----\n    Uses the interpolation algorithm (Armijo backtracking) as suggested by\n    Wright and Nocedal in 'Numerical Optimization', 1999, pg. 56-57\n\n    ")
    
    # Assigning a Call to a Name (line 639):
    
    # Assigning a Call to a Name (line 639):
    
    # Call to atleast_1d(...): (line 639)
    # Processing the call arguments (line 639)
    # Getting the type of 'xk' (line 639)
    xk_170444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 23), 'xk', False)
    # Processing the call keyword arguments (line 639)
    kwargs_170445 = {}
    # Getting the type of 'np' (line 639)
    np_170442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 9), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 639)
    atleast_1d_170443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 9), np_170442, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 639)
    atleast_1d_call_result_170446 = invoke(stypy.reporting.localization.Localization(__file__, 639, 9), atleast_1d_170443, *[xk_170444], **kwargs_170445)
    
    # Assigning a type to the variable 'xk' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'xk', atleast_1d_call_result_170446)
    
    # Assigning a List to a Name (line 640):
    
    # Assigning a List to a Name (line 640):
    
    # Obtaining an instance of the builtin type 'list' (line 640)
    list_170447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 640)
    # Adding element type (line 640)
    int_170448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 9), list_170447, int_170448)
    
    # Assigning a type to the variable 'fc' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'fc', list_170447)

    @norecursion
    def phi(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'phi'
        module_type_store = module_type_store.open_function_context('phi', 642, 4, False)
        
        # Passed parameters checking function
        phi.stypy_localization = localization
        phi.stypy_type_of_self = None
        phi.stypy_type_store = module_type_store
        phi.stypy_function_name = 'phi'
        phi.stypy_param_names_list = ['alpha1']
        phi.stypy_varargs_param_name = None
        phi.stypy_kwargs_param_name = None
        phi.stypy_call_defaults = defaults
        phi.stypy_call_varargs = varargs
        phi.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'phi', ['alpha1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'phi', localization, ['alpha1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'phi(...)' code ##################

        
        # Getting the type of 'fc' (line 643)
        fc_170449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'fc')
        
        # Obtaining the type of the subscript
        int_170450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 11), 'int')
        # Getting the type of 'fc' (line 643)
        fc_170451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'fc')
        # Obtaining the member '__getitem__' of a type (line 643)
        getitem___170452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), fc_170451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 643)
        subscript_call_result_170453 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), getitem___170452, int_170450)
        
        int_170454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 17), 'int')
        # Applying the binary operator '+=' (line 643)
        result_iadd_170455 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 8), '+=', subscript_call_result_170453, int_170454)
        # Getting the type of 'fc' (line 643)
        fc_170456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'fc')
        int_170457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 11), 'int')
        # Storing an element on a container (line 643)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 8), fc_170456, (int_170457, result_iadd_170455))
        
        
        # Call to f(...): (line 644)
        # Processing the call arguments (line 644)
        # Getting the type of 'xk' (line 644)
        xk_170459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 17), 'xk', False)
        # Getting the type of 'alpha1' (line 644)
        alpha1_170460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 22), 'alpha1', False)
        # Getting the type of 'pk' (line 644)
        pk_170461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 29), 'pk', False)
        # Applying the binary operator '*' (line 644)
        result_mul_170462 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 22), '*', alpha1_170460, pk_170461)
        
        # Applying the binary operator '+' (line 644)
        result_add_170463 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 17), '+', xk_170459, result_mul_170462)
        
        # Getting the type of 'args' (line 644)
        args_170464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 34), 'args', False)
        # Processing the call keyword arguments (line 644)
        kwargs_170465 = {}
        # Getting the type of 'f' (line 644)
        f_170458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 15), 'f', False)
        # Calling f(args, kwargs) (line 644)
        f_call_result_170466 = invoke(stypy.reporting.localization.Localization(__file__, 644, 15), f_170458, *[result_add_170463, args_170464], **kwargs_170465)
        
        # Assigning a type to the variable 'stypy_return_type' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'stypy_return_type', f_call_result_170466)
        
        # ################# End of 'phi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'phi' in the type store
        # Getting the type of 'stypy_return_type' (line 642)
        stypy_return_type_170467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_170467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'phi'
        return stypy_return_type_170467

    # Assigning a type to the variable 'phi' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'phi', phi)
    
    # Type idiom detected: calculating its left and rigth part (line 646)
    # Getting the type of 'old_fval' (line 646)
    old_fval_170468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 7), 'old_fval')
    # Getting the type of 'None' (line 646)
    None_170469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 19), 'None')
    
    (may_be_170470, more_types_in_union_170471) = may_be_none(old_fval_170468, None_170469)

    if may_be_170470:

        if more_types_in_union_170471:
            # Runtime conditional SSA (line 646)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 647):
        
        # Assigning a Call to a Name (line 647):
        
        # Call to phi(...): (line 647)
        # Processing the call arguments (line 647)
        float_170473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 19), 'float')
        # Processing the call keyword arguments (line 647)
        kwargs_170474 = {}
        # Getting the type of 'phi' (line 647)
        phi_170472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'phi', False)
        # Calling phi(args, kwargs) (line 647)
        phi_call_result_170475 = invoke(stypy.reporting.localization.Localization(__file__, 647, 15), phi_170472, *[float_170473], **kwargs_170474)
        
        # Assigning a type to the variable 'phi0' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'phi0', phi_call_result_170475)

        if more_types_in_union_170471:
            # Runtime conditional SSA for else branch (line 646)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_170470) or more_types_in_union_170471):
        
        # Assigning a Name to a Name (line 649):
        
        # Assigning a Name to a Name (line 649):
        # Getting the type of 'old_fval' (line 649)
        old_fval_170476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 15), 'old_fval')
        # Assigning a type to the variable 'phi0' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'phi0', old_fval_170476)

        if (may_be_170470 and more_types_in_union_170471):
            # SSA join for if statement (line 646)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 651):
    
    # Assigning a Call to a Name (line 651):
    
    # Call to dot(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'gfk' (line 651)
    gfk_170479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 21), 'gfk', False)
    # Getting the type of 'pk' (line 651)
    pk_170480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 26), 'pk', False)
    # Processing the call keyword arguments (line 651)
    kwargs_170481 = {}
    # Getting the type of 'np' (line 651)
    np_170477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 14), 'np', False)
    # Obtaining the member 'dot' of a type (line 651)
    dot_170478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 14), np_170477, 'dot')
    # Calling dot(args, kwargs) (line 651)
    dot_call_result_170482 = invoke(stypy.reporting.localization.Localization(__file__, 651, 14), dot_170478, *[gfk_170479, pk_170480], **kwargs_170481)
    
    # Assigning a type to the variable 'derphi0' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 4), 'derphi0', dot_call_result_170482)
    
    # Assigning a Call to a Tuple (line 652):
    
    # Assigning a Subscript to a Name (line 652):
    
    # Obtaining the type of the subscript
    int_170483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 4), 'int')
    
    # Call to scalar_search_armijo(...): (line 652)
    # Processing the call arguments (line 652)
    # Getting the type of 'phi' (line 652)
    phi_170485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 39), 'phi', False)
    # Getting the type of 'phi0' (line 652)
    phi0_170486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 44), 'phi0', False)
    # Getting the type of 'derphi0' (line 652)
    derphi0_170487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 50), 'derphi0', False)
    # Processing the call keyword arguments (line 652)
    # Getting the type of 'c1' (line 652)
    c1_170488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 62), 'c1', False)
    keyword_170489 = c1_170488
    # Getting the type of 'alpha0' (line 653)
    alpha0_170490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 46), 'alpha0', False)
    keyword_170491 = alpha0_170490
    kwargs_170492 = {'c1': keyword_170489, 'alpha0': keyword_170491}
    # Getting the type of 'scalar_search_armijo' (line 652)
    scalar_search_armijo_170484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 18), 'scalar_search_armijo', False)
    # Calling scalar_search_armijo(args, kwargs) (line 652)
    scalar_search_armijo_call_result_170493 = invoke(stypy.reporting.localization.Localization(__file__, 652, 18), scalar_search_armijo_170484, *[phi_170485, phi0_170486, derphi0_170487], **kwargs_170492)
    
    # Obtaining the member '__getitem__' of a type (line 652)
    getitem___170494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 4), scalar_search_armijo_call_result_170493, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 652)
    subscript_call_result_170495 = invoke(stypy.reporting.localization.Localization(__file__, 652, 4), getitem___170494, int_170483)
    
    # Assigning a type to the variable 'tuple_var_assignment_169058' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'tuple_var_assignment_169058', subscript_call_result_170495)
    
    # Assigning a Subscript to a Name (line 652):
    
    # Obtaining the type of the subscript
    int_170496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 4), 'int')
    
    # Call to scalar_search_armijo(...): (line 652)
    # Processing the call arguments (line 652)
    # Getting the type of 'phi' (line 652)
    phi_170498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 39), 'phi', False)
    # Getting the type of 'phi0' (line 652)
    phi0_170499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 44), 'phi0', False)
    # Getting the type of 'derphi0' (line 652)
    derphi0_170500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 50), 'derphi0', False)
    # Processing the call keyword arguments (line 652)
    # Getting the type of 'c1' (line 652)
    c1_170501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 62), 'c1', False)
    keyword_170502 = c1_170501
    # Getting the type of 'alpha0' (line 653)
    alpha0_170503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 46), 'alpha0', False)
    keyword_170504 = alpha0_170503
    kwargs_170505 = {'c1': keyword_170502, 'alpha0': keyword_170504}
    # Getting the type of 'scalar_search_armijo' (line 652)
    scalar_search_armijo_170497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 18), 'scalar_search_armijo', False)
    # Calling scalar_search_armijo(args, kwargs) (line 652)
    scalar_search_armijo_call_result_170506 = invoke(stypy.reporting.localization.Localization(__file__, 652, 18), scalar_search_armijo_170497, *[phi_170498, phi0_170499, derphi0_170500], **kwargs_170505)
    
    # Obtaining the member '__getitem__' of a type (line 652)
    getitem___170507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 4), scalar_search_armijo_call_result_170506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 652)
    subscript_call_result_170508 = invoke(stypy.reporting.localization.Localization(__file__, 652, 4), getitem___170507, int_170496)
    
    # Assigning a type to the variable 'tuple_var_assignment_169059' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'tuple_var_assignment_169059', subscript_call_result_170508)
    
    # Assigning a Name to a Name (line 652):
    # Getting the type of 'tuple_var_assignment_169058' (line 652)
    tuple_var_assignment_169058_170509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'tuple_var_assignment_169058')
    # Assigning a type to the variable 'alpha' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'alpha', tuple_var_assignment_169058_170509)
    
    # Assigning a Name to a Name (line 652):
    # Getting the type of 'tuple_var_assignment_169059' (line 652)
    tuple_var_assignment_169059_170510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'tuple_var_assignment_169059')
    # Assigning a type to the variable 'phi1' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 11), 'phi1', tuple_var_assignment_169059_170510)
    
    # Obtaining an instance of the builtin type 'tuple' (line 654)
    tuple_170511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 654)
    # Adding element type (line 654)
    # Getting the type of 'alpha' (line 654)
    alpha_170512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 11), 'alpha')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 11), tuple_170511, alpha_170512)
    # Adding element type (line 654)
    
    # Obtaining the type of the subscript
    int_170513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 21), 'int')
    # Getting the type of 'fc' (line 654)
    fc_170514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 18), 'fc')
    # Obtaining the member '__getitem__' of a type (line 654)
    getitem___170515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 18), fc_170514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 654)
    subscript_call_result_170516 = invoke(stypy.reporting.localization.Localization(__file__, 654, 18), getitem___170515, int_170513)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 11), tuple_170511, subscript_call_result_170516)
    # Adding element type (line 654)
    # Getting the type of 'phi1' (line 654)
    phi1_170517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 25), 'phi1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 11), tuple_170511, phi1_170517)
    
    # Assigning a type to the variable 'stypy_return_type' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'stypy_return_type', tuple_170511)
    
    # ################# End of 'line_search_armijo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'line_search_armijo' in the type store
    # Getting the type of 'stypy_return_type' (line 605)
    stypy_return_type_170518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170518)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'line_search_armijo'
    return stypy_return_type_170518

# Assigning a type to the variable 'line_search_armijo' (line 605)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 0), 'line_search_armijo', line_search_armijo)

@norecursion
def line_search_BFGS(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 657)
    tuple_170519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 657)
    
    float_170520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 59), 'float')
    int_170521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 72), 'int')
    defaults = [tuple_170519, float_170520, int_170521]
    # Create a new context for function 'line_search_BFGS'
    module_type_store = module_type_store.open_function_context('line_search_BFGS', 657, 0, False)
    
    # Passed parameters checking function
    line_search_BFGS.stypy_localization = localization
    line_search_BFGS.stypy_type_of_self = None
    line_search_BFGS.stypy_type_store = module_type_store
    line_search_BFGS.stypy_function_name = 'line_search_BFGS'
    line_search_BFGS.stypy_param_names_list = ['f', 'xk', 'pk', 'gfk', 'old_fval', 'args', 'c1', 'alpha0']
    line_search_BFGS.stypy_varargs_param_name = None
    line_search_BFGS.stypy_kwargs_param_name = None
    line_search_BFGS.stypy_call_defaults = defaults
    line_search_BFGS.stypy_call_varargs = varargs
    line_search_BFGS.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'line_search_BFGS', ['f', 'xk', 'pk', 'gfk', 'old_fval', 'args', 'c1', 'alpha0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'line_search_BFGS', localization, ['f', 'xk', 'pk', 'gfk', 'old_fval', 'args', 'c1', 'alpha0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'line_search_BFGS(...)' code ##################

    str_170522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, (-1)), 'str', '\n    Compatibility wrapper for `line_search_armijo`\n    ')
    
    # Assigning a Call to a Name (line 661):
    
    # Assigning a Call to a Name (line 661):
    
    # Call to line_search_armijo(...): (line 661)
    # Processing the call arguments (line 661)
    # Getting the type of 'f' (line 661)
    f_170524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 27), 'f', False)
    # Getting the type of 'xk' (line 661)
    xk_170525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 30), 'xk', False)
    # Getting the type of 'pk' (line 661)
    pk_170526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 34), 'pk', False)
    # Getting the type of 'gfk' (line 661)
    gfk_170527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 38), 'gfk', False)
    # Getting the type of 'old_fval' (line 661)
    old_fval_170528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 43), 'old_fval', False)
    # Processing the call keyword arguments (line 661)
    # Getting the type of 'args' (line 661)
    args_170529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 58), 'args', False)
    keyword_170530 = args_170529
    # Getting the type of 'c1' (line 661)
    c1_170531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 67), 'c1', False)
    keyword_170532 = c1_170531
    # Getting the type of 'alpha0' (line 662)
    alpha0_170533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 34), 'alpha0', False)
    keyword_170534 = alpha0_170533
    kwargs_170535 = {'c1': keyword_170532, 'args': keyword_170530, 'alpha0': keyword_170534}
    # Getting the type of 'line_search_armijo' (line 661)
    line_search_armijo_170523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'line_search_armijo', False)
    # Calling line_search_armijo(args, kwargs) (line 661)
    line_search_armijo_call_result_170536 = invoke(stypy.reporting.localization.Localization(__file__, 661, 8), line_search_armijo_170523, *[f_170524, xk_170525, pk_170526, gfk_170527, old_fval_170528], **kwargs_170535)
    
    # Assigning a type to the variable 'r' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'r', line_search_armijo_call_result_170536)
    
    # Obtaining an instance of the builtin type 'tuple' (line 663)
    tuple_170537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 663)
    # Adding element type (line 663)
    
    # Obtaining the type of the subscript
    int_170538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 13), 'int')
    # Getting the type of 'r' (line 663)
    r_170539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 11), 'r')
    # Obtaining the member '__getitem__' of a type (line 663)
    getitem___170540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 11), r_170539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 663)
    subscript_call_result_170541 = invoke(stypy.reporting.localization.Localization(__file__, 663, 11), getitem___170540, int_170538)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 11), tuple_170537, subscript_call_result_170541)
    # Adding element type (line 663)
    
    # Obtaining the type of the subscript
    int_170542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 19), 'int')
    # Getting the type of 'r' (line 663)
    r_170543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 17), 'r')
    # Obtaining the member '__getitem__' of a type (line 663)
    getitem___170544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 17), r_170543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 663)
    subscript_call_result_170545 = invoke(stypy.reporting.localization.Localization(__file__, 663, 17), getitem___170544, int_170542)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 11), tuple_170537, subscript_call_result_170545)
    # Adding element type (line 663)
    int_170546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 11), tuple_170537, int_170546)
    # Adding element type (line 663)
    
    # Obtaining the type of the subscript
    int_170547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 28), 'int')
    # Getting the type of 'r' (line 663)
    r_170548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 26), 'r')
    # Obtaining the member '__getitem__' of a type (line 663)
    getitem___170549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 26), r_170548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 663)
    subscript_call_result_170550 = invoke(stypy.reporting.localization.Localization(__file__, 663, 26), getitem___170549, int_170547)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 11), tuple_170537, subscript_call_result_170550)
    
    # Assigning a type to the variable 'stypy_return_type' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 'stypy_return_type', tuple_170537)
    
    # ################# End of 'line_search_BFGS(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'line_search_BFGS' in the type store
    # Getting the type of 'stypy_return_type' (line 657)
    stypy_return_type_170551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'line_search_BFGS'
    return stypy_return_type_170551

# Assigning a type to the variable 'line_search_BFGS' (line 657)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 0), 'line_search_BFGS', line_search_BFGS)

@norecursion
def scalar_search_armijo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_170552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 48), 'float')
    int_170553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 61), 'int')
    int_170554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 69), 'int')
    defaults = [float_170552, int_170553, int_170554]
    # Create a new context for function 'scalar_search_armijo'
    module_type_store = module_type_store.open_function_context('scalar_search_armijo', 666, 0, False)
    
    # Passed parameters checking function
    scalar_search_armijo.stypy_localization = localization
    scalar_search_armijo.stypy_type_of_self = None
    scalar_search_armijo.stypy_type_store = module_type_store
    scalar_search_armijo.stypy_function_name = 'scalar_search_armijo'
    scalar_search_armijo.stypy_param_names_list = ['phi', 'phi0', 'derphi0', 'c1', 'alpha0', 'amin']
    scalar_search_armijo.stypy_varargs_param_name = None
    scalar_search_armijo.stypy_kwargs_param_name = None
    scalar_search_armijo.stypy_call_defaults = defaults
    scalar_search_armijo.stypy_call_varargs = varargs
    scalar_search_armijo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'scalar_search_armijo', ['phi', 'phi0', 'derphi0', 'c1', 'alpha0', 'amin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'scalar_search_armijo', localization, ['phi', 'phi0', 'derphi0', 'c1', 'alpha0', 'amin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'scalar_search_armijo(...)' code ##################

    str_170555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, (-1)), 'str', "Minimize over alpha, the function ``phi(alpha)``.\n\n    Uses the interpolation algorithm (Armijo backtracking) as suggested by\n    Wright and Nocedal in 'Numerical Optimization', 1999, pg. 56-57\n\n    alpha > 0 is assumed to be a descent direction.\n\n    Returns\n    -------\n    alpha\n    phi1\n\n    ")
    
    # Assigning a Call to a Name (line 680):
    
    # Assigning a Call to a Name (line 680):
    
    # Call to phi(...): (line 680)
    # Processing the call arguments (line 680)
    # Getting the type of 'alpha0' (line 680)
    alpha0_170557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 17), 'alpha0', False)
    # Processing the call keyword arguments (line 680)
    kwargs_170558 = {}
    # Getting the type of 'phi' (line 680)
    phi_170556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 13), 'phi', False)
    # Calling phi(args, kwargs) (line 680)
    phi_call_result_170559 = invoke(stypy.reporting.localization.Localization(__file__, 680, 13), phi_170556, *[alpha0_170557], **kwargs_170558)
    
    # Assigning a type to the variable 'phi_a0' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'phi_a0', phi_call_result_170559)
    
    
    # Getting the type of 'phi_a0' (line 681)
    phi_a0_170560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 7), 'phi_a0')
    # Getting the type of 'phi0' (line 681)
    phi0_170561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 17), 'phi0')
    # Getting the type of 'c1' (line 681)
    c1_170562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 24), 'c1')
    # Getting the type of 'alpha0' (line 681)
    alpha0_170563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 27), 'alpha0')
    # Applying the binary operator '*' (line 681)
    result_mul_170564 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 24), '*', c1_170562, alpha0_170563)
    
    # Getting the type of 'derphi0' (line 681)
    derphi0_170565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 34), 'derphi0')
    # Applying the binary operator '*' (line 681)
    result_mul_170566 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 33), '*', result_mul_170564, derphi0_170565)
    
    # Applying the binary operator '+' (line 681)
    result_add_170567 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 17), '+', phi0_170561, result_mul_170566)
    
    # Applying the binary operator '<=' (line 681)
    result_le_170568 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 7), '<=', phi_a0_170560, result_add_170567)
    
    # Testing the type of an if condition (line 681)
    if_condition_170569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 681, 4), result_le_170568)
    # Assigning a type to the variable 'if_condition_170569' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'if_condition_170569', if_condition_170569)
    # SSA begins for if statement (line 681)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 682)
    tuple_170570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 682)
    # Adding element type (line 682)
    # Getting the type of 'alpha0' (line 682)
    alpha0_170571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 15), 'alpha0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 15), tuple_170570, alpha0_170571)
    # Adding element type (line 682)
    # Getting the type of 'phi_a0' (line 682)
    phi_a0_170572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 23), 'phi_a0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 15), tuple_170570, phi_a0_170572)
    
    # Assigning a type to the variable 'stypy_return_type' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'stypy_return_type', tuple_170570)
    # SSA join for if statement (line 681)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 686):
    
    # Assigning a BinOp to a Name (line 686):
    
    # Getting the type of 'derphi0' (line 686)
    derphi0_170573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 15), 'derphi0')
    # Applying the 'usub' unary operator (line 686)
    result___neg___170574 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 13), 'usub', derphi0_170573)
    
    # Getting the type of 'alpha0' (line 686)
    alpha0_170575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 26), 'alpha0')
    int_170576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 34), 'int')
    # Applying the binary operator '**' (line 686)
    result_pow_170577 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 26), '**', alpha0_170575, int_170576)
    
    # Applying the binary operator '*' (line 686)
    result_mul_170578 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 13), '*', result___neg___170574, result_pow_170577)
    
    float_170579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 38), 'float')
    # Applying the binary operator 'div' (line 686)
    result_div_170580 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 36), 'div', result_mul_170578, float_170579)
    
    # Getting the type of 'phi_a0' (line 686)
    phi_a0_170581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 45), 'phi_a0')
    # Getting the type of 'phi0' (line 686)
    phi0_170582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 54), 'phi0')
    # Applying the binary operator '-' (line 686)
    result_sub_170583 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 45), '-', phi_a0_170581, phi0_170582)
    
    # Getting the type of 'derphi0' (line 686)
    derphi0_170584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 61), 'derphi0')
    # Getting the type of 'alpha0' (line 686)
    alpha0_170585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 71), 'alpha0')
    # Applying the binary operator '*' (line 686)
    result_mul_170586 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 61), '*', derphi0_170584, alpha0_170585)
    
    # Applying the binary operator '-' (line 686)
    result_sub_170587 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 59), '-', result_sub_170583, result_mul_170586)
    
    # Applying the binary operator 'div' (line 686)
    result_div_170588 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 42), 'div', result_div_170580, result_sub_170587)
    
    # Assigning a type to the variable 'alpha1' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'alpha1', result_div_170588)
    
    # Assigning a Call to a Name (line 687):
    
    # Assigning a Call to a Name (line 687):
    
    # Call to phi(...): (line 687)
    # Processing the call arguments (line 687)
    # Getting the type of 'alpha1' (line 687)
    alpha1_170590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 17), 'alpha1', False)
    # Processing the call keyword arguments (line 687)
    kwargs_170591 = {}
    # Getting the type of 'phi' (line 687)
    phi_170589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 13), 'phi', False)
    # Calling phi(args, kwargs) (line 687)
    phi_call_result_170592 = invoke(stypy.reporting.localization.Localization(__file__, 687, 13), phi_170589, *[alpha1_170590], **kwargs_170591)
    
    # Assigning a type to the variable 'phi_a1' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'phi_a1', phi_call_result_170592)
    
    
    # Getting the type of 'phi_a1' (line 689)
    phi_a1_170593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'phi_a1')
    # Getting the type of 'phi0' (line 689)
    phi0_170594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 18), 'phi0')
    # Getting the type of 'c1' (line 689)
    c1_170595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 25), 'c1')
    # Getting the type of 'alpha1' (line 689)
    alpha1_170596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 28), 'alpha1')
    # Applying the binary operator '*' (line 689)
    result_mul_170597 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 25), '*', c1_170595, alpha1_170596)
    
    # Getting the type of 'derphi0' (line 689)
    derphi0_170598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 35), 'derphi0')
    # Applying the binary operator '*' (line 689)
    result_mul_170599 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 34), '*', result_mul_170597, derphi0_170598)
    
    # Applying the binary operator '+' (line 689)
    result_add_170600 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 18), '+', phi0_170594, result_mul_170599)
    
    # Applying the binary operator '<=' (line 689)
    result_le_170601 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 8), '<=', phi_a1_170593, result_add_170600)
    
    # Testing the type of an if condition (line 689)
    if_condition_170602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 689, 4), result_le_170601)
    # Assigning a type to the variable 'if_condition_170602' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'if_condition_170602', if_condition_170602)
    # SSA begins for if statement (line 689)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 690)
    tuple_170603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 690)
    # Adding element type (line 690)
    # Getting the type of 'alpha1' (line 690)
    alpha1_170604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 15), 'alpha1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 690, 15), tuple_170603, alpha1_170604)
    # Adding element type (line 690)
    # Getting the type of 'phi_a1' (line 690)
    phi_a1_170605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 23), 'phi_a1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 690, 15), tuple_170603, phi_a1_170605)
    
    # Assigning a type to the variable 'stypy_return_type' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'stypy_return_type', tuple_170603)
    # SSA join for if statement (line 689)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'alpha1' (line 697)
    alpha1_170606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 10), 'alpha1')
    # Getting the type of 'amin' (line 697)
    amin_170607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 19), 'amin')
    # Applying the binary operator '>' (line 697)
    result_gt_170608 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 10), '>', alpha1_170606, amin_170607)
    
    # Testing the type of an if condition (line 697)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 4), result_gt_170608)
    # SSA begins for while statement (line 697)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 698):
    
    # Assigning a BinOp to a Name (line 698):
    # Getting the type of 'alpha0' (line 698)
    alpha0_170609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 17), 'alpha0')
    int_170610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 25), 'int')
    # Applying the binary operator '**' (line 698)
    result_pow_170611 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 17), '**', alpha0_170609, int_170610)
    
    # Getting the type of 'alpha1' (line 698)
    alpha1_170612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 29), 'alpha1')
    int_170613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 37), 'int')
    # Applying the binary operator '**' (line 698)
    result_pow_170614 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 29), '**', alpha1_170612, int_170613)
    
    # Applying the binary operator '*' (line 698)
    result_mul_170615 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 17), '*', result_pow_170611, result_pow_170614)
    
    # Getting the type of 'alpha1' (line 698)
    alpha1_170616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 42), 'alpha1')
    # Getting the type of 'alpha0' (line 698)
    alpha0_170617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 49), 'alpha0')
    # Applying the binary operator '-' (line 698)
    result_sub_170618 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 42), '-', alpha1_170616, alpha0_170617)
    
    # Applying the binary operator '*' (line 698)
    result_mul_170619 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 39), '*', result_mul_170615, result_sub_170618)
    
    # Assigning a type to the variable 'factor' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'factor', result_mul_170619)
    
    # Assigning a BinOp to a Name (line 699):
    
    # Assigning a BinOp to a Name (line 699):
    # Getting the type of 'alpha0' (line 699)
    alpha0_170620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'alpha0')
    int_170621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 20), 'int')
    # Applying the binary operator '**' (line 699)
    result_pow_170622 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 12), '**', alpha0_170620, int_170621)
    
    # Getting the type of 'phi_a1' (line 699)
    phi_a1_170623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 25), 'phi_a1')
    # Getting the type of 'phi0' (line 699)
    phi0_170624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 34), 'phi0')
    # Applying the binary operator '-' (line 699)
    result_sub_170625 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 25), '-', phi_a1_170623, phi0_170624)
    
    # Getting the type of 'derphi0' (line 699)
    derphi0_170626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 41), 'derphi0')
    # Getting the type of 'alpha1' (line 699)
    alpha1_170627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 49), 'alpha1')
    # Applying the binary operator '*' (line 699)
    result_mul_170628 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 41), '*', derphi0_170626, alpha1_170627)
    
    # Applying the binary operator '-' (line 699)
    result_sub_170629 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 39), '-', result_sub_170625, result_mul_170628)
    
    # Applying the binary operator '*' (line 699)
    result_mul_170630 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 12), '*', result_pow_170622, result_sub_170629)
    
    # Getting the type of 'alpha1' (line 700)
    alpha1_170631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'alpha1')
    int_170632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 20), 'int')
    # Applying the binary operator '**' (line 700)
    result_pow_170633 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 12), '**', alpha1_170631, int_170632)
    
    # Getting the type of 'phi_a0' (line 700)
    phi_a0_170634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 25), 'phi_a0')
    # Getting the type of 'phi0' (line 700)
    phi0_170635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 34), 'phi0')
    # Applying the binary operator '-' (line 700)
    result_sub_170636 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 25), '-', phi_a0_170634, phi0_170635)
    
    # Getting the type of 'derphi0' (line 700)
    derphi0_170637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 41), 'derphi0')
    # Getting the type of 'alpha0' (line 700)
    alpha0_170638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 49), 'alpha0')
    # Applying the binary operator '*' (line 700)
    result_mul_170639 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 41), '*', derphi0_170637, alpha0_170638)
    
    # Applying the binary operator '-' (line 700)
    result_sub_170640 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 39), '-', result_sub_170636, result_mul_170639)
    
    # Applying the binary operator '*' (line 700)
    result_mul_170641 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 12), '*', result_pow_170633, result_sub_170640)
    
    # Applying the binary operator '-' (line 699)
    result_sub_170642 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 12), '-', result_mul_170630, result_mul_170641)
    
    # Assigning a type to the variable 'a' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'a', result_sub_170642)
    
    # Assigning a BinOp to a Name (line 701):
    
    # Assigning a BinOp to a Name (line 701):
    # Getting the type of 'a' (line 701)
    a_170643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 12), 'a')
    # Getting the type of 'factor' (line 701)
    factor_170644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'factor')
    # Applying the binary operator 'div' (line 701)
    result_div_170645 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 12), 'div', a_170643, factor_170644)
    
    # Assigning a type to the variable 'a' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'a', result_div_170645)
    
    # Assigning a BinOp to a Name (line 702):
    
    # Assigning a BinOp to a Name (line 702):
    
    # Getting the type of 'alpha0' (line 702)
    alpha0_170646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 13), 'alpha0')
    int_170647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 21), 'int')
    # Applying the binary operator '**' (line 702)
    result_pow_170648 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 13), '**', alpha0_170646, int_170647)
    
    # Applying the 'usub' unary operator (line 702)
    result___neg___170649 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 12), 'usub', result_pow_170648)
    
    # Getting the type of 'phi_a1' (line 702)
    phi_a1_170650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 26), 'phi_a1')
    # Getting the type of 'phi0' (line 702)
    phi0_170651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 35), 'phi0')
    # Applying the binary operator '-' (line 702)
    result_sub_170652 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 26), '-', phi_a1_170650, phi0_170651)
    
    # Getting the type of 'derphi0' (line 702)
    derphi0_170653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 42), 'derphi0')
    # Getting the type of 'alpha1' (line 702)
    alpha1_170654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 50), 'alpha1')
    # Applying the binary operator '*' (line 702)
    result_mul_170655 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 42), '*', derphi0_170653, alpha1_170654)
    
    # Applying the binary operator '-' (line 702)
    result_sub_170656 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 40), '-', result_sub_170652, result_mul_170655)
    
    # Applying the binary operator '*' (line 702)
    result_mul_170657 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 12), '*', result___neg___170649, result_sub_170656)
    
    # Getting the type of 'alpha1' (line 703)
    alpha1_170658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), 'alpha1')
    int_170659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 20), 'int')
    # Applying the binary operator '**' (line 703)
    result_pow_170660 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 12), '**', alpha1_170658, int_170659)
    
    # Getting the type of 'phi_a0' (line 703)
    phi_a0_170661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 25), 'phi_a0')
    # Getting the type of 'phi0' (line 703)
    phi0_170662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 34), 'phi0')
    # Applying the binary operator '-' (line 703)
    result_sub_170663 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 25), '-', phi_a0_170661, phi0_170662)
    
    # Getting the type of 'derphi0' (line 703)
    derphi0_170664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 41), 'derphi0')
    # Getting the type of 'alpha0' (line 703)
    alpha0_170665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 49), 'alpha0')
    # Applying the binary operator '*' (line 703)
    result_mul_170666 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 41), '*', derphi0_170664, alpha0_170665)
    
    # Applying the binary operator '-' (line 703)
    result_sub_170667 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 39), '-', result_sub_170663, result_mul_170666)
    
    # Applying the binary operator '*' (line 703)
    result_mul_170668 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 12), '*', result_pow_170660, result_sub_170667)
    
    # Applying the binary operator '+' (line 702)
    result_add_170669 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 12), '+', result_mul_170657, result_mul_170668)
    
    # Assigning a type to the variable 'b' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'b', result_add_170669)
    
    # Assigning a BinOp to a Name (line 704):
    
    # Assigning a BinOp to a Name (line 704):
    # Getting the type of 'b' (line 704)
    b_170670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'b')
    # Getting the type of 'factor' (line 704)
    factor_170671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 16), 'factor')
    # Applying the binary operator 'div' (line 704)
    result_div_170672 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 12), 'div', b_170670, factor_170671)
    
    # Assigning a type to the variable 'b' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'b', result_div_170672)
    
    # Assigning a BinOp to a Name (line 706):
    
    # Assigning a BinOp to a Name (line 706):
    
    # Getting the type of 'b' (line 706)
    b_170673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 19), 'b')
    # Applying the 'usub' unary operator (line 706)
    result___neg___170674 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 18), 'usub', b_170673)
    
    
    # Call to sqrt(...): (line 706)
    # Processing the call arguments (line 706)
    
    # Call to abs(...): (line 706)
    # Processing the call arguments (line 706)
    # Getting the type of 'b' (line 706)
    b_170678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 35), 'b', False)
    int_170679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 38), 'int')
    # Applying the binary operator '**' (line 706)
    result_pow_170680 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 35), '**', b_170678, int_170679)
    
    int_170681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 42), 'int')
    # Getting the type of 'a' (line 706)
    a_170682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 46), 'a', False)
    # Applying the binary operator '*' (line 706)
    result_mul_170683 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 42), '*', int_170681, a_170682)
    
    # Getting the type of 'derphi0' (line 706)
    derphi0_170684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 50), 'derphi0', False)
    # Applying the binary operator '*' (line 706)
    result_mul_170685 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 48), '*', result_mul_170683, derphi0_170684)
    
    # Applying the binary operator '-' (line 706)
    result_sub_170686 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 35), '-', result_pow_170680, result_mul_170685)
    
    # Processing the call keyword arguments (line 706)
    kwargs_170687 = {}
    # Getting the type of 'abs' (line 706)
    abs_170677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 31), 'abs', False)
    # Calling abs(args, kwargs) (line 706)
    abs_call_result_170688 = invoke(stypy.reporting.localization.Localization(__file__, 706, 31), abs_170677, *[result_sub_170686], **kwargs_170687)
    
    # Processing the call keyword arguments (line 706)
    kwargs_170689 = {}
    # Getting the type of 'np' (line 706)
    np_170675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 23), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 706)
    sqrt_170676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 23), np_170675, 'sqrt')
    # Calling sqrt(args, kwargs) (line 706)
    sqrt_call_result_170690 = invoke(stypy.reporting.localization.Localization(__file__, 706, 23), sqrt_170676, *[abs_call_result_170688], **kwargs_170689)
    
    # Applying the binary operator '+' (line 706)
    result_add_170691 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 18), '+', result___neg___170674, sqrt_call_result_170690)
    
    float_170692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 64), 'float')
    # Getting the type of 'a' (line 706)
    a_170693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 68), 'a')
    # Applying the binary operator '*' (line 706)
    result_mul_170694 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 64), '*', float_170692, a_170693)
    
    # Applying the binary operator 'div' (line 706)
    result_div_170695 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 17), 'div', result_add_170691, result_mul_170694)
    
    # Assigning a type to the variable 'alpha2' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'alpha2', result_div_170695)
    
    # Assigning a Call to a Name (line 707):
    
    # Assigning a Call to a Name (line 707):
    
    # Call to phi(...): (line 707)
    # Processing the call arguments (line 707)
    # Getting the type of 'alpha2' (line 707)
    alpha2_170697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 21), 'alpha2', False)
    # Processing the call keyword arguments (line 707)
    kwargs_170698 = {}
    # Getting the type of 'phi' (line 707)
    phi_170696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 17), 'phi', False)
    # Calling phi(args, kwargs) (line 707)
    phi_call_result_170699 = invoke(stypy.reporting.localization.Localization(__file__, 707, 17), phi_170696, *[alpha2_170697], **kwargs_170698)
    
    # Assigning a type to the variable 'phi_a2' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'phi_a2', phi_call_result_170699)
    
    
    # Getting the type of 'phi_a2' (line 709)
    phi_a2_170700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 12), 'phi_a2')
    # Getting the type of 'phi0' (line 709)
    phi0_170701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 22), 'phi0')
    # Getting the type of 'c1' (line 709)
    c1_170702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 29), 'c1')
    # Getting the type of 'alpha2' (line 709)
    alpha2_170703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 32), 'alpha2')
    # Applying the binary operator '*' (line 709)
    result_mul_170704 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 29), '*', c1_170702, alpha2_170703)
    
    # Getting the type of 'derphi0' (line 709)
    derphi0_170705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 39), 'derphi0')
    # Applying the binary operator '*' (line 709)
    result_mul_170706 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 38), '*', result_mul_170704, derphi0_170705)
    
    # Applying the binary operator '+' (line 709)
    result_add_170707 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 22), '+', phi0_170701, result_mul_170706)
    
    # Applying the binary operator '<=' (line 709)
    result_le_170708 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 12), '<=', phi_a2_170700, result_add_170707)
    
    # Testing the type of an if condition (line 709)
    if_condition_170709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 709, 8), result_le_170708)
    # Assigning a type to the variable 'if_condition_170709' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'if_condition_170709', if_condition_170709)
    # SSA begins for if statement (line 709)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 710)
    tuple_170710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 710)
    # Adding element type (line 710)
    # Getting the type of 'alpha2' (line 710)
    alpha2_170711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 19), 'alpha2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 19), tuple_170710, alpha2_170711)
    # Adding element type (line 710)
    # Getting the type of 'phi_a2' (line 710)
    phi_a2_170712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 27), 'phi_a2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 19), tuple_170710, phi_a2_170712)
    
    # Assigning a type to the variable 'stypy_return_type' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 12), 'stypy_return_type', tuple_170710)
    # SSA join for if statement (line 709)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'alpha1' (line 712)
    alpha1_170713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 12), 'alpha1')
    # Getting the type of 'alpha2' (line 712)
    alpha2_170714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 21), 'alpha2')
    # Applying the binary operator '-' (line 712)
    result_sub_170715 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 12), '-', alpha1_170713, alpha2_170714)
    
    # Getting the type of 'alpha1' (line 712)
    alpha1_170716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 31), 'alpha1')
    float_170717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 40), 'float')
    # Applying the binary operator 'div' (line 712)
    result_div_170718 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 31), 'div', alpha1_170716, float_170717)
    
    # Applying the binary operator '>' (line 712)
    result_gt_170719 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 11), '>', result_sub_170715, result_div_170718)
    
    
    int_170720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 48), 'int')
    # Getting the type of 'alpha2' (line 712)
    alpha2_170721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 52), 'alpha2')
    # Getting the type of 'alpha1' (line 712)
    alpha1_170722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 59), 'alpha1')
    # Applying the binary operator 'div' (line 712)
    result_div_170723 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 52), 'div', alpha2_170721, alpha1_170722)
    
    # Applying the binary operator '-' (line 712)
    result_sub_170724 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 48), '-', int_170720, result_div_170723)
    
    float_170725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 69), 'float')
    # Applying the binary operator '<' (line 712)
    result_lt_170726 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 47), '<', result_sub_170724, float_170725)
    
    # Applying the binary operator 'or' (line 712)
    result_or_keyword_170727 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 11), 'or', result_gt_170719, result_lt_170726)
    
    # Testing the type of an if condition (line 712)
    if_condition_170728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 712, 8), result_or_keyword_170727)
    # Assigning a type to the variable 'if_condition_170728' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'if_condition_170728', if_condition_170728)
    # SSA begins for if statement (line 712)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 713):
    
    # Assigning a BinOp to a Name (line 713):
    # Getting the type of 'alpha1' (line 713)
    alpha1_170729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 21), 'alpha1')
    float_170730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 30), 'float')
    # Applying the binary operator 'div' (line 713)
    result_div_170731 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 21), 'div', alpha1_170729, float_170730)
    
    # Assigning a type to the variable 'alpha2' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 12), 'alpha2', result_div_170731)
    # SSA join for if statement (line 712)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 715):
    
    # Assigning a Name to a Name (line 715):
    # Getting the type of 'alpha1' (line 715)
    alpha1_170732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 17), 'alpha1')
    # Assigning a type to the variable 'alpha0' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'alpha0', alpha1_170732)
    
    # Assigning a Name to a Name (line 716):
    
    # Assigning a Name to a Name (line 716):
    # Getting the type of 'alpha2' (line 716)
    alpha2_170733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 17), 'alpha2')
    # Assigning a type to the variable 'alpha1' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'alpha1', alpha2_170733)
    
    # Assigning a Name to a Name (line 717):
    
    # Assigning a Name to a Name (line 717):
    # Getting the type of 'phi_a1' (line 717)
    phi_a1_170734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 17), 'phi_a1')
    # Assigning a type to the variable 'phi_a0' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'phi_a0', phi_a1_170734)
    
    # Assigning a Name to a Name (line 718):
    
    # Assigning a Name to a Name (line 718):
    # Getting the type of 'phi_a2' (line 718)
    phi_a2_170735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 17), 'phi_a2')
    # Assigning a type to the variable 'phi_a1' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'phi_a1', phi_a2_170735)
    # SSA join for while statement (line 697)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 721)
    tuple_170736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 721)
    # Adding element type (line 721)
    # Getting the type of 'None' (line 721)
    None_170737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 11), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 721, 11), tuple_170736, None_170737)
    # Adding element type (line 721)
    # Getting the type of 'phi_a1' (line 721)
    phi_a1_170738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 17), 'phi_a1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 721, 11), tuple_170736, phi_a1_170738)
    
    # Assigning a type to the variable 'stypy_return_type' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'stypy_return_type', tuple_170736)
    
    # ################# End of 'scalar_search_armijo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'scalar_search_armijo' in the type store
    # Getting the type of 'stypy_return_type' (line 666)
    stypy_return_type_170739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170739)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'scalar_search_armijo'
    return stypy_return_type_170739

# Assigning a type to the variable 'scalar_search_armijo' (line 666)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 0), 'scalar_search_armijo', scalar_search_armijo)

@norecursion
def _nonmonotone_line_search_cruz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_170740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 40), 'float')
    float_170741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 54), 'float')
    float_170742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 67), 'float')
    defaults = [float_170740, float_170741, float_170742]
    # Create a new context for function '_nonmonotone_line_search_cruz'
    module_type_store = module_type_store.open_function_context('_nonmonotone_line_search_cruz', 728, 0, False)
    
    # Passed parameters checking function
    _nonmonotone_line_search_cruz.stypy_localization = localization
    _nonmonotone_line_search_cruz.stypy_type_of_self = None
    _nonmonotone_line_search_cruz.stypy_type_store = module_type_store
    _nonmonotone_line_search_cruz.stypy_function_name = '_nonmonotone_line_search_cruz'
    _nonmonotone_line_search_cruz.stypy_param_names_list = ['f', 'x_k', 'd', 'prev_fs', 'eta', 'gamma', 'tau_min', 'tau_max']
    _nonmonotone_line_search_cruz.stypy_varargs_param_name = None
    _nonmonotone_line_search_cruz.stypy_kwargs_param_name = None
    _nonmonotone_line_search_cruz.stypy_call_defaults = defaults
    _nonmonotone_line_search_cruz.stypy_call_varargs = varargs
    _nonmonotone_line_search_cruz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nonmonotone_line_search_cruz', ['f', 'x_k', 'd', 'prev_fs', 'eta', 'gamma', 'tau_min', 'tau_max'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nonmonotone_line_search_cruz', localization, ['f', 'x_k', 'd', 'prev_fs', 'eta', 'gamma', 'tau_min', 'tau_max'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nonmonotone_line_search_cruz(...)' code ##################

    str_170743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, (-1)), 'str', '\n    Nonmonotone backtracking line search as described in [1]_\n\n    Parameters\n    ----------\n    f : callable\n        Function returning a tuple ``(f, F)`` where ``f`` is the value\n        of a merit function and ``F`` the residual.\n    x_k : ndarray\n        Initial position\n    d : ndarray\n        Search direction\n    prev_fs : float\n        List of previous merit function values. Should have ``len(prev_fs) <= M``\n        where ``M`` is the nonmonotonicity window parameter.\n    eta : float\n        Allowed merit function increase, see [1]_\n    gamma, tau_min, tau_max : float, optional\n        Search parameters, see [1]_\n\n    Returns\n    -------\n    alpha : float\n        Step length\n    xp : ndarray\n        Next position\n    fp : float\n        Merit function value at next position\n    Fp : ndarray\n        Residual at next position\n\n    References\n    ----------\n    [1] "Spectral residual method without gradient information for solving\n        large-scale nonlinear systems of equations." W. La Cruz,\n        J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).\n\n    ')
    
    # Assigning a Subscript to a Name (line 768):
    
    # Assigning a Subscript to a Name (line 768):
    
    # Obtaining the type of the subscript
    int_170744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 18), 'int')
    # Getting the type of 'prev_fs' (line 768)
    prev_fs_170745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 10), 'prev_fs')
    # Obtaining the member '__getitem__' of a type (line 768)
    getitem___170746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 10), prev_fs_170745, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 768)
    subscript_call_result_170747 = invoke(stypy.reporting.localization.Localization(__file__, 768, 10), getitem___170746, int_170744)
    
    # Assigning a type to the variable 'f_k' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'f_k', subscript_call_result_170747)
    
    # Assigning a Call to a Name (line 769):
    
    # Assigning a Call to a Name (line 769):
    
    # Call to max(...): (line 769)
    # Processing the call arguments (line 769)
    # Getting the type of 'prev_fs' (line 769)
    prev_fs_170749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 16), 'prev_fs', False)
    # Processing the call keyword arguments (line 769)
    kwargs_170750 = {}
    # Getting the type of 'max' (line 769)
    max_170748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'max', False)
    # Calling max(args, kwargs) (line 769)
    max_call_result_170751 = invoke(stypy.reporting.localization.Localization(__file__, 769, 12), max_170748, *[prev_fs_170749], **kwargs_170750)
    
    # Assigning a type to the variable 'f_bar' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'f_bar', max_call_result_170751)
    
    # Assigning a Num to a Name (line 771):
    
    # Assigning a Num to a Name (line 771):
    int_170752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 14), 'int')
    # Assigning a type to the variable 'alpha_p' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'alpha_p', int_170752)
    
    # Assigning a Num to a Name (line 772):
    
    # Assigning a Num to a Name (line 772):
    int_170753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 14), 'int')
    # Assigning a type to the variable 'alpha_m' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'alpha_m', int_170753)
    
    # Assigning a Num to a Name (line 773):
    
    # Assigning a Num to a Name (line 773):
    int_170754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 12), 'int')
    # Assigning a type to the variable 'alpha' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'alpha', int_170754)
    
    # Getting the type of 'True' (line 775)
    True_170755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 10), 'True')
    # Testing the type of an if condition (line 775)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 775, 4), True_170755)
    # SSA begins for while statement (line 775)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 776):
    
    # Assigning a BinOp to a Name (line 776):
    # Getting the type of 'x_k' (line 776)
    x_k_170756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 13), 'x_k')
    # Getting the type of 'alpha_p' (line 776)
    alpha_p_170757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 19), 'alpha_p')
    # Getting the type of 'd' (line 776)
    d_170758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 29), 'd')
    # Applying the binary operator '*' (line 776)
    result_mul_170759 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 19), '*', alpha_p_170757, d_170758)
    
    # Applying the binary operator '+' (line 776)
    result_add_170760 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 13), '+', x_k_170756, result_mul_170759)
    
    # Assigning a type to the variable 'xp' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'xp', result_add_170760)
    
    # Assigning a Call to a Tuple (line 777):
    
    # Assigning a Subscript to a Name (line 777):
    
    # Obtaining the type of the subscript
    int_170761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 8), 'int')
    
    # Call to f(...): (line 777)
    # Processing the call arguments (line 777)
    # Getting the type of 'xp' (line 777)
    xp_170763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 19), 'xp', False)
    # Processing the call keyword arguments (line 777)
    kwargs_170764 = {}
    # Getting the type of 'f' (line 777)
    f_170762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 17), 'f', False)
    # Calling f(args, kwargs) (line 777)
    f_call_result_170765 = invoke(stypy.reporting.localization.Localization(__file__, 777, 17), f_170762, *[xp_170763], **kwargs_170764)
    
    # Obtaining the member '__getitem__' of a type (line 777)
    getitem___170766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 8), f_call_result_170765, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 777)
    subscript_call_result_170767 = invoke(stypy.reporting.localization.Localization(__file__, 777, 8), getitem___170766, int_170761)
    
    # Assigning a type to the variable 'tuple_var_assignment_169060' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'tuple_var_assignment_169060', subscript_call_result_170767)
    
    # Assigning a Subscript to a Name (line 777):
    
    # Obtaining the type of the subscript
    int_170768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 8), 'int')
    
    # Call to f(...): (line 777)
    # Processing the call arguments (line 777)
    # Getting the type of 'xp' (line 777)
    xp_170770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 19), 'xp', False)
    # Processing the call keyword arguments (line 777)
    kwargs_170771 = {}
    # Getting the type of 'f' (line 777)
    f_170769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 17), 'f', False)
    # Calling f(args, kwargs) (line 777)
    f_call_result_170772 = invoke(stypy.reporting.localization.Localization(__file__, 777, 17), f_170769, *[xp_170770], **kwargs_170771)
    
    # Obtaining the member '__getitem__' of a type (line 777)
    getitem___170773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 8), f_call_result_170772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 777)
    subscript_call_result_170774 = invoke(stypy.reporting.localization.Localization(__file__, 777, 8), getitem___170773, int_170768)
    
    # Assigning a type to the variable 'tuple_var_assignment_169061' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'tuple_var_assignment_169061', subscript_call_result_170774)
    
    # Assigning a Name to a Name (line 777):
    # Getting the type of 'tuple_var_assignment_169060' (line 777)
    tuple_var_assignment_169060_170775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'tuple_var_assignment_169060')
    # Assigning a type to the variable 'fp' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'fp', tuple_var_assignment_169060_170775)
    
    # Assigning a Name to a Name (line 777):
    # Getting the type of 'tuple_var_assignment_169061' (line 777)
    tuple_var_assignment_169061_170776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'tuple_var_assignment_169061')
    # Assigning a type to the variable 'Fp' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 12), 'Fp', tuple_var_assignment_169061_170776)
    
    
    # Getting the type of 'fp' (line 779)
    fp_170777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 11), 'fp')
    # Getting the type of 'f_bar' (line 779)
    f_bar_170778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 17), 'f_bar')
    # Getting the type of 'eta' (line 779)
    eta_170779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 25), 'eta')
    # Applying the binary operator '+' (line 779)
    result_add_170780 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 17), '+', f_bar_170778, eta_170779)
    
    # Getting the type of 'gamma' (line 779)
    gamma_170781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 31), 'gamma')
    # Getting the type of 'alpha_p' (line 779)
    alpha_p_170782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 39), 'alpha_p')
    int_170783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 48), 'int')
    # Applying the binary operator '**' (line 779)
    result_pow_170784 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 39), '**', alpha_p_170782, int_170783)
    
    # Applying the binary operator '*' (line 779)
    result_mul_170785 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 31), '*', gamma_170781, result_pow_170784)
    
    # Getting the type of 'f_k' (line 779)
    f_k_170786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 52), 'f_k')
    # Applying the binary operator '*' (line 779)
    result_mul_170787 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 50), '*', result_mul_170785, f_k_170786)
    
    # Applying the binary operator '-' (line 779)
    result_sub_170788 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 29), '-', result_add_170780, result_mul_170787)
    
    # Applying the binary operator '<=' (line 779)
    result_le_170789 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 11), '<=', fp_170777, result_sub_170788)
    
    # Testing the type of an if condition (line 779)
    if_condition_170790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 779, 8), result_le_170789)
    # Assigning a type to the variable 'if_condition_170790' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'if_condition_170790', if_condition_170790)
    # SSA begins for if statement (line 779)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 780):
    
    # Assigning a Name to a Name (line 780):
    # Getting the type of 'alpha_p' (line 780)
    alpha_p_170791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 20), 'alpha_p')
    # Assigning a type to the variable 'alpha' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 12), 'alpha', alpha_p_170791)
    # SSA join for if statement (line 779)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 783):
    
    # Assigning a BinOp to a Name (line 783):
    # Getting the type of 'alpha_p' (line 783)
    alpha_p_170792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 19), 'alpha_p')
    int_170793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 28), 'int')
    # Applying the binary operator '**' (line 783)
    result_pow_170794 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 19), '**', alpha_p_170792, int_170793)
    
    # Getting the type of 'f_k' (line 783)
    f_k_170795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 32), 'f_k')
    # Applying the binary operator '*' (line 783)
    result_mul_170796 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 19), '*', result_pow_170794, f_k_170795)
    
    # Getting the type of 'fp' (line 783)
    fp_170797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 39), 'fp')
    int_170798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 45), 'int')
    # Getting the type of 'alpha_p' (line 783)
    alpha_p_170799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 47), 'alpha_p')
    # Applying the binary operator '*' (line 783)
    result_mul_170800 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 45), '*', int_170798, alpha_p_170799)
    
    int_170801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 57), 'int')
    # Applying the binary operator '-' (line 783)
    result_sub_170802 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 45), '-', result_mul_170800, int_170801)
    
    # Getting the type of 'f_k' (line 783)
    f_k_170803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 60), 'f_k')
    # Applying the binary operator '*' (line 783)
    result_mul_170804 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 44), '*', result_sub_170802, f_k_170803)
    
    # Applying the binary operator '+' (line 783)
    result_add_170805 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 39), '+', fp_170797, result_mul_170804)
    
    # Applying the binary operator 'div' (line 783)
    result_div_170806 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 36), 'div', result_mul_170796, result_add_170805)
    
    # Assigning a type to the variable 'alpha_tp' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'alpha_tp', result_div_170806)
    
    # Assigning a BinOp to a Name (line 785):
    
    # Assigning a BinOp to a Name (line 785):
    # Getting the type of 'x_k' (line 785)
    x_k_170807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 13), 'x_k')
    # Getting the type of 'alpha_m' (line 785)
    alpha_m_170808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 19), 'alpha_m')
    # Getting the type of 'd' (line 785)
    d_170809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 29), 'd')
    # Applying the binary operator '*' (line 785)
    result_mul_170810 = python_operator(stypy.reporting.localization.Localization(__file__, 785, 19), '*', alpha_m_170808, d_170809)
    
    # Applying the binary operator '-' (line 785)
    result_sub_170811 = python_operator(stypy.reporting.localization.Localization(__file__, 785, 13), '-', x_k_170807, result_mul_170810)
    
    # Assigning a type to the variable 'xp' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'xp', result_sub_170811)
    
    # Assigning a Call to a Tuple (line 786):
    
    # Assigning a Subscript to a Name (line 786):
    
    # Obtaining the type of the subscript
    int_170812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 8), 'int')
    
    # Call to f(...): (line 786)
    # Processing the call arguments (line 786)
    # Getting the type of 'xp' (line 786)
    xp_170814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 19), 'xp', False)
    # Processing the call keyword arguments (line 786)
    kwargs_170815 = {}
    # Getting the type of 'f' (line 786)
    f_170813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 17), 'f', False)
    # Calling f(args, kwargs) (line 786)
    f_call_result_170816 = invoke(stypy.reporting.localization.Localization(__file__, 786, 17), f_170813, *[xp_170814], **kwargs_170815)
    
    # Obtaining the member '__getitem__' of a type (line 786)
    getitem___170817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 8), f_call_result_170816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 786)
    subscript_call_result_170818 = invoke(stypy.reporting.localization.Localization(__file__, 786, 8), getitem___170817, int_170812)
    
    # Assigning a type to the variable 'tuple_var_assignment_169062' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'tuple_var_assignment_169062', subscript_call_result_170818)
    
    # Assigning a Subscript to a Name (line 786):
    
    # Obtaining the type of the subscript
    int_170819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 8), 'int')
    
    # Call to f(...): (line 786)
    # Processing the call arguments (line 786)
    # Getting the type of 'xp' (line 786)
    xp_170821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 19), 'xp', False)
    # Processing the call keyword arguments (line 786)
    kwargs_170822 = {}
    # Getting the type of 'f' (line 786)
    f_170820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 17), 'f', False)
    # Calling f(args, kwargs) (line 786)
    f_call_result_170823 = invoke(stypy.reporting.localization.Localization(__file__, 786, 17), f_170820, *[xp_170821], **kwargs_170822)
    
    # Obtaining the member '__getitem__' of a type (line 786)
    getitem___170824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 8), f_call_result_170823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 786)
    subscript_call_result_170825 = invoke(stypy.reporting.localization.Localization(__file__, 786, 8), getitem___170824, int_170819)
    
    # Assigning a type to the variable 'tuple_var_assignment_169063' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'tuple_var_assignment_169063', subscript_call_result_170825)
    
    # Assigning a Name to a Name (line 786):
    # Getting the type of 'tuple_var_assignment_169062' (line 786)
    tuple_var_assignment_169062_170826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'tuple_var_assignment_169062')
    # Assigning a type to the variable 'fp' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'fp', tuple_var_assignment_169062_170826)
    
    # Assigning a Name to a Name (line 786):
    # Getting the type of 'tuple_var_assignment_169063' (line 786)
    tuple_var_assignment_169063_170827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'tuple_var_assignment_169063')
    # Assigning a type to the variable 'Fp' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 12), 'Fp', tuple_var_assignment_169063_170827)
    
    
    # Getting the type of 'fp' (line 788)
    fp_170828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 11), 'fp')
    # Getting the type of 'f_bar' (line 788)
    f_bar_170829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 17), 'f_bar')
    # Getting the type of 'eta' (line 788)
    eta_170830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 25), 'eta')
    # Applying the binary operator '+' (line 788)
    result_add_170831 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 17), '+', f_bar_170829, eta_170830)
    
    # Getting the type of 'gamma' (line 788)
    gamma_170832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 31), 'gamma')
    # Getting the type of 'alpha_m' (line 788)
    alpha_m_170833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 39), 'alpha_m')
    int_170834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 48), 'int')
    # Applying the binary operator '**' (line 788)
    result_pow_170835 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 39), '**', alpha_m_170833, int_170834)
    
    # Applying the binary operator '*' (line 788)
    result_mul_170836 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 31), '*', gamma_170832, result_pow_170835)
    
    # Getting the type of 'f_k' (line 788)
    f_k_170837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 52), 'f_k')
    # Applying the binary operator '*' (line 788)
    result_mul_170838 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 50), '*', result_mul_170836, f_k_170837)
    
    # Applying the binary operator '-' (line 788)
    result_sub_170839 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 29), '-', result_add_170831, result_mul_170838)
    
    # Applying the binary operator '<=' (line 788)
    result_le_170840 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 11), '<=', fp_170828, result_sub_170839)
    
    # Testing the type of an if condition (line 788)
    if_condition_170841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 788, 8), result_le_170840)
    # Assigning a type to the variable 'if_condition_170841' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'if_condition_170841', if_condition_170841)
    # SSA begins for if statement (line 788)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 789):
    
    # Assigning a UnaryOp to a Name (line 789):
    
    # Getting the type of 'alpha_m' (line 789)
    alpha_m_170842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 21), 'alpha_m')
    # Applying the 'usub' unary operator (line 789)
    result___neg___170843 = python_operator(stypy.reporting.localization.Localization(__file__, 789, 20), 'usub', alpha_m_170842)
    
    # Assigning a type to the variable 'alpha' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'alpha', result___neg___170843)
    # SSA join for if statement (line 788)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 792):
    
    # Assigning a BinOp to a Name (line 792):
    # Getting the type of 'alpha_m' (line 792)
    alpha_m_170844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 19), 'alpha_m')
    int_170845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 28), 'int')
    # Applying the binary operator '**' (line 792)
    result_pow_170846 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 19), '**', alpha_m_170844, int_170845)
    
    # Getting the type of 'f_k' (line 792)
    f_k_170847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 32), 'f_k')
    # Applying the binary operator '*' (line 792)
    result_mul_170848 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 19), '*', result_pow_170846, f_k_170847)
    
    # Getting the type of 'fp' (line 792)
    fp_170849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 39), 'fp')
    int_170850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 45), 'int')
    # Getting the type of 'alpha_m' (line 792)
    alpha_m_170851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 47), 'alpha_m')
    # Applying the binary operator '*' (line 792)
    result_mul_170852 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 45), '*', int_170850, alpha_m_170851)
    
    int_170853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 57), 'int')
    # Applying the binary operator '-' (line 792)
    result_sub_170854 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 45), '-', result_mul_170852, int_170853)
    
    # Getting the type of 'f_k' (line 792)
    f_k_170855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 60), 'f_k')
    # Applying the binary operator '*' (line 792)
    result_mul_170856 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 44), '*', result_sub_170854, f_k_170855)
    
    # Applying the binary operator '+' (line 792)
    result_add_170857 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 39), '+', fp_170849, result_mul_170856)
    
    # Applying the binary operator 'div' (line 792)
    result_div_170858 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 36), 'div', result_mul_170848, result_add_170857)
    
    # Assigning a type to the variable 'alpha_tm' (line 792)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'alpha_tm', result_div_170858)
    
    # Assigning a Call to a Name (line 794):
    
    # Assigning a Call to a Name (line 794):
    
    # Call to clip(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'alpha_tp' (line 794)
    alpha_tp_170861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 26), 'alpha_tp', False)
    # Getting the type of 'tau_min' (line 794)
    tau_min_170862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 36), 'tau_min', False)
    # Getting the type of 'alpha_p' (line 794)
    alpha_p_170863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 46), 'alpha_p', False)
    # Applying the binary operator '*' (line 794)
    result_mul_170864 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 36), '*', tau_min_170862, alpha_p_170863)
    
    # Getting the type of 'tau_max' (line 794)
    tau_max_170865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 55), 'tau_max', False)
    # Getting the type of 'alpha_p' (line 794)
    alpha_p_170866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 65), 'alpha_p', False)
    # Applying the binary operator '*' (line 794)
    result_mul_170867 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 55), '*', tau_max_170865, alpha_p_170866)
    
    # Processing the call keyword arguments (line 794)
    kwargs_170868 = {}
    # Getting the type of 'np' (line 794)
    np_170859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 18), 'np', False)
    # Obtaining the member 'clip' of a type (line 794)
    clip_170860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 18), np_170859, 'clip')
    # Calling clip(args, kwargs) (line 794)
    clip_call_result_170869 = invoke(stypy.reporting.localization.Localization(__file__, 794, 18), clip_170860, *[alpha_tp_170861, result_mul_170864, result_mul_170867], **kwargs_170868)
    
    # Assigning a type to the variable 'alpha_p' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'alpha_p', clip_call_result_170869)
    
    # Assigning a Call to a Name (line 795):
    
    # Assigning a Call to a Name (line 795):
    
    # Call to clip(...): (line 795)
    # Processing the call arguments (line 795)
    # Getting the type of 'alpha_tm' (line 795)
    alpha_tm_170872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 26), 'alpha_tm', False)
    # Getting the type of 'tau_min' (line 795)
    tau_min_170873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 36), 'tau_min', False)
    # Getting the type of 'alpha_m' (line 795)
    alpha_m_170874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 46), 'alpha_m', False)
    # Applying the binary operator '*' (line 795)
    result_mul_170875 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 36), '*', tau_min_170873, alpha_m_170874)
    
    # Getting the type of 'tau_max' (line 795)
    tau_max_170876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 55), 'tau_max', False)
    # Getting the type of 'alpha_m' (line 795)
    alpha_m_170877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 65), 'alpha_m', False)
    # Applying the binary operator '*' (line 795)
    result_mul_170878 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 55), '*', tau_max_170876, alpha_m_170877)
    
    # Processing the call keyword arguments (line 795)
    kwargs_170879 = {}
    # Getting the type of 'np' (line 795)
    np_170870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 18), 'np', False)
    # Obtaining the member 'clip' of a type (line 795)
    clip_170871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 18), np_170870, 'clip')
    # Calling clip(args, kwargs) (line 795)
    clip_call_result_170880 = invoke(stypy.reporting.localization.Localization(__file__, 795, 18), clip_170871, *[alpha_tm_170872, result_mul_170875, result_mul_170878], **kwargs_170879)
    
    # Assigning a type to the variable 'alpha_m' (line 795)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'alpha_m', clip_call_result_170880)
    # SSA join for while statement (line 775)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 797)
    tuple_170881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 797)
    # Adding element type (line 797)
    # Getting the type of 'alpha' (line 797)
    alpha_170882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 11), 'alpha')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 11), tuple_170881, alpha_170882)
    # Adding element type (line 797)
    # Getting the type of 'xp' (line 797)
    xp_170883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 18), 'xp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 11), tuple_170881, xp_170883)
    # Adding element type (line 797)
    # Getting the type of 'fp' (line 797)
    fp_170884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 22), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 11), tuple_170881, fp_170884)
    # Adding element type (line 797)
    # Getting the type of 'Fp' (line 797)
    Fp_170885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 26), 'Fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 11), tuple_170881, Fp_170885)
    
    # Assigning a type to the variable 'stypy_return_type' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'stypy_return_type', tuple_170881)
    
    # ################# End of '_nonmonotone_line_search_cruz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nonmonotone_line_search_cruz' in the type store
    # Getting the type of 'stypy_return_type' (line 728)
    stypy_return_type_170886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nonmonotone_line_search_cruz'
    return stypy_return_type_170886

# Assigning a type to the variable '_nonmonotone_line_search_cruz' (line 728)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 0), '_nonmonotone_line_search_cruz', _nonmonotone_line_search_cruz)

@norecursion
def _nonmonotone_line_search_cheng(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_170887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 41), 'float')
    float_170888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 55), 'float')
    float_170889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 68), 'float')
    float_170890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 38), 'float')
    defaults = [float_170887, float_170888, float_170889, float_170890]
    # Create a new context for function '_nonmonotone_line_search_cheng'
    module_type_store = module_type_store.open_function_context('_nonmonotone_line_search_cheng', 800, 0, False)
    
    # Passed parameters checking function
    _nonmonotone_line_search_cheng.stypy_localization = localization
    _nonmonotone_line_search_cheng.stypy_type_of_self = None
    _nonmonotone_line_search_cheng.stypy_type_store = module_type_store
    _nonmonotone_line_search_cheng.stypy_function_name = '_nonmonotone_line_search_cheng'
    _nonmonotone_line_search_cheng.stypy_param_names_list = ['f', 'x_k', 'd', 'f_k', 'C', 'Q', 'eta', 'gamma', 'tau_min', 'tau_max', 'nu']
    _nonmonotone_line_search_cheng.stypy_varargs_param_name = None
    _nonmonotone_line_search_cheng.stypy_kwargs_param_name = None
    _nonmonotone_line_search_cheng.stypy_call_defaults = defaults
    _nonmonotone_line_search_cheng.stypy_call_varargs = varargs
    _nonmonotone_line_search_cheng.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nonmonotone_line_search_cheng', ['f', 'x_k', 'd', 'f_k', 'C', 'Q', 'eta', 'gamma', 'tau_min', 'tau_max', 'nu'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nonmonotone_line_search_cheng', localization, ['f', 'x_k', 'd', 'f_k', 'C', 'Q', 'eta', 'gamma', 'tau_min', 'tau_max', 'nu'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nonmonotone_line_search_cheng(...)' code ##################

    str_170891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, (-1)), 'str', "\n    Nonmonotone line search from [1]\n\n    Parameters\n    ----------\n    f : callable\n        Function returning a tuple ``(f, F)`` where ``f`` is the value\n        of a merit function and ``F`` the residual.\n    x_k : ndarray\n        Initial position\n    d : ndarray\n        Search direction\n    f_k : float\n        Initial merit function value\n    C, Q : float\n        Control parameters. On the first iteration, give values\n        Q=1.0, C=f_k\n    eta : float\n        Allowed merit function increase, see [1]_\n    nu, gamma, tau_min, tau_max : float, optional\n        Search parameters, see [1]_\n\n    Returns\n    -------\n    alpha : float\n        Step length\n    xp : ndarray\n        Next position\n    fp : float\n        Merit function value at next position\n    Fp : ndarray\n        Residual at next position\n    C : float\n        New value for the control parameter C\n    Q : float\n        New value for the control parameter Q\n\n    References\n    ----------\n    .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line\n           search and its application to the spectral residual\n           method'', IMA J. Numer. Anal. 29, 814 (2009).\n\n    ")
    
    # Assigning a Num to a Name (line 847):
    
    # Assigning a Num to a Name (line 847):
    int_170892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 14), 'int')
    # Assigning a type to the variable 'alpha_p' (line 847)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 4), 'alpha_p', int_170892)
    
    # Assigning a Num to a Name (line 848):
    
    # Assigning a Num to a Name (line 848):
    int_170893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 14), 'int')
    # Assigning a type to the variable 'alpha_m' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'alpha_m', int_170893)
    
    # Assigning a Num to a Name (line 849):
    
    # Assigning a Num to a Name (line 849):
    int_170894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 12), 'int')
    # Assigning a type to the variable 'alpha' (line 849)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 4), 'alpha', int_170894)
    
    # Getting the type of 'True' (line 851)
    True_170895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 10), 'True')
    # Testing the type of an if condition (line 851)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 851, 4), True_170895)
    # SSA begins for while statement (line 851)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 852):
    
    # Assigning a BinOp to a Name (line 852):
    # Getting the type of 'x_k' (line 852)
    x_k_170896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 13), 'x_k')
    # Getting the type of 'alpha_p' (line 852)
    alpha_p_170897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 19), 'alpha_p')
    # Getting the type of 'd' (line 852)
    d_170898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 29), 'd')
    # Applying the binary operator '*' (line 852)
    result_mul_170899 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 19), '*', alpha_p_170897, d_170898)
    
    # Applying the binary operator '+' (line 852)
    result_add_170900 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 13), '+', x_k_170896, result_mul_170899)
    
    # Assigning a type to the variable 'xp' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'xp', result_add_170900)
    
    # Assigning a Call to a Tuple (line 853):
    
    # Assigning a Subscript to a Name (line 853):
    
    # Obtaining the type of the subscript
    int_170901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 8), 'int')
    
    # Call to f(...): (line 853)
    # Processing the call arguments (line 853)
    # Getting the type of 'xp' (line 853)
    xp_170903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 19), 'xp', False)
    # Processing the call keyword arguments (line 853)
    kwargs_170904 = {}
    # Getting the type of 'f' (line 853)
    f_170902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 17), 'f', False)
    # Calling f(args, kwargs) (line 853)
    f_call_result_170905 = invoke(stypy.reporting.localization.Localization(__file__, 853, 17), f_170902, *[xp_170903], **kwargs_170904)
    
    # Obtaining the member '__getitem__' of a type (line 853)
    getitem___170906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 8), f_call_result_170905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 853)
    subscript_call_result_170907 = invoke(stypy.reporting.localization.Localization(__file__, 853, 8), getitem___170906, int_170901)
    
    # Assigning a type to the variable 'tuple_var_assignment_169064' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'tuple_var_assignment_169064', subscript_call_result_170907)
    
    # Assigning a Subscript to a Name (line 853):
    
    # Obtaining the type of the subscript
    int_170908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 8), 'int')
    
    # Call to f(...): (line 853)
    # Processing the call arguments (line 853)
    # Getting the type of 'xp' (line 853)
    xp_170910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 19), 'xp', False)
    # Processing the call keyword arguments (line 853)
    kwargs_170911 = {}
    # Getting the type of 'f' (line 853)
    f_170909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 17), 'f', False)
    # Calling f(args, kwargs) (line 853)
    f_call_result_170912 = invoke(stypy.reporting.localization.Localization(__file__, 853, 17), f_170909, *[xp_170910], **kwargs_170911)
    
    # Obtaining the member '__getitem__' of a type (line 853)
    getitem___170913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 8), f_call_result_170912, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 853)
    subscript_call_result_170914 = invoke(stypy.reporting.localization.Localization(__file__, 853, 8), getitem___170913, int_170908)
    
    # Assigning a type to the variable 'tuple_var_assignment_169065' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'tuple_var_assignment_169065', subscript_call_result_170914)
    
    # Assigning a Name to a Name (line 853):
    # Getting the type of 'tuple_var_assignment_169064' (line 853)
    tuple_var_assignment_169064_170915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'tuple_var_assignment_169064')
    # Assigning a type to the variable 'fp' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'fp', tuple_var_assignment_169064_170915)
    
    # Assigning a Name to a Name (line 853):
    # Getting the type of 'tuple_var_assignment_169065' (line 853)
    tuple_var_assignment_169065_170916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'tuple_var_assignment_169065')
    # Assigning a type to the variable 'Fp' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 12), 'Fp', tuple_var_assignment_169065_170916)
    
    
    # Getting the type of 'fp' (line 855)
    fp_170917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 11), 'fp')
    # Getting the type of 'C' (line 855)
    C_170918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 17), 'C')
    # Getting the type of 'eta' (line 855)
    eta_170919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 21), 'eta')
    # Applying the binary operator '+' (line 855)
    result_add_170920 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 17), '+', C_170918, eta_170919)
    
    # Getting the type of 'gamma' (line 855)
    gamma_170921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 27), 'gamma')
    # Getting the type of 'alpha_p' (line 855)
    alpha_p_170922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 35), 'alpha_p')
    int_170923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 44), 'int')
    # Applying the binary operator '**' (line 855)
    result_pow_170924 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 35), '**', alpha_p_170922, int_170923)
    
    # Applying the binary operator '*' (line 855)
    result_mul_170925 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 27), '*', gamma_170921, result_pow_170924)
    
    # Getting the type of 'f_k' (line 855)
    f_k_170926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 48), 'f_k')
    # Applying the binary operator '*' (line 855)
    result_mul_170927 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 46), '*', result_mul_170925, f_k_170926)
    
    # Applying the binary operator '-' (line 855)
    result_sub_170928 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 25), '-', result_add_170920, result_mul_170927)
    
    # Applying the binary operator '<=' (line 855)
    result_le_170929 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 11), '<=', fp_170917, result_sub_170928)
    
    # Testing the type of an if condition (line 855)
    if_condition_170930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 855, 8), result_le_170929)
    # Assigning a type to the variable 'if_condition_170930' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 8), 'if_condition_170930', if_condition_170930)
    # SSA begins for if statement (line 855)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 856):
    
    # Assigning a Name to a Name (line 856):
    # Getting the type of 'alpha_p' (line 856)
    alpha_p_170931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 20), 'alpha_p')
    # Assigning a type to the variable 'alpha' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 12), 'alpha', alpha_p_170931)
    # SSA join for if statement (line 855)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 859):
    
    # Assigning a BinOp to a Name (line 859):
    # Getting the type of 'alpha_p' (line 859)
    alpha_p_170932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 19), 'alpha_p')
    int_170933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 28), 'int')
    # Applying the binary operator '**' (line 859)
    result_pow_170934 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 19), '**', alpha_p_170932, int_170933)
    
    # Getting the type of 'f_k' (line 859)
    f_k_170935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 32), 'f_k')
    # Applying the binary operator '*' (line 859)
    result_mul_170936 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 19), '*', result_pow_170934, f_k_170935)
    
    # Getting the type of 'fp' (line 859)
    fp_170937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 39), 'fp')
    int_170938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 45), 'int')
    # Getting the type of 'alpha_p' (line 859)
    alpha_p_170939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 47), 'alpha_p')
    # Applying the binary operator '*' (line 859)
    result_mul_170940 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 45), '*', int_170938, alpha_p_170939)
    
    int_170941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 57), 'int')
    # Applying the binary operator '-' (line 859)
    result_sub_170942 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 45), '-', result_mul_170940, int_170941)
    
    # Getting the type of 'f_k' (line 859)
    f_k_170943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 60), 'f_k')
    # Applying the binary operator '*' (line 859)
    result_mul_170944 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 44), '*', result_sub_170942, f_k_170943)
    
    # Applying the binary operator '+' (line 859)
    result_add_170945 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 39), '+', fp_170937, result_mul_170944)
    
    # Applying the binary operator 'div' (line 859)
    result_div_170946 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 36), 'div', result_mul_170936, result_add_170945)
    
    # Assigning a type to the variable 'alpha_tp' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 8), 'alpha_tp', result_div_170946)
    
    # Assigning a BinOp to a Name (line 861):
    
    # Assigning a BinOp to a Name (line 861):
    # Getting the type of 'x_k' (line 861)
    x_k_170947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 13), 'x_k')
    # Getting the type of 'alpha_m' (line 861)
    alpha_m_170948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 19), 'alpha_m')
    # Getting the type of 'd' (line 861)
    d_170949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 29), 'd')
    # Applying the binary operator '*' (line 861)
    result_mul_170950 = python_operator(stypy.reporting.localization.Localization(__file__, 861, 19), '*', alpha_m_170948, d_170949)
    
    # Applying the binary operator '-' (line 861)
    result_sub_170951 = python_operator(stypy.reporting.localization.Localization(__file__, 861, 13), '-', x_k_170947, result_mul_170950)
    
    # Assigning a type to the variable 'xp' (line 861)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 8), 'xp', result_sub_170951)
    
    # Assigning a Call to a Tuple (line 862):
    
    # Assigning a Subscript to a Name (line 862):
    
    # Obtaining the type of the subscript
    int_170952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 8), 'int')
    
    # Call to f(...): (line 862)
    # Processing the call arguments (line 862)
    # Getting the type of 'xp' (line 862)
    xp_170954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 19), 'xp', False)
    # Processing the call keyword arguments (line 862)
    kwargs_170955 = {}
    # Getting the type of 'f' (line 862)
    f_170953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 17), 'f', False)
    # Calling f(args, kwargs) (line 862)
    f_call_result_170956 = invoke(stypy.reporting.localization.Localization(__file__, 862, 17), f_170953, *[xp_170954], **kwargs_170955)
    
    # Obtaining the member '__getitem__' of a type (line 862)
    getitem___170957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 8), f_call_result_170956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 862)
    subscript_call_result_170958 = invoke(stypy.reporting.localization.Localization(__file__, 862, 8), getitem___170957, int_170952)
    
    # Assigning a type to the variable 'tuple_var_assignment_169066' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'tuple_var_assignment_169066', subscript_call_result_170958)
    
    # Assigning a Subscript to a Name (line 862):
    
    # Obtaining the type of the subscript
    int_170959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 8), 'int')
    
    # Call to f(...): (line 862)
    # Processing the call arguments (line 862)
    # Getting the type of 'xp' (line 862)
    xp_170961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 19), 'xp', False)
    # Processing the call keyword arguments (line 862)
    kwargs_170962 = {}
    # Getting the type of 'f' (line 862)
    f_170960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 17), 'f', False)
    # Calling f(args, kwargs) (line 862)
    f_call_result_170963 = invoke(stypy.reporting.localization.Localization(__file__, 862, 17), f_170960, *[xp_170961], **kwargs_170962)
    
    # Obtaining the member '__getitem__' of a type (line 862)
    getitem___170964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 8), f_call_result_170963, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 862)
    subscript_call_result_170965 = invoke(stypy.reporting.localization.Localization(__file__, 862, 8), getitem___170964, int_170959)
    
    # Assigning a type to the variable 'tuple_var_assignment_169067' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'tuple_var_assignment_169067', subscript_call_result_170965)
    
    # Assigning a Name to a Name (line 862):
    # Getting the type of 'tuple_var_assignment_169066' (line 862)
    tuple_var_assignment_169066_170966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'tuple_var_assignment_169066')
    # Assigning a type to the variable 'fp' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'fp', tuple_var_assignment_169066_170966)
    
    # Assigning a Name to a Name (line 862):
    # Getting the type of 'tuple_var_assignment_169067' (line 862)
    tuple_var_assignment_169067_170967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'tuple_var_assignment_169067')
    # Assigning a type to the variable 'Fp' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 12), 'Fp', tuple_var_assignment_169067_170967)
    
    
    # Getting the type of 'fp' (line 864)
    fp_170968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 11), 'fp')
    # Getting the type of 'C' (line 864)
    C_170969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 17), 'C')
    # Getting the type of 'eta' (line 864)
    eta_170970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 21), 'eta')
    # Applying the binary operator '+' (line 864)
    result_add_170971 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 17), '+', C_170969, eta_170970)
    
    # Getting the type of 'gamma' (line 864)
    gamma_170972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 27), 'gamma')
    # Getting the type of 'alpha_m' (line 864)
    alpha_m_170973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 35), 'alpha_m')
    int_170974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 44), 'int')
    # Applying the binary operator '**' (line 864)
    result_pow_170975 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 35), '**', alpha_m_170973, int_170974)
    
    # Applying the binary operator '*' (line 864)
    result_mul_170976 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 27), '*', gamma_170972, result_pow_170975)
    
    # Getting the type of 'f_k' (line 864)
    f_k_170977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 48), 'f_k')
    # Applying the binary operator '*' (line 864)
    result_mul_170978 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 46), '*', result_mul_170976, f_k_170977)
    
    # Applying the binary operator '-' (line 864)
    result_sub_170979 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 25), '-', result_add_170971, result_mul_170978)
    
    # Applying the binary operator '<=' (line 864)
    result_le_170980 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 11), '<=', fp_170968, result_sub_170979)
    
    # Testing the type of an if condition (line 864)
    if_condition_170981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 864, 8), result_le_170980)
    # Assigning a type to the variable 'if_condition_170981' (line 864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'if_condition_170981', if_condition_170981)
    # SSA begins for if statement (line 864)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 865):
    
    # Assigning a UnaryOp to a Name (line 865):
    
    # Getting the type of 'alpha_m' (line 865)
    alpha_m_170982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 21), 'alpha_m')
    # Applying the 'usub' unary operator (line 865)
    result___neg___170983 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 20), 'usub', alpha_m_170982)
    
    # Assigning a type to the variable 'alpha' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 12), 'alpha', result___neg___170983)
    # SSA join for if statement (line 864)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 868):
    
    # Assigning a BinOp to a Name (line 868):
    # Getting the type of 'alpha_m' (line 868)
    alpha_m_170984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 19), 'alpha_m')
    int_170985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 28), 'int')
    # Applying the binary operator '**' (line 868)
    result_pow_170986 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 19), '**', alpha_m_170984, int_170985)
    
    # Getting the type of 'f_k' (line 868)
    f_k_170987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 32), 'f_k')
    # Applying the binary operator '*' (line 868)
    result_mul_170988 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 19), '*', result_pow_170986, f_k_170987)
    
    # Getting the type of 'fp' (line 868)
    fp_170989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 39), 'fp')
    int_170990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 45), 'int')
    # Getting the type of 'alpha_m' (line 868)
    alpha_m_170991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 47), 'alpha_m')
    # Applying the binary operator '*' (line 868)
    result_mul_170992 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 45), '*', int_170990, alpha_m_170991)
    
    int_170993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 57), 'int')
    # Applying the binary operator '-' (line 868)
    result_sub_170994 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 45), '-', result_mul_170992, int_170993)
    
    # Getting the type of 'f_k' (line 868)
    f_k_170995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 60), 'f_k')
    # Applying the binary operator '*' (line 868)
    result_mul_170996 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 44), '*', result_sub_170994, f_k_170995)
    
    # Applying the binary operator '+' (line 868)
    result_add_170997 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 39), '+', fp_170989, result_mul_170996)
    
    # Applying the binary operator 'div' (line 868)
    result_div_170998 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 36), 'div', result_mul_170988, result_add_170997)
    
    # Assigning a type to the variable 'alpha_tm' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'alpha_tm', result_div_170998)
    
    # Assigning a Call to a Name (line 870):
    
    # Assigning a Call to a Name (line 870):
    
    # Call to clip(...): (line 870)
    # Processing the call arguments (line 870)
    # Getting the type of 'alpha_tp' (line 870)
    alpha_tp_171001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 26), 'alpha_tp', False)
    # Getting the type of 'tau_min' (line 870)
    tau_min_171002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 36), 'tau_min', False)
    # Getting the type of 'alpha_p' (line 870)
    alpha_p_171003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 46), 'alpha_p', False)
    # Applying the binary operator '*' (line 870)
    result_mul_171004 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 36), '*', tau_min_171002, alpha_p_171003)
    
    # Getting the type of 'tau_max' (line 870)
    tau_max_171005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 55), 'tau_max', False)
    # Getting the type of 'alpha_p' (line 870)
    alpha_p_171006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 65), 'alpha_p', False)
    # Applying the binary operator '*' (line 870)
    result_mul_171007 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 55), '*', tau_max_171005, alpha_p_171006)
    
    # Processing the call keyword arguments (line 870)
    kwargs_171008 = {}
    # Getting the type of 'np' (line 870)
    np_170999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 18), 'np', False)
    # Obtaining the member 'clip' of a type (line 870)
    clip_171000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 18), np_170999, 'clip')
    # Calling clip(args, kwargs) (line 870)
    clip_call_result_171009 = invoke(stypy.reporting.localization.Localization(__file__, 870, 18), clip_171000, *[alpha_tp_171001, result_mul_171004, result_mul_171007], **kwargs_171008)
    
    # Assigning a type to the variable 'alpha_p' (line 870)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'alpha_p', clip_call_result_171009)
    
    # Assigning a Call to a Name (line 871):
    
    # Assigning a Call to a Name (line 871):
    
    # Call to clip(...): (line 871)
    # Processing the call arguments (line 871)
    # Getting the type of 'alpha_tm' (line 871)
    alpha_tm_171012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 26), 'alpha_tm', False)
    # Getting the type of 'tau_min' (line 871)
    tau_min_171013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 36), 'tau_min', False)
    # Getting the type of 'alpha_m' (line 871)
    alpha_m_171014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 46), 'alpha_m', False)
    # Applying the binary operator '*' (line 871)
    result_mul_171015 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 36), '*', tau_min_171013, alpha_m_171014)
    
    # Getting the type of 'tau_max' (line 871)
    tau_max_171016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 55), 'tau_max', False)
    # Getting the type of 'alpha_m' (line 871)
    alpha_m_171017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 65), 'alpha_m', False)
    # Applying the binary operator '*' (line 871)
    result_mul_171018 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 55), '*', tau_max_171016, alpha_m_171017)
    
    # Processing the call keyword arguments (line 871)
    kwargs_171019 = {}
    # Getting the type of 'np' (line 871)
    np_171010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 18), 'np', False)
    # Obtaining the member 'clip' of a type (line 871)
    clip_171011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 18), np_171010, 'clip')
    # Calling clip(args, kwargs) (line 871)
    clip_call_result_171020 = invoke(stypy.reporting.localization.Localization(__file__, 871, 18), clip_171011, *[alpha_tm_171012, result_mul_171015, result_mul_171018], **kwargs_171019)
    
    # Assigning a type to the variable 'alpha_m' (line 871)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'alpha_m', clip_call_result_171020)
    # SSA join for while statement (line 851)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 874):
    
    # Assigning a BinOp to a Name (line 874):
    # Getting the type of 'nu' (line 874)
    nu_171021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 13), 'nu')
    # Getting the type of 'Q' (line 874)
    Q_171022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 18), 'Q')
    # Applying the binary operator '*' (line 874)
    result_mul_171023 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 13), '*', nu_171021, Q_171022)
    
    int_171024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 22), 'int')
    # Applying the binary operator '+' (line 874)
    result_add_171025 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 13), '+', result_mul_171023, int_171024)
    
    # Assigning a type to the variable 'Q_next' (line 874)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 4), 'Q_next', result_add_171025)
    
    # Assigning a BinOp to a Name (line 875):
    
    # Assigning a BinOp to a Name (line 875):
    # Getting the type of 'nu' (line 875)
    nu_171026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 9), 'nu')
    # Getting the type of 'Q' (line 875)
    Q_171027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 14), 'Q')
    # Applying the binary operator '*' (line 875)
    result_mul_171028 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 9), '*', nu_171026, Q_171027)
    
    # Getting the type of 'C' (line 875)
    C_171029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 19), 'C')
    # Getting the type of 'eta' (line 875)
    eta_171030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 23), 'eta')
    # Applying the binary operator '+' (line 875)
    result_add_171031 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 19), '+', C_171029, eta_171030)
    
    # Applying the binary operator '*' (line 875)
    result_mul_171032 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 16), '*', result_mul_171028, result_add_171031)
    
    # Getting the type of 'fp' (line 875)
    fp_171033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 30), 'fp')
    # Applying the binary operator '+' (line 875)
    result_add_171034 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 9), '+', result_mul_171032, fp_171033)
    
    # Getting the type of 'Q_next' (line 875)
    Q_next_171035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 36), 'Q_next')
    # Applying the binary operator 'div' (line 875)
    result_div_171036 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 8), 'div', result_add_171034, Q_next_171035)
    
    # Assigning a type to the variable 'C' (line 875)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 4), 'C', result_div_171036)
    
    # Assigning a Name to a Name (line 876):
    
    # Assigning a Name to a Name (line 876):
    # Getting the type of 'Q_next' (line 876)
    Q_next_171037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'Q_next')
    # Assigning a type to the variable 'Q' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 4), 'Q', Q_next_171037)
    
    # Obtaining an instance of the builtin type 'tuple' (line 878)
    tuple_171038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 878)
    # Adding element type (line 878)
    # Getting the type of 'alpha' (line 878)
    alpha_171039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 11), 'alpha')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 878, 11), tuple_171038, alpha_171039)
    # Adding element type (line 878)
    # Getting the type of 'xp' (line 878)
    xp_171040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 18), 'xp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 878, 11), tuple_171038, xp_171040)
    # Adding element type (line 878)
    # Getting the type of 'fp' (line 878)
    fp_171041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 22), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 878, 11), tuple_171038, fp_171041)
    # Adding element type (line 878)
    # Getting the type of 'Fp' (line 878)
    Fp_171042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 26), 'Fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 878, 11), tuple_171038, Fp_171042)
    # Adding element type (line 878)
    # Getting the type of 'C' (line 878)
    C_171043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 30), 'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 878, 11), tuple_171038, C_171043)
    # Adding element type (line 878)
    # Getting the type of 'Q' (line 878)
    Q_171044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 33), 'Q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 878, 11), tuple_171038, Q_171044)
    
    # Assigning a type to the variable 'stypy_return_type' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 4), 'stypy_return_type', tuple_171038)
    
    # ################# End of '_nonmonotone_line_search_cheng(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nonmonotone_line_search_cheng' in the type store
    # Getting the type of 'stypy_return_type' (line 800)
    stypy_return_type_171045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171045)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nonmonotone_line_search_cheng'
    return stypy_return_type_171045

# Assigning a type to the variable '_nonmonotone_line_search_cheng' (line 800)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 0), '_nonmonotone_line_search_cheng', _nonmonotone_line_search_cheng)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
