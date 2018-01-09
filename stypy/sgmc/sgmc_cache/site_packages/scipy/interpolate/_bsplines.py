
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import functools
4: import operator
5: 
6: import numpy as np
7: from scipy.linalg import (get_lapack_funcs, LinAlgError,
8:                           cholesky_banded, cho_solve_banded)
9: from . import _bspl
10: from . import _fitpack_impl
11: from . import _fitpack as _dierckx
12: 
13: __all__ = ["BSpline", "make_interp_spline", "make_lsq_spline"]
14: 
15: 
16: # copy-paste from interpolate.py
17: def prod(x):
18:     '''Product of a list of numbers; ~40x faster vs np.prod for Python tuples'''
19:     if len(x) == 0:
20:         return 1
21:     return functools.reduce(operator.mul, x)
22: 
23: 
24: def _get_dtype(dtype):
25:     '''Return np.complex128 for complex dtypes, np.float64 otherwise.'''
26:     if np.issubdtype(dtype, np.complexfloating):
27:         return np.complex_
28:     else:
29:         return np.float_
30: 
31: 
32: def _as_float_array(x, check_finite=False):
33:     '''Convert the input into a C contiguous float array.
34: 
35:     NB: Upcasts half- and single-precision floats to double precision.
36:     '''
37:     x = np.ascontiguousarray(x)
38:     dtyp = _get_dtype(x.dtype)
39:     x = x.astype(dtyp, copy=False)
40:     if check_finite and not np.isfinite(x).all():
41:         raise ValueError("Array must not contain infs or nans.")
42:     return x
43: 
44: 
45: class BSpline(object):
46:     r'''Univariate spline in the B-spline basis.
47: 
48:     .. math::
49: 
50:         S(x) = \sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)
51: 
52:     where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`
53:     and knots `t`.
54: 
55:     Parameters
56:     ----------
57:     t : ndarray, shape (n+k+1,)
58:         knots
59:     c : ndarray, shape (>=n, ...)
60:         spline coefficients
61:     k : int
62:         B-spline order
63:     extrapolate : bool or 'periodic', optional
64:         whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,
65:         or to return nans.
66:         If True, extrapolates the first and last polynomial pieces of b-spline
67:         functions active on the base interval.
68:         If 'periodic', periodic extrapolation is used.
69:         Default is True.
70:     axis : int, optional
71:         Interpolation axis. Default is zero.
72: 
73:     Attributes
74:     ----------
75:     t : ndarray
76:         knot vector
77:     c : ndarray
78:         spline coefficients
79:     k : int
80:         spline degree
81:     extrapolate : bool
82:         If True, extrapolates the first and last polynomial pieces of b-spline
83:         functions active on the base interval.
84:     axis : int
85:         Interpolation axis.
86:     tck : tuple
87:         A read-only equivalent of ``(self.t, self.c, self.k)``
88: 
89:     Methods
90:     -------
91:     __call__
92:     basis_element
93:     derivative
94:     antiderivative
95:     integrate
96:     construct_fast
97: 
98:     Notes
99:     -----
100:     B-spline basis elements are defined via
101: 
102:     .. math::
103: 
104:         B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}
105: 
106:         B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
107:                  + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)
108: 
109:     **Implementation details**
110: 
111:     - At least ``k+1`` coefficients are required for a spline of degree `k`,
112:       so that ``n >= k+1``. Additional coefficients, ``c[j]`` with
113:       ``j > n``, are ignored.
114: 
115:     - B-spline basis elements of degree `k` form a partition of unity on the
116:       *base interval*, ``t[k] <= x <= t[n]``.
117: 
118: 
119:     Examples
120:     --------
121: 
122:     Translating the recursive definition of B-splines into Python code, we have:
123: 
124:     >>> def B(x, k, i, t):
125:     ...    if k == 0:
126:     ...       return 1.0 if t[i] <= x < t[i+1] else 0.0
127:     ...    if t[i+k] == t[i]:
128:     ...       c1 = 0.0
129:     ...    else:
130:     ...       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
131:     ...    if t[i+k+1] == t[i+1]:
132:     ...       c2 = 0.0
133:     ...    else:
134:     ...       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
135:     ...    return c1 + c2
136: 
137:     >>> def bspline(x, t, c, k):
138:     ...    n = len(t) - k - 1
139:     ...    assert (n >= k+1) and (len(c) >= n)
140:     ...    return sum(c[i] * B(x, k, i, t) for i in range(n))
141: 
142:     Note that this is an inefficient (if straightforward) way to
143:     evaluate B-splines --- this spline class does it in an equivalent,
144:     but much more efficient way.
145: 
146:     Here we construct a quadratic spline function on the base interval
147:     ``2 <= x <= 4`` and compare with the naive way of evaluating the spline:
148: 
149:     >>> from scipy.interpolate import BSpline
150:     >>> k = 2
151:     >>> t = [0, 1, 2, 3, 4, 5, 6]
152:     >>> c = [-1, 2, 0, -1]
153:     >>> spl = BSpline(t, c, k)
154:     >>> spl(2.5)
155:     array(1.375)
156:     >>> bspline(2.5, t, c, k)
157:     1.375
158: 
159:     Note that outside of the base interval results differ. This is because
160:     `BSpline` extrapolates the first and last polynomial pieces of b-spline
161:     functions active on the base interval.
162: 
163:     >>> import matplotlib.pyplot as plt
164:     >>> fig, ax = plt.subplots()
165:     >>> xx = np.linspace(1.5, 4.5, 50)
166:     >>> ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
167:     >>> ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
168:     >>> ax.grid(True)
169:     >>> ax.legend(loc='best')
170:     >>> plt.show()
171: 
172: 
173:     References
174:     ----------
175:     .. [1] Tom Lyche and Knut Morken, Spline methods,
176:         http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/
177:     .. [2] Carl de Boor, A practical guide to splines, Springer, 2001.
178: 
179:     '''
180:     def __init__(self, t, c, k, extrapolate=True, axis=0):
181:         super(BSpline, self).__init__()
182: 
183:         self.k = int(k)
184:         self.c = np.asarray(c)
185:         self.t = np.ascontiguousarray(t, dtype=np.float64)
186: 
187:         if extrapolate == 'periodic':
188:             self.extrapolate = extrapolate
189:         else:
190:             self.extrapolate = bool(extrapolate)
191: 
192:         n = self.t.shape[0] - self.k - 1
193: 
194:         if not (0 <= axis < self.c.ndim):
195:             raise ValueError("%s must be between 0 and %s" % (axis, c.ndim))
196: 
197:         self.axis = axis
198:         if axis != 0:
199:             # roll the interpolation axis to be the first one in self.c
200:             # More specifically, the target shape for self.c is (n, ...),
201:             # and axis !=0 means that we have c.shape (..., n, ...)
202:             #                                               ^
203:             #                                              axis
204:             self.c = np.rollaxis(self.c, axis)
205: 
206:         if k < 0:
207:             raise ValueError("Spline order cannot be negative.")
208:         if int(k) != k:
209:             raise ValueError("Spline order must be integer.")
210:         if self.t.ndim != 1:
211:             raise ValueError("Knot vector must be one-dimensional.")
212:         if n < self.k + 1:
213:             raise ValueError("Need at least %d knots for degree %d" %
214:                     (2*k + 2, k))
215:         if (np.diff(self.t) < 0).any():
216:             raise ValueError("Knots must be in a non-decreasing order.")
217:         if len(np.unique(self.t[k:n+1])) < 2:
218:             raise ValueError("Need at least two internal knots.")
219:         if not np.isfinite(self.t).all():
220:             raise ValueError("Knots should not have nans or infs.")
221:         if self.c.ndim < 1:
222:             raise ValueError("Coefficients must be at least 1-dimensional.")
223:         if self.c.shape[0] < n:
224:             raise ValueError("Knots, coefficients and degree are inconsistent.")
225: 
226:         dt = _get_dtype(self.c.dtype)
227:         self.c = np.ascontiguousarray(self.c, dtype=dt)
228: 
229:     @classmethod
230:     def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
231:         '''Construct a spline without making checks.
232: 
233:         Accepts same parameters as the regular constructor. Input arrays
234:         `t` and `c` must of correct shape and dtype.
235:         '''
236:         self = object.__new__(cls)
237:         self.t, self.c, self.k = t, c, k
238:         self.extrapolate = extrapolate
239:         self.axis = axis
240:         return self
241: 
242:     @property
243:     def tck(self):
244:         '''Equvalent to ``(self.t, self.c, self.k)`` (read-only).
245:         '''
246:         return self.t, self.c, self.k
247: 
248:     @classmethod
249:     def basis_element(cls, t, extrapolate=True):
250:         '''Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.
251: 
252:         Parameters
253:         ----------
254:         t : ndarray, shape (k+1,)
255:             internal knots
256:         extrapolate : bool or 'periodic', optional
257:             whether to extrapolate beyond the base interval, ``t[0] .. t[k+1]``,
258:             or to return nans.
259:             If 'periodic', periodic extrapolation is used.
260:             Default is True.
261: 
262:         Returns
263:         -------
264:         basis_element : callable
265:             A callable representing a B-spline basis element for the knot
266:             vector `t`.
267: 
268:         Notes
269:         -----
270:         The order of the b-spline, `k`, is inferred from the length of `t` as
271:         ``len(t)-2``. The knot vector is constructed by appending and prepending
272:         ``k+1`` elements to internal knots `t`.
273: 
274:         Examples
275:         --------
276: 
277:         Construct a cubic b-spline:
278: 
279:         >>> from scipy.interpolate import BSpline
280:         >>> b = BSpline.basis_element([0, 1, 2, 3, 4])
281:         >>> k = b.k
282:         >>> b.t[k:-k]
283:         array([ 0.,  1.,  2.,  3.,  4.])
284:         >>> k
285:         3
286: 
287:         Construct a second order b-spline on ``[0, 1, 1, 2]``, and compare
288:         to its explicit form:
289: 
290:         >>> t = [-1, 0, 1, 1, 2]
291:         >>> b = BSpline.basis_element(t[1:])
292:         >>> def f(x):
293:         ...     return np.where(x < 1, x*x, (2. - x)**2)
294: 
295:         >>> import matplotlib.pyplot as plt
296:         >>> fig, ax = plt.subplots()
297:         >>> x = np.linspace(0, 2, 51)
298:         >>> ax.plot(x, b(x), 'g', lw=3)
299:         >>> ax.plot(x, f(x), 'r', lw=8, alpha=0.4)
300:         >>> ax.grid(True)
301:         >>> plt.show()
302: 
303:         '''
304:         k = len(t) - 2
305:         t = _as_float_array(t)
306:         t = np.r_[(t[0]-1,) * k, t, (t[-1]+1,) * k]
307:         c = np.zeros_like(t)
308:         c[k] = 1.
309:         return cls.construct_fast(t, c, k, extrapolate)
310: 
311:     def __call__(self, x, nu=0, extrapolate=None):
312:         '''
313:         Evaluate a spline function.
314: 
315:         Parameters
316:         ----------
317:         x : array_like
318:             points to evaluate the spline at.
319:         nu: int, optional
320:             derivative to evaluate (default is 0).
321:         extrapolate : bool or 'periodic', optional
322:             whether to extrapolate based on the first and last intervals
323:             or return nans. If 'periodic', periodic extrapolation is used.
324:             Default is `self.extrapolate`.
325: 
326:         Returns
327:         -------
328:         y : array_like
329:             Shape is determined by replacing the interpolation axis
330:             in the coefficient array with the shape of `x`.
331: 
332:         '''
333:         if extrapolate is None:
334:             extrapolate = self.extrapolate
335:         x = np.asarray(x)
336:         x_shape, x_ndim = x.shape, x.ndim
337:         x = np.ascontiguousarray(x.ravel(), dtype=np.float_)
338: 
339:         # With periodic extrapolation we map x to the segment
340:         # [self.t[k], self.t[n]].
341:         if extrapolate == 'periodic':
342:             n = self.t.size - self.k - 1
343:             x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] -
344:                                                          self.t[self.k])
345:             extrapolate = False
346: 
347:         out = np.empty((len(x), prod(self.c.shape[1:])), dtype=self.c.dtype)
348:         self._ensure_c_contiguous()
349:         self._evaluate(x, nu, extrapolate, out)
350:         out = out.reshape(x_shape + self.c.shape[1:])
351:         if self.axis != 0:
352:             # transpose to move the calculated values to the interpolation axis
353:             l = list(range(out.ndim))
354:             l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
355:             out = out.transpose(l)
356:         return out
357: 
358:     def _evaluate(self, xp, nu, extrapolate, out):
359:         _bspl.evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1),
360:                 self.k, xp, nu, extrapolate, out)
361: 
362:     def _ensure_c_contiguous(self):
363:         '''
364:         c and t may be modified by the user. The Cython code expects
365:         that they are C contiguous.
366: 
367:         '''
368:         if not self.t.flags.c_contiguous:
369:             self.t = self.t.copy()
370:         if not self.c.flags.c_contiguous:
371:             self.c = self.c.copy()
372: 
373:     def derivative(self, nu=1):
374:         '''Return a b-spline representing the derivative.
375: 
376:         Parameters
377:         ----------
378:         nu : int, optional
379:             Derivative order.
380:             Default is 1.
381: 
382:         Returns
383:         -------
384:         b : BSpline object
385:             A new instance representing the derivative.
386: 
387:         See Also
388:         --------
389:         splder, splantider
390: 
391:         '''
392:         c = self.c
393:         # pad the c array if needed
394:         ct = len(self.t) - len(c)
395:         if ct > 0:
396:             c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
397:         tck = _fitpack_impl.splder((self.t, c, self.k), nu)
398:         return self.construct_fast(*tck, extrapolate=self.extrapolate,
399:                                     axis=self.axis)
400: 
401:     def antiderivative(self, nu=1):
402:         '''Return a b-spline representing the antiderivative.
403: 
404:         Parameters
405:         ----------
406:         nu : int, optional
407:             Antiderivative order. Default is 1.
408: 
409:         Returns
410:         -------
411:         b : BSpline object
412:             A new instance representing the antiderivative.
413: 
414:         Notes
415:         -----
416:         If antiderivative is computed and ``self.extrapolate='periodic'``,
417:         it will be set to False for the returned instance. This is done because
418:         the antiderivative is no longer periodic and its correct evaluation
419:         outside of the initially given x interval is difficult.
420: 
421:         See Also
422:         --------
423:         splder, splantider
424: 
425:         '''
426:         c = self.c
427:         # pad the c array if needed
428:         ct = len(self.t) - len(c)
429:         if ct > 0:
430:             c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
431:         tck = _fitpack_impl.splantider((self.t, c, self.k), nu)
432: 
433:         if self.extrapolate == 'periodic':
434:             extrapolate = False
435:         else:
436:             extrapolate = self.extrapolate
437: 
438:         return self.construct_fast(*tck, extrapolate=extrapolate,
439:                                    axis=self.axis)
440: 
441:     def integrate(self, a, b, extrapolate=None):
442:         '''Compute a definite integral of the spline.
443: 
444:         Parameters
445:         ----------
446:         a : float
447:             Lower limit of integration.
448:         b : float
449:             Upper limit of integration.
450:         extrapolate : bool or 'periodic', optional
451:             whether to extrapolate beyond the base interval,
452:             ``t[k] .. t[-k-1]``, or take the spline to be zero outside of the
453:             base interval. If 'periodic', periodic extrapolation is used.
454:             If None (default), use `self.extrapolate`.
455: 
456:         Returns
457:         -------
458:         I : array_like
459:             Definite integral of the spline over the interval ``[a, b]``.
460: 
461:         Examples
462:         --------
463:         Construct the linear spline ``x if x < 1 else 2 - x`` on the base
464:         interval :math:`[0, 2]`, and integrate it
465: 
466:         >>> from scipy.interpolate import BSpline
467:         >>> b = BSpline.basis_element([0, 1, 2])
468:         >>> b.integrate(0, 1)
469:         array(0.5)
470: 
471:         If the integration limits are outside of the base interval, the result
472:         is controlled by the `extrapolate` parameter
473: 
474:         >>> b.integrate(-1, 1)
475:         array(0.0)
476:         >>> b.integrate(-1, 1, extrapolate=False)
477:         array(0.5)
478: 
479:         >>> import matplotlib.pyplot as plt
480:         >>> fig, ax = plt.subplots()
481:         >>> ax.grid(True)
482:         >>> ax.axvline(0, c='r', lw=5, alpha=0.5)  # base interval
483:         >>> ax.axvline(2, c='r', lw=5, alpha=0.5)
484:         >>> xx = [-1, 1, 2]
485:         >>> ax.plot(xx, b(xx))
486:         >>> plt.show()
487: 
488:         '''
489:         if extrapolate is None:
490:             extrapolate = self.extrapolate
491: 
492:         # Prepare self.t and self.c.
493:         self._ensure_c_contiguous()
494: 
495:         # Swap integration bounds if needed.
496:         sign = 1
497:         if b < a:
498:             a, b = b, a
499:             sign = -1
500:         n = self.t.size - self.k - 1
501: 
502:         if extrapolate != "periodic" and not extrapolate:
503:             # Shrink the integration interval, if needed.
504:             a = max(a, self.t[self.k])
505:             b = min(b, self.t[n])
506: 
507:             if self.c.ndim == 1:
508:                 # Fast path: use FITPACK's routine
509:                 # (cf _fitpack_impl.splint).
510:                 t, c, k = self.tck
511:                 integral, wrk = _dierckx._splint(t, c, k, a, b)
512:                 return integral * sign 
513: 
514:         out = np.empty((2, prod(self.c.shape[1:])), dtype=self.c.dtype)
515: 
516:         # Compute the antiderivative.
517:         c = self.c
518:         ct = len(self.t) - len(c)
519:         if ct > 0:
520:             c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
521:         ta, ca, ka = _fitpack_impl.splantider((self.t, c, self.k), 1)
522: 
523:         if extrapolate == 'periodic':
524:             # Split the integral into the part over period (can be several
525:             # of them) and the remaining part.
526: 
527:             ts, te = self.t[self.k], self.t[n]
528:             period = te - ts
529:             interval = b - a
530:             n_periods, left = divmod(interval, period)
531: 
532:             if n_periods > 0:
533:                 # Evaluate the difference of antiderivatives.
534:                 x = np.asarray([ts, te], dtype=np.float_)
535:                 _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
536:                                       ka, x, 0, False, out)
537:                 integral = out[1] - out[0]
538:                 integral *= n_periods
539:             else:
540:                 integral = np.zeros((1, prod(self.c.shape[1:])),
541:                                     dtype=self.c.dtype)
542: 
543:             # Map a to [ts, te], b is always a + left.
544:             a = ts + (a - ts) % period
545:             b = a + left
546: 
547:             # If b <= te then we need to integrate over [a, b], otherwise
548:             # over [a, te] and from xs to what is remained.
549:             if b <= te:
550:                 x = np.asarray([a, b], dtype=np.float_)
551:                 _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
552:                                       ka, x, 0, False, out)
553:                 integral += out[1] - out[0]
554:             else:
555:                 x = np.asarray([a, te], dtype=np.float_)
556:                 _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
557:                                       ka, x, 0, False, out)
558:                 integral += out[1] - out[0]
559: 
560:                 x = np.asarray([ts, ts + b - te], dtype=np.float_)
561:                 _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
562:                                       ka, x, 0, False, out)
563:                 integral += out[1] - out[0]
564:         else:
565:             # Evaluate the difference of antiderivatives.
566:             x = np.asarray([a, b], dtype=np.float_)
567:             _bspl.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
568:                                   ka, x, 0, extrapolate, out)
569:             integral = out[1] - out[0]
570: 
571:         integral *= sign
572:         return integral.reshape(ca.shape[1:])
573: 
574: 
575: #################################
576: #  Interpolating spline helpers #
577: #################################
578: 
579: def _not_a_knot(x, k):
580:     '''Given data x, construct the knot vector w/ not-a-knot BC.
581:     cf de Boor, XIII(12).'''
582:     x = np.asarray(x)
583:     if k % 2 != 1:
584:         raise ValueError("Odd degree for now only. Got %s." % k)
585: 
586:     m = (k - 1) // 2
587:     t = x[m+1:-m-1]
588:     t = np.r_[(x[0],)*(k+1), t, (x[-1],)*(k+1)]
589:     return t
590: 
591: 
592: def _augknt(x, k):
593:     '''Construct a knot vector appropriate for the order-k interpolation.'''
594:     return np.r_[(x[0],)*k, x, (x[-1],)*k]
595: 
596: 
597: def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0,
598:                        check_finite=True):
599:     '''Compute the (coefficients of) interpolating B-spline.
600: 
601:     Parameters
602:     ----------
603:     x : array_like, shape (n,)
604:         Abscissas.
605:     y : array_like, shape (n, ...)
606:         Ordinates.
607:     k : int, optional
608:         B-spline degree. Default is cubic, k=3.
609:     t : array_like, shape (nt + k + 1,), optional.
610:         Knots.
611:         The number of knots needs to agree with the number of datapoints and
612:         the number of derivatives at the edges. Specifically, ``nt - n`` must
613:         equal ``len(deriv_l) + len(deriv_r)``.
614:     bc_type : 2-tuple or None
615:         Boundary conditions.
616:         Default is None, which means choosing the boundary conditions
617:         automatically. Otherwise, it must be a length-two tuple where the first
618:         element sets the boundary conditions at ``x[0]`` and the second
619:         element sets the boundary conditions at ``x[-1]``. Each of these must
620:         be an iterable of pairs ``(order, value)`` which gives the values of
621:         derivatives of specified orders at the given edge of the interpolation
622:         interval.
623:     axis : int, optional
624:         Interpolation axis. Default is 0.
625:     check_finite : bool, optional
626:         Whether to check that the input arrays contain only finite numbers.
627:         Disabling may give a performance gain, but may result in problems
628:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
629:         Default is True.
630: 
631:     Returns
632:     -------
633:     b : a BSpline object of the degree ``k`` and with knots ``t``.
634: 
635:     Examples
636:     --------
637: 
638:     Use cubic interpolation on Chebyshev nodes:
639: 
640:     >>> def cheb_nodes(N):
641:     ...     jj = 2.*np.arange(N) + 1
642:     ...     x = np.cos(np.pi * jj / 2 / N)[::-1]
643:     ...     return x
644: 
645:     >>> x = cheb_nodes(20)
646:     >>> y = np.sqrt(1 - x**2)
647: 
648:     >>> from scipy.interpolate import BSpline, make_interp_spline
649:     >>> b = make_interp_spline(x, y)
650:     >>> np.allclose(b(x), y)
651:     True
652: 
653:     Note that the default is a cubic spline with a not-a-knot boundary condition
654: 
655:     >>> b.k
656:     3
657: 
658:     Here we use a 'natural' spline, with zero 2nd derivatives at edges:
659: 
660:     >>> l, r = [(2, 0)], [(2, 0)]
661:     >>> b_n = make_interp_spline(x, y, bc_type=(l, r))
662:     >>> np.allclose(b_n(x), y)
663:     True
664:     >>> x0, x1 = x[0], x[-1]
665:     >>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
666:     True
667: 
668:     Interpolation of parametric curves is also supported. As an example, we
669:     compute a discretization of a snail curve in polar coordinates
670: 
671:     >>> phi = np.linspace(0, 2.*np.pi, 40)
672:     >>> r = 0.3 + np.cos(phi)
673:     >>> x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates
674: 
675:     Build an interpolating curve, parameterizing it by the angle
676: 
677:     >>> from scipy.interpolate import make_interp_spline
678:     >>> spl = make_interp_spline(phi, np.c_[x, y])
679: 
680:     Evaluate the interpolant on a finer grid (note that we transpose the result
681:     to unpack it into a pair of x- and y-arrays)
682: 
683:     >>> phi_new = np.linspace(0, 2.*np.pi, 100)
684:     >>> x_new, y_new = spl(phi_new).T
685: 
686:     Plot the result
687: 
688:     >>> import matplotlib.pyplot as plt
689:     >>> plt.plot(x, y, 'o')
690:     >>> plt.plot(x_new, y_new, '-')
691:     >>> plt.show()
692: 
693:     See Also
694:     --------
695:     BSpline : base class representing the B-spline objects
696:     CubicSpline : a cubic spline in the polynomial basis
697:     make_lsq_spline : a similar factory function for spline fitting
698:     UnivariateSpline : a wrapper over FITPACK spline fitting routines
699:     splrep : a wrapper over FITPACK spline fitting routines
700: 
701:     '''
702:     if bc_type is None:
703:         bc_type = (None, None)
704:     deriv_l, deriv_r = bc_type
705: 
706:     # special-case k=0 right away
707:     if k == 0:
708:         if any(_ is not None for _ in (t, deriv_l, deriv_r)):
709:             raise ValueError("Too much info for k=0: t and bc_type can only "
710:                              "be None.")
711:         x = _as_float_array(x, check_finite)
712:         t = np.r_[x, x[-1]]
713:         c = np.asarray(y)
714:         c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
715:         return BSpline.construct_fast(t, c, k, axis=axis)
716: 
717:     # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
718:     if k == 1 and t is None:
719:         if not (deriv_l is None and deriv_r is None):
720:             raise ValueError("Too much info for k=1: bc_type can only be None.")
721:         x = _as_float_array(x, check_finite)
722:         t = np.r_[x[0], x, x[-1]]
723:         c = np.asarray(y)
724:         c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
725:         return BSpline.construct_fast(t, c, k, axis=axis)
726: 
727:     # come up with a sensible knot vector, if needed
728:     if t is None:
729:         if deriv_l is None and deriv_r is None:
730:             if k == 2:
731:                 # OK, it's a bit ad hoc: Greville sites + omit
732:                 # 2nd and 2nd-to-last points, a la not-a-knot
733:                 t = (x[1:] + x[:-1]) / 2.
734:                 t = np.r_[(x[0],)*(k+1),
735:                            t[1:-1],
736:                            (x[-1],)*(k+1)]
737:             else:
738:                 t = _not_a_knot(x, k)
739:         else:
740:             t = _augknt(x, k)
741: 
742:     x = _as_float_array(x, check_finite)
743:     y = _as_float_array(y, check_finite)
744:     t = _as_float_array(t, check_finite)
745:     k = int(k)
746: 
747:     axis = axis % y.ndim
748:     y = np.rollaxis(y, axis)    # now internally interp axis is zero
749: 
750:     if x.ndim != 1 or np.any(x[1:] <= x[:-1]):
751:         raise ValueError("Expect x to be a 1-D sorted array_like.")
752:     if k < 0:
753:         raise ValueError("Expect non-negative k.")
754:     if t.ndim != 1 or np.any(t[1:] < t[:-1]):
755:         raise ValueError("Expect t to be a 1-D sorted array_like.")
756:     if x.size != y.shape[0]:
757:         raise ValueError('x and y are incompatible.')
758:     if t.size < x.size + k + 1:
759:         raise ValueError('Got %d knots, need at least %d.' %
760:                          (t.size, x.size + k + 1))
761:     if (x[0] < t[k]) or (x[-1] > t[-k]):
762:         raise ValueError('Out of bounds w/ x = %s.' % x)
763: 
764:     # Here : deriv_l, r = [(nu, value), ...]
765:     if deriv_l is not None:
766:         deriv_l_ords, deriv_l_vals = zip(*deriv_l)
767:     else:
768:         deriv_l_ords, deriv_l_vals = [], []
769:     deriv_l_ords, deriv_l_vals = np.atleast_1d(deriv_l_ords, deriv_l_vals)
770:     nleft = deriv_l_ords.shape[0]
771: 
772:     if deriv_r is not None:
773:         deriv_r_ords, deriv_r_vals = zip(*deriv_r)
774:     else:
775:         deriv_r_ords, deriv_r_vals = [], []
776:     deriv_r_ords, deriv_r_vals = np.atleast_1d(deriv_r_ords, deriv_r_vals)
777:     nright = deriv_r_ords.shape[0]
778: 
779:     # have `n` conditions for `nt` coefficients; need nt-n derivatives
780:     n = x.size
781:     nt = t.size - k - 1
782: 
783:     if nt - n != nleft + nright:
784:         raise ValueError("number of derivatives at boundaries.")
785: 
786:     # set up the LHS: the collocation matrix + derivatives at boundaries
787:     kl = ku = k
788:     ab = np.zeros((2*kl + ku + 1, nt), dtype=np.float_, order='F')
789:     _bspl._colloc(x, t, k, ab, offset=nleft)
790:     if nleft > 0:
791:         _bspl._handle_lhs_derivatives(t, k, x[0], ab, kl, ku, deriv_l_ords)
792:     if nright > 0:
793:         _bspl._handle_lhs_derivatives(t, k, x[-1], ab, kl, ku, deriv_r_ords,
794:                                 offset=nt-nright)
795: 
796:     # set up the RHS: values to interpolate (+ derivative values, if any)
797:     extradim = prod(y.shape[1:])
798:     rhs = np.empty((nt, extradim), dtype=y.dtype)
799:     if nleft > 0:
800:         rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
801:     rhs[nleft:nt - nright] = y.reshape(-1, extradim)
802:     if nright > 0:
803:         rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)
804: 
805:     # solve Ab @ x = rhs; this is the relevant part of linalg.solve_banded
806:     if check_finite:
807:         ab, rhs = map(np.asarray_chkfinite, (ab, rhs))
808:     gbsv, = get_lapack_funcs(('gbsv',), (ab, rhs))
809:     lu, piv, c, info = gbsv(kl, ku, ab, rhs,
810:             overwrite_ab=True, overwrite_b=True)
811: 
812:     if info > 0:
813:         raise LinAlgError("Collocation matix is singular.")
814:     elif info < 0:
815:         raise ValueError('illegal value in %d-th argument of internal gbsv' % -info)
816: 
817:     c = np.ascontiguousarray(c.reshape((nt,) + y.shape[1:]))
818:     return BSpline.construct_fast(t, c, k, axis=axis)
819: 
820: 
821: def make_lsq_spline(x, y, t, k=3, w=None, axis=0, check_finite=True):
822:     r'''Compute the (coefficients of) an LSQ B-spline.
823: 
824:     The result is a linear combination
825: 
826:     .. math::
827: 
828:             S(x) = \sum_j c_j B_j(x; t)
829: 
830:     of the B-spline basis elements, :math:`B_j(x; t)`, which minimizes
831: 
832:     .. math::
833: 
834:         \sum_{j} \left( w_j \times (S(x_j) - y_j) \right)^2
835: 
836:     Parameters
837:     ----------
838:     x : array_like, shape (m,)
839:         Abscissas.
840:     y : array_like, shape (m, ...)
841:         Ordinates.
842:     t : array_like, shape (n + k + 1,).
843:         Knots.
844:         Knots and data points must satisfy Schoenberg-Whitney conditions.
845:     k : int, optional
846:         B-spline degree. Default is cubic, k=3.
847:     w : array_like, shape (n,), optional
848:         Weights for spline fitting. Must be positive. If ``None``,
849:         then weights are all equal.
850:         Default is ``None``.
851:     axis : int, optional
852:         Interpolation axis. Default is zero.
853:     check_finite : bool, optional
854:         Whether to check that the input arrays contain only finite numbers.
855:         Disabling may give a performance gain, but may result in problems
856:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
857:         Default is True.
858: 
859:     Returns
860:     -------
861:     b : a BSpline object of the degree `k` with knots `t`.
862: 
863:     Notes
864:     -----
865: 
866:     The number of data points must be larger than the spline degree `k`.
867: 
868:     Knots `t` must satisfy the Schoenberg-Whitney conditions,
869:     i.e., there must be a subset of data points ``x[j]`` such that
870:     ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.
871: 
872:     Examples
873:     --------
874:     Generate some noisy data:
875: 
876:     >>> x = np.linspace(-3, 3, 50)
877:     >>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)
878: 
879:     Now fit a smoothing cubic spline with a pre-defined internal knots.
880:     Here we make the knot vector (k+1)-regular by adding boundary knots:
881: 
882:     >>> from scipy.interpolate import make_lsq_spline, BSpline
883:     >>> t = [-1, 0, 1]
884:     >>> k = 3
885:     >>> t = np.r_[(x[0],)*(k+1),
886:     ...           t,
887:     ...           (x[-1],)*(k+1)]
888:     >>> spl = make_lsq_spline(x, y, t, k)
889: 
890:     For comparison, we also construct an interpolating spline for the same
891:     set of data:
892: 
893:     >>> from scipy.interpolate import make_interp_spline
894:     >>> spl_i = make_interp_spline(x, y)
895: 
896:     Plot both:
897: 
898:     >>> import matplotlib.pyplot as plt
899:     >>> xs = np.linspace(-3, 3, 100)
900:     >>> plt.plot(x, y, 'ro', ms=5)
901:     >>> plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')
902:     >>> plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')
903:     >>> plt.legend(loc='best')
904:     >>> plt.show()
905: 
906:     **NaN handling**: If the input arrays contain ``nan`` values, the result is
907:     not useful since the underlying spline fitting routines cannot deal with
908:     ``nan``. A workaround is to use zero weights for not-a-number data points:
909: 
910:     >>> y[8] = np.nan
911:     >>> w = np.isnan(y)
912:     >>> y[w] = 0.
913:     >>> tck = make_lsq_spline(x, y, t, w=~w)
914: 
915:     Notice the need to replace a ``nan`` by a numerical value (precise value
916:     does not matter as long as the corresponding weight is zero.)
917: 
918:     See Also
919:     --------
920:     BSpline : base class representing the B-spline objects
921:     make_interp_spline : a similar factory function for interpolating splines
922:     LSQUnivariateSpline : a FITPACK-based spline fitting routine
923:     splrep : a FITPACK-based fitting routine
924: 
925:     '''
926:     x = _as_float_array(x, check_finite)
927:     y = _as_float_array(y, check_finite)
928:     t = _as_float_array(t, check_finite)
929:     if w is not None:
930:         w = _as_float_array(w, check_finite)
931:     else:
932:         w = np.ones_like(x)
933:     k = int(k)
934: 
935:     axis = axis % y.ndim
936:     y = np.rollaxis(y, axis)    # now internally interp axis is zero
937: 
938:     if x.ndim != 1 or np.any(x[1:] - x[:-1] <= 0):
939:         raise ValueError("Expect x to be a 1-D sorted array_like.")
940:     if x.shape[0] < k+1:
941:         raise ValueError("Need more x points.")
942:     if k < 0:
943:         raise ValueError("Expect non-negative k.")
944:     if t.ndim != 1 or np.any(t[1:] - t[:-1] < 0):
945:         raise ValueError("Expect t to be a 1-D sorted array_like.")
946:     if x.size != y.shape[0]:
947:         raise ValueError('x & y are incompatible.')
948:     if k > 0 and np.any((x < t[k]) | (x > t[-k])):
949:         raise ValueError('Out of bounds w/ x = %s.' % x)
950:     if x.size != w.size:
951:         raise ValueError('Incompatible weights.')
952: 
953:     # number of coefficients
954:     n = t.size - k - 1
955: 
956:     # construct A.T @ A and rhs with A the collocation matrix, and
957:     # rhs = A.T @ y for solving the LSQ problem  ``A.T @ A @ c = A.T @ y``
958:     lower = True
959:     extradim = prod(y.shape[1:])
960:     ab = np.zeros((k+1, n), dtype=np.float_, order='F')
961:     rhs = np.zeros((n, extradim), dtype=y.dtype, order='F')
962:     _bspl._norm_eq_lsq(x, t, k,
963:                       y.reshape(-1, extradim),
964:                       w,
965:                       ab, rhs)
966:     rhs = rhs.reshape((n,) + y.shape[1:])
967: 
968:     # have observation matrix & rhs, can solve the LSQ problem
969:     cho_decomp = cholesky_banded(ab, overwrite_ab=True, lower=lower,
970:                                  check_finite=check_finite)
971:     c = cho_solve_banded((cho_decomp, lower), rhs, overwrite_b=True,
972:                          check_finite=check_finite)
973: 
974:     c = np.ascontiguousarray(c)
975:     return BSpline.construct_fast(t, c, k, axis=axis)
976: 
977: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import functools' statement (line 3)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import operator' statement (line 4)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_73711 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_73711) is not StypyTypeError):

    if (import_73711 != 'pyd_module'):
        __import__(import_73711)
        sys_modules_73712 = sys.modules[import_73711]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_73712.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_73711)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.linalg import get_lapack_funcs, LinAlgError, cholesky_banded, cho_solve_banded' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_73713 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg')

if (type(import_73713) is not StypyTypeError):

    if (import_73713 != 'pyd_module'):
        __import__(import_73713)
        sys_modules_73714 = sys.modules[import_73713]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', sys_modules_73714.module_type_store, module_type_store, ['get_lapack_funcs', 'LinAlgError', 'cholesky_banded', 'cho_solve_banded'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_73714, sys_modules_73714.module_type_store, module_type_store)
    else:
        from scipy.linalg import get_lapack_funcs, LinAlgError, cholesky_banded, cho_solve_banded

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', None, module_type_store, ['get_lapack_funcs', 'LinAlgError', 'cholesky_banded', 'cho_solve_banded'], [get_lapack_funcs, LinAlgError, cholesky_banded, cho_solve_banded])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', import_73713)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.interpolate import _bspl' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_73715 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.interpolate')

if (type(import_73715) is not StypyTypeError):

    if (import_73715 != 'pyd_module'):
        __import__(import_73715)
        sys_modules_73716 = sys.modules[import_73715]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.interpolate', sys_modules_73716.module_type_store, module_type_store, ['_bspl'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_73716, sys_modules_73716.module_type_store, module_type_store)
    else:
        from scipy.interpolate import _bspl

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.interpolate', None, module_type_store, ['_bspl'], [_bspl])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.interpolate', import_73715)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.interpolate import _fitpack_impl' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_73717 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate')

if (type(import_73717) is not StypyTypeError):

    if (import_73717 != 'pyd_module'):
        __import__(import_73717)
        sys_modules_73718 = sys.modules[import_73717]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate', sys_modules_73718.module_type_store, module_type_store, ['_fitpack_impl'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_73718, sys_modules_73718.module_type_store, module_type_store)
    else:
        from scipy.interpolate import _fitpack_impl

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate', None, module_type_store, ['_fitpack_impl'], [_fitpack_impl])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate', import_73717)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.interpolate import _dierckx' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_73719 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate')

if (type(import_73719) is not StypyTypeError):

    if (import_73719 != 'pyd_module'):
        __import__(import_73719)
        sys_modules_73720 = sys.modules[import_73719]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate', sys_modules_73720.module_type_store, module_type_store, ['_fitpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_73720, sys_modules_73720.module_type_store, module_type_store)
    else:
        from scipy.interpolate import _fitpack as _dierckx

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate', None, module_type_store, ['_fitpack'], [_dierckx])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate', import_73719)

# Adding an alias
module_type_store.add_alias('_dierckx', '_fitpack')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['BSpline', 'make_interp_spline', 'make_lsq_spline']
module_type_store.set_exportable_members(['BSpline', 'make_interp_spline', 'make_lsq_spline'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_73721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_73722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'BSpline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_73721, str_73722)
# Adding element type (line 13)
str_73723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'str', 'make_interp_spline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_73721, str_73723)
# Adding element type (line 13)
str_73724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 44), 'str', 'make_lsq_spline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_73721, str_73724)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_73721)

@norecursion
def prod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'prod'
    module_type_store = module_type_store.open_function_context('prod', 17, 0, False)
    
    # Passed parameters checking function
    prod.stypy_localization = localization
    prod.stypy_type_of_self = None
    prod.stypy_type_store = module_type_store
    prod.stypy_function_name = 'prod'
    prod.stypy_param_names_list = ['x']
    prod.stypy_varargs_param_name = None
    prod.stypy_kwargs_param_name = None
    prod.stypy_call_defaults = defaults
    prod.stypy_call_varargs = varargs
    prod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'prod', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'prod', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'prod(...)' code ##################

    str_73725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', 'Product of a list of numbers; ~40x faster vs np.prod for Python tuples')
    
    
    
    # Call to len(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'x' (line 19)
    x_73727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'x', False)
    # Processing the call keyword arguments (line 19)
    kwargs_73728 = {}
    # Getting the type of 'len' (line 19)
    len_73726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'len', False)
    # Calling len(args, kwargs) (line 19)
    len_call_result_73729 = invoke(stypy.reporting.localization.Localization(__file__, 19, 7), len_73726, *[x_73727], **kwargs_73728)
    
    int_73730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
    # Applying the binary operator '==' (line 19)
    result_eq_73731 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 7), '==', len_call_result_73729, int_73730)
    
    # Testing the type of an if condition (line 19)
    if_condition_73732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 4), result_eq_73731)
    # Assigning a type to the variable 'if_condition_73732' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'if_condition_73732', if_condition_73732)
    # SSA begins for if statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_73733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', int_73733)
    # SSA join for if statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reduce(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'operator' (line 21)
    operator_73736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 28), 'operator', False)
    # Obtaining the member 'mul' of a type (line 21)
    mul_73737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 28), operator_73736, 'mul')
    # Getting the type of 'x' (line 21)
    x_73738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 42), 'x', False)
    # Processing the call keyword arguments (line 21)
    kwargs_73739 = {}
    # Getting the type of 'functools' (line 21)
    functools_73734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'functools', False)
    # Obtaining the member 'reduce' of a type (line 21)
    reduce_73735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), functools_73734, 'reduce')
    # Calling reduce(args, kwargs) (line 21)
    reduce_call_result_73740 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), reduce_73735, *[mul_73737, x_73738], **kwargs_73739)
    
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', reduce_call_result_73740)
    
    # ################# End of 'prod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'prod' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_73741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_73741)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'prod'
    return stypy_return_type_73741

# Assigning a type to the variable 'prod' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'prod', prod)

@norecursion
def _get_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_dtype'
    module_type_store = module_type_store.open_function_context('_get_dtype', 24, 0, False)
    
    # Passed parameters checking function
    _get_dtype.stypy_localization = localization
    _get_dtype.stypy_type_of_self = None
    _get_dtype.stypy_type_store = module_type_store
    _get_dtype.stypy_function_name = '_get_dtype'
    _get_dtype.stypy_param_names_list = ['dtype']
    _get_dtype.stypy_varargs_param_name = None
    _get_dtype.stypy_kwargs_param_name = None
    _get_dtype.stypy_call_defaults = defaults
    _get_dtype.stypy_call_varargs = varargs
    _get_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_dtype', ['dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_dtype', localization, ['dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_dtype(...)' code ##################

    str_73742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', 'Return np.complex128 for complex dtypes, np.float64 otherwise.')
    
    
    # Call to issubdtype(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'dtype' (line 26)
    dtype_73745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'dtype', False)
    # Getting the type of 'np' (line 26)
    np_73746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 26)
    complexfloating_73747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 28), np_73746, 'complexfloating')
    # Processing the call keyword arguments (line 26)
    kwargs_73748 = {}
    # Getting the type of 'np' (line 26)
    np_73743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 7), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 26)
    issubdtype_73744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 7), np_73743, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 26)
    issubdtype_call_result_73749 = invoke(stypy.reporting.localization.Localization(__file__, 26, 7), issubdtype_73744, *[dtype_73745, complexfloating_73747], **kwargs_73748)
    
    # Testing the type of an if condition (line 26)
    if_condition_73750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 4), issubdtype_call_result_73749)
    # Assigning a type to the variable 'if_condition_73750' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'if_condition_73750', if_condition_73750)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 27)
    np_73751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'np')
    # Obtaining the member 'complex_' of a type (line 27)
    complex__73752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), np_73751, 'complex_')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', complex__73752)
    # SSA branch for the else part of an if statement (line 26)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'np' (line 29)
    np_73753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'np')
    # Obtaining the member 'float_' of a type (line 29)
    float__73754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), np_73753, 'float_')
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', float__73754)
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_get_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_73755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_73755)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_dtype'
    return stypy_return_type_73755

# Assigning a type to the variable '_get_dtype' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '_get_dtype', _get_dtype)

@norecursion
def _as_float_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 32)
    False_73756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 36), 'False')
    defaults = [False_73756]
    # Create a new context for function '_as_float_array'
    module_type_store = module_type_store.open_function_context('_as_float_array', 32, 0, False)
    
    # Passed parameters checking function
    _as_float_array.stypy_localization = localization
    _as_float_array.stypy_type_of_self = None
    _as_float_array.stypy_type_store = module_type_store
    _as_float_array.stypy_function_name = '_as_float_array'
    _as_float_array.stypy_param_names_list = ['x', 'check_finite']
    _as_float_array.stypy_varargs_param_name = None
    _as_float_array.stypy_kwargs_param_name = None
    _as_float_array.stypy_call_defaults = defaults
    _as_float_array.stypy_call_varargs = varargs
    _as_float_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_as_float_array', ['x', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_as_float_array', localization, ['x', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_as_float_array(...)' code ##################

    str_73757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', 'Convert the input into a C contiguous float array.\n\n    NB: Upcasts half- and single-precision floats to double precision.\n    ')
    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to ascontiguousarray(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'x' (line 37)
    x_73760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'x', False)
    # Processing the call keyword arguments (line 37)
    kwargs_73761 = {}
    # Getting the type of 'np' (line 37)
    np_73758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 37)
    ascontiguousarray_73759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), np_73758, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 37)
    ascontiguousarray_call_result_73762 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), ascontiguousarray_73759, *[x_73760], **kwargs_73761)
    
    # Assigning a type to the variable 'x' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'x', ascontiguousarray_call_result_73762)
    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to _get_dtype(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'x' (line 38)
    x_73764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'x', False)
    # Obtaining the member 'dtype' of a type (line 38)
    dtype_73765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 22), x_73764, 'dtype')
    # Processing the call keyword arguments (line 38)
    kwargs_73766 = {}
    # Getting the type of '_get_dtype' (line 38)
    _get_dtype_73763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), '_get_dtype', False)
    # Calling _get_dtype(args, kwargs) (line 38)
    _get_dtype_call_result_73767 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), _get_dtype_73763, *[dtype_73765], **kwargs_73766)
    
    # Assigning a type to the variable 'dtyp' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'dtyp', _get_dtype_call_result_73767)
    
    # Assigning a Call to a Name (line 39):
    
    # Assigning a Call to a Name (line 39):
    
    # Call to astype(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'dtyp' (line 39)
    dtyp_73770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'dtyp', False)
    # Processing the call keyword arguments (line 39)
    # Getting the type of 'False' (line 39)
    False_73771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'False', False)
    keyword_73772 = False_73771
    kwargs_73773 = {'copy': keyword_73772}
    # Getting the type of 'x' (line 39)
    x_73768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'x', False)
    # Obtaining the member 'astype' of a type (line 39)
    astype_73769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), x_73768, 'astype')
    # Calling astype(args, kwargs) (line 39)
    astype_call_result_73774 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), astype_73769, *[dtyp_73770], **kwargs_73773)
    
    # Assigning a type to the variable 'x' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'x', astype_call_result_73774)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'check_finite' (line 40)
    check_finite_73775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 7), 'check_finite')
    
    
    # Call to all(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_73782 = {}
    
    # Call to isfinite(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'x' (line 40)
    x_73778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 40), 'x', False)
    # Processing the call keyword arguments (line 40)
    kwargs_73779 = {}
    # Getting the type of 'np' (line 40)
    np_73776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 40)
    isfinite_73777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 28), np_73776, 'isfinite')
    # Calling isfinite(args, kwargs) (line 40)
    isfinite_call_result_73780 = invoke(stypy.reporting.localization.Localization(__file__, 40, 28), isfinite_73777, *[x_73778], **kwargs_73779)
    
    # Obtaining the member 'all' of a type (line 40)
    all_73781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 28), isfinite_call_result_73780, 'all')
    # Calling all(args, kwargs) (line 40)
    all_call_result_73783 = invoke(stypy.reporting.localization.Localization(__file__, 40, 28), all_73781, *[], **kwargs_73782)
    
    # Applying the 'not' unary operator (line 40)
    result_not__73784 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 24), 'not', all_call_result_73783)
    
    # Applying the binary operator 'and' (line 40)
    result_and_keyword_73785 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 7), 'and', check_finite_73775, result_not__73784)
    
    # Testing the type of an if condition (line 40)
    if_condition_73786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 4), result_and_keyword_73785)
    # Assigning a type to the variable 'if_condition_73786' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'if_condition_73786', if_condition_73786)
    # SSA begins for if statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 41)
    # Processing the call arguments (line 41)
    str_73788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'str', 'Array must not contain infs or nans.')
    # Processing the call keyword arguments (line 41)
    kwargs_73789 = {}
    # Getting the type of 'ValueError' (line 41)
    ValueError_73787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 41)
    ValueError_call_result_73790 = invoke(stypy.reporting.localization.Localization(__file__, 41, 14), ValueError_73787, *[str_73788], **kwargs_73789)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 41, 8), ValueError_call_result_73790, 'raise parameter', BaseException)
    # SSA join for if statement (line 40)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 42)
    x_73791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', x_73791)
    
    # ################# End of '_as_float_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_as_float_array' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_73792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_73792)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_as_float_array'
    return stypy_return_type_73792

# Assigning a type to the variable '_as_float_array' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '_as_float_array', _as_float_array)
# Declaration of the 'BSpline' class

class BSpline(object, ):
    str_73793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, (-1)), 'str', "Univariate spline in the B-spline basis.\n\n    .. math::\n\n        S(x) = \\sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)\n\n    where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`\n    and knots `t`.\n\n    Parameters\n    ----------\n    t : ndarray, shape (n+k+1,)\n        knots\n    c : ndarray, shape (>=n, ...)\n        spline coefficients\n    k : int\n        B-spline order\n    extrapolate : bool or 'periodic', optional\n        whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,\n        or to return nans.\n        If True, extrapolates the first and last polynomial pieces of b-spline\n        functions active on the base interval.\n        If 'periodic', periodic extrapolation is used.\n        Default is True.\n    axis : int, optional\n        Interpolation axis. Default is zero.\n\n    Attributes\n    ----------\n    t : ndarray\n        knot vector\n    c : ndarray\n        spline coefficients\n    k : int\n        spline degree\n    extrapolate : bool\n        If True, extrapolates the first and last polynomial pieces of b-spline\n        functions active on the base interval.\n    axis : int\n        Interpolation axis.\n    tck : tuple\n        A read-only equivalent of ``(self.t, self.c, self.k)``\n\n    Methods\n    -------\n    __call__\n    basis_element\n    derivative\n    antiderivative\n    integrate\n    construct_fast\n\n    Notes\n    -----\n    B-spline basis elements are defined via\n\n    .. math::\n\n        B_{i, 0}(x) = 1, \\textrm{if $t_i \\le x < t_{i+1}$, otherwise $0$,}\n\n        B_{i, k}(x) = \\frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)\n                 + \\frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)\n\n    **Implementation details**\n\n    - At least ``k+1`` coefficients are required for a spline of degree `k`,\n      so that ``n >= k+1``. Additional coefficients, ``c[j]`` with\n      ``j > n``, are ignored.\n\n    - B-spline basis elements of degree `k` form a partition of unity on the\n      *base interval*, ``t[k] <= x <= t[n]``.\n\n\n    Examples\n    --------\n\n    Translating the recursive definition of B-splines into Python code, we have:\n\n    >>> def B(x, k, i, t):\n    ...    if k == 0:\n    ...       return 1.0 if t[i] <= x < t[i+1] else 0.0\n    ...    if t[i+k] == t[i]:\n    ...       c1 = 0.0\n    ...    else:\n    ...       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)\n    ...    if t[i+k+1] == t[i+1]:\n    ...       c2 = 0.0\n    ...    else:\n    ...       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)\n    ...    return c1 + c2\n\n    >>> def bspline(x, t, c, k):\n    ...    n = len(t) - k - 1\n    ...    assert (n >= k+1) and (len(c) >= n)\n    ...    return sum(c[i] * B(x, k, i, t) for i in range(n))\n\n    Note that this is an inefficient (if straightforward) way to\n    evaluate B-splines --- this spline class does it in an equivalent,\n    but much more efficient way.\n\n    Here we construct a quadratic spline function on the base interval\n    ``2 <= x <= 4`` and compare with the naive way of evaluating the spline:\n\n    >>> from scipy.interpolate import BSpline\n    >>> k = 2\n    >>> t = [0, 1, 2, 3, 4, 5, 6]\n    >>> c = [-1, 2, 0, -1]\n    >>> spl = BSpline(t, c, k)\n    >>> spl(2.5)\n    array(1.375)\n    >>> bspline(2.5, t, c, k)\n    1.375\n\n    Note that outside of the base interval results differ. This is because\n    `BSpline` extrapolates the first and last polynomial pieces of b-spline\n    functions active on the base interval.\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> xx = np.linspace(1.5, 4.5, 50)\n    >>> ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')\n    >>> ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')\n    >>> ax.grid(True)\n    >>> ax.legend(loc='best')\n    >>> plt.show()\n\n\n    References\n    ----------\n    .. [1] Tom Lyche and Knut Morken, Spline methods,\n        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/\n    .. [2] Carl de Boor, A practical guide to splines, Springer, 2001.\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 180)
        True_73794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 44), 'True')
        int_73795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 55), 'int')
        defaults = [True_73794, int_73795]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline.__init__', ['t', 'c', 'k', 'extrapolate', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['t', 'c', 'k', 'extrapolate', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_73802 = {}
        
        # Call to super(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'BSpline' (line 181)
        BSpline_73797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'BSpline', False)
        # Getting the type of 'self' (line 181)
        self_73798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'self', False)
        # Processing the call keyword arguments (line 181)
        kwargs_73799 = {}
        # Getting the type of 'super' (line 181)
        super_73796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'super', False)
        # Calling super(args, kwargs) (line 181)
        super_call_result_73800 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), super_73796, *[BSpline_73797, self_73798], **kwargs_73799)
        
        # Obtaining the member '__init__' of a type (line 181)
        init___73801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), super_call_result_73800, '__init__')
        # Calling __init__(args, kwargs) (line 181)
        init___call_result_73803 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), init___73801, *[], **kwargs_73802)
        
        
        # Assigning a Call to a Attribute (line 183):
        
        # Assigning a Call to a Attribute (line 183):
        
        # Call to int(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'k' (line 183)
        k_73805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'k', False)
        # Processing the call keyword arguments (line 183)
        kwargs_73806 = {}
        # Getting the type of 'int' (line 183)
        int_73804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'int', False)
        # Calling int(args, kwargs) (line 183)
        int_call_result_73807 = invoke(stypy.reporting.localization.Localization(__file__, 183, 17), int_73804, *[k_73805], **kwargs_73806)
        
        # Getting the type of 'self' (line 183)
        self_73808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self')
        # Setting the type of the member 'k' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_73808, 'k', int_call_result_73807)
        
        # Assigning a Call to a Attribute (line 184):
        
        # Assigning a Call to a Attribute (line 184):
        
        # Call to asarray(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'c' (line 184)
        c_73811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'c', False)
        # Processing the call keyword arguments (line 184)
        kwargs_73812 = {}
        # Getting the type of 'np' (line 184)
        np_73809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 17), 'np', False)
        # Obtaining the member 'asarray' of a type (line 184)
        asarray_73810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 17), np_73809, 'asarray')
        # Calling asarray(args, kwargs) (line 184)
        asarray_call_result_73813 = invoke(stypy.reporting.localization.Localization(__file__, 184, 17), asarray_73810, *[c_73811], **kwargs_73812)
        
        # Getting the type of 'self' (line 184)
        self_73814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self')
        # Setting the type of the member 'c' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_73814, 'c', asarray_call_result_73813)
        
        # Assigning a Call to a Attribute (line 185):
        
        # Assigning a Call to a Attribute (line 185):
        
        # Call to ascontiguousarray(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 't' (line 185)
        t_73817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 38), 't', False)
        # Processing the call keyword arguments (line 185)
        # Getting the type of 'np' (line 185)
        np_73818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 47), 'np', False)
        # Obtaining the member 'float64' of a type (line 185)
        float64_73819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 47), np_73818, 'float64')
        keyword_73820 = float64_73819
        kwargs_73821 = {'dtype': keyword_73820}
        # Getting the type of 'np' (line 185)
        np_73815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 185)
        ascontiguousarray_73816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 17), np_73815, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 185)
        ascontiguousarray_call_result_73822 = invoke(stypy.reporting.localization.Localization(__file__, 185, 17), ascontiguousarray_73816, *[t_73817], **kwargs_73821)
        
        # Getting the type of 'self' (line 185)
        self_73823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'self')
        # Setting the type of the member 't' of a type (line 185)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), self_73823, 't', ascontiguousarray_call_result_73822)
        
        
        # Getting the type of 'extrapolate' (line 187)
        extrapolate_73824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'extrapolate')
        str_73825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'str', 'periodic')
        # Applying the binary operator '==' (line 187)
        result_eq_73826 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 11), '==', extrapolate_73824, str_73825)
        
        # Testing the type of an if condition (line 187)
        if_condition_73827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), result_eq_73826)
        # Assigning a type to the variable 'if_condition_73827' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_73827', if_condition_73827)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 188):
        
        # Assigning a Name to a Attribute (line 188):
        # Getting the type of 'extrapolate' (line 188)
        extrapolate_73828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'extrapolate')
        # Getting the type of 'self' (line 188)
        self_73829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'self')
        # Setting the type of the member 'extrapolate' of a type (line 188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), self_73829, 'extrapolate', extrapolate_73828)
        # SSA branch for the else part of an if statement (line 187)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 190):
        
        # Assigning a Call to a Attribute (line 190):
        
        # Call to bool(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'extrapolate' (line 190)
        extrapolate_73831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 36), 'extrapolate', False)
        # Processing the call keyword arguments (line 190)
        kwargs_73832 = {}
        # Getting the type of 'bool' (line 190)
        bool_73830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 31), 'bool', False)
        # Calling bool(args, kwargs) (line 190)
        bool_call_result_73833 = invoke(stypy.reporting.localization.Localization(__file__, 190, 31), bool_73830, *[extrapolate_73831], **kwargs_73832)
        
        # Getting the type of 'self' (line 190)
        self_73834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'self')
        # Setting the type of the member 'extrapolate' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 12), self_73834, 'extrapolate', bool_call_result_73833)
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 192):
        
        # Assigning a BinOp to a Name (line 192):
        
        # Obtaining the type of the subscript
        int_73835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 25), 'int')
        # Getting the type of 'self' (line 192)
        self_73836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'self')
        # Obtaining the member 't' of a type (line 192)
        t_73837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), self_73836, 't')
        # Obtaining the member 'shape' of a type (line 192)
        shape_73838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), t_73837, 'shape')
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___73839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), shape_73838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_73840 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), getitem___73839, int_73835)
        
        # Getting the type of 'self' (line 192)
        self_73841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'self')
        # Obtaining the member 'k' of a type (line 192)
        k_73842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 30), self_73841, 'k')
        # Applying the binary operator '-' (line 192)
        result_sub_73843 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 12), '-', subscript_call_result_73840, k_73842)
        
        int_73844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 39), 'int')
        # Applying the binary operator '-' (line 192)
        result_sub_73845 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 37), '-', result_sub_73843, int_73844)
        
        # Assigning a type to the variable 'n' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'n', result_sub_73845)
        
        
        
        int_73846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 16), 'int')
        # Getting the type of 'axis' (line 194)
        axis_73847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 21), 'axis')
        # Applying the binary operator '<=' (line 194)
        result_le_73848 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 16), '<=', int_73846, axis_73847)
        # Getting the type of 'self' (line 194)
        self_73849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'self')
        # Obtaining the member 'c' of a type (line 194)
        c_73850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 28), self_73849, 'c')
        # Obtaining the member 'ndim' of a type (line 194)
        ndim_73851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 28), c_73850, 'ndim')
        # Applying the binary operator '<' (line 194)
        result_lt_73852 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 16), '<', axis_73847, ndim_73851)
        # Applying the binary operator '&' (line 194)
        result_and__73853 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 16), '&', result_le_73848, result_lt_73852)
        
        # Applying the 'not' unary operator (line 194)
        result_not__73854 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), 'not', result_and__73853)
        
        # Testing the type of an if condition (line 194)
        if_condition_73855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), result_not__73854)
        # Assigning a type to the variable 'if_condition_73855' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_73855', if_condition_73855)
        # SSA begins for if statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 195)
        # Processing the call arguments (line 195)
        str_73857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 29), 'str', '%s must be between 0 and %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 195)
        tuple_73858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 195)
        # Adding element type (line 195)
        # Getting the type of 'axis' (line 195)
        axis_73859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 62), 'axis', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 62), tuple_73858, axis_73859)
        # Adding element type (line 195)
        # Getting the type of 'c' (line 195)
        c_73860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 68), 'c', False)
        # Obtaining the member 'ndim' of a type (line 195)
        ndim_73861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 68), c_73860, 'ndim')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 62), tuple_73858, ndim_73861)
        
        # Applying the binary operator '%' (line 195)
        result_mod_73862 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 29), '%', str_73857, tuple_73858)
        
        # Processing the call keyword arguments (line 195)
        kwargs_73863 = {}
        # Getting the type of 'ValueError' (line 195)
        ValueError_73856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 195)
        ValueError_call_result_73864 = invoke(stypy.reporting.localization.Localization(__file__, 195, 18), ValueError_73856, *[result_mod_73862], **kwargs_73863)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 195, 12), ValueError_call_result_73864, 'raise parameter', BaseException)
        # SSA join for if statement (line 194)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 197):
        
        # Assigning a Name to a Attribute (line 197):
        # Getting the type of 'axis' (line 197)
        axis_73865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'axis')
        # Getting the type of 'self' (line 197)
        self_73866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_73866, 'axis', axis_73865)
        
        
        # Getting the type of 'axis' (line 198)
        axis_73867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'axis')
        int_73868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 19), 'int')
        # Applying the binary operator '!=' (line 198)
        result_ne_73869 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '!=', axis_73867, int_73868)
        
        # Testing the type of an if condition (line 198)
        if_condition_73870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_ne_73869)
        # Assigning a type to the variable 'if_condition_73870' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_73870', if_condition_73870)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 204):
        
        # Assigning a Call to a Attribute (line 204):
        
        # Call to rollaxis(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'self' (line 204)
        self_73873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'self', False)
        # Obtaining the member 'c' of a type (line 204)
        c_73874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 33), self_73873, 'c')
        # Getting the type of 'axis' (line 204)
        axis_73875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 41), 'axis', False)
        # Processing the call keyword arguments (line 204)
        kwargs_73876 = {}
        # Getting the type of 'np' (line 204)
        np_73871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'np', False)
        # Obtaining the member 'rollaxis' of a type (line 204)
        rollaxis_73872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), np_73871, 'rollaxis')
        # Calling rollaxis(args, kwargs) (line 204)
        rollaxis_call_result_73877 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), rollaxis_73872, *[c_73874, axis_73875], **kwargs_73876)
        
        # Getting the type of 'self' (line 204)
        self_73878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'self')
        # Setting the type of the member 'c' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), self_73878, 'c', rollaxis_call_result_73877)
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'k' (line 206)
        k_73879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'k')
        int_73880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 15), 'int')
        # Applying the binary operator '<' (line 206)
        result_lt_73881 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 11), '<', k_73879, int_73880)
        
        # Testing the type of an if condition (line 206)
        if_condition_73882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), result_lt_73881)
        # Assigning a type to the variable 'if_condition_73882' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'if_condition_73882', if_condition_73882)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 207)
        # Processing the call arguments (line 207)
        str_73884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 29), 'str', 'Spline order cannot be negative.')
        # Processing the call keyword arguments (line 207)
        kwargs_73885 = {}
        # Getting the type of 'ValueError' (line 207)
        ValueError_73883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 207)
        ValueError_call_result_73886 = invoke(stypy.reporting.localization.Localization(__file__, 207, 18), ValueError_73883, *[str_73884], **kwargs_73885)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 207, 12), ValueError_call_result_73886, 'raise parameter', BaseException)
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to int(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'k' (line 208)
        k_73888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'k', False)
        # Processing the call keyword arguments (line 208)
        kwargs_73889 = {}
        # Getting the type of 'int' (line 208)
        int_73887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'int', False)
        # Calling int(args, kwargs) (line 208)
        int_call_result_73890 = invoke(stypy.reporting.localization.Localization(__file__, 208, 11), int_73887, *[k_73888], **kwargs_73889)
        
        # Getting the type of 'k' (line 208)
        k_73891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 21), 'k')
        # Applying the binary operator '!=' (line 208)
        result_ne_73892 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 11), '!=', int_call_result_73890, k_73891)
        
        # Testing the type of an if condition (line 208)
        if_condition_73893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), result_ne_73892)
        # Assigning a type to the variable 'if_condition_73893' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_73893', if_condition_73893)
        # SSA begins for if statement (line 208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 209)
        # Processing the call arguments (line 209)
        str_73895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 29), 'str', 'Spline order must be integer.')
        # Processing the call keyword arguments (line 209)
        kwargs_73896 = {}
        # Getting the type of 'ValueError' (line 209)
        ValueError_73894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 209)
        ValueError_call_result_73897 = invoke(stypy.reporting.localization.Localization(__file__, 209, 18), ValueError_73894, *[str_73895], **kwargs_73896)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 209, 12), ValueError_call_result_73897, 'raise parameter', BaseException)
        # SSA join for if statement (line 208)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 210)
        self_73898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'self')
        # Obtaining the member 't' of a type (line 210)
        t_73899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 11), self_73898, 't')
        # Obtaining the member 'ndim' of a type (line 210)
        ndim_73900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 11), t_73899, 'ndim')
        int_73901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 26), 'int')
        # Applying the binary operator '!=' (line 210)
        result_ne_73902 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 11), '!=', ndim_73900, int_73901)
        
        # Testing the type of an if condition (line 210)
        if_condition_73903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), result_ne_73902)
        # Assigning a type to the variable 'if_condition_73903' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_73903', if_condition_73903)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 211)
        # Processing the call arguments (line 211)
        str_73905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 29), 'str', 'Knot vector must be one-dimensional.')
        # Processing the call keyword arguments (line 211)
        kwargs_73906 = {}
        # Getting the type of 'ValueError' (line 211)
        ValueError_73904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 211)
        ValueError_call_result_73907 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), ValueError_73904, *[str_73905], **kwargs_73906)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 211, 12), ValueError_call_result_73907, 'raise parameter', BaseException)
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'n' (line 212)
        n_73908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'n')
        # Getting the type of 'self' (line 212)
        self_73909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'self')
        # Obtaining the member 'k' of a type (line 212)
        k_73910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), self_73909, 'k')
        int_73911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 24), 'int')
        # Applying the binary operator '+' (line 212)
        result_add_73912 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 15), '+', k_73910, int_73911)
        
        # Applying the binary operator '<' (line 212)
        result_lt_73913 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 11), '<', n_73908, result_add_73912)
        
        # Testing the type of an if condition (line 212)
        if_condition_73914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 8), result_lt_73913)
        # Assigning a type to the variable 'if_condition_73914' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'if_condition_73914', if_condition_73914)
        # SSA begins for if statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 213)
        # Processing the call arguments (line 213)
        str_73916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 29), 'str', 'Need at least %d knots for degree %d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 214)
        tuple_73917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 214)
        # Adding element type (line 214)
        int_73918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 21), 'int')
        # Getting the type of 'k' (line 214)
        k_73919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'k', False)
        # Applying the binary operator '*' (line 214)
        result_mul_73920 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 21), '*', int_73918, k_73919)
        
        int_73921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 27), 'int')
        # Applying the binary operator '+' (line 214)
        result_add_73922 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 21), '+', result_mul_73920, int_73921)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 21), tuple_73917, result_add_73922)
        # Adding element type (line 214)
        # Getting the type of 'k' (line 214)
        k_73923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 21), tuple_73917, k_73923)
        
        # Applying the binary operator '%' (line 213)
        result_mod_73924 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 29), '%', str_73916, tuple_73917)
        
        # Processing the call keyword arguments (line 213)
        kwargs_73925 = {}
        # Getting the type of 'ValueError' (line 213)
        ValueError_73915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 213)
        ValueError_call_result_73926 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), ValueError_73915, *[result_mod_73924], **kwargs_73925)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 213, 12), ValueError_call_result_73926, 'raise parameter', BaseException)
        # SSA join for if statement (line 212)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to any(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_73936 = {}
        
        
        # Call to diff(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'self' (line 215)
        self_73929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'self', False)
        # Obtaining the member 't' of a type (line 215)
        t_73930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 20), self_73929, 't')
        # Processing the call keyword arguments (line 215)
        kwargs_73931 = {}
        # Getting the type of 'np' (line 215)
        np_73927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'np', False)
        # Obtaining the member 'diff' of a type (line 215)
        diff_73928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), np_73927, 'diff')
        # Calling diff(args, kwargs) (line 215)
        diff_call_result_73932 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), diff_73928, *[t_73930], **kwargs_73931)
        
        int_73933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 30), 'int')
        # Applying the binary operator '<' (line 215)
        result_lt_73934 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 12), '<', diff_call_result_73932, int_73933)
        
        # Obtaining the member 'any' of a type (line 215)
        any_73935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), result_lt_73934, 'any')
        # Calling any(args, kwargs) (line 215)
        any_call_result_73937 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), any_73935, *[], **kwargs_73936)
        
        # Testing the type of an if condition (line 215)
        if_condition_73938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), any_call_result_73937)
        # Assigning a type to the variable 'if_condition_73938' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_73938', if_condition_73938)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 216)
        # Processing the call arguments (line 216)
        str_73940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 29), 'str', 'Knots must be in a non-decreasing order.')
        # Processing the call keyword arguments (line 216)
        kwargs_73941 = {}
        # Getting the type of 'ValueError' (line 216)
        ValueError_73939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 216)
        ValueError_call_result_73942 = invoke(stypy.reporting.localization.Localization(__file__, 216, 18), ValueError_73939, *[str_73940], **kwargs_73941)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 216, 12), ValueError_call_result_73942, 'raise parameter', BaseException)
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Call to unique(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 217)
        k_73946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'k', False)
        # Getting the type of 'n' (line 217)
        n_73947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'n', False)
        int_73948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 36), 'int')
        # Applying the binary operator '+' (line 217)
        result_add_73949 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 34), '+', n_73947, int_73948)
        
        slice_73950 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 217, 25), k_73946, result_add_73949, None)
        # Getting the type of 'self' (line 217)
        self_73951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'self', False)
        # Obtaining the member 't' of a type (line 217)
        t_73952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 25), self_73951, 't')
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___73953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 25), t_73952, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_73954 = invoke(stypy.reporting.localization.Localization(__file__, 217, 25), getitem___73953, slice_73950)
        
        # Processing the call keyword arguments (line 217)
        kwargs_73955 = {}
        # Getting the type of 'np' (line 217)
        np_73944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'np', False)
        # Obtaining the member 'unique' of a type (line 217)
        unique_73945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), np_73944, 'unique')
        # Calling unique(args, kwargs) (line 217)
        unique_call_result_73956 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), unique_73945, *[subscript_call_result_73954], **kwargs_73955)
        
        # Processing the call keyword arguments (line 217)
        kwargs_73957 = {}
        # Getting the type of 'len' (line 217)
        len_73943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'len', False)
        # Calling len(args, kwargs) (line 217)
        len_call_result_73958 = invoke(stypy.reporting.localization.Localization(__file__, 217, 11), len_73943, *[unique_call_result_73956], **kwargs_73957)
        
        int_73959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 43), 'int')
        # Applying the binary operator '<' (line 217)
        result_lt_73960 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 11), '<', len_call_result_73958, int_73959)
        
        # Testing the type of an if condition (line 217)
        if_condition_73961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 8), result_lt_73960)
        # Assigning a type to the variable 'if_condition_73961' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'if_condition_73961', if_condition_73961)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 218)
        # Processing the call arguments (line 218)
        str_73963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 29), 'str', 'Need at least two internal knots.')
        # Processing the call keyword arguments (line 218)
        kwargs_73964 = {}
        # Getting the type of 'ValueError' (line 218)
        ValueError_73962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 218)
        ValueError_call_result_73965 = invoke(stypy.reporting.localization.Localization(__file__, 218, 18), ValueError_73962, *[str_73963], **kwargs_73964)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 218, 12), ValueError_call_result_73965, 'raise parameter', BaseException)
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to all(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_73973 = {}
        
        # Call to isfinite(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'self' (line 219)
        self_73968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 27), 'self', False)
        # Obtaining the member 't' of a type (line 219)
        t_73969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 27), self_73968, 't')
        # Processing the call keyword arguments (line 219)
        kwargs_73970 = {}
        # Getting the type of 'np' (line 219)
        np_73966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 219)
        isfinite_73967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), np_73966, 'isfinite')
        # Calling isfinite(args, kwargs) (line 219)
        isfinite_call_result_73971 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), isfinite_73967, *[t_73969], **kwargs_73970)
        
        # Obtaining the member 'all' of a type (line 219)
        all_73972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), isfinite_call_result_73971, 'all')
        # Calling all(args, kwargs) (line 219)
        all_call_result_73974 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), all_73972, *[], **kwargs_73973)
        
        # Applying the 'not' unary operator (line 219)
        result_not__73975 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), 'not', all_call_result_73974)
        
        # Testing the type of an if condition (line 219)
        if_condition_73976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 8), result_not__73975)
        # Assigning a type to the variable 'if_condition_73976' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'if_condition_73976', if_condition_73976)
        # SSA begins for if statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 220)
        # Processing the call arguments (line 220)
        str_73978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 29), 'str', 'Knots should not have nans or infs.')
        # Processing the call keyword arguments (line 220)
        kwargs_73979 = {}
        # Getting the type of 'ValueError' (line 220)
        ValueError_73977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 220)
        ValueError_call_result_73980 = invoke(stypy.reporting.localization.Localization(__file__, 220, 18), ValueError_73977, *[str_73978], **kwargs_73979)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 220, 12), ValueError_call_result_73980, 'raise parameter', BaseException)
        # SSA join for if statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 221)
        self_73981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'self')
        # Obtaining the member 'c' of a type (line 221)
        c_73982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 11), self_73981, 'c')
        # Obtaining the member 'ndim' of a type (line 221)
        ndim_73983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 11), c_73982, 'ndim')
        int_73984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 25), 'int')
        # Applying the binary operator '<' (line 221)
        result_lt_73985 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 11), '<', ndim_73983, int_73984)
        
        # Testing the type of an if condition (line 221)
        if_condition_73986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), result_lt_73985)
        # Assigning a type to the variable 'if_condition_73986' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_73986', if_condition_73986)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 222)
        # Processing the call arguments (line 222)
        str_73988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 29), 'str', 'Coefficients must be at least 1-dimensional.')
        # Processing the call keyword arguments (line 222)
        kwargs_73989 = {}
        # Getting the type of 'ValueError' (line 222)
        ValueError_73987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 222)
        ValueError_call_result_73990 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), ValueError_73987, *[str_73988], **kwargs_73989)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 222, 12), ValueError_call_result_73990, 'raise parameter', BaseException)
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_73991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 24), 'int')
        # Getting the type of 'self' (line 223)
        self_73992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'self')
        # Obtaining the member 'c' of a type (line 223)
        c_73993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 11), self_73992, 'c')
        # Obtaining the member 'shape' of a type (line 223)
        shape_73994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 11), c_73993, 'shape')
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___73995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 11), shape_73994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_73996 = invoke(stypy.reporting.localization.Localization(__file__, 223, 11), getitem___73995, int_73991)
        
        # Getting the type of 'n' (line 223)
        n_73997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'n')
        # Applying the binary operator '<' (line 223)
        result_lt_73998 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 11), '<', subscript_call_result_73996, n_73997)
        
        # Testing the type of an if condition (line 223)
        if_condition_73999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_lt_73998)
        # Assigning a type to the variable 'if_condition_73999' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_73999', if_condition_73999)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 224)
        # Processing the call arguments (line 224)
        str_74001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 29), 'str', 'Knots, coefficients and degree are inconsistent.')
        # Processing the call keyword arguments (line 224)
        kwargs_74002 = {}
        # Getting the type of 'ValueError' (line 224)
        ValueError_74000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 224)
        ValueError_call_result_74003 = invoke(stypy.reporting.localization.Localization(__file__, 224, 18), ValueError_74000, *[str_74001], **kwargs_74002)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 224, 12), ValueError_call_result_74003, 'raise parameter', BaseException)
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to _get_dtype(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'self' (line 226)
        self_74005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'self', False)
        # Obtaining the member 'c' of a type (line 226)
        c_74006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 24), self_74005, 'c')
        # Obtaining the member 'dtype' of a type (line 226)
        dtype_74007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 24), c_74006, 'dtype')
        # Processing the call keyword arguments (line 226)
        kwargs_74008 = {}
        # Getting the type of '_get_dtype' (line 226)
        _get_dtype_74004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), '_get_dtype', False)
        # Calling _get_dtype(args, kwargs) (line 226)
        _get_dtype_call_result_74009 = invoke(stypy.reporting.localization.Localization(__file__, 226, 13), _get_dtype_74004, *[dtype_74007], **kwargs_74008)
        
        # Assigning a type to the variable 'dt' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'dt', _get_dtype_call_result_74009)
        
        # Assigning a Call to a Attribute (line 227):
        
        # Assigning a Call to a Attribute (line 227):
        
        # Call to ascontiguousarray(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'self' (line 227)
        self_74012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 38), 'self', False)
        # Obtaining the member 'c' of a type (line 227)
        c_74013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 38), self_74012, 'c')
        # Processing the call keyword arguments (line 227)
        # Getting the type of 'dt' (line 227)
        dt_74014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'dt', False)
        keyword_74015 = dt_74014
        kwargs_74016 = {'dtype': keyword_74015}
        # Getting the type of 'np' (line 227)
        np_74010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 17), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 227)
        ascontiguousarray_74011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 17), np_74010, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 227)
        ascontiguousarray_call_result_74017 = invoke(stypy.reporting.localization.Localization(__file__, 227, 17), ascontiguousarray_74011, *[c_74013], **kwargs_74016)
        
        # Getting the type of 'self' (line 227)
        self_74018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self')
        # Setting the type of the member 'c' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_74018, 'c', ascontiguousarray_call_result_74017)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def construct_fast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 230)
        True_74019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 49), 'True')
        int_74020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 60), 'int')
        defaults = [True_74019, int_74020]
        # Create a new context for function 'construct_fast'
        module_type_store = module_type_store.open_function_context('construct_fast', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline.construct_fast.__dict__.__setitem__('stypy_localization', localization)
        BSpline.construct_fast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline.construct_fast.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline.construct_fast.__dict__.__setitem__('stypy_function_name', 'BSpline.construct_fast')
        BSpline.construct_fast.__dict__.__setitem__('stypy_param_names_list', ['t', 'c', 'k', 'extrapolate', 'axis'])
        BSpline.construct_fast.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline.construct_fast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline.construct_fast.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline.construct_fast.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline.construct_fast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline.construct_fast.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline.construct_fast', ['t', 'c', 'k', 'extrapolate', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'construct_fast', localization, ['t', 'c', 'k', 'extrapolate', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'construct_fast(...)' code ##################

        str_74021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, (-1)), 'str', 'Construct a spline without making checks.\n\n        Accepts same parameters as the regular constructor. Input arrays\n        `t` and `c` must of correct shape and dtype.\n        ')
        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to __new__(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'cls' (line 236)
        cls_74024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), 'cls', False)
        # Processing the call keyword arguments (line 236)
        kwargs_74025 = {}
        # Getting the type of 'object' (line 236)
        object_74022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'object', False)
        # Obtaining the member '__new__' of a type (line 236)
        new___74023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), object_74022, '__new__')
        # Calling __new__(args, kwargs) (line 236)
        new___call_result_74026 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), new___74023, *[cls_74024], **kwargs_74025)
        
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self', new___call_result_74026)
        
        # Assigning a Tuple to a Tuple (line 237):
        
        # Assigning a Name to a Name (line 237):
        # Getting the type of 't' (line 237)
        t_74027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 't')
        # Assigning a type to the variable 'tuple_assignment_73671' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_assignment_73671', t_74027)
        
        # Assigning a Name to a Name (line 237):
        # Getting the type of 'c' (line 237)
        c_74028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 36), 'c')
        # Assigning a type to the variable 'tuple_assignment_73672' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_assignment_73672', c_74028)
        
        # Assigning a Name to a Name (line 237):
        # Getting the type of 'k' (line 237)
        k_74029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 39), 'k')
        # Assigning a type to the variable 'tuple_assignment_73673' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_assignment_73673', k_74029)
        
        # Assigning a Name to a Attribute (line 237):
        # Getting the type of 'tuple_assignment_73671' (line 237)
        tuple_assignment_73671_74030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_assignment_73671')
        # Getting the type of 'self' (line 237)
        self_74031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self')
        # Setting the type of the member 't' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_74031, 't', tuple_assignment_73671_74030)
        
        # Assigning a Name to a Attribute (line 237):
        # Getting the type of 'tuple_assignment_73672' (line 237)
        tuple_assignment_73672_74032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_assignment_73672')
        # Getting the type of 'self' (line 237)
        self_74033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'self')
        # Setting the type of the member 'c' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), self_74033, 'c', tuple_assignment_73672_74032)
        
        # Assigning a Name to a Attribute (line 237):
        # Getting the type of 'tuple_assignment_73673' (line 237)
        tuple_assignment_73673_74034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_assignment_73673')
        # Getting the type of 'self' (line 237)
        self_74035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 24), 'self')
        # Setting the type of the member 'k' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 24), self_74035, 'k', tuple_assignment_73673_74034)
        
        # Assigning a Name to a Attribute (line 238):
        
        # Assigning a Name to a Attribute (line 238):
        # Getting the type of 'extrapolate' (line 238)
        extrapolate_74036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'extrapolate')
        # Getting the type of 'self' (line 238)
        self_74037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self')
        # Setting the type of the member 'extrapolate' of a type (line 238)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_74037, 'extrapolate', extrapolate_74036)
        
        # Assigning a Name to a Attribute (line 239):
        
        # Assigning a Name to a Attribute (line 239):
        # Getting the type of 'axis' (line 239)
        axis_74038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'axis')
        # Getting the type of 'self' (line 239)
        self_74039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_74039, 'axis', axis_74038)
        # Getting the type of 'self' (line 240)
        self_74040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'stypy_return_type', self_74040)
        
        # ################# End of 'construct_fast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'construct_fast' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_74041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'construct_fast'
        return stypy_return_type_74041


    @norecursion
    def tck(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tck'
        module_type_store = module_type_store.open_function_context('tck', 242, 4, False)
        # Assigning a type to the variable 'self' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline.tck.__dict__.__setitem__('stypy_localization', localization)
        BSpline.tck.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline.tck.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline.tck.__dict__.__setitem__('stypy_function_name', 'BSpline.tck')
        BSpline.tck.__dict__.__setitem__('stypy_param_names_list', [])
        BSpline.tck.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline.tck.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline.tck.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline.tck.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline.tck.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline.tck.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline.tck', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tck', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tck(...)' code ##################

        str_74042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, (-1)), 'str', 'Equvalent to ``(self.t, self.c, self.k)`` (read-only).\n        ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 246)
        tuple_74043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 246)
        # Adding element type (line 246)
        # Getting the type of 'self' (line 246)
        self_74044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'self')
        # Obtaining the member 't' of a type (line 246)
        t_74045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), self_74044, 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), tuple_74043, t_74045)
        # Adding element type (line 246)
        # Getting the type of 'self' (line 246)
        self_74046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'self')
        # Obtaining the member 'c' of a type (line 246)
        c_74047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 23), self_74046, 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), tuple_74043, c_74047)
        # Adding element type (line 246)
        # Getting the type of 'self' (line 246)
        self_74048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'self')
        # Obtaining the member 'k' of a type (line 246)
        k_74049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 31), self_74048, 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), tuple_74043, k_74049)
        
        # Assigning a type to the variable 'stypy_return_type' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'stypy_return_type', tuple_74043)
        
        # ################# End of 'tck(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tck' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_74050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tck'
        return stypy_return_type_74050


    @norecursion
    def basis_element(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 249)
        True_74051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'True')
        defaults = [True_74051]
        # Create a new context for function 'basis_element'
        module_type_store = module_type_store.open_function_context('basis_element', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline.basis_element.__dict__.__setitem__('stypy_localization', localization)
        BSpline.basis_element.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline.basis_element.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline.basis_element.__dict__.__setitem__('stypy_function_name', 'BSpline.basis_element')
        BSpline.basis_element.__dict__.__setitem__('stypy_param_names_list', ['t', 'extrapolate'])
        BSpline.basis_element.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline.basis_element.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline.basis_element.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline.basis_element.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline.basis_element.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline.basis_element.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline.basis_element', ['t', 'extrapolate'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'basis_element', localization, ['t', 'extrapolate'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'basis_element(...)' code ##################

        str_74052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, (-1)), 'str', "Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.\n\n        Parameters\n        ----------\n        t : ndarray, shape (k+1,)\n            internal knots\n        extrapolate : bool or 'periodic', optional\n            whether to extrapolate beyond the base interval, ``t[0] .. t[k+1]``,\n            or to return nans.\n            If 'periodic', periodic extrapolation is used.\n            Default is True.\n\n        Returns\n        -------\n        basis_element : callable\n            A callable representing a B-spline basis element for the knot\n            vector `t`.\n\n        Notes\n        -----\n        The order of the b-spline, `k`, is inferred from the length of `t` as\n        ``len(t)-2``. The knot vector is constructed by appending and prepending\n        ``k+1`` elements to internal knots `t`.\n\n        Examples\n        --------\n\n        Construct a cubic b-spline:\n\n        >>> from scipy.interpolate import BSpline\n        >>> b = BSpline.basis_element([0, 1, 2, 3, 4])\n        >>> k = b.k\n        >>> b.t[k:-k]\n        array([ 0.,  1.,  2.,  3.,  4.])\n        >>> k\n        3\n\n        Construct a second order b-spline on ``[0, 1, 1, 2]``, and compare\n        to its explicit form:\n\n        >>> t = [-1, 0, 1, 1, 2]\n        >>> b = BSpline.basis_element(t[1:])\n        >>> def f(x):\n        ...     return np.where(x < 1, x*x, (2. - x)**2)\n\n        >>> import matplotlib.pyplot as plt\n        >>> fig, ax = plt.subplots()\n        >>> x = np.linspace(0, 2, 51)\n        >>> ax.plot(x, b(x), 'g', lw=3)\n        >>> ax.plot(x, f(x), 'r', lw=8, alpha=0.4)\n        >>> ax.grid(True)\n        >>> plt.show()\n\n        ")
        
        # Assigning a BinOp to a Name (line 304):
        
        # Assigning a BinOp to a Name (line 304):
        
        # Call to len(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 't' (line 304)
        t_74054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 't', False)
        # Processing the call keyword arguments (line 304)
        kwargs_74055 = {}
        # Getting the type of 'len' (line 304)
        len_74053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'len', False)
        # Calling len(args, kwargs) (line 304)
        len_call_result_74056 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), len_74053, *[t_74054], **kwargs_74055)
        
        int_74057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 21), 'int')
        # Applying the binary operator '-' (line 304)
        result_sub_74058 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 12), '-', len_call_result_74056, int_74057)
        
        # Assigning a type to the variable 'k' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'k', result_sub_74058)
        
        # Assigning a Call to a Name (line 305):
        
        # Assigning a Call to a Name (line 305):
        
        # Call to _as_float_array(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 't' (line 305)
        t_74060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 28), 't', False)
        # Processing the call keyword arguments (line 305)
        kwargs_74061 = {}
        # Getting the type of '_as_float_array' (line 305)
        _as_float_array_74059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), '_as_float_array', False)
        # Calling _as_float_array(args, kwargs) (line 305)
        _as_float_array_call_result_74062 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), _as_float_array_74059, *[t_74060], **kwargs_74061)
        
        # Assigning a type to the variable 't' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 't', _as_float_array_call_result_74062)
        
        # Assigning a Subscript to a Name (line 306):
        
        # Assigning a Subscript to a Name (line 306):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 306)
        tuple_74063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 306)
        # Adding element type (line 306)
        
        # Obtaining an instance of the builtin type 'tuple' (line 306)
        tuple_74064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 306)
        # Adding element type (line 306)
        
        # Obtaining the type of the subscript
        int_74065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 21), 'int')
        # Getting the type of 't' (line 306)
        t_74066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 't')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___74067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), t_74066, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_74068 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), getitem___74067, int_74065)
        
        int_74069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 24), 'int')
        # Applying the binary operator '-' (line 306)
        result_sub_74070 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 19), '-', subscript_call_result_74068, int_74069)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 19), tuple_74064, result_sub_74070)
        
        # Getting the type of 'k' (line 306)
        k_74071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'k')
        # Applying the binary operator '*' (line 306)
        result_mul_74072 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 18), '*', tuple_74064, k_74071)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 18), tuple_74063, result_mul_74072)
        # Adding element type (line 306)
        # Getting the type of 't' (line 306)
        t_74073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 33), 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 18), tuple_74063, t_74073)
        # Adding element type (line 306)
        
        # Obtaining an instance of the builtin type 'tuple' (line 306)
        tuple_74074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 306)
        # Adding element type (line 306)
        
        # Obtaining the type of the subscript
        int_74075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 39), 'int')
        # Getting the type of 't' (line 306)
        t_74076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 37), 't')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___74077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 37), t_74076, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_74078 = invoke(stypy.reporting.localization.Localization(__file__, 306, 37), getitem___74077, int_74075)
        
        int_74079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 43), 'int')
        # Applying the binary operator '+' (line 306)
        result_add_74080 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 37), '+', subscript_call_result_74078, int_74079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 37), tuple_74074, result_add_74080)
        
        # Getting the type of 'k' (line 306)
        k_74081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 49), 'k')
        # Applying the binary operator '*' (line 306)
        result_mul_74082 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 36), '*', tuple_74074, k_74081)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 18), tuple_74063, result_mul_74082)
        
        # Getting the type of 'np' (line 306)
        np_74083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'np')
        # Obtaining the member 'r_' of a type (line 306)
        r__74084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), np_74083, 'r_')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___74085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), r__74084, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_74086 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), getitem___74085, tuple_74063)
        
        # Assigning a type to the variable 't' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 't', subscript_call_result_74086)
        
        # Assigning a Call to a Name (line 307):
        
        # Assigning a Call to a Name (line 307):
        
        # Call to zeros_like(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 't' (line 307)
        t_74089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 26), 't', False)
        # Processing the call keyword arguments (line 307)
        kwargs_74090 = {}
        # Getting the type of 'np' (line 307)
        np_74087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 307)
        zeros_like_74088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), np_74087, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 307)
        zeros_like_call_result_74091 = invoke(stypy.reporting.localization.Localization(__file__, 307, 12), zeros_like_74088, *[t_74089], **kwargs_74090)
        
        # Assigning a type to the variable 'c' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'c', zeros_like_call_result_74091)
        
        # Assigning a Num to a Subscript (line 308):
        
        # Assigning a Num to a Subscript (line 308):
        float_74092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 15), 'float')
        # Getting the type of 'c' (line 308)
        c_74093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'c')
        # Getting the type of 'k' (line 308)
        k_74094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 10), 'k')
        # Storing an element on a container (line 308)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 8), c_74093, (k_74094, float_74092))
        
        # Call to construct_fast(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 't' (line 309)
        t_74097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 34), 't', False)
        # Getting the type of 'c' (line 309)
        c_74098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 37), 'c', False)
        # Getting the type of 'k' (line 309)
        k_74099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 40), 'k', False)
        # Getting the type of 'extrapolate' (line 309)
        extrapolate_74100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 43), 'extrapolate', False)
        # Processing the call keyword arguments (line 309)
        kwargs_74101 = {}
        # Getting the type of 'cls' (line 309)
        cls_74095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 15), 'cls', False)
        # Obtaining the member 'construct_fast' of a type (line 309)
        construct_fast_74096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 15), cls_74095, 'construct_fast')
        # Calling construct_fast(args, kwargs) (line 309)
        construct_fast_call_result_74102 = invoke(stypy.reporting.localization.Localization(__file__, 309, 15), construct_fast_74096, *[t_74097, c_74098, k_74099, extrapolate_74100], **kwargs_74101)
        
        # Assigning a type to the variable 'stypy_return_type' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'stypy_return_type', construct_fast_call_result_74102)
        
        # ################# End of 'basis_element(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'basis_element' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_74103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74103)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'basis_element'
        return stypy_return_type_74103


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_74104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 29), 'int')
        # Getting the type of 'None' (line 311)
        None_74105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 44), 'None')
        defaults = [int_74104, None_74105]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 311, 4, False)
        # Assigning a type to the variable 'self' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline.__call__.__dict__.__setitem__('stypy_localization', localization)
        BSpline.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline.__call__.__dict__.__setitem__('stypy_function_name', 'BSpline.__call__')
        BSpline.__call__.__dict__.__setitem__('stypy_param_names_list', ['x', 'nu', 'extrapolate'])
        BSpline.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline.__call__', ['x', 'nu', 'extrapolate'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x', 'nu', 'extrapolate'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_74106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, (-1)), 'str', "\n        Evaluate a spline function.\n\n        Parameters\n        ----------\n        x : array_like\n            points to evaluate the spline at.\n        nu: int, optional\n            derivative to evaluate (default is 0).\n        extrapolate : bool or 'periodic', optional\n            whether to extrapolate based on the first and last intervals\n            or return nans. If 'periodic', periodic extrapolation is used.\n            Default is `self.extrapolate`.\n\n        Returns\n        -------\n        y : array_like\n            Shape is determined by replacing the interpolation axis\n            in the coefficient array with the shape of `x`.\n\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 333)
        # Getting the type of 'extrapolate' (line 333)
        extrapolate_74107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'extrapolate')
        # Getting the type of 'None' (line 333)
        None_74108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 26), 'None')
        
        (may_be_74109, more_types_in_union_74110) = may_be_none(extrapolate_74107, None_74108)

        if may_be_74109:

            if more_types_in_union_74110:
                # Runtime conditional SSA (line 333)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 334):
            
            # Assigning a Attribute to a Name (line 334):
            # Getting the type of 'self' (line 334)
            self_74111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 26), 'self')
            # Obtaining the member 'extrapolate' of a type (line 334)
            extrapolate_74112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 26), self_74111, 'extrapolate')
            # Assigning a type to the variable 'extrapolate' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'extrapolate', extrapolate_74112)

            if more_types_in_union_74110:
                # SSA join for if statement (line 333)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to asarray(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'x' (line 335)
        x_74115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 23), 'x', False)
        # Processing the call keyword arguments (line 335)
        kwargs_74116 = {}
        # Getting the type of 'np' (line 335)
        np_74113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 335)
        asarray_74114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), np_74113, 'asarray')
        # Calling asarray(args, kwargs) (line 335)
        asarray_call_result_74117 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), asarray_74114, *[x_74115], **kwargs_74116)
        
        # Assigning a type to the variable 'x' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'x', asarray_call_result_74117)
        
        # Assigning a Tuple to a Tuple (line 336):
        
        # Assigning a Attribute to a Name (line 336):
        # Getting the type of 'x' (line 336)
        x_74118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 26), 'x')
        # Obtaining the member 'shape' of a type (line 336)
        shape_74119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 26), x_74118, 'shape')
        # Assigning a type to the variable 'tuple_assignment_73674' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_assignment_73674', shape_74119)
        
        # Assigning a Attribute to a Name (line 336):
        # Getting the type of 'x' (line 336)
        x_74120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 35), 'x')
        # Obtaining the member 'ndim' of a type (line 336)
        ndim_74121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 35), x_74120, 'ndim')
        # Assigning a type to the variable 'tuple_assignment_73675' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_assignment_73675', ndim_74121)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_assignment_73674' (line 336)
        tuple_assignment_73674_74122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_assignment_73674')
        # Assigning a type to the variable 'x_shape' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'x_shape', tuple_assignment_73674_74122)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_assignment_73675' (line 336)
        tuple_assignment_73675_74123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_assignment_73675')
        # Assigning a type to the variable 'x_ndim' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 17), 'x_ndim', tuple_assignment_73675_74123)
        
        # Assigning a Call to a Name (line 337):
        
        # Assigning a Call to a Name (line 337):
        
        # Call to ascontiguousarray(...): (line 337)
        # Processing the call arguments (line 337)
        
        # Call to ravel(...): (line 337)
        # Processing the call keyword arguments (line 337)
        kwargs_74128 = {}
        # Getting the type of 'x' (line 337)
        x_74126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 33), 'x', False)
        # Obtaining the member 'ravel' of a type (line 337)
        ravel_74127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 33), x_74126, 'ravel')
        # Calling ravel(args, kwargs) (line 337)
        ravel_call_result_74129 = invoke(stypy.reporting.localization.Localization(__file__, 337, 33), ravel_74127, *[], **kwargs_74128)
        
        # Processing the call keyword arguments (line 337)
        # Getting the type of 'np' (line 337)
        np_74130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 50), 'np', False)
        # Obtaining the member 'float_' of a type (line 337)
        float__74131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 50), np_74130, 'float_')
        keyword_74132 = float__74131
        kwargs_74133 = {'dtype': keyword_74132}
        # Getting the type of 'np' (line 337)
        np_74124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 337)
        ascontiguousarray_74125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), np_74124, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 337)
        ascontiguousarray_call_result_74134 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), ascontiguousarray_74125, *[ravel_call_result_74129], **kwargs_74133)
        
        # Assigning a type to the variable 'x' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'x', ascontiguousarray_call_result_74134)
        
        
        # Getting the type of 'extrapolate' (line 341)
        extrapolate_74135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 11), 'extrapolate')
        str_74136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 26), 'str', 'periodic')
        # Applying the binary operator '==' (line 341)
        result_eq_74137 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 11), '==', extrapolate_74135, str_74136)
        
        # Testing the type of an if condition (line 341)
        if_condition_74138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 8), result_eq_74137)
        # Assigning a type to the variable 'if_condition_74138' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'if_condition_74138', if_condition_74138)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 342):
        
        # Assigning a BinOp to a Name (line 342):
        # Getting the type of 'self' (line 342)
        self_74139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'self')
        # Obtaining the member 't' of a type (line 342)
        t_74140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 16), self_74139, 't')
        # Obtaining the member 'size' of a type (line 342)
        size_74141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 16), t_74140, 'size')
        # Getting the type of 'self' (line 342)
        self_74142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 30), 'self')
        # Obtaining the member 'k' of a type (line 342)
        k_74143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 30), self_74142, 'k')
        # Applying the binary operator '-' (line 342)
        result_sub_74144 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 16), '-', size_74141, k_74143)
        
        int_74145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 39), 'int')
        # Applying the binary operator '-' (line 342)
        result_sub_74146 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 37), '-', result_sub_74144, int_74145)
        
        # Assigning a type to the variable 'n' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'n', result_sub_74146)
        
        # Assigning a BinOp to a Name (line 343):
        
        # Assigning a BinOp to a Name (line 343):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 343)
        self_74147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'self')
        # Obtaining the member 'k' of a type (line 343)
        k_74148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 23), self_74147, 'k')
        # Getting the type of 'self' (line 343)
        self_74149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'self')
        # Obtaining the member 't' of a type (line 343)
        t_74150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 16), self_74149, 't')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___74151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 16), t_74150, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_74152 = invoke(stypy.reporting.localization.Localization(__file__, 343, 16), getitem___74151, k_74148)
        
        # Getting the type of 'x' (line 343)
        x_74153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 34), 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 343)
        self_74154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 45), 'self')
        # Obtaining the member 'k' of a type (line 343)
        k_74155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 45), self_74154, 'k')
        # Getting the type of 'self' (line 343)
        self_74156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 38), 'self')
        # Obtaining the member 't' of a type (line 343)
        t_74157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 38), self_74156, 't')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___74158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 38), t_74157, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_74159 = invoke(stypy.reporting.localization.Localization(__file__, 343, 38), getitem___74158, k_74155)
        
        # Applying the binary operator '-' (line 343)
        result_sub_74160 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 34), '-', x_74153, subscript_call_result_74159)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 343)
        n_74161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 64), 'n')
        # Getting the type of 'self' (line 343)
        self_74162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 57), 'self')
        # Obtaining the member 't' of a type (line 343)
        t_74163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 57), self_74162, 't')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___74164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 57), t_74163, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_74165 = invoke(stypy.reporting.localization.Localization(__file__, 343, 57), getitem___74164, n_74161)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 344)
        self_74166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 64), 'self')
        # Obtaining the member 'k' of a type (line 344)
        k_74167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 64), self_74166, 'k')
        # Getting the type of 'self' (line 344)
        self_74168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 57), 'self')
        # Obtaining the member 't' of a type (line 344)
        t_74169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 57), self_74168, 't')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___74170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 57), t_74169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_74171 = invoke(stypy.reporting.localization.Localization(__file__, 344, 57), getitem___74170, k_74167)
        
        # Applying the binary operator '-' (line 343)
        result_sub_74172 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 57), '-', subscript_call_result_74165, subscript_call_result_74171)
        
        # Applying the binary operator '%' (line 343)
        result_mod_74173 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 33), '%', result_sub_74160, result_sub_74172)
        
        # Applying the binary operator '+' (line 343)
        result_add_74174 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 16), '+', subscript_call_result_74152, result_mod_74173)
        
        # Assigning a type to the variable 'x' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'x', result_add_74174)
        
        # Assigning a Name to a Name (line 345):
        
        # Assigning a Name to a Name (line 345):
        # Getting the type of 'False' (line 345)
        False_74175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 26), 'False')
        # Assigning a type to the variable 'extrapolate' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'extrapolate', False_74175)
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to empty(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Obtaining an instance of the builtin type 'tuple' (line 347)
        tuple_74178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 347)
        # Adding element type (line 347)
        
        # Call to len(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'x' (line 347)
        x_74180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 'x', False)
        # Processing the call keyword arguments (line 347)
        kwargs_74181 = {}
        # Getting the type of 'len' (line 347)
        len_74179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'len', False)
        # Calling len(args, kwargs) (line 347)
        len_call_result_74182 = invoke(stypy.reporting.localization.Localization(__file__, 347, 24), len_74179, *[x_74180], **kwargs_74181)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 24), tuple_74178, len_call_result_74182)
        # Adding element type (line 347)
        
        # Call to prod(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Obtaining the type of the subscript
        int_74184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 50), 'int')
        slice_74185 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 347, 37), int_74184, None, None)
        # Getting the type of 'self' (line 347)
        self_74186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 37), 'self', False)
        # Obtaining the member 'c' of a type (line 347)
        c_74187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 37), self_74186, 'c')
        # Obtaining the member 'shape' of a type (line 347)
        shape_74188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 37), c_74187, 'shape')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___74189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 37), shape_74188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_74190 = invoke(stypy.reporting.localization.Localization(__file__, 347, 37), getitem___74189, slice_74185)
        
        # Processing the call keyword arguments (line 347)
        kwargs_74191 = {}
        # Getting the type of 'prod' (line 347)
        prod_74183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 32), 'prod', False)
        # Calling prod(args, kwargs) (line 347)
        prod_call_result_74192 = invoke(stypy.reporting.localization.Localization(__file__, 347, 32), prod_74183, *[subscript_call_result_74190], **kwargs_74191)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 24), tuple_74178, prod_call_result_74192)
        
        # Processing the call keyword arguments (line 347)
        # Getting the type of 'self' (line 347)
        self_74193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 63), 'self', False)
        # Obtaining the member 'c' of a type (line 347)
        c_74194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 63), self_74193, 'c')
        # Obtaining the member 'dtype' of a type (line 347)
        dtype_74195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 63), c_74194, 'dtype')
        keyword_74196 = dtype_74195
        kwargs_74197 = {'dtype': keyword_74196}
        # Getting the type of 'np' (line 347)
        np_74176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 14), 'np', False)
        # Obtaining the member 'empty' of a type (line 347)
        empty_74177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 14), np_74176, 'empty')
        # Calling empty(args, kwargs) (line 347)
        empty_call_result_74198 = invoke(stypy.reporting.localization.Localization(__file__, 347, 14), empty_74177, *[tuple_74178], **kwargs_74197)
        
        # Assigning a type to the variable 'out' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'out', empty_call_result_74198)
        
        # Call to _ensure_c_contiguous(...): (line 348)
        # Processing the call keyword arguments (line 348)
        kwargs_74201 = {}
        # Getting the type of 'self' (line 348)
        self_74199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self', False)
        # Obtaining the member '_ensure_c_contiguous' of a type (line 348)
        _ensure_c_contiguous_74200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_74199, '_ensure_c_contiguous')
        # Calling _ensure_c_contiguous(args, kwargs) (line 348)
        _ensure_c_contiguous_call_result_74202 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), _ensure_c_contiguous_74200, *[], **kwargs_74201)
        
        
        # Call to _evaluate(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'x' (line 349)
        x_74205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'x', False)
        # Getting the type of 'nu' (line 349)
        nu_74206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 26), 'nu', False)
        # Getting the type of 'extrapolate' (line 349)
        extrapolate_74207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 30), 'extrapolate', False)
        # Getting the type of 'out' (line 349)
        out_74208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 43), 'out', False)
        # Processing the call keyword arguments (line 349)
        kwargs_74209 = {}
        # Getting the type of 'self' (line 349)
        self_74203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'self', False)
        # Obtaining the member '_evaluate' of a type (line 349)
        _evaluate_74204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), self_74203, '_evaluate')
        # Calling _evaluate(args, kwargs) (line 349)
        _evaluate_call_result_74210 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), _evaluate_74204, *[x_74205, nu_74206, extrapolate_74207, out_74208], **kwargs_74209)
        
        
        # Assigning a Call to a Name (line 350):
        
        # Assigning a Call to a Name (line 350):
        
        # Call to reshape(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'x_shape' (line 350)
        x_shape_74213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'x_shape', False)
        
        # Obtaining the type of the subscript
        int_74214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 49), 'int')
        slice_74215 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 350, 36), int_74214, None, None)
        # Getting the type of 'self' (line 350)
        self_74216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), 'self', False)
        # Obtaining the member 'c' of a type (line 350)
        c_74217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 36), self_74216, 'c')
        # Obtaining the member 'shape' of a type (line 350)
        shape_74218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 36), c_74217, 'shape')
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___74219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 36), shape_74218, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_74220 = invoke(stypy.reporting.localization.Localization(__file__, 350, 36), getitem___74219, slice_74215)
        
        # Applying the binary operator '+' (line 350)
        result_add_74221 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 26), '+', x_shape_74213, subscript_call_result_74220)
        
        # Processing the call keyword arguments (line 350)
        kwargs_74222 = {}
        # Getting the type of 'out' (line 350)
        out_74211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 14), 'out', False)
        # Obtaining the member 'reshape' of a type (line 350)
        reshape_74212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 14), out_74211, 'reshape')
        # Calling reshape(args, kwargs) (line 350)
        reshape_call_result_74223 = invoke(stypy.reporting.localization.Localization(__file__, 350, 14), reshape_74212, *[result_add_74221], **kwargs_74222)
        
        # Assigning a type to the variable 'out' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'out', reshape_call_result_74223)
        
        
        # Getting the type of 'self' (line 351)
        self_74224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 11), 'self')
        # Obtaining the member 'axis' of a type (line 351)
        axis_74225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 11), self_74224, 'axis')
        int_74226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 24), 'int')
        # Applying the binary operator '!=' (line 351)
        result_ne_74227 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 11), '!=', axis_74225, int_74226)
        
        # Testing the type of an if condition (line 351)
        if_condition_74228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 8), result_ne_74227)
        # Assigning a type to the variable 'if_condition_74228' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'if_condition_74228', if_condition_74228)
        # SSA begins for if statement (line 351)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Call to list(...): (line 353)
        # Processing the call arguments (line 353)
        
        # Call to range(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'out' (line 353)
        out_74231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 27), 'out', False)
        # Obtaining the member 'ndim' of a type (line 353)
        ndim_74232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 27), out_74231, 'ndim')
        # Processing the call keyword arguments (line 353)
        kwargs_74233 = {}
        # Getting the type of 'range' (line 353)
        range_74230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'range', False)
        # Calling range(args, kwargs) (line 353)
        range_call_result_74234 = invoke(stypy.reporting.localization.Localization(__file__, 353, 21), range_74230, *[ndim_74232], **kwargs_74233)
        
        # Processing the call keyword arguments (line 353)
        kwargs_74235 = {}
        # Getting the type of 'list' (line 353)
        list_74229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'list', False)
        # Calling list(args, kwargs) (line 353)
        list_call_result_74236 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), list_74229, *[range_call_result_74234], **kwargs_74235)
        
        # Assigning a type to the variable 'l' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'l', list_call_result_74236)
        
        # Assigning a BinOp to a Name (line 354):
        
        # Assigning a BinOp to a Name (line 354):
        
        # Obtaining the type of the subscript
        # Getting the type of 'x_ndim' (line 354)
        x_ndim_74237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'x_ndim')
        # Getting the type of 'x_ndim' (line 354)
        x_ndim_74238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 25), 'x_ndim')
        # Getting the type of 'self' (line 354)
        self_74239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 32), 'self')
        # Obtaining the member 'axis' of a type (line 354)
        axis_74240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 32), self_74239, 'axis')
        # Applying the binary operator '+' (line 354)
        result_add_74241 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 25), '+', x_ndim_74238, axis_74240)
        
        slice_74242 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 16), x_ndim_74237, result_add_74241, None)
        # Getting the type of 'l' (line 354)
        l_74243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'l')
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___74244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 16), l_74243, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_74245 = invoke(stypy.reporting.localization.Localization(__file__, 354, 16), getitem___74244, slice_74242)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'x_ndim' (line 354)
        x_ndim_74246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 48), 'x_ndim')
        slice_74247 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 45), None, x_ndim_74246, None)
        # Getting the type of 'l' (line 354)
        l_74248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 45), 'l')
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___74249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 45), l_74248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_74250 = invoke(stypy.reporting.localization.Localization(__file__, 354, 45), getitem___74249, slice_74247)
        
        # Applying the binary operator '+' (line 354)
        result_add_74251 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 16), '+', subscript_call_result_74245, subscript_call_result_74250)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'x_ndim' (line 354)
        x_ndim_74252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 60), 'x_ndim')
        # Getting the type of 'self' (line 354)
        self_74253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 67), 'self')
        # Obtaining the member 'axis' of a type (line 354)
        axis_74254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 67), self_74253, 'axis')
        # Applying the binary operator '+' (line 354)
        result_add_74255 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 60), '+', x_ndim_74252, axis_74254)
        
        slice_74256 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 58), result_add_74255, None, None)
        # Getting the type of 'l' (line 354)
        l_74257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 58), 'l')
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___74258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 58), l_74257, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_74259 = invoke(stypy.reporting.localization.Localization(__file__, 354, 58), getitem___74258, slice_74256)
        
        # Applying the binary operator '+' (line 354)
        result_add_74260 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 56), '+', result_add_74251, subscript_call_result_74259)
        
        # Assigning a type to the variable 'l' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'l', result_add_74260)
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Call to transpose(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'l' (line 355)
        l_74263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 32), 'l', False)
        # Processing the call keyword arguments (line 355)
        kwargs_74264 = {}
        # Getting the type of 'out' (line 355)
        out_74261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 18), 'out', False)
        # Obtaining the member 'transpose' of a type (line 355)
        transpose_74262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 18), out_74261, 'transpose')
        # Calling transpose(args, kwargs) (line 355)
        transpose_call_result_74265 = invoke(stypy.reporting.localization.Localization(__file__, 355, 18), transpose_74262, *[l_74263], **kwargs_74264)
        
        # Assigning a type to the variable 'out' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'out', transpose_call_result_74265)
        # SSA join for if statement (line 351)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'out' (line 356)
        out_74266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'stypy_return_type', out_74266)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 311)
        stypy_return_type_74267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_74267


    @norecursion
    def _evaluate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_evaluate'
        module_type_store = module_type_store.open_function_context('_evaluate', 358, 4, False)
        # Assigning a type to the variable 'self' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline._evaluate.__dict__.__setitem__('stypy_localization', localization)
        BSpline._evaluate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline._evaluate.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline._evaluate.__dict__.__setitem__('stypy_function_name', 'BSpline._evaluate')
        BSpline._evaluate.__dict__.__setitem__('stypy_param_names_list', ['xp', 'nu', 'extrapolate', 'out'])
        BSpline._evaluate.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline._evaluate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline._evaluate.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline._evaluate.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline._evaluate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline._evaluate.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline._evaluate', ['xp', 'nu', 'extrapolate', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_evaluate', localization, ['xp', 'nu', 'extrapolate', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_evaluate(...)' code ##################

        
        # Call to evaluate_spline(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'self' (line 359)
        self_74270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 30), 'self', False)
        # Obtaining the member 't' of a type (line 359)
        t_74271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 30), self_74270, 't')
        
        # Call to reshape(...): (line 359)
        # Processing the call arguments (line 359)
        
        # Obtaining the type of the subscript
        int_74275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 66), 'int')
        # Getting the type of 'self' (line 359)
        self_74276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 53), 'self', False)
        # Obtaining the member 'c' of a type (line 359)
        c_74277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 53), self_74276, 'c')
        # Obtaining the member 'shape' of a type (line 359)
        shape_74278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 53), c_74277, 'shape')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___74279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 53), shape_74278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_74280 = invoke(stypy.reporting.localization.Localization(__file__, 359, 53), getitem___74279, int_74275)
        
        int_74281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 70), 'int')
        # Processing the call keyword arguments (line 359)
        kwargs_74282 = {}
        # Getting the type of 'self' (line 359)
        self_74272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 38), 'self', False)
        # Obtaining the member 'c' of a type (line 359)
        c_74273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 38), self_74272, 'c')
        # Obtaining the member 'reshape' of a type (line 359)
        reshape_74274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 38), c_74273, 'reshape')
        # Calling reshape(args, kwargs) (line 359)
        reshape_call_result_74283 = invoke(stypy.reporting.localization.Localization(__file__, 359, 38), reshape_74274, *[subscript_call_result_74280, int_74281], **kwargs_74282)
        
        # Getting the type of 'self' (line 360)
        self_74284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'self', False)
        # Obtaining the member 'k' of a type (line 360)
        k_74285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 16), self_74284, 'k')
        # Getting the type of 'xp' (line 360)
        xp_74286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 24), 'xp', False)
        # Getting the type of 'nu' (line 360)
        nu_74287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'nu', False)
        # Getting the type of 'extrapolate' (line 360)
        extrapolate_74288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 32), 'extrapolate', False)
        # Getting the type of 'out' (line 360)
        out_74289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 45), 'out', False)
        # Processing the call keyword arguments (line 359)
        kwargs_74290 = {}
        # Getting the type of '_bspl' (line 359)
        _bspl_74268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), '_bspl', False)
        # Obtaining the member 'evaluate_spline' of a type (line 359)
        evaluate_spline_74269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), _bspl_74268, 'evaluate_spline')
        # Calling evaluate_spline(args, kwargs) (line 359)
        evaluate_spline_call_result_74291 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), evaluate_spline_74269, *[t_74271, reshape_call_result_74283, k_74285, xp_74286, nu_74287, extrapolate_74288, out_74289], **kwargs_74290)
        
        
        # ################# End of '_evaluate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_evaluate' in the type store
        # Getting the type of 'stypy_return_type' (line 358)
        stypy_return_type_74292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_evaluate'
        return stypy_return_type_74292


    @norecursion
    def _ensure_c_contiguous(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ensure_c_contiguous'
        module_type_store = module_type_store.open_function_context('_ensure_c_contiguous', 362, 4, False)
        # Assigning a type to the variable 'self' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_localization', localization)
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_function_name', 'BSpline._ensure_c_contiguous')
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_param_names_list', [])
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline._ensure_c_contiguous.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline._ensure_c_contiguous', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ensure_c_contiguous', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ensure_c_contiguous(...)' code ##################

        str_74293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, (-1)), 'str', '\n        c and t may be modified by the user. The Cython code expects\n        that they are C contiguous.\n\n        ')
        
        
        # Getting the type of 'self' (line 368)
        self_74294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), 'self')
        # Obtaining the member 't' of a type (line 368)
        t_74295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), self_74294, 't')
        # Obtaining the member 'flags' of a type (line 368)
        flags_74296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), t_74295, 'flags')
        # Obtaining the member 'c_contiguous' of a type (line 368)
        c_contiguous_74297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), flags_74296, 'c_contiguous')
        # Applying the 'not' unary operator (line 368)
        result_not__74298 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 11), 'not', c_contiguous_74297)
        
        # Testing the type of an if condition (line 368)
        if_condition_74299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 8), result_not__74298)
        # Assigning a type to the variable 'if_condition_74299' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'if_condition_74299', if_condition_74299)
        # SSA begins for if statement (line 368)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 369):
        
        # Assigning a Call to a Attribute (line 369):
        
        # Call to copy(...): (line 369)
        # Processing the call keyword arguments (line 369)
        kwargs_74303 = {}
        # Getting the type of 'self' (line 369)
        self_74300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'self', False)
        # Obtaining the member 't' of a type (line 369)
        t_74301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), self_74300, 't')
        # Obtaining the member 'copy' of a type (line 369)
        copy_74302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), t_74301, 'copy')
        # Calling copy(args, kwargs) (line 369)
        copy_call_result_74304 = invoke(stypy.reporting.localization.Localization(__file__, 369, 21), copy_74302, *[], **kwargs_74303)
        
        # Getting the type of 'self' (line 369)
        self_74305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'self')
        # Setting the type of the member 't' of a type (line 369)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), self_74305, 't', copy_call_result_74304)
        # SSA join for if statement (line 368)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 370)
        self_74306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'self')
        # Obtaining the member 'c' of a type (line 370)
        c_74307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), self_74306, 'c')
        # Obtaining the member 'flags' of a type (line 370)
        flags_74308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), c_74307, 'flags')
        # Obtaining the member 'c_contiguous' of a type (line 370)
        c_contiguous_74309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), flags_74308, 'c_contiguous')
        # Applying the 'not' unary operator (line 370)
        result_not__74310 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 11), 'not', c_contiguous_74309)
        
        # Testing the type of an if condition (line 370)
        if_condition_74311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 8), result_not__74310)
        # Assigning a type to the variable 'if_condition_74311' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'if_condition_74311', if_condition_74311)
        # SSA begins for if statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 371):
        
        # Assigning a Call to a Attribute (line 371):
        
        # Call to copy(...): (line 371)
        # Processing the call keyword arguments (line 371)
        kwargs_74315 = {}
        # Getting the type of 'self' (line 371)
        self_74312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'self', False)
        # Obtaining the member 'c' of a type (line 371)
        c_74313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 21), self_74312, 'c')
        # Obtaining the member 'copy' of a type (line 371)
        copy_74314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 21), c_74313, 'copy')
        # Calling copy(args, kwargs) (line 371)
        copy_call_result_74316 = invoke(stypy.reporting.localization.Localization(__file__, 371, 21), copy_74314, *[], **kwargs_74315)
        
        # Getting the type of 'self' (line 371)
        self_74317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'self')
        # Setting the type of the member 'c' of a type (line 371)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 12), self_74317, 'c', copy_call_result_74316)
        # SSA join for if statement (line 370)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_ensure_c_contiguous(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ensure_c_contiguous' in the type store
        # Getting the type of 'stypy_return_type' (line 362)
        stypy_return_type_74318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ensure_c_contiguous'
        return stypy_return_type_74318


    @norecursion
    def derivative(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_74319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 28), 'int')
        defaults = [int_74319]
        # Create a new context for function 'derivative'
        module_type_store = module_type_store.open_function_context('derivative', 373, 4, False)
        # Assigning a type to the variable 'self' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline.derivative.__dict__.__setitem__('stypy_localization', localization)
        BSpline.derivative.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline.derivative.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline.derivative.__dict__.__setitem__('stypy_function_name', 'BSpline.derivative')
        BSpline.derivative.__dict__.__setitem__('stypy_param_names_list', ['nu'])
        BSpline.derivative.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline.derivative.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline.derivative.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline.derivative.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline.derivative.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline.derivative.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline.derivative', ['nu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'derivative', localization, ['nu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'derivative(...)' code ##################

        str_74320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, (-1)), 'str', 'Return a b-spline representing the derivative.\n\n        Parameters\n        ----------\n        nu : int, optional\n            Derivative order.\n            Default is 1.\n\n        Returns\n        -------\n        b : BSpline object\n            A new instance representing the derivative.\n\n        See Also\n        --------\n        splder, splantider\n\n        ')
        
        # Assigning a Attribute to a Name (line 392):
        
        # Assigning a Attribute to a Name (line 392):
        # Getting the type of 'self' (line 392)
        self_74321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'self')
        # Obtaining the member 'c' of a type (line 392)
        c_74322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), self_74321, 'c')
        # Assigning a type to the variable 'c' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'c', c_74322)
        
        # Assigning a BinOp to a Name (line 394):
        
        # Assigning a BinOp to a Name (line 394):
        
        # Call to len(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'self' (line 394)
        self_74324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 17), 'self', False)
        # Obtaining the member 't' of a type (line 394)
        t_74325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 17), self_74324, 't')
        # Processing the call keyword arguments (line 394)
        kwargs_74326 = {}
        # Getting the type of 'len' (line 394)
        len_74323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 13), 'len', False)
        # Calling len(args, kwargs) (line 394)
        len_call_result_74327 = invoke(stypy.reporting.localization.Localization(__file__, 394, 13), len_74323, *[t_74325], **kwargs_74326)
        
        
        # Call to len(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'c' (line 394)
        c_74329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 31), 'c', False)
        # Processing the call keyword arguments (line 394)
        kwargs_74330 = {}
        # Getting the type of 'len' (line 394)
        len_74328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 27), 'len', False)
        # Calling len(args, kwargs) (line 394)
        len_call_result_74331 = invoke(stypy.reporting.localization.Localization(__file__, 394, 27), len_74328, *[c_74329], **kwargs_74330)
        
        # Applying the binary operator '-' (line 394)
        result_sub_74332 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 13), '-', len_call_result_74327, len_call_result_74331)
        
        # Assigning a type to the variable 'ct' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'ct', result_sub_74332)
        
        
        # Getting the type of 'ct' (line 395)
        ct_74333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'ct')
        int_74334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 16), 'int')
        # Applying the binary operator '>' (line 395)
        result_gt_74335 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 11), '>', ct_74333, int_74334)
        
        # Testing the type of an if condition (line 395)
        if_condition_74336 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 8), result_gt_74335)
        # Assigning a type to the variable 'if_condition_74336' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'if_condition_74336', if_condition_74336)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 396):
        
        # Assigning a Subscript to a Name (line 396):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 396)
        tuple_74337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 396)
        # Adding element type (line 396)
        # Getting the type of 'c' (line 396)
        c_74338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 22), tuple_74337, c_74338)
        # Adding element type (line 396)
        
        # Call to zeros(...): (line 396)
        # Processing the call arguments (line 396)
        
        # Obtaining an instance of the builtin type 'tuple' (line 396)
        tuple_74341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 396)
        # Adding element type (line 396)
        # Getting the type of 'ct' (line 396)
        ct_74342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 35), 'ct', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 35), tuple_74341, ct_74342)
        
        
        # Obtaining the type of the subscript
        int_74343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 50), 'int')
        slice_74344 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 396, 42), int_74343, None, None)
        # Getting the type of 'c' (line 396)
        c_74345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 42), 'c', False)
        # Obtaining the member 'shape' of a type (line 396)
        shape_74346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 42), c_74345, 'shape')
        # Obtaining the member '__getitem__' of a type (line 396)
        getitem___74347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 42), shape_74346, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 396)
        subscript_call_result_74348 = invoke(stypy.reporting.localization.Localization(__file__, 396, 42), getitem___74347, slice_74344)
        
        # Applying the binary operator '+' (line 396)
        result_add_74349 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 34), '+', tuple_74341, subscript_call_result_74348)
        
        # Processing the call keyword arguments (line 396)
        kwargs_74350 = {}
        # Getting the type of 'np' (line 396)
        np_74339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 25), 'np', False)
        # Obtaining the member 'zeros' of a type (line 396)
        zeros_74340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 25), np_74339, 'zeros')
        # Calling zeros(args, kwargs) (line 396)
        zeros_call_result_74351 = invoke(stypy.reporting.localization.Localization(__file__, 396, 25), zeros_74340, *[result_add_74349], **kwargs_74350)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 22), tuple_74337, zeros_call_result_74351)
        
        # Getting the type of 'np' (line 396)
        np_74352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 'np')
        # Obtaining the member 'r_' of a type (line 396)
        r__74353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), np_74352, 'r_')
        # Obtaining the member '__getitem__' of a type (line 396)
        getitem___74354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), r__74353, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 396)
        subscript_call_result_74355 = invoke(stypy.reporting.localization.Localization(__file__, 396, 16), getitem___74354, tuple_74337)
        
        # Assigning a type to the variable 'c' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'c', subscript_call_result_74355)
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to splder(...): (line 397)
        # Processing the call arguments (line 397)
        
        # Obtaining an instance of the builtin type 'tuple' (line 397)
        tuple_74358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 397)
        # Adding element type (line 397)
        # Getting the type of 'self' (line 397)
        self_74359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 36), 'self', False)
        # Obtaining the member 't' of a type (line 397)
        t_74360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 36), self_74359, 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 36), tuple_74358, t_74360)
        # Adding element type (line 397)
        # Getting the type of 'c' (line 397)
        c_74361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 44), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 36), tuple_74358, c_74361)
        # Adding element type (line 397)
        # Getting the type of 'self' (line 397)
        self_74362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 47), 'self', False)
        # Obtaining the member 'k' of a type (line 397)
        k_74363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 47), self_74362, 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 36), tuple_74358, k_74363)
        
        # Getting the type of 'nu' (line 397)
        nu_74364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 56), 'nu', False)
        # Processing the call keyword arguments (line 397)
        kwargs_74365 = {}
        # Getting the type of '_fitpack_impl' (line 397)
        _fitpack_impl_74356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 14), '_fitpack_impl', False)
        # Obtaining the member 'splder' of a type (line 397)
        splder_74357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 14), _fitpack_impl_74356, 'splder')
        # Calling splder(args, kwargs) (line 397)
        splder_call_result_74366 = invoke(stypy.reporting.localization.Localization(__file__, 397, 14), splder_74357, *[tuple_74358, nu_74364], **kwargs_74365)
        
        # Assigning a type to the variable 'tck' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'tck', splder_call_result_74366)
        
        # Call to construct_fast(...): (line 398)
        # Getting the type of 'tck' (line 398)
        tck_74369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 36), 'tck', False)
        # Processing the call keyword arguments (line 398)
        # Getting the type of 'self' (line 398)
        self_74370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 53), 'self', False)
        # Obtaining the member 'extrapolate' of a type (line 398)
        extrapolate_74371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 53), self_74370, 'extrapolate')
        keyword_74372 = extrapolate_74371
        # Getting the type of 'self' (line 399)
        self_74373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 41), 'self', False)
        # Obtaining the member 'axis' of a type (line 399)
        axis_74374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 41), self_74373, 'axis')
        keyword_74375 = axis_74374
        kwargs_74376 = {'extrapolate': keyword_74372, 'axis': keyword_74375}
        # Getting the type of 'self' (line 398)
        self_74367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'self', False)
        # Obtaining the member 'construct_fast' of a type (line 398)
        construct_fast_74368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 15), self_74367, 'construct_fast')
        # Calling construct_fast(args, kwargs) (line 398)
        construct_fast_call_result_74377 = invoke(stypy.reporting.localization.Localization(__file__, 398, 15), construct_fast_74368, *[tck_74369], **kwargs_74376)
        
        # Assigning a type to the variable 'stypy_return_type' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'stypy_return_type', construct_fast_call_result_74377)
        
        # ################# End of 'derivative(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'derivative' in the type store
        # Getting the type of 'stypy_return_type' (line 373)
        stypy_return_type_74378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74378)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'derivative'
        return stypy_return_type_74378


    @norecursion
    def antiderivative(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_74379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 32), 'int')
        defaults = [int_74379]
        # Create a new context for function 'antiderivative'
        module_type_store = module_type_store.open_function_context('antiderivative', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline.antiderivative.__dict__.__setitem__('stypy_localization', localization)
        BSpline.antiderivative.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline.antiderivative.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline.antiderivative.__dict__.__setitem__('stypy_function_name', 'BSpline.antiderivative')
        BSpline.antiderivative.__dict__.__setitem__('stypy_param_names_list', ['nu'])
        BSpline.antiderivative.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline.antiderivative.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline.antiderivative.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline.antiderivative.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline.antiderivative.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline.antiderivative.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline.antiderivative', ['nu'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'antiderivative', localization, ['nu'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'antiderivative(...)' code ##################

        str_74380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, (-1)), 'str', "Return a b-spline representing the antiderivative.\n\n        Parameters\n        ----------\n        nu : int, optional\n            Antiderivative order. Default is 1.\n\n        Returns\n        -------\n        b : BSpline object\n            A new instance representing the antiderivative.\n\n        Notes\n        -----\n        If antiderivative is computed and ``self.extrapolate='periodic'``,\n        it will be set to False for the returned instance. This is done because\n        the antiderivative is no longer periodic and its correct evaluation\n        outside of the initially given x interval is difficult.\n\n        See Also\n        --------\n        splder, splantider\n\n        ")
        
        # Assigning a Attribute to a Name (line 426):
        
        # Assigning a Attribute to a Name (line 426):
        # Getting the type of 'self' (line 426)
        self_74381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'self')
        # Obtaining the member 'c' of a type (line 426)
        c_74382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), self_74381, 'c')
        # Assigning a type to the variable 'c' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'c', c_74382)
        
        # Assigning a BinOp to a Name (line 428):
        
        # Assigning a BinOp to a Name (line 428):
        
        # Call to len(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'self' (line 428)
        self_74384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 17), 'self', False)
        # Obtaining the member 't' of a type (line 428)
        t_74385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 17), self_74384, 't')
        # Processing the call keyword arguments (line 428)
        kwargs_74386 = {}
        # Getting the type of 'len' (line 428)
        len_74383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 13), 'len', False)
        # Calling len(args, kwargs) (line 428)
        len_call_result_74387 = invoke(stypy.reporting.localization.Localization(__file__, 428, 13), len_74383, *[t_74385], **kwargs_74386)
        
        
        # Call to len(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'c' (line 428)
        c_74389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 31), 'c', False)
        # Processing the call keyword arguments (line 428)
        kwargs_74390 = {}
        # Getting the type of 'len' (line 428)
        len_74388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 27), 'len', False)
        # Calling len(args, kwargs) (line 428)
        len_call_result_74391 = invoke(stypy.reporting.localization.Localization(__file__, 428, 27), len_74388, *[c_74389], **kwargs_74390)
        
        # Applying the binary operator '-' (line 428)
        result_sub_74392 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 13), '-', len_call_result_74387, len_call_result_74391)
        
        # Assigning a type to the variable 'ct' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'ct', result_sub_74392)
        
        
        # Getting the type of 'ct' (line 429)
        ct_74393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 11), 'ct')
        int_74394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 16), 'int')
        # Applying the binary operator '>' (line 429)
        result_gt_74395 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 11), '>', ct_74393, int_74394)
        
        # Testing the type of an if condition (line 429)
        if_condition_74396 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 429, 8), result_gt_74395)
        # Assigning a type to the variable 'if_condition_74396' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'if_condition_74396', if_condition_74396)
        # SSA begins for if statement (line 429)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 430):
        
        # Assigning a Subscript to a Name (line 430):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 430)
        tuple_74397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 430)
        # Adding element type (line 430)
        # Getting the type of 'c' (line 430)
        c_74398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 22), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 22), tuple_74397, c_74398)
        # Adding element type (line 430)
        
        # Call to zeros(...): (line 430)
        # Processing the call arguments (line 430)
        
        # Obtaining an instance of the builtin type 'tuple' (line 430)
        tuple_74401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 430)
        # Adding element type (line 430)
        # Getting the type of 'ct' (line 430)
        ct_74402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 35), 'ct', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 35), tuple_74401, ct_74402)
        
        
        # Obtaining the type of the subscript
        int_74403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 50), 'int')
        slice_74404 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 430, 42), int_74403, None, None)
        # Getting the type of 'c' (line 430)
        c_74405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 42), 'c', False)
        # Obtaining the member 'shape' of a type (line 430)
        shape_74406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 42), c_74405, 'shape')
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___74407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 42), shape_74406, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_74408 = invoke(stypy.reporting.localization.Localization(__file__, 430, 42), getitem___74407, slice_74404)
        
        # Applying the binary operator '+' (line 430)
        result_add_74409 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 34), '+', tuple_74401, subscript_call_result_74408)
        
        # Processing the call keyword arguments (line 430)
        kwargs_74410 = {}
        # Getting the type of 'np' (line 430)
        np_74399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 25), 'np', False)
        # Obtaining the member 'zeros' of a type (line 430)
        zeros_74400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 25), np_74399, 'zeros')
        # Calling zeros(args, kwargs) (line 430)
        zeros_call_result_74411 = invoke(stypy.reporting.localization.Localization(__file__, 430, 25), zeros_74400, *[result_add_74409], **kwargs_74410)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 22), tuple_74397, zeros_call_result_74411)
        
        # Getting the type of 'np' (line 430)
        np_74412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'np')
        # Obtaining the member 'r_' of a type (line 430)
        r__74413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 16), np_74412, 'r_')
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___74414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 16), r__74413, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_74415 = invoke(stypy.reporting.localization.Localization(__file__, 430, 16), getitem___74414, tuple_74397)
        
        # Assigning a type to the variable 'c' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'c', subscript_call_result_74415)
        # SSA join for if statement (line 429)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to splantider(...): (line 431)
        # Processing the call arguments (line 431)
        
        # Obtaining an instance of the builtin type 'tuple' (line 431)
        tuple_74418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 431)
        # Adding element type (line 431)
        # Getting the type of 'self' (line 431)
        self_74419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 40), 'self', False)
        # Obtaining the member 't' of a type (line 431)
        t_74420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 40), self_74419, 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 40), tuple_74418, t_74420)
        # Adding element type (line 431)
        # Getting the type of 'c' (line 431)
        c_74421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 48), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 40), tuple_74418, c_74421)
        # Adding element type (line 431)
        # Getting the type of 'self' (line 431)
        self_74422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 51), 'self', False)
        # Obtaining the member 'k' of a type (line 431)
        k_74423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 51), self_74422, 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 40), tuple_74418, k_74423)
        
        # Getting the type of 'nu' (line 431)
        nu_74424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 60), 'nu', False)
        # Processing the call keyword arguments (line 431)
        kwargs_74425 = {}
        # Getting the type of '_fitpack_impl' (line 431)
        _fitpack_impl_74416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 14), '_fitpack_impl', False)
        # Obtaining the member 'splantider' of a type (line 431)
        splantider_74417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 14), _fitpack_impl_74416, 'splantider')
        # Calling splantider(args, kwargs) (line 431)
        splantider_call_result_74426 = invoke(stypy.reporting.localization.Localization(__file__, 431, 14), splantider_74417, *[tuple_74418, nu_74424], **kwargs_74425)
        
        # Assigning a type to the variable 'tck' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'tck', splantider_call_result_74426)
        
        
        # Getting the type of 'self' (line 433)
        self_74427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'self')
        # Obtaining the member 'extrapolate' of a type (line 433)
        extrapolate_74428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 11), self_74427, 'extrapolate')
        str_74429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 31), 'str', 'periodic')
        # Applying the binary operator '==' (line 433)
        result_eq_74430 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 11), '==', extrapolate_74428, str_74429)
        
        # Testing the type of an if condition (line 433)
        if_condition_74431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 8), result_eq_74430)
        # Assigning a type to the variable 'if_condition_74431' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'if_condition_74431', if_condition_74431)
        # SSA begins for if statement (line 433)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 434):
        
        # Assigning a Name to a Name (line 434):
        # Getting the type of 'False' (line 434)
        False_74432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 26), 'False')
        # Assigning a type to the variable 'extrapolate' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'extrapolate', False_74432)
        # SSA branch for the else part of an if statement (line 433)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 436):
        
        # Assigning a Attribute to a Name (line 436):
        # Getting the type of 'self' (line 436)
        self_74433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 26), 'self')
        # Obtaining the member 'extrapolate' of a type (line 436)
        extrapolate_74434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 26), self_74433, 'extrapolate')
        # Assigning a type to the variable 'extrapolate' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'extrapolate', extrapolate_74434)
        # SSA join for if statement (line 433)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to construct_fast(...): (line 438)
        # Getting the type of 'tck' (line 438)
        tck_74437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 36), 'tck', False)
        # Processing the call keyword arguments (line 438)
        # Getting the type of 'extrapolate' (line 438)
        extrapolate_74438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 53), 'extrapolate', False)
        keyword_74439 = extrapolate_74438
        # Getting the type of 'self' (line 439)
        self_74440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 40), 'self', False)
        # Obtaining the member 'axis' of a type (line 439)
        axis_74441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 40), self_74440, 'axis')
        keyword_74442 = axis_74441
        kwargs_74443 = {'extrapolate': keyword_74439, 'axis': keyword_74442}
        # Getting the type of 'self' (line 438)
        self_74435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'self', False)
        # Obtaining the member 'construct_fast' of a type (line 438)
        construct_fast_74436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 15), self_74435, 'construct_fast')
        # Calling construct_fast(args, kwargs) (line 438)
        construct_fast_call_result_74444 = invoke(stypy.reporting.localization.Localization(__file__, 438, 15), construct_fast_74436, *[tck_74437], **kwargs_74443)
        
        # Assigning a type to the variable 'stypy_return_type' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'stypy_return_type', construct_fast_call_result_74444)
        
        # ################# End of 'antiderivative(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'antiderivative' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_74445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74445)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'antiderivative'
        return stypy_return_type_74445


    @norecursion
    def integrate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 441)
        None_74446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 42), 'None')
        defaults = [None_74446]
        # Create a new context for function 'integrate'
        module_type_store = module_type_store.open_function_context('integrate', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BSpline.integrate.__dict__.__setitem__('stypy_localization', localization)
        BSpline.integrate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BSpline.integrate.__dict__.__setitem__('stypy_type_store', module_type_store)
        BSpline.integrate.__dict__.__setitem__('stypy_function_name', 'BSpline.integrate')
        BSpline.integrate.__dict__.__setitem__('stypy_param_names_list', ['a', 'b', 'extrapolate'])
        BSpline.integrate.__dict__.__setitem__('stypy_varargs_param_name', None)
        BSpline.integrate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BSpline.integrate.__dict__.__setitem__('stypy_call_defaults', defaults)
        BSpline.integrate.__dict__.__setitem__('stypy_call_varargs', varargs)
        BSpline.integrate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BSpline.integrate.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BSpline.integrate', ['a', 'b', 'extrapolate'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate', localization, ['a', 'b', 'extrapolate'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate(...)' code ##################

        str_74447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, (-1)), 'str', "Compute a definite integral of the spline.\n\n        Parameters\n        ----------\n        a : float\n            Lower limit of integration.\n        b : float\n            Upper limit of integration.\n        extrapolate : bool or 'periodic', optional\n            whether to extrapolate beyond the base interval,\n            ``t[k] .. t[-k-1]``, or take the spline to be zero outside of the\n            base interval. If 'periodic', periodic extrapolation is used.\n            If None (default), use `self.extrapolate`.\n\n        Returns\n        -------\n        I : array_like\n            Definite integral of the spline over the interval ``[a, b]``.\n\n        Examples\n        --------\n        Construct the linear spline ``x if x < 1 else 2 - x`` on the base\n        interval :math:`[0, 2]`, and integrate it\n\n        >>> from scipy.interpolate import BSpline\n        >>> b = BSpline.basis_element([0, 1, 2])\n        >>> b.integrate(0, 1)\n        array(0.5)\n\n        If the integration limits are outside of the base interval, the result\n        is controlled by the `extrapolate` parameter\n\n        >>> b.integrate(-1, 1)\n        array(0.0)\n        >>> b.integrate(-1, 1, extrapolate=False)\n        array(0.5)\n\n        >>> import matplotlib.pyplot as plt\n        >>> fig, ax = plt.subplots()\n        >>> ax.grid(True)\n        >>> ax.axvline(0, c='r', lw=5, alpha=0.5)  # base interval\n        >>> ax.axvline(2, c='r', lw=5, alpha=0.5)\n        >>> xx = [-1, 1, 2]\n        >>> ax.plot(xx, b(xx))\n        >>> plt.show()\n\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 489)
        # Getting the type of 'extrapolate' (line 489)
        extrapolate_74448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 11), 'extrapolate')
        # Getting the type of 'None' (line 489)
        None_74449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 26), 'None')
        
        (may_be_74450, more_types_in_union_74451) = may_be_none(extrapolate_74448, None_74449)

        if may_be_74450:

            if more_types_in_union_74451:
                # Runtime conditional SSA (line 489)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 490):
            
            # Assigning a Attribute to a Name (line 490):
            # Getting the type of 'self' (line 490)
            self_74452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 26), 'self')
            # Obtaining the member 'extrapolate' of a type (line 490)
            extrapolate_74453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 26), self_74452, 'extrapolate')
            # Assigning a type to the variable 'extrapolate' (line 490)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'extrapolate', extrapolate_74453)

            if more_types_in_union_74451:
                # SSA join for if statement (line 489)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _ensure_c_contiguous(...): (line 493)
        # Processing the call keyword arguments (line 493)
        kwargs_74456 = {}
        # Getting the type of 'self' (line 493)
        self_74454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'self', False)
        # Obtaining the member '_ensure_c_contiguous' of a type (line 493)
        _ensure_c_contiguous_74455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), self_74454, '_ensure_c_contiguous')
        # Calling _ensure_c_contiguous(args, kwargs) (line 493)
        _ensure_c_contiguous_call_result_74457 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), _ensure_c_contiguous_74455, *[], **kwargs_74456)
        
        
        # Assigning a Num to a Name (line 496):
        
        # Assigning a Num to a Name (line 496):
        int_74458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 15), 'int')
        # Assigning a type to the variable 'sign' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'sign', int_74458)
        
        
        # Getting the type of 'b' (line 497)
        b_74459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 11), 'b')
        # Getting the type of 'a' (line 497)
        a_74460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), 'a')
        # Applying the binary operator '<' (line 497)
        result_lt_74461 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 11), '<', b_74459, a_74460)
        
        # Testing the type of an if condition (line 497)
        if_condition_74462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 8), result_lt_74461)
        # Assigning a type to the variable 'if_condition_74462' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'if_condition_74462', if_condition_74462)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 498):
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'b' (line 498)
        b_74463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 19), 'b')
        # Assigning a type to the variable 'tuple_assignment_73676' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'tuple_assignment_73676', b_74463)
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'a' (line 498)
        a_74464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 22), 'a')
        # Assigning a type to the variable 'tuple_assignment_73677' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'tuple_assignment_73677', a_74464)
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'tuple_assignment_73676' (line 498)
        tuple_assignment_73676_74465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'tuple_assignment_73676')
        # Assigning a type to the variable 'a' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'a', tuple_assignment_73676_74465)
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'tuple_assignment_73677' (line 498)
        tuple_assignment_73677_74466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'tuple_assignment_73677')
        # Assigning a type to the variable 'b' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 15), 'b', tuple_assignment_73677_74466)
        
        # Assigning a Num to a Name (line 499):
        
        # Assigning a Num to a Name (line 499):
        int_74467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 19), 'int')
        # Assigning a type to the variable 'sign' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'sign', int_74467)
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 500):
        
        # Assigning a BinOp to a Name (line 500):
        # Getting the type of 'self' (line 500)
        self_74468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'self')
        # Obtaining the member 't' of a type (line 500)
        t_74469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), self_74468, 't')
        # Obtaining the member 'size' of a type (line 500)
        size_74470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), t_74469, 'size')
        # Getting the type of 'self' (line 500)
        self_74471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 26), 'self')
        # Obtaining the member 'k' of a type (line 500)
        k_74472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 26), self_74471, 'k')
        # Applying the binary operator '-' (line 500)
        result_sub_74473 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 12), '-', size_74470, k_74472)
        
        int_74474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 35), 'int')
        # Applying the binary operator '-' (line 500)
        result_sub_74475 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 33), '-', result_sub_74473, int_74474)
        
        # Assigning a type to the variable 'n' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'n', result_sub_74475)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'extrapolate' (line 502)
        extrapolate_74476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'extrapolate')
        str_74477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 26), 'str', 'periodic')
        # Applying the binary operator '!=' (line 502)
        result_ne_74478 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 11), '!=', extrapolate_74476, str_74477)
        
        
        # Getting the type of 'extrapolate' (line 502)
        extrapolate_74479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 45), 'extrapolate')
        # Applying the 'not' unary operator (line 502)
        result_not__74480 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 41), 'not', extrapolate_74479)
        
        # Applying the binary operator 'and' (line 502)
        result_and_keyword_74481 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 11), 'and', result_ne_74478, result_not__74480)
        
        # Testing the type of an if condition (line 502)
        if_condition_74482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 8), result_and_keyword_74481)
        # Assigning a type to the variable 'if_condition_74482' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'if_condition_74482', if_condition_74482)
        # SSA begins for if statement (line 502)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 504):
        
        # Assigning a Call to a Name (line 504):
        
        # Call to max(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'a' (line 504)
        a_74484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 20), 'a', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 504)
        self_74485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 30), 'self', False)
        # Obtaining the member 'k' of a type (line 504)
        k_74486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 30), self_74485, 'k')
        # Getting the type of 'self' (line 504)
        self_74487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 23), 'self', False)
        # Obtaining the member 't' of a type (line 504)
        t_74488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 23), self_74487, 't')
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___74489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 23), t_74488, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_74490 = invoke(stypy.reporting.localization.Localization(__file__, 504, 23), getitem___74489, k_74486)
        
        # Processing the call keyword arguments (line 504)
        kwargs_74491 = {}
        # Getting the type of 'max' (line 504)
        max_74483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 16), 'max', False)
        # Calling max(args, kwargs) (line 504)
        max_call_result_74492 = invoke(stypy.reporting.localization.Localization(__file__, 504, 16), max_74483, *[a_74484, subscript_call_result_74490], **kwargs_74491)
        
        # Assigning a type to the variable 'a' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'a', max_call_result_74492)
        
        # Assigning a Call to a Name (line 505):
        
        # Assigning a Call to a Name (line 505):
        
        # Call to min(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'b' (line 505)
        b_74494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 20), 'b', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 505)
        n_74495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 30), 'n', False)
        # Getting the type of 'self' (line 505)
        self_74496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 23), 'self', False)
        # Obtaining the member 't' of a type (line 505)
        t_74497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 23), self_74496, 't')
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___74498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 23), t_74497, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_74499 = invoke(stypy.reporting.localization.Localization(__file__, 505, 23), getitem___74498, n_74495)
        
        # Processing the call keyword arguments (line 505)
        kwargs_74500 = {}
        # Getting the type of 'min' (line 505)
        min_74493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 'min', False)
        # Calling min(args, kwargs) (line 505)
        min_call_result_74501 = invoke(stypy.reporting.localization.Localization(__file__, 505, 16), min_74493, *[b_74494, subscript_call_result_74499], **kwargs_74500)
        
        # Assigning a type to the variable 'b' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'b', min_call_result_74501)
        
        
        # Getting the type of 'self' (line 507)
        self_74502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 15), 'self')
        # Obtaining the member 'c' of a type (line 507)
        c_74503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 15), self_74502, 'c')
        # Obtaining the member 'ndim' of a type (line 507)
        ndim_74504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 15), c_74503, 'ndim')
        int_74505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 30), 'int')
        # Applying the binary operator '==' (line 507)
        result_eq_74506 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 15), '==', ndim_74504, int_74505)
        
        # Testing the type of an if condition (line 507)
        if_condition_74507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 12), result_eq_74506)
        # Assigning a type to the variable 'if_condition_74507' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'if_condition_74507', if_condition_74507)
        # SSA begins for if statement (line 507)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 510):
        
        # Assigning a Subscript to a Name (line 510):
        
        # Obtaining the type of the subscript
        int_74508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 16), 'int')
        # Getting the type of 'self' (line 510)
        self_74509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'self')
        # Obtaining the member 'tck' of a type (line 510)
        tck_74510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 26), self_74509, 'tck')
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___74511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), tck_74510, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_74512 = invoke(stypy.reporting.localization.Localization(__file__, 510, 16), getitem___74511, int_74508)
        
        # Assigning a type to the variable 'tuple_var_assignment_73678' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'tuple_var_assignment_73678', subscript_call_result_74512)
        
        # Assigning a Subscript to a Name (line 510):
        
        # Obtaining the type of the subscript
        int_74513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 16), 'int')
        # Getting the type of 'self' (line 510)
        self_74514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'self')
        # Obtaining the member 'tck' of a type (line 510)
        tck_74515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 26), self_74514, 'tck')
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___74516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), tck_74515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_74517 = invoke(stypy.reporting.localization.Localization(__file__, 510, 16), getitem___74516, int_74513)
        
        # Assigning a type to the variable 'tuple_var_assignment_73679' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'tuple_var_assignment_73679', subscript_call_result_74517)
        
        # Assigning a Subscript to a Name (line 510):
        
        # Obtaining the type of the subscript
        int_74518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 16), 'int')
        # Getting the type of 'self' (line 510)
        self_74519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'self')
        # Obtaining the member 'tck' of a type (line 510)
        tck_74520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 26), self_74519, 'tck')
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___74521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), tck_74520, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_74522 = invoke(stypy.reporting.localization.Localization(__file__, 510, 16), getitem___74521, int_74518)
        
        # Assigning a type to the variable 'tuple_var_assignment_73680' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'tuple_var_assignment_73680', subscript_call_result_74522)
        
        # Assigning a Name to a Name (line 510):
        # Getting the type of 'tuple_var_assignment_73678' (line 510)
        tuple_var_assignment_73678_74523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'tuple_var_assignment_73678')
        # Assigning a type to the variable 't' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 't', tuple_var_assignment_73678_74523)
        
        # Assigning a Name to a Name (line 510):
        # Getting the type of 'tuple_var_assignment_73679' (line 510)
        tuple_var_assignment_73679_74524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'tuple_var_assignment_73679')
        # Assigning a type to the variable 'c' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 19), 'c', tuple_var_assignment_73679_74524)
        
        # Assigning a Name to a Name (line 510):
        # Getting the type of 'tuple_var_assignment_73680' (line 510)
        tuple_var_assignment_73680_74525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'tuple_var_assignment_73680')
        # Assigning a type to the variable 'k' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 22), 'k', tuple_var_assignment_73680_74525)
        
        # Assigning a Call to a Tuple (line 511):
        
        # Assigning a Subscript to a Name (line 511):
        
        # Obtaining the type of the subscript
        int_74526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 16), 'int')
        
        # Call to _splint(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 't' (line 511)
        t_74529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 49), 't', False)
        # Getting the type of 'c' (line 511)
        c_74530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 52), 'c', False)
        # Getting the type of 'k' (line 511)
        k_74531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 55), 'k', False)
        # Getting the type of 'a' (line 511)
        a_74532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 58), 'a', False)
        # Getting the type of 'b' (line 511)
        b_74533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 61), 'b', False)
        # Processing the call keyword arguments (line 511)
        kwargs_74534 = {}
        # Getting the type of '_dierckx' (line 511)
        _dierckx_74527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 32), '_dierckx', False)
        # Obtaining the member '_splint' of a type (line 511)
        _splint_74528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 32), _dierckx_74527, '_splint')
        # Calling _splint(args, kwargs) (line 511)
        _splint_call_result_74535 = invoke(stypy.reporting.localization.Localization(__file__, 511, 32), _splint_74528, *[t_74529, c_74530, k_74531, a_74532, b_74533], **kwargs_74534)
        
        # Obtaining the member '__getitem__' of a type (line 511)
        getitem___74536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 16), _splint_call_result_74535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 511)
        subscript_call_result_74537 = invoke(stypy.reporting.localization.Localization(__file__, 511, 16), getitem___74536, int_74526)
        
        # Assigning a type to the variable 'tuple_var_assignment_73681' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 16), 'tuple_var_assignment_73681', subscript_call_result_74537)
        
        # Assigning a Subscript to a Name (line 511):
        
        # Obtaining the type of the subscript
        int_74538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 16), 'int')
        
        # Call to _splint(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 't' (line 511)
        t_74541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 49), 't', False)
        # Getting the type of 'c' (line 511)
        c_74542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 52), 'c', False)
        # Getting the type of 'k' (line 511)
        k_74543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 55), 'k', False)
        # Getting the type of 'a' (line 511)
        a_74544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 58), 'a', False)
        # Getting the type of 'b' (line 511)
        b_74545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 61), 'b', False)
        # Processing the call keyword arguments (line 511)
        kwargs_74546 = {}
        # Getting the type of '_dierckx' (line 511)
        _dierckx_74539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 32), '_dierckx', False)
        # Obtaining the member '_splint' of a type (line 511)
        _splint_74540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 32), _dierckx_74539, '_splint')
        # Calling _splint(args, kwargs) (line 511)
        _splint_call_result_74547 = invoke(stypy.reporting.localization.Localization(__file__, 511, 32), _splint_74540, *[t_74541, c_74542, k_74543, a_74544, b_74545], **kwargs_74546)
        
        # Obtaining the member '__getitem__' of a type (line 511)
        getitem___74548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 16), _splint_call_result_74547, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 511)
        subscript_call_result_74549 = invoke(stypy.reporting.localization.Localization(__file__, 511, 16), getitem___74548, int_74538)
        
        # Assigning a type to the variable 'tuple_var_assignment_73682' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 16), 'tuple_var_assignment_73682', subscript_call_result_74549)
        
        # Assigning a Name to a Name (line 511):
        # Getting the type of 'tuple_var_assignment_73681' (line 511)
        tuple_var_assignment_73681_74550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 16), 'tuple_var_assignment_73681')
        # Assigning a type to the variable 'integral' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 16), 'integral', tuple_var_assignment_73681_74550)
        
        # Assigning a Name to a Name (line 511):
        # Getting the type of 'tuple_var_assignment_73682' (line 511)
        tuple_var_assignment_73682_74551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 16), 'tuple_var_assignment_73682')
        # Assigning a type to the variable 'wrk' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 26), 'wrk', tuple_var_assignment_73682_74551)
        # Getting the type of 'integral' (line 512)
        integral_74552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 23), 'integral')
        # Getting the type of 'sign' (line 512)
        sign_74553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 34), 'sign')
        # Applying the binary operator '*' (line 512)
        result_mul_74554 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 23), '*', integral_74552, sign_74553)
        
        # Assigning a type to the variable 'stypy_return_type' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 16), 'stypy_return_type', result_mul_74554)
        # SSA join for if statement (line 507)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 502)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to empty(...): (line 514)
        # Processing the call arguments (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 514)
        tuple_74557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 514)
        # Adding element type (line 514)
        int_74558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 24), tuple_74557, int_74558)
        # Adding element type (line 514)
        
        # Call to prod(...): (line 514)
        # Processing the call arguments (line 514)
        
        # Obtaining the type of the subscript
        int_74560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 45), 'int')
        slice_74561 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 514, 32), int_74560, None, None)
        # Getting the type of 'self' (line 514)
        self_74562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 32), 'self', False)
        # Obtaining the member 'c' of a type (line 514)
        c_74563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 32), self_74562, 'c')
        # Obtaining the member 'shape' of a type (line 514)
        shape_74564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 32), c_74563, 'shape')
        # Obtaining the member '__getitem__' of a type (line 514)
        getitem___74565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 32), shape_74564, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 514)
        subscript_call_result_74566 = invoke(stypy.reporting.localization.Localization(__file__, 514, 32), getitem___74565, slice_74561)
        
        # Processing the call keyword arguments (line 514)
        kwargs_74567 = {}
        # Getting the type of 'prod' (line 514)
        prod_74559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 27), 'prod', False)
        # Calling prod(args, kwargs) (line 514)
        prod_call_result_74568 = invoke(stypy.reporting.localization.Localization(__file__, 514, 27), prod_74559, *[subscript_call_result_74566], **kwargs_74567)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 24), tuple_74557, prod_call_result_74568)
        
        # Processing the call keyword arguments (line 514)
        # Getting the type of 'self' (line 514)
        self_74569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 58), 'self', False)
        # Obtaining the member 'c' of a type (line 514)
        c_74570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 58), self_74569, 'c')
        # Obtaining the member 'dtype' of a type (line 514)
        dtype_74571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 58), c_74570, 'dtype')
        keyword_74572 = dtype_74571
        kwargs_74573 = {'dtype': keyword_74572}
        # Getting the type of 'np' (line 514)
        np_74555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 14), 'np', False)
        # Obtaining the member 'empty' of a type (line 514)
        empty_74556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 14), np_74555, 'empty')
        # Calling empty(args, kwargs) (line 514)
        empty_call_result_74574 = invoke(stypy.reporting.localization.Localization(__file__, 514, 14), empty_74556, *[tuple_74557], **kwargs_74573)
        
        # Assigning a type to the variable 'out' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'out', empty_call_result_74574)
        
        # Assigning a Attribute to a Name (line 517):
        
        # Assigning a Attribute to a Name (line 517):
        # Getting the type of 'self' (line 517)
        self_74575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'self')
        # Obtaining the member 'c' of a type (line 517)
        c_74576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 12), self_74575, 'c')
        # Assigning a type to the variable 'c' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'c', c_74576)
        
        # Assigning a BinOp to a Name (line 518):
        
        # Assigning a BinOp to a Name (line 518):
        
        # Call to len(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'self' (line 518)
        self_74578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 17), 'self', False)
        # Obtaining the member 't' of a type (line 518)
        t_74579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 17), self_74578, 't')
        # Processing the call keyword arguments (line 518)
        kwargs_74580 = {}
        # Getting the type of 'len' (line 518)
        len_74577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 13), 'len', False)
        # Calling len(args, kwargs) (line 518)
        len_call_result_74581 = invoke(stypy.reporting.localization.Localization(__file__, 518, 13), len_74577, *[t_74579], **kwargs_74580)
        
        
        # Call to len(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'c' (line 518)
        c_74583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 31), 'c', False)
        # Processing the call keyword arguments (line 518)
        kwargs_74584 = {}
        # Getting the type of 'len' (line 518)
        len_74582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 27), 'len', False)
        # Calling len(args, kwargs) (line 518)
        len_call_result_74585 = invoke(stypy.reporting.localization.Localization(__file__, 518, 27), len_74582, *[c_74583], **kwargs_74584)
        
        # Applying the binary operator '-' (line 518)
        result_sub_74586 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 13), '-', len_call_result_74581, len_call_result_74585)
        
        # Assigning a type to the variable 'ct' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'ct', result_sub_74586)
        
        
        # Getting the type of 'ct' (line 519)
        ct_74587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 11), 'ct')
        int_74588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 16), 'int')
        # Applying the binary operator '>' (line 519)
        result_gt_74589 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 11), '>', ct_74587, int_74588)
        
        # Testing the type of an if condition (line 519)
        if_condition_74590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 519, 8), result_gt_74589)
        # Assigning a type to the variable 'if_condition_74590' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'if_condition_74590', if_condition_74590)
        # SSA begins for if statement (line 519)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 520):
        
        # Assigning a Subscript to a Name (line 520):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 520)
        tuple_74591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 520)
        # Adding element type (line 520)
        # Getting the type of 'c' (line 520)
        c_74592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 22), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 22), tuple_74591, c_74592)
        # Adding element type (line 520)
        
        # Call to zeros(...): (line 520)
        # Processing the call arguments (line 520)
        
        # Obtaining an instance of the builtin type 'tuple' (line 520)
        tuple_74595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 520)
        # Adding element type (line 520)
        # Getting the type of 'ct' (line 520)
        ct_74596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 35), 'ct', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 35), tuple_74595, ct_74596)
        
        
        # Obtaining the type of the subscript
        int_74597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 50), 'int')
        slice_74598 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 520, 42), int_74597, None, None)
        # Getting the type of 'c' (line 520)
        c_74599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 42), 'c', False)
        # Obtaining the member 'shape' of a type (line 520)
        shape_74600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 42), c_74599, 'shape')
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___74601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 42), shape_74600, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 520)
        subscript_call_result_74602 = invoke(stypy.reporting.localization.Localization(__file__, 520, 42), getitem___74601, slice_74598)
        
        # Applying the binary operator '+' (line 520)
        result_add_74603 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 34), '+', tuple_74595, subscript_call_result_74602)
        
        # Processing the call keyword arguments (line 520)
        kwargs_74604 = {}
        # Getting the type of 'np' (line 520)
        np_74593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 25), 'np', False)
        # Obtaining the member 'zeros' of a type (line 520)
        zeros_74594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 25), np_74593, 'zeros')
        # Calling zeros(args, kwargs) (line 520)
        zeros_call_result_74605 = invoke(stypy.reporting.localization.Localization(__file__, 520, 25), zeros_74594, *[result_add_74603], **kwargs_74604)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 22), tuple_74591, zeros_call_result_74605)
        
        # Getting the type of 'np' (line 520)
        np_74606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'np')
        # Obtaining the member 'r_' of a type (line 520)
        r__74607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 16), np_74606, 'r_')
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___74608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 16), r__74607, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 520)
        subscript_call_result_74609 = invoke(stypy.reporting.localization.Localization(__file__, 520, 16), getitem___74608, tuple_74591)
        
        # Assigning a type to the variable 'c' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'c', subscript_call_result_74609)
        # SSA join for if statement (line 519)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 521):
        
        # Assigning a Subscript to a Name (line 521):
        
        # Obtaining the type of the subscript
        int_74610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 8), 'int')
        
        # Call to splantider(...): (line 521)
        # Processing the call arguments (line 521)
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_74613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        # Getting the type of 'self' (line 521)
        self_74614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 47), 'self', False)
        # Obtaining the member 't' of a type (line 521)
        t_74615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 47), self_74614, 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74613, t_74615)
        # Adding element type (line 521)
        # Getting the type of 'c' (line 521)
        c_74616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 55), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74613, c_74616)
        # Adding element type (line 521)
        # Getting the type of 'self' (line 521)
        self_74617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 58), 'self', False)
        # Obtaining the member 'k' of a type (line 521)
        k_74618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 58), self_74617, 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74613, k_74618)
        
        int_74619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 67), 'int')
        # Processing the call keyword arguments (line 521)
        kwargs_74620 = {}
        # Getting the type of '_fitpack_impl' (line 521)
        _fitpack_impl_74611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 21), '_fitpack_impl', False)
        # Obtaining the member 'splantider' of a type (line 521)
        splantider_74612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 21), _fitpack_impl_74611, 'splantider')
        # Calling splantider(args, kwargs) (line 521)
        splantider_call_result_74621 = invoke(stypy.reporting.localization.Localization(__file__, 521, 21), splantider_74612, *[tuple_74613, int_74619], **kwargs_74620)
        
        # Obtaining the member '__getitem__' of a type (line 521)
        getitem___74622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 8), splantider_call_result_74621, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 521)
        subscript_call_result_74623 = invoke(stypy.reporting.localization.Localization(__file__, 521, 8), getitem___74622, int_74610)
        
        # Assigning a type to the variable 'tuple_var_assignment_73683' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'tuple_var_assignment_73683', subscript_call_result_74623)
        
        # Assigning a Subscript to a Name (line 521):
        
        # Obtaining the type of the subscript
        int_74624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 8), 'int')
        
        # Call to splantider(...): (line 521)
        # Processing the call arguments (line 521)
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_74627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        # Getting the type of 'self' (line 521)
        self_74628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 47), 'self', False)
        # Obtaining the member 't' of a type (line 521)
        t_74629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 47), self_74628, 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74627, t_74629)
        # Adding element type (line 521)
        # Getting the type of 'c' (line 521)
        c_74630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 55), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74627, c_74630)
        # Adding element type (line 521)
        # Getting the type of 'self' (line 521)
        self_74631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 58), 'self', False)
        # Obtaining the member 'k' of a type (line 521)
        k_74632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 58), self_74631, 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74627, k_74632)
        
        int_74633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 67), 'int')
        # Processing the call keyword arguments (line 521)
        kwargs_74634 = {}
        # Getting the type of '_fitpack_impl' (line 521)
        _fitpack_impl_74625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 21), '_fitpack_impl', False)
        # Obtaining the member 'splantider' of a type (line 521)
        splantider_74626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 21), _fitpack_impl_74625, 'splantider')
        # Calling splantider(args, kwargs) (line 521)
        splantider_call_result_74635 = invoke(stypy.reporting.localization.Localization(__file__, 521, 21), splantider_74626, *[tuple_74627, int_74633], **kwargs_74634)
        
        # Obtaining the member '__getitem__' of a type (line 521)
        getitem___74636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 8), splantider_call_result_74635, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 521)
        subscript_call_result_74637 = invoke(stypy.reporting.localization.Localization(__file__, 521, 8), getitem___74636, int_74624)
        
        # Assigning a type to the variable 'tuple_var_assignment_73684' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'tuple_var_assignment_73684', subscript_call_result_74637)
        
        # Assigning a Subscript to a Name (line 521):
        
        # Obtaining the type of the subscript
        int_74638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 8), 'int')
        
        # Call to splantider(...): (line 521)
        # Processing the call arguments (line 521)
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_74641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        # Getting the type of 'self' (line 521)
        self_74642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 47), 'self', False)
        # Obtaining the member 't' of a type (line 521)
        t_74643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 47), self_74642, 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74641, t_74643)
        # Adding element type (line 521)
        # Getting the type of 'c' (line 521)
        c_74644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 55), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74641, c_74644)
        # Adding element type (line 521)
        # Getting the type of 'self' (line 521)
        self_74645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 58), 'self', False)
        # Obtaining the member 'k' of a type (line 521)
        k_74646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 58), self_74645, 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 47), tuple_74641, k_74646)
        
        int_74647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 67), 'int')
        # Processing the call keyword arguments (line 521)
        kwargs_74648 = {}
        # Getting the type of '_fitpack_impl' (line 521)
        _fitpack_impl_74639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 21), '_fitpack_impl', False)
        # Obtaining the member 'splantider' of a type (line 521)
        splantider_74640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 21), _fitpack_impl_74639, 'splantider')
        # Calling splantider(args, kwargs) (line 521)
        splantider_call_result_74649 = invoke(stypy.reporting.localization.Localization(__file__, 521, 21), splantider_74640, *[tuple_74641, int_74647], **kwargs_74648)
        
        # Obtaining the member '__getitem__' of a type (line 521)
        getitem___74650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 8), splantider_call_result_74649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 521)
        subscript_call_result_74651 = invoke(stypy.reporting.localization.Localization(__file__, 521, 8), getitem___74650, int_74638)
        
        # Assigning a type to the variable 'tuple_var_assignment_73685' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'tuple_var_assignment_73685', subscript_call_result_74651)
        
        # Assigning a Name to a Name (line 521):
        # Getting the type of 'tuple_var_assignment_73683' (line 521)
        tuple_var_assignment_73683_74652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'tuple_var_assignment_73683')
        # Assigning a type to the variable 'ta' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'ta', tuple_var_assignment_73683_74652)
        
        # Assigning a Name to a Name (line 521):
        # Getting the type of 'tuple_var_assignment_73684' (line 521)
        tuple_var_assignment_73684_74653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'tuple_var_assignment_73684')
        # Assigning a type to the variable 'ca' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'ca', tuple_var_assignment_73684_74653)
        
        # Assigning a Name to a Name (line 521):
        # Getting the type of 'tuple_var_assignment_73685' (line 521)
        tuple_var_assignment_73685_74654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'tuple_var_assignment_73685')
        # Assigning a type to the variable 'ka' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 16), 'ka', tuple_var_assignment_73685_74654)
        
        
        # Getting the type of 'extrapolate' (line 523)
        extrapolate_74655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'extrapolate')
        str_74656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 26), 'str', 'periodic')
        # Applying the binary operator '==' (line 523)
        result_eq_74657 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 11), '==', extrapolate_74655, str_74656)
        
        # Testing the type of an if condition (line 523)
        if_condition_74658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 523, 8), result_eq_74657)
        # Assigning a type to the variable 'if_condition_74658' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'if_condition_74658', if_condition_74658)
        # SSA begins for if statement (line 523)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 527):
        
        # Assigning a Subscript to a Name (line 527):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 527)
        self_74659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 28), 'self')
        # Obtaining the member 'k' of a type (line 527)
        k_74660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 28), self_74659, 'k')
        # Getting the type of 'self' (line 527)
        self_74661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 21), 'self')
        # Obtaining the member 't' of a type (line 527)
        t_74662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 21), self_74661, 't')
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___74663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 21), t_74662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_74664 = invoke(stypy.reporting.localization.Localization(__file__, 527, 21), getitem___74663, k_74660)
        
        # Assigning a type to the variable 'tuple_assignment_73686' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'tuple_assignment_73686', subscript_call_result_74664)
        
        # Assigning a Subscript to a Name (line 527):
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 527)
        n_74665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 44), 'n')
        # Getting the type of 'self' (line 527)
        self_74666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 37), 'self')
        # Obtaining the member 't' of a type (line 527)
        t_74667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 37), self_74666, 't')
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___74668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 37), t_74667, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_74669 = invoke(stypy.reporting.localization.Localization(__file__, 527, 37), getitem___74668, n_74665)
        
        # Assigning a type to the variable 'tuple_assignment_73687' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'tuple_assignment_73687', subscript_call_result_74669)
        
        # Assigning a Name to a Name (line 527):
        # Getting the type of 'tuple_assignment_73686' (line 527)
        tuple_assignment_73686_74670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'tuple_assignment_73686')
        # Assigning a type to the variable 'ts' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'ts', tuple_assignment_73686_74670)
        
        # Assigning a Name to a Name (line 527):
        # Getting the type of 'tuple_assignment_73687' (line 527)
        tuple_assignment_73687_74671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'tuple_assignment_73687')
        # Assigning a type to the variable 'te' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 16), 'te', tuple_assignment_73687_74671)
        
        # Assigning a BinOp to a Name (line 528):
        
        # Assigning a BinOp to a Name (line 528):
        # Getting the type of 'te' (line 528)
        te_74672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 21), 'te')
        # Getting the type of 'ts' (line 528)
        ts_74673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 26), 'ts')
        # Applying the binary operator '-' (line 528)
        result_sub_74674 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 21), '-', te_74672, ts_74673)
        
        # Assigning a type to the variable 'period' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'period', result_sub_74674)
        
        # Assigning a BinOp to a Name (line 529):
        
        # Assigning a BinOp to a Name (line 529):
        # Getting the type of 'b' (line 529)
        b_74675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 23), 'b')
        # Getting the type of 'a' (line 529)
        a_74676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 27), 'a')
        # Applying the binary operator '-' (line 529)
        result_sub_74677 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 23), '-', b_74675, a_74676)
        
        # Assigning a type to the variable 'interval' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'interval', result_sub_74677)
        
        # Assigning a Call to a Tuple (line 530):
        
        # Assigning a Subscript to a Name (line 530):
        
        # Obtaining the type of the subscript
        int_74678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 12), 'int')
        
        # Call to divmod(...): (line 530)
        # Processing the call arguments (line 530)
        # Getting the type of 'interval' (line 530)
        interval_74680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 37), 'interval', False)
        # Getting the type of 'period' (line 530)
        period_74681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 47), 'period', False)
        # Processing the call keyword arguments (line 530)
        kwargs_74682 = {}
        # Getting the type of 'divmod' (line 530)
        divmod_74679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 30), 'divmod', False)
        # Calling divmod(args, kwargs) (line 530)
        divmod_call_result_74683 = invoke(stypy.reporting.localization.Localization(__file__, 530, 30), divmod_74679, *[interval_74680, period_74681], **kwargs_74682)
        
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___74684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 12), divmod_call_result_74683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 530)
        subscript_call_result_74685 = invoke(stypy.reporting.localization.Localization(__file__, 530, 12), getitem___74684, int_74678)
        
        # Assigning a type to the variable 'tuple_var_assignment_73688' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'tuple_var_assignment_73688', subscript_call_result_74685)
        
        # Assigning a Subscript to a Name (line 530):
        
        # Obtaining the type of the subscript
        int_74686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 12), 'int')
        
        # Call to divmod(...): (line 530)
        # Processing the call arguments (line 530)
        # Getting the type of 'interval' (line 530)
        interval_74688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 37), 'interval', False)
        # Getting the type of 'period' (line 530)
        period_74689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 47), 'period', False)
        # Processing the call keyword arguments (line 530)
        kwargs_74690 = {}
        # Getting the type of 'divmod' (line 530)
        divmod_74687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 30), 'divmod', False)
        # Calling divmod(args, kwargs) (line 530)
        divmod_call_result_74691 = invoke(stypy.reporting.localization.Localization(__file__, 530, 30), divmod_74687, *[interval_74688, period_74689], **kwargs_74690)
        
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___74692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 12), divmod_call_result_74691, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 530)
        subscript_call_result_74693 = invoke(stypy.reporting.localization.Localization(__file__, 530, 12), getitem___74692, int_74686)
        
        # Assigning a type to the variable 'tuple_var_assignment_73689' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'tuple_var_assignment_73689', subscript_call_result_74693)
        
        # Assigning a Name to a Name (line 530):
        # Getting the type of 'tuple_var_assignment_73688' (line 530)
        tuple_var_assignment_73688_74694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'tuple_var_assignment_73688')
        # Assigning a type to the variable 'n_periods' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'n_periods', tuple_var_assignment_73688_74694)
        
        # Assigning a Name to a Name (line 530):
        # Getting the type of 'tuple_var_assignment_73689' (line 530)
        tuple_var_assignment_73689_74695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'tuple_var_assignment_73689')
        # Assigning a type to the variable 'left' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 23), 'left', tuple_var_assignment_73689_74695)
        
        
        # Getting the type of 'n_periods' (line 532)
        n_periods_74696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 15), 'n_periods')
        int_74697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 27), 'int')
        # Applying the binary operator '>' (line 532)
        result_gt_74698 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 15), '>', n_periods_74696, int_74697)
        
        # Testing the type of an if condition (line 532)
        if_condition_74699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 12), result_gt_74698)
        # Assigning a type to the variable 'if_condition_74699' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'if_condition_74699', if_condition_74699)
        # SSA begins for if statement (line 532)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 534):
        
        # Assigning a Call to a Name (line 534):
        
        # Call to asarray(...): (line 534)
        # Processing the call arguments (line 534)
        
        # Obtaining an instance of the builtin type 'list' (line 534)
        list_74702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 534)
        # Adding element type (line 534)
        # Getting the type of 'ts' (line 534)
        ts_74703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 32), 'ts', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 31), list_74702, ts_74703)
        # Adding element type (line 534)
        # Getting the type of 'te' (line 534)
        te_74704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 36), 'te', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 31), list_74702, te_74704)
        
        # Processing the call keyword arguments (line 534)
        # Getting the type of 'np' (line 534)
        np_74705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 47), 'np', False)
        # Obtaining the member 'float_' of a type (line 534)
        float__74706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 47), np_74705, 'float_')
        keyword_74707 = float__74706
        kwargs_74708 = {'dtype': keyword_74707}
        # Getting the type of 'np' (line 534)
        np_74700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 20), 'np', False)
        # Obtaining the member 'asarray' of a type (line 534)
        asarray_74701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 20), np_74700, 'asarray')
        # Calling asarray(args, kwargs) (line 534)
        asarray_call_result_74709 = invoke(stypy.reporting.localization.Localization(__file__, 534, 20), asarray_74701, *[list_74702], **kwargs_74708)
        
        # Assigning a type to the variable 'x' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'x', asarray_call_result_74709)
        
        # Call to evaluate_spline(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'ta' (line 535)
        ta_74712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 38), 'ta', False)
        
        # Call to reshape(...): (line 535)
        # Processing the call arguments (line 535)
        
        # Obtaining the type of the subscript
        int_74715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 62), 'int')
        # Getting the type of 'ca' (line 535)
        ca_74716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 53), 'ca', False)
        # Obtaining the member 'shape' of a type (line 535)
        shape_74717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 53), ca_74716, 'shape')
        # Obtaining the member '__getitem__' of a type (line 535)
        getitem___74718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 53), shape_74717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 535)
        subscript_call_result_74719 = invoke(stypy.reporting.localization.Localization(__file__, 535, 53), getitem___74718, int_74715)
        
        int_74720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 66), 'int')
        # Processing the call keyword arguments (line 535)
        kwargs_74721 = {}
        # Getting the type of 'ca' (line 535)
        ca_74713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 42), 'ca', False)
        # Obtaining the member 'reshape' of a type (line 535)
        reshape_74714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 42), ca_74713, 'reshape')
        # Calling reshape(args, kwargs) (line 535)
        reshape_call_result_74722 = invoke(stypy.reporting.localization.Localization(__file__, 535, 42), reshape_74714, *[subscript_call_result_74719, int_74720], **kwargs_74721)
        
        # Getting the type of 'ka' (line 536)
        ka_74723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 38), 'ka', False)
        # Getting the type of 'x' (line 536)
        x_74724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 42), 'x', False)
        int_74725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 45), 'int')
        # Getting the type of 'False' (line 536)
        False_74726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 48), 'False', False)
        # Getting the type of 'out' (line 536)
        out_74727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 55), 'out', False)
        # Processing the call keyword arguments (line 535)
        kwargs_74728 = {}
        # Getting the type of '_bspl' (line 535)
        _bspl_74710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 16), '_bspl', False)
        # Obtaining the member 'evaluate_spline' of a type (line 535)
        evaluate_spline_74711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 16), _bspl_74710, 'evaluate_spline')
        # Calling evaluate_spline(args, kwargs) (line 535)
        evaluate_spline_call_result_74729 = invoke(stypy.reporting.localization.Localization(__file__, 535, 16), evaluate_spline_74711, *[ta_74712, reshape_call_result_74722, ka_74723, x_74724, int_74725, False_74726, out_74727], **kwargs_74728)
        
        
        # Assigning a BinOp to a Name (line 537):
        
        # Assigning a BinOp to a Name (line 537):
        
        # Obtaining the type of the subscript
        int_74730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 31), 'int')
        # Getting the type of 'out' (line 537)
        out_74731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 27), 'out')
        # Obtaining the member '__getitem__' of a type (line 537)
        getitem___74732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 27), out_74731, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 537)
        subscript_call_result_74733 = invoke(stypy.reporting.localization.Localization(__file__, 537, 27), getitem___74732, int_74730)
        
        
        # Obtaining the type of the subscript
        int_74734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 40), 'int')
        # Getting the type of 'out' (line 537)
        out_74735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 36), 'out')
        # Obtaining the member '__getitem__' of a type (line 537)
        getitem___74736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 36), out_74735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 537)
        subscript_call_result_74737 = invoke(stypy.reporting.localization.Localization(__file__, 537, 36), getitem___74736, int_74734)
        
        # Applying the binary operator '-' (line 537)
        result_sub_74738 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 27), '-', subscript_call_result_74733, subscript_call_result_74737)
        
        # Assigning a type to the variable 'integral' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 16), 'integral', result_sub_74738)
        
        # Getting the type of 'integral' (line 538)
        integral_74739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 16), 'integral')
        # Getting the type of 'n_periods' (line 538)
        n_periods_74740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 28), 'n_periods')
        # Applying the binary operator '*=' (line 538)
        result_imul_74741 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 16), '*=', integral_74739, n_periods_74740)
        # Assigning a type to the variable 'integral' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 16), 'integral', result_imul_74741)
        
        # SSA branch for the else part of an if statement (line 532)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 540):
        
        # Assigning a Call to a Name (line 540):
        
        # Call to zeros(...): (line 540)
        # Processing the call arguments (line 540)
        
        # Obtaining an instance of the builtin type 'tuple' (line 540)
        tuple_74744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 540)
        # Adding element type (line 540)
        int_74745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 37), tuple_74744, int_74745)
        # Adding element type (line 540)
        
        # Call to prod(...): (line 540)
        # Processing the call arguments (line 540)
        
        # Obtaining the type of the subscript
        int_74747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 58), 'int')
        slice_74748 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 540, 45), int_74747, None, None)
        # Getting the type of 'self' (line 540)
        self_74749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 45), 'self', False)
        # Obtaining the member 'c' of a type (line 540)
        c_74750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 45), self_74749, 'c')
        # Obtaining the member 'shape' of a type (line 540)
        shape_74751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 45), c_74750, 'shape')
        # Obtaining the member '__getitem__' of a type (line 540)
        getitem___74752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 45), shape_74751, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 540)
        subscript_call_result_74753 = invoke(stypy.reporting.localization.Localization(__file__, 540, 45), getitem___74752, slice_74748)
        
        # Processing the call keyword arguments (line 540)
        kwargs_74754 = {}
        # Getting the type of 'prod' (line 540)
        prod_74746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 40), 'prod', False)
        # Calling prod(args, kwargs) (line 540)
        prod_call_result_74755 = invoke(stypy.reporting.localization.Localization(__file__, 540, 40), prod_74746, *[subscript_call_result_74753], **kwargs_74754)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 37), tuple_74744, prod_call_result_74755)
        
        # Processing the call keyword arguments (line 540)
        # Getting the type of 'self' (line 541)
        self_74756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 42), 'self', False)
        # Obtaining the member 'c' of a type (line 541)
        c_74757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 42), self_74756, 'c')
        # Obtaining the member 'dtype' of a type (line 541)
        dtype_74758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 42), c_74757, 'dtype')
        keyword_74759 = dtype_74758
        kwargs_74760 = {'dtype': keyword_74759}
        # Getting the type of 'np' (line 540)
        np_74742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'np', False)
        # Obtaining the member 'zeros' of a type (line 540)
        zeros_74743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 27), np_74742, 'zeros')
        # Calling zeros(args, kwargs) (line 540)
        zeros_call_result_74761 = invoke(stypy.reporting.localization.Localization(__file__, 540, 27), zeros_74743, *[tuple_74744], **kwargs_74760)
        
        # Assigning a type to the variable 'integral' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'integral', zeros_call_result_74761)
        # SSA join for if statement (line 532)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 544):
        
        # Assigning a BinOp to a Name (line 544):
        # Getting the type of 'ts' (line 544)
        ts_74762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'ts')
        # Getting the type of 'a' (line 544)
        a_74763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 22), 'a')
        # Getting the type of 'ts' (line 544)
        ts_74764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 26), 'ts')
        # Applying the binary operator '-' (line 544)
        result_sub_74765 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 22), '-', a_74763, ts_74764)
        
        # Getting the type of 'period' (line 544)
        period_74766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 32), 'period')
        # Applying the binary operator '%' (line 544)
        result_mod_74767 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 21), '%', result_sub_74765, period_74766)
        
        # Applying the binary operator '+' (line 544)
        result_add_74768 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 16), '+', ts_74762, result_mod_74767)
        
        # Assigning a type to the variable 'a' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'a', result_add_74768)
        
        # Assigning a BinOp to a Name (line 545):
        
        # Assigning a BinOp to a Name (line 545):
        # Getting the type of 'a' (line 545)
        a_74769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'a')
        # Getting the type of 'left' (line 545)
        left_74770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 20), 'left')
        # Applying the binary operator '+' (line 545)
        result_add_74771 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 16), '+', a_74769, left_74770)
        
        # Assigning a type to the variable 'b' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'b', result_add_74771)
        
        
        # Getting the type of 'b' (line 549)
        b_74772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 15), 'b')
        # Getting the type of 'te' (line 549)
        te_74773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'te')
        # Applying the binary operator '<=' (line 549)
        result_le_74774 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 15), '<=', b_74772, te_74773)
        
        # Testing the type of an if condition (line 549)
        if_condition_74775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 549, 12), result_le_74774)
        # Assigning a type to the variable 'if_condition_74775' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'if_condition_74775', if_condition_74775)
        # SSA begins for if statement (line 549)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 550):
        
        # Assigning a Call to a Name (line 550):
        
        # Call to asarray(...): (line 550)
        # Processing the call arguments (line 550)
        
        # Obtaining an instance of the builtin type 'list' (line 550)
        list_74778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 550)
        # Adding element type (line 550)
        # Getting the type of 'a' (line 550)
        a_74779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 32), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 31), list_74778, a_74779)
        # Adding element type (line 550)
        # Getting the type of 'b' (line 550)
        b_74780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 31), list_74778, b_74780)
        
        # Processing the call keyword arguments (line 550)
        # Getting the type of 'np' (line 550)
        np_74781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 45), 'np', False)
        # Obtaining the member 'float_' of a type (line 550)
        float__74782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 45), np_74781, 'float_')
        keyword_74783 = float__74782
        kwargs_74784 = {'dtype': keyword_74783}
        # Getting the type of 'np' (line 550)
        np_74776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 20), 'np', False)
        # Obtaining the member 'asarray' of a type (line 550)
        asarray_74777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 20), np_74776, 'asarray')
        # Calling asarray(args, kwargs) (line 550)
        asarray_call_result_74785 = invoke(stypy.reporting.localization.Localization(__file__, 550, 20), asarray_74777, *[list_74778], **kwargs_74784)
        
        # Assigning a type to the variable 'x' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 16), 'x', asarray_call_result_74785)
        
        # Call to evaluate_spline(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'ta' (line 551)
        ta_74788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 38), 'ta', False)
        
        # Call to reshape(...): (line 551)
        # Processing the call arguments (line 551)
        
        # Obtaining the type of the subscript
        int_74791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 62), 'int')
        # Getting the type of 'ca' (line 551)
        ca_74792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 53), 'ca', False)
        # Obtaining the member 'shape' of a type (line 551)
        shape_74793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 53), ca_74792, 'shape')
        # Obtaining the member '__getitem__' of a type (line 551)
        getitem___74794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 53), shape_74793, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 551)
        subscript_call_result_74795 = invoke(stypy.reporting.localization.Localization(__file__, 551, 53), getitem___74794, int_74791)
        
        int_74796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 66), 'int')
        # Processing the call keyword arguments (line 551)
        kwargs_74797 = {}
        # Getting the type of 'ca' (line 551)
        ca_74789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 42), 'ca', False)
        # Obtaining the member 'reshape' of a type (line 551)
        reshape_74790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 42), ca_74789, 'reshape')
        # Calling reshape(args, kwargs) (line 551)
        reshape_call_result_74798 = invoke(stypy.reporting.localization.Localization(__file__, 551, 42), reshape_74790, *[subscript_call_result_74795, int_74796], **kwargs_74797)
        
        # Getting the type of 'ka' (line 552)
        ka_74799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 38), 'ka', False)
        # Getting the type of 'x' (line 552)
        x_74800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 42), 'x', False)
        int_74801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 45), 'int')
        # Getting the type of 'False' (line 552)
        False_74802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 48), 'False', False)
        # Getting the type of 'out' (line 552)
        out_74803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 55), 'out', False)
        # Processing the call keyword arguments (line 551)
        kwargs_74804 = {}
        # Getting the type of '_bspl' (line 551)
        _bspl_74786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), '_bspl', False)
        # Obtaining the member 'evaluate_spline' of a type (line 551)
        evaluate_spline_74787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 16), _bspl_74786, 'evaluate_spline')
        # Calling evaluate_spline(args, kwargs) (line 551)
        evaluate_spline_call_result_74805 = invoke(stypy.reporting.localization.Localization(__file__, 551, 16), evaluate_spline_74787, *[ta_74788, reshape_call_result_74798, ka_74799, x_74800, int_74801, False_74802, out_74803], **kwargs_74804)
        
        
        # Getting the type of 'integral' (line 553)
        integral_74806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'integral')
        
        # Obtaining the type of the subscript
        int_74807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 32), 'int')
        # Getting the type of 'out' (line 553)
        out_74808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 28), 'out')
        # Obtaining the member '__getitem__' of a type (line 553)
        getitem___74809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 28), out_74808, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 553)
        subscript_call_result_74810 = invoke(stypy.reporting.localization.Localization(__file__, 553, 28), getitem___74809, int_74807)
        
        
        # Obtaining the type of the subscript
        int_74811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 41), 'int')
        # Getting the type of 'out' (line 553)
        out_74812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 37), 'out')
        # Obtaining the member '__getitem__' of a type (line 553)
        getitem___74813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 37), out_74812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 553)
        subscript_call_result_74814 = invoke(stypy.reporting.localization.Localization(__file__, 553, 37), getitem___74813, int_74811)
        
        # Applying the binary operator '-' (line 553)
        result_sub_74815 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 28), '-', subscript_call_result_74810, subscript_call_result_74814)
        
        # Applying the binary operator '+=' (line 553)
        result_iadd_74816 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 16), '+=', integral_74806, result_sub_74815)
        # Assigning a type to the variable 'integral' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'integral', result_iadd_74816)
        
        # SSA branch for the else part of an if statement (line 549)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 555):
        
        # Assigning a Call to a Name (line 555):
        
        # Call to asarray(...): (line 555)
        # Processing the call arguments (line 555)
        
        # Obtaining an instance of the builtin type 'list' (line 555)
        list_74819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 555)
        # Adding element type (line 555)
        # Getting the type of 'a' (line 555)
        a_74820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 32), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 31), list_74819, a_74820)
        # Adding element type (line 555)
        # Getting the type of 'te' (line 555)
        te_74821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 35), 'te', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 31), list_74819, te_74821)
        
        # Processing the call keyword arguments (line 555)
        # Getting the type of 'np' (line 555)
        np_74822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 46), 'np', False)
        # Obtaining the member 'float_' of a type (line 555)
        float__74823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 46), np_74822, 'float_')
        keyword_74824 = float__74823
        kwargs_74825 = {'dtype': keyword_74824}
        # Getting the type of 'np' (line 555)
        np_74817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 20), 'np', False)
        # Obtaining the member 'asarray' of a type (line 555)
        asarray_74818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 20), np_74817, 'asarray')
        # Calling asarray(args, kwargs) (line 555)
        asarray_call_result_74826 = invoke(stypy.reporting.localization.Localization(__file__, 555, 20), asarray_74818, *[list_74819], **kwargs_74825)
        
        # Assigning a type to the variable 'x' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 16), 'x', asarray_call_result_74826)
        
        # Call to evaluate_spline(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'ta' (line 556)
        ta_74829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 38), 'ta', False)
        
        # Call to reshape(...): (line 556)
        # Processing the call arguments (line 556)
        
        # Obtaining the type of the subscript
        int_74832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 62), 'int')
        # Getting the type of 'ca' (line 556)
        ca_74833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 53), 'ca', False)
        # Obtaining the member 'shape' of a type (line 556)
        shape_74834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 53), ca_74833, 'shape')
        # Obtaining the member '__getitem__' of a type (line 556)
        getitem___74835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 53), shape_74834, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 556)
        subscript_call_result_74836 = invoke(stypy.reporting.localization.Localization(__file__, 556, 53), getitem___74835, int_74832)
        
        int_74837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 66), 'int')
        # Processing the call keyword arguments (line 556)
        kwargs_74838 = {}
        # Getting the type of 'ca' (line 556)
        ca_74830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 42), 'ca', False)
        # Obtaining the member 'reshape' of a type (line 556)
        reshape_74831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 42), ca_74830, 'reshape')
        # Calling reshape(args, kwargs) (line 556)
        reshape_call_result_74839 = invoke(stypy.reporting.localization.Localization(__file__, 556, 42), reshape_74831, *[subscript_call_result_74836, int_74837], **kwargs_74838)
        
        # Getting the type of 'ka' (line 557)
        ka_74840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 38), 'ka', False)
        # Getting the type of 'x' (line 557)
        x_74841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 42), 'x', False)
        int_74842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 45), 'int')
        # Getting the type of 'False' (line 557)
        False_74843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 48), 'False', False)
        # Getting the type of 'out' (line 557)
        out_74844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 55), 'out', False)
        # Processing the call keyword arguments (line 556)
        kwargs_74845 = {}
        # Getting the type of '_bspl' (line 556)
        _bspl_74827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), '_bspl', False)
        # Obtaining the member 'evaluate_spline' of a type (line 556)
        evaluate_spline_74828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 16), _bspl_74827, 'evaluate_spline')
        # Calling evaluate_spline(args, kwargs) (line 556)
        evaluate_spline_call_result_74846 = invoke(stypy.reporting.localization.Localization(__file__, 556, 16), evaluate_spline_74828, *[ta_74829, reshape_call_result_74839, ka_74840, x_74841, int_74842, False_74843, out_74844], **kwargs_74845)
        
        
        # Getting the type of 'integral' (line 558)
        integral_74847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'integral')
        
        # Obtaining the type of the subscript
        int_74848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 32), 'int')
        # Getting the type of 'out' (line 558)
        out_74849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 28), 'out')
        # Obtaining the member '__getitem__' of a type (line 558)
        getitem___74850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 28), out_74849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 558)
        subscript_call_result_74851 = invoke(stypy.reporting.localization.Localization(__file__, 558, 28), getitem___74850, int_74848)
        
        
        # Obtaining the type of the subscript
        int_74852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 41), 'int')
        # Getting the type of 'out' (line 558)
        out_74853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 37), 'out')
        # Obtaining the member '__getitem__' of a type (line 558)
        getitem___74854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 37), out_74853, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 558)
        subscript_call_result_74855 = invoke(stypy.reporting.localization.Localization(__file__, 558, 37), getitem___74854, int_74852)
        
        # Applying the binary operator '-' (line 558)
        result_sub_74856 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 28), '-', subscript_call_result_74851, subscript_call_result_74855)
        
        # Applying the binary operator '+=' (line 558)
        result_iadd_74857 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 16), '+=', integral_74847, result_sub_74856)
        # Assigning a type to the variable 'integral' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'integral', result_iadd_74857)
        
        
        # Assigning a Call to a Name (line 560):
        
        # Assigning a Call to a Name (line 560):
        
        # Call to asarray(...): (line 560)
        # Processing the call arguments (line 560)
        
        # Obtaining an instance of the builtin type 'list' (line 560)
        list_74860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 560)
        # Adding element type (line 560)
        # Getting the type of 'ts' (line 560)
        ts_74861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 32), 'ts', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 31), list_74860, ts_74861)
        # Adding element type (line 560)
        # Getting the type of 'ts' (line 560)
        ts_74862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 36), 'ts', False)
        # Getting the type of 'b' (line 560)
        b_74863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 41), 'b', False)
        # Applying the binary operator '+' (line 560)
        result_add_74864 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 36), '+', ts_74862, b_74863)
        
        # Getting the type of 'te' (line 560)
        te_74865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 45), 'te', False)
        # Applying the binary operator '-' (line 560)
        result_sub_74866 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 43), '-', result_add_74864, te_74865)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 31), list_74860, result_sub_74866)
        
        # Processing the call keyword arguments (line 560)
        # Getting the type of 'np' (line 560)
        np_74867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 56), 'np', False)
        # Obtaining the member 'float_' of a type (line 560)
        float__74868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 56), np_74867, 'float_')
        keyword_74869 = float__74868
        kwargs_74870 = {'dtype': keyword_74869}
        # Getting the type of 'np' (line 560)
        np_74858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 'np', False)
        # Obtaining the member 'asarray' of a type (line 560)
        asarray_74859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 20), np_74858, 'asarray')
        # Calling asarray(args, kwargs) (line 560)
        asarray_call_result_74871 = invoke(stypy.reporting.localization.Localization(__file__, 560, 20), asarray_74859, *[list_74860], **kwargs_74870)
        
        # Assigning a type to the variable 'x' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'x', asarray_call_result_74871)
        
        # Call to evaluate_spline(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'ta' (line 561)
        ta_74874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 38), 'ta', False)
        
        # Call to reshape(...): (line 561)
        # Processing the call arguments (line 561)
        
        # Obtaining the type of the subscript
        int_74877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 62), 'int')
        # Getting the type of 'ca' (line 561)
        ca_74878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 53), 'ca', False)
        # Obtaining the member 'shape' of a type (line 561)
        shape_74879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 53), ca_74878, 'shape')
        # Obtaining the member '__getitem__' of a type (line 561)
        getitem___74880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 53), shape_74879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 561)
        subscript_call_result_74881 = invoke(stypy.reporting.localization.Localization(__file__, 561, 53), getitem___74880, int_74877)
        
        int_74882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 66), 'int')
        # Processing the call keyword arguments (line 561)
        kwargs_74883 = {}
        # Getting the type of 'ca' (line 561)
        ca_74875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 42), 'ca', False)
        # Obtaining the member 'reshape' of a type (line 561)
        reshape_74876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 42), ca_74875, 'reshape')
        # Calling reshape(args, kwargs) (line 561)
        reshape_call_result_74884 = invoke(stypy.reporting.localization.Localization(__file__, 561, 42), reshape_74876, *[subscript_call_result_74881, int_74882], **kwargs_74883)
        
        # Getting the type of 'ka' (line 562)
        ka_74885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 38), 'ka', False)
        # Getting the type of 'x' (line 562)
        x_74886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 42), 'x', False)
        int_74887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 45), 'int')
        # Getting the type of 'False' (line 562)
        False_74888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 48), 'False', False)
        # Getting the type of 'out' (line 562)
        out_74889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 55), 'out', False)
        # Processing the call keyword arguments (line 561)
        kwargs_74890 = {}
        # Getting the type of '_bspl' (line 561)
        _bspl_74872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), '_bspl', False)
        # Obtaining the member 'evaluate_spline' of a type (line 561)
        evaluate_spline_74873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 16), _bspl_74872, 'evaluate_spline')
        # Calling evaluate_spline(args, kwargs) (line 561)
        evaluate_spline_call_result_74891 = invoke(stypy.reporting.localization.Localization(__file__, 561, 16), evaluate_spline_74873, *[ta_74874, reshape_call_result_74884, ka_74885, x_74886, int_74887, False_74888, out_74889], **kwargs_74890)
        
        
        # Getting the type of 'integral' (line 563)
        integral_74892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'integral')
        
        # Obtaining the type of the subscript
        int_74893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 32), 'int')
        # Getting the type of 'out' (line 563)
        out_74894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 28), 'out')
        # Obtaining the member '__getitem__' of a type (line 563)
        getitem___74895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 28), out_74894, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 563)
        subscript_call_result_74896 = invoke(stypy.reporting.localization.Localization(__file__, 563, 28), getitem___74895, int_74893)
        
        
        # Obtaining the type of the subscript
        int_74897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 41), 'int')
        # Getting the type of 'out' (line 563)
        out_74898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 37), 'out')
        # Obtaining the member '__getitem__' of a type (line 563)
        getitem___74899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 37), out_74898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 563)
        subscript_call_result_74900 = invoke(stypy.reporting.localization.Localization(__file__, 563, 37), getitem___74899, int_74897)
        
        # Applying the binary operator '-' (line 563)
        result_sub_74901 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 28), '-', subscript_call_result_74896, subscript_call_result_74900)
        
        # Applying the binary operator '+=' (line 563)
        result_iadd_74902 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 16), '+=', integral_74892, result_sub_74901)
        # Assigning a type to the variable 'integral' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'integral', result_iadd_74902)
        
        # SSA join for if statement (line 549)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 523)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 566):
        
        # Assigning a Call to a Name (line 566):
        
        # Call to asarray(...): (line 566)
        # Processing the call arguments (line 566)
        
        # Obtaining an instance of the builtin type 'list' (line 566)
        list_74905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 566)
        # Adding element type (line 566)
        # Getting the type of 'a' (line 566)
        a_74906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 28), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 27), list_74905, a_74906)
        # Adding element type (line 566)
        # Getting the type of 'b' (line 566)
        b_74907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 31), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 27), list_74905, b_74907)
        
        # Processing the call keyword arguments (line 566)
        # Getting the type of 'np' (line 566)
        np_74908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 41), 'np', False)
        # Obtaining the member 'float_' of a type (line 566)
        float__74909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 41), np_74908, 'float_')
        keyword_74910 = float__74909
        kwargs_74911 = {'dtype': keyword_74910}
        # Getting the type of 'np' (line 566)
        np_74903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 566)
        asarray_74904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 16), np_74903, 'asarray')
        # Calling asarray(args, kwargs) (line 566)
        asarray_call_result_74912 = invoke(stypy.reporting.localization.Localization(__file__, 566, 16), asarray_74904, *[list_74905], **kwargs_74911)
        
        # Assigning a type to the variable 'x' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'x', asarray_call_result_74912)
        
        # Call to evaluate_spline(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'ta' (line 567)
        ta_74915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 34), 'ta', False)
        
        # Call to reshape(...): (line 567)
        # Processing the call arguments (line 567)
        
        # Obtaining the type of the subscript
        int_74918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 58), 'int')
        # Getting the type of 'ca' (line 567)
        ca_74919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 49), 'ca', False)
        # Obtaining the member 'shape' of a type (line 567)
        shape_74920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 49), ca_74919, 'shape')
        # Obtaining the member '__getitem__' of a type (line 567)
        getitem___74921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 49), shape_74920, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 567)
        subscript_call_result_74922 = invoke(stypy.reporting.localization.Localization(__file__, 567, 49), getitem___74921, int_74918)
        
        int_74923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 62), 'int')
        # Processing the call keyword arguments (line 567)
        kwargs_74924 = {}
        # Getting the type of 'ca' (line 567)
        ca_74916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 38), 'ca', False)
        # Obtaining the member 'reshape' of a type (line 567)
        reshape_74917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 38), ca_74916, 'reshape')
        # Calling reshape(args, kwargs) (line 567)
        reshape_call_result_74925 = invoke(stypy.reporting.localization.Localization(__file__, 567, 38), reshape_74917, *[subscript_call_result_74922, int_74923], **kwargs_74924)
        
        # Getting the type of 'ka' (line 568)
        ka_74926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 34), 'ka', False)
        # Getting the type of 'x' (line 568)
        x_74927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 38), 'x', False)
        int_74928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 41), 'int')
        # Getting the type of 'extrapolate' (line 568)
        extrapolate_74929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 44), 'extrapolate', False)
        # Getting the type of 'out' (line 568)
        out_74930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 57), 'out', False)
        # Processing the call keyword arguments (line 567)
        kwargs_74931 = {}
        # Getting the type of '_bspl' (line 567)
        _bspl_74913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), '_bspl', False)
        # Obtaining the member 'evaluate_spline' of a type (line 567)
        evaluate_spline_74914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 12), _bspl_74913, 'evaluate_spline')
        # Calling evaluate_spline(args, kwargs) (line 567)
        evaluate_spline_call_result_74932 = invoke(stypy.reporting.localization.Localization(__file__, 567, 12), evaluate_spline_74914, *[ta_74915, reshape_call_result_74925, ka_74926, x_74927, int_74928, extrapolate_74929, out_74930], **kwargs_74931)
        
        
        # Assigning a BinOp to a Name (line 569):
        
        # Assigning a BinOp to a Name (line 569):
        
        # Obtaining the type of the subscript
        int_74933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 27), 'int')
        # Getting the type of 'out' (line 569)
        out_74934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 23), 'out')
        # Obtaining the member '__getitem__' of a type (line 569)
        getitem___74935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 23), out_74934, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 569)
        subscript_call_result_74936 = invoke(stypy.reporting.localization.Localization(__file__, 569, 23), getitem___74935, int_74933)
        
        
        # Obtaining the type of the subscript
        int_74937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 36), 'int')
        # Getting the type of 'out' (line 569)
        out_74938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 32), 'out')
        # Obtaining the member '__getitem__' of a type (line 569)
        getitem___74939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 32), out_74938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 569)
        subscript_call_result_74940 = invoke(stypy.reporting.localization.Localization(__file__, 569, 32), getitem___74939, int_74937)
        
        # Applying the binary operator '-' (line 569)
        result_sub_74941 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 23), '-', subscript_call_result_74936, subscript_call_result_74940)
        
        # Assigning a type to the variable 'integral' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'integral', result_sub_74941)
        # SSA join for if statement (line 523)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'integral' (line 571)
        integral_74942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'integral')
        # Getting the type of 'sign' (line 571)
        sign_74943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'sign')
        # Applying the binary operator '*=' (line 571)
        result_imul_74944 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 8), '*=', integral_74942, sign_74943)
        # Assigning a type to the variable 'integral' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'integral', result_imul_74944)
        
        
        # Call to reshape(...): (line 572)
        # Processing the call arguments (line 572)
        
        # Obtaining the type of the subscript
        int_74947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 41), 'int')
        slice_74948 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 572, 32), int_74947, None, None)
        # Getting the type of 'ca' (line 572)
        ca_74949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 32), 'ca', False)
        # Obtaining the member 'shape' of a type (line 572)
        shape_74950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 32), ca_74949, 'shape')
        # Obtaining the member '__getitem__' of a type (line 572)
        getitem___74951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 32), shape_74950, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 572)
        subscript_call_result_74952 = invoke(stypy.reporting.localization.Localization(__file__, 572, 32), getitem___74951, slice_74948)
        
        # Processing the call keyword arguments (line 572)
        kwargs_74953 = {}
        # Getting the type of 'integral' (line 572)
        integral_74945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'integral', False)
        # Obtaining the member 'reshape' of a type (line 572)
        reshape_74946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 15), integral_74945, 'reshape')
        # Calling reshape(args, kwargs) (line 572)
        reshape_call_result_74954 = invoke(stypy.reporting.localization.Localization(__file__, 572, 15), reshape_74946, *[subscript_call_result_74952], **kwargs_74953)
        
        # Assigning a type to the variable 'stypy_return_type' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'stypy_return_type', reshape_call_result_74954)
        
        # ################# End of 'integrate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_74955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_74955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate'
        return stypy_return_type_74955


# Assigning a type to the variable 'BSpline' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'BSpline', BSpline)

@norecursion
def _not_a_knot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_not_a_knot'
    module_type_store = module_type_store.open_function_context('_not_a_knot', 579, 0, False)
    
    # Passed parameters checking function
    _not_a_knot.stypy_localization = localization
    _not_a_knot.stypy_type_of_self = None
    _not_a_knot.stypy_type_store = module_type_store
    _not_a_knot.stypy_function_name = '_not_a_knot'
    _not_a_knot.stypy_param_names_list = ['x', 'k']
    _not_a_knot.stypy_varargs_param_name = None
    _not_a_knot.stypy_kwargs_param_name = None
    _not_a_knot.stypy_call_defaults = defaults
    _not_a_knot.stypy_call_varargs = varargs
    _not_a_knot.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_not_a_knot', ['x', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_not_a_knot', localization, ['x', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_not_a_knot(...)' code ##################

    str_74956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, (-1)), 'str', 'Given data x, construct the knot vector w/ not-a-knot BC.\n    cf de Boor, XIII(12).')
    
    # Assigning a Call to a Name (line 582):
    
    # Assigning a Call to a Name (line 582):
    
    # Call to asarray(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'x' (line 582)
    x_74959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 19), 'x', False)
    # Processing the call keyword arguments (line 582)
    kwargs_74960 = {}
    # Getting the type of 'np' (line 582)
    np_74957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 582)
    asarray_74958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 8), np_74957, 'asarray')
    # Calling asarray(args, kwargs) (line 582)
    asarray_call_result_74961 = invoke(stypy.reporting.localization.Localization(__file__, 582, 8), asarray_74958, *[x_74959], **kwargs_74960)
    
    # Assigning a type to the variable 'x' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'x', asarray_call_result_74961)
    
    
    # Getting the type of 'k' (line 583)
    k_74962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 7), 'k')
    int_74963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 11), 'int')
    # Applying the binary operator '%' (line 583)
    result_mod_74964 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 7), '%', k_74962, int_74963)
    
    int_74965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 16), 'int')
    # Applying the binary operator '!=' (line 583)
    result_ne_74966 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 7), '!=', result_mod_74964, int_74965)
    
    # Testing the type of an if condition (line 583)
    if_condition_74967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 4), result_ne_74966)
    # Assigning a type to the variable 'if_condition_74967' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'if_condition_74967', if_condition_74967)
    # SSA begins for if statement (line 583)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 584)
    # Processing the call arguments (line 584)
    str_74969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 25), 'str', 'Odd degree for now only. Got %s.')
    # Getting the type of 'k' (line 584)
    k_74970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 62), 'k', False)
    # Applying the binary operator '%' (line 584)
    result_mod_74971 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 25), '%', str_74969, k_74970)
    
    # Processing the call keyword arguments (line 584)
    kwargs_74972 = {}
    # Getting the type of 'ValueError' (line 584)
    ValueError_74968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 584)
    ValueError_call_result_74973 = invoke(stypy.reporting.localization.Localization(__file__, 584, 14), ValueError_74968, *[result_mod_74971], **kwargs_74972)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 584, 8), ValueError_call_result_74973, 'raise parameter', BaseException)
    # SSA join for if statement (line 583)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 586):
    
    # Assigning a BinOp to a Name (line 586):
    # Getting the type of 'k' (line 586)
    k_74974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 9), 'k')
    int_74975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 13), 'int')
    # Applying the binary operator '-' (line 586)
    result_sub_74976 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 9), '-', k_74974, int_74975)
    
    int_74977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 19), 'int')
    # Applying the binary operator '//' (line 586)
    result_floordiv_74978 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 8), '//', result_sub_74976, int_74977)
    
    # Assigning a type to the variable 'm' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'm', result_floordiv_74978)
    
    # Assigning a Subscript to a Name (line 587):
    
    # Assigning a Subscript to a Name (line 587):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 587)
    m_74979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 10), 'm')
    int_74980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 12), 'int')
    # Applying the binary operator '+' (line 587)
    result_add_74981 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 10), '+', m_74979, int_74980)
    
    
    # Getting the type of 'm' (line 587)
    m_74982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'm')
    # Applying the 'usub' unary operator (line 587)
    result___neg___74983 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 14), 'usub', m_74982)
    
    int_74984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 17), 'int')
    # Applying the binary operator '-' (line 587)
    result_sub_74985 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 14), '-', result___neg___74983, int_74984)
    
    slice_74986 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 587, 8), result_add_74981, result_sub_74985, None)
    # Getting the type of 'x' (line 587)
    x_74987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___74988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 8), x_74987, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_74989 = invoke(stypy.reporting.localization.Localization(__file__, 587, 8), getitem___74988, slice_74986)
    
    # Assigning a type to the variable 't' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 't', subscript_call_result_74989)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 588)
    tuple_74990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 588)
    # Adding element type (line 588)
    
    # Obtaining an instance of the builtin type 'tuple' (line 588)
    tuple_74991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 588)
    # Adding element type (line 588)
    
    # Obtaining the type of the subscript
    int_74992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 17), 'int')
    # Getting the type of 'x' (line 588)
    x_74993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 15), 'x')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___74994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 15), x_74993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_74995 = invoke(stypy.reporting.localization.Localization(__file__, 588, 15), getitem___74994, int_74992)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 15), tuple_74991, subscript_call_result_74995)
    
    # Getting the type of 'k' (line 588)
    k_74996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 23), 'k')
    int_74997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 25), 'int')
    # Applying the binary operator '+' (line 588)
    result_add_74998 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 23), '+', k_74996, int_74997)
    
    # Applying the binary operator '*' (line 588)
    result_mul_74999 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 14), '*', tuple_74991, result_add_74998)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 14), tuple_74990, result_mul_74999)
    # Adding element type (line 588)
    # Getting the type of 't' (line 588)
    t_75000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 29), 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 14), tuple_74990, t_75000)
    # Adding element type (line 588)
    
    # Obtaining an instance of the builtin type 'tuple' (line 588)
    tuple_75001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 588)
    # Adding element type (line 588)
    
    # Obtaining the type of the subscript
    int_75002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 35), 'int')
    # Getting the type of 'x' (line 588)
    x_75003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 33), 'x')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___75004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 33), x_75003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_75005 = invoke(stypy.reporting.localization.Localization(__file__, 588, 33), getitem___75004, int_75002)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 33), tuple_75001, subscript_call_result_75005)
    
    # Getting the type of 'k' (line 588)
    k_75006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 42), 'k')
    int_75007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 44), 'int')
    # Applying the binary operator '+' (line 588)
    result_add_75008 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 42), '+', k_75006, int_75007)
    
    # Applying the binary operator '*' (line 588)
    result_mul_75009 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 32), '*', tuple_75001, result_add_75008)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 14), tuple_74990, result_mul_75009)
    
    # Getting the type of 'np' (line 588)
    np_75010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'np')
    # Obtaining the member 'r_' of a type (line 588)
    r__75011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), np_75010, 'r_')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___75012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), r__75011, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_75013 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___75012, tuple_74990)
    
    # Assigning a type to the variable 't' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 't', subscript_call_result_75013)
    # Getting the type of 't' (line 589)
    t_75014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 11), 't')
    # Assigning a type to the variable 'stypy_return_type' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'stypy_return_type', t_75014)
    
    # ################# End of '_not_a_knot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_not_a_knot' in the type store
    # Getting the type of 'stypy_return_type' (line 579)
    stypy_return_type_75015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_75015)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_not_a_knot'
    return stypy_return_type_75015

# Assigning a type to the variable '_not_a_knot' (line 579)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 0), '_not_a_knot', _not_a_knot)

@norecursion
def _augknt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_augknt'
    module_type_store = module_type_store.open_function_context('_augknt', 592, 0, False)
    
    # Passed parameters checking function
    _augknt.stypy_localization = localization
    _augknt.stypy_type_of_self = None
    _augknt.stypy_type_store = module_type_store
    _augknt.stypy_function_name = '_augknt'
    _augknt.stypy_param_names_list = ['x', 'k']
    _augknt.stypy_varargs_param_name = None
    _augknt.stypy_kwargs_param_name = None
    _augknt.stypy_call_defaults = defaults
    _augknt.stypy_call_varargs = varargs
    _augknt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_augknt', ['x', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_augknt', localization, ['x', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_augknt(...)' code ##################

    str_75016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 4), 'str', 'Construct a knot vector appropriate for the order-k interpolation.')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 594)
    tuple_75017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 594)
    # Adding element type (line 594)
    
    # Obtaining an instance of the builtin type 'tuple' (line 594)
    tuple_75018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 594)
    # Adding element type (line 594)
    
    # Obtaining the type of the subscript
    int_75019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 20), 'int')
    # Getting the type of 'x' (line 594)
    x_75020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 18), 'x')
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___75021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 18), x_75020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_75022 = invoke(stypy.reporting.localization.Localization(__file__, 594, 18), getitem___75021, int_75019)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 18), tuple_75018, subscript_call_result_75022)
    
    # Getting the type of 'k' (line 594)
    k_75023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 25), 'k')
    # Applying the binary operator '*' (line 594)
    result_mul_75024 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 17), '*', tuple_75018, k_75023)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 17), tuple_75017, result_mul_75024)
    # Adding element type (line 594)
    # Getting the type of 'x' (line 594)
    x_75025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 28), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 17), tuple_75017, x_75025)
    # Adding element type (line 594)
    
    # Obtaining an instance of the builtin type 'tuple' (line 594)
    tuple_75026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 594)
    # Adding element type (line 594)
    
    # Obtaining the type of the subscript
    int_75027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 34), 'int')
    # Getting the type of 'x' (line 594)
    x_75028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 32), 'x')
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___75029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 32), x_75028, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_75030 = invoke(stypy.reporting.localization.Localization(__file__, 594, 32), getitem___75029, int_75027)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 32), tuple_75026, subscript_call_result_75030)
    
    # Getting the type of 'k' (line 594)
    k_75031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 40), 'k')
    # Applying the binary operator '*' (line 594)
    result_mul_75032 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 31), '*', tuple_75026, k_75031)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 17), tuple_75017, result_mul_75032)
    
    # Getting the type of 'np' (line 594)
    np_75033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 11), 'np')
    # Obtaining the member 'r_' of a type (line 594)
    r__75034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 11), np_75033, 'r_')
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___75035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 11), r__75034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_75036 = invoke(stypy.reporting.localization.Localization(__file__, 594, 11), getitem___75035, tuple_75017)
    
    # Assigning a type to the variable 'stypy_return_type' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'stypy_return_type', subscript_call_result_75036)
    
    # ################# End of '_augknt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_augknt' in the type store
    # Getting the type of 'stypy_return_type' (line 592)
    stypy_return_type_75037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_75037)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_augknt'
    return stypy_return_type_75037

# Assigning a type to the variable '_augknt' (line 592)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), '_augknt', _augknt)

@norecursion
def make_interp_spline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_75038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 31), 'int')
    # Getting the type of 'None' (line 597)
    None_75039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'None')
    # Getting the type of 'None' (line 597)
    None_75040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 50), 'None')
    int_75041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 61), 'int')
    # Getting the type of 'True' (line 598)
    True_75042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 36), 'True')
    defaults = [int_75038, None_75039, None_75040, int_75041, True_75042]
    # Create a new context for function 'make_interp_spline'
    module_type_store = module_type_store.open_function_context('make_interp_spline', 597, 0, False)
    
    # Passed parameters checking function
    make_interp_spline.stypy_localization = localization
    make_interp_spline.stypy_type_of_self = None
    make_interp_spline.stypy_type_store = module_type_store
    make_interp_spline.stypy_function_name = 'make_interp_spline'
    make_interp_spline.stypy_param_names_list = ['x', 'y', 'k', 't', 'bc_type', 'axis', 'check_finite']
    make_interp_spline.stypy_varargs_param_name = None
    make_interp_spline.stypy_kwargs_param_name = None
    make_interp_spline.stypy_call_defaults = defaults
    make_interp_spline.stypy_call_varargs = varargs
    make_interp_spline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_interp_spline', ['x', 'y', 'k', 't', 'bc_type', 'axis', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_interp_spline', localization, ['x', 'y', 'k', 't', 'bc_type', 'axis', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_interp_spline(...)' code ##################

    str_75043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, (-1)), 'str', "Compute the (coefficients of) interpolating B-spline.\n\n    Parameters\n    ----------\n    x : array_like, shape (n,)\n        Abscissas.\n    y : array_like, shape (n, ...)\n        Ordinates.\n    k : int, optional\n        B-spline degree. Default is cubic, k=3.\n    t : array_like, shape (nt + k + 1,), optional.\n        Knots.\n        The number of knots needs to agree with the number of datapoints and\n        the number of derivatives at the edges. Specifically, ``nt - n`` must\n        equal ``len(deriv_l) + len(deriv_r)``.\n    bc_type : 2-tuple or None\n        Boundary conditions.\n        Default is None, which means choosing the boundary conditions\n        automatically. Otherwise, it must be a length-two tuple where the first\n        element sets the boundary conditions at ``x[0]`` and the second\n        element sets the boundary conditions at ``x[-1]``. Each of these must\n        be an iterable of pairs ``(order, value)`` which gives the values of\n        derivatives of specified orders at the given edge of the interpolation\n        interval.\n    axis : int, optional\n        Interpolation axis. Default is 0.\n    check_finite : bool, optional\n        Whether to check that the input arrays contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default is True.\n\n    Returns\n    -------\n    b : a BSpline object of the degree ``k`` and with knots ``t``.\n\n    Examples\n    --------\n\n    Use cubic interpolation on Chebyshev nodes:\n\n    >>> def cheb_nodes(N):\n    ...     jj = 2.*np.arange(N) + 1\n    ...     x = np.cos(np.pi * jj / 2 / N)[::-1]\n    ...     return x\n\n    >>> x = cheb_nodes(20)\n    >>> y = np.sqrt(1 - x**2)\n\n    >>> from scipy.interpolate import BSpline, make_interp_spline\n    >>> b = make_interp_spline(x, y)\n    >>> np.allclose(b(x), y)\n    True\n\n    Note that the default is a cubic spline with a not-a-knot boundary condition\n\n    >>> b.k\n    3\n\n    Here we use a 'natural' spline, with zero 2nd derivatives at edges:\n\n    >>> l, r = [(2, 0)], [(2, 0)]\n    >>> b_n = make_interp_spline(x, y, bc_type=(l, r))\n    >>> np.allclose(b_n(x), y)\n    True\n    >>> x0, x1 = x[0], x[-1]\n    >>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])\n    True\n\n    Interpolation of parametric curves is also supported. As an example, we\n    compute a discretization of a snail curve in polar coordinates\n\n    >>> phi = np.linspace(0, 2.*np.pi, 40)\n    >>> r = 0.3 + np.cos(phi)\n    >>> x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates\n\n    Build an interpolating curve, parameterizing it by the angle\n\n    >>> from scipy.interpolate import make_interp_spline\n    >>> spl = make_interp_spline(phi, np.c_[x, y])\n\n    Evaluate the interpolant on a finer grid (note that we transpose the result\n    to unpack it into a pair of x- and y-arrays)\n\n    >>> phi_new = np.linspace(0, 2.*np.pi, 100)\n    >>> x_new, y_new = spl(phi_new).T\n\n    Plot the result\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(x, y, 'o')\n    >>> plt.plot(x_new, y_new, '-')\n    >>> plt.show()\n\n    See Also\n    --------\n    BSpline : base class representing the B-spline objects\n    CubicSpline : a cubic spline in the polynomial basis\n    make_lsq_spline : a similar factory function for spline fitting\n    UnivariateSpline : a wrapper over FITPACK spline fitting routines\n    splrep : a wrapper over FITPACK spline fitting routines\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 702)
    # Getting the type of 'bc_type' (line 702)
    bc_type_75044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 7), 'bc_type')
    # Getting the type of 'None' (line 702)
    None_75045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 18), 'None')
    
    (may_be_75046, more_types_in_union_75047) = may_be_none(bc_type_75044, None_75045)

    if may_be_75046:

        if more_types_in_union_75047:
            # Runtime conditional SSA (line 702)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Tuple to a Name (line 703):
        
        # Assigning a Tuple to a Name (line 703):
        
        # Obtaining an instance of the builtin type 'tuple' (line 703)
        tuple_75048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 703)
        # Adding element type (line 703)
        # Getting the type of 'None' (line 703)
        None_75049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 19), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 19), tuple_75048, None_75049)
        # Adding element type (line 703)
        # Getting the type of 'None' (line 703)
        None_75050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 25), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 19), tuple_75048, None_75050)
        
        # Assigning a type to the variable 'bc_type' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'bc_type', tuple_75048)

        if more_types_in_union_75047:
            # SSA join for if statement (line 702)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Tuple (line 704):
    
    # Assigning a Subscript to a Name (line 704):
    
    # Obtaining the type of the subscript
    int_75051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 4), 'int')
    # Getting the type of 'bc_type' (line 704)
    bc_type_75052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 23), 'bc_type')
    # Obtaining the member '__getitem__' of a type (line 704)
    getitem___75053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 4), bc_type_75052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 704)
    subscript_call_result_75054 = invoke(stypy.reporting.localization.Localization(__file__, 704, 4), getitem___75053, int_75051)
    
    # Assigning a type to the variable 'tuple_var_assignment_73690' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'tuple_var_assignment_73690', subscript_call_result_75054)
    
    # Assigning a Subscript to a Name (line 704):
    
    # Obtaining the type of the subscript
    int_75055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 4), 'int')
    # Getting the type of 'bc_type' (line 704)
    bc_type_75056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 23), 'bc_type')
    # Obtaining the member '__getitem__' of a type (line 704)
    getitem___75057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 4), bc_type_75056, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 704)
    subscript_call_result_75058 = invoke(stypy.reporting.localization.Localization(__file__, 704, 4), getitem___75057, int_75055)
    
    # Assigning a type to the variable 'tuple_var_assignment_73691' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'tuple_var_assignment_73691', subscript_call_result_75058)
    
    # Assigning a Name to a Name (line 704):
    # Getting the type of 'tuple_var_assignment_73690' (line 704)
    tuple_var_assignment_73690_75059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'tuple_var_assignment_73690')
    # Assigning a type to the variable 'deriv_l' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'deriv_l', tuple_var_assignment_73690_75059)
    
    # Assigning a Name to a Name (line 704):
    # Getting the type of 'tuple_var_assignment_73691' (line 704)
    tuple_var_assignment_73691_75060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'tuple_var_assignment_73691')
    # Assigning a type to the variable 'deriv_r' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 13), 'deriv_r', tuple_var_assignment_73691_75060)
    
    
    # Getting the type of 'k' (line 707)
    k_75061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 7), 'k')
    int_75062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 12), 'int')
    # Applying the binary operator '==' (line 707)
    result_eq_75063 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 7), '==', k_75061, int_75062)
    
    # Testing the type of an if condition (line 707)
    if_condition_75064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 707, 4), result_eq_75063)
    # Assigning a type to the variable 'if_condition_75064' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'if_condition_75064', if_condition_75064)
    # SSA begins for if statement (line 707)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to any(...): (line 708)
    # Processing the call arguments (line 708)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 708, 15, True)
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'tuple' (line 708)
    tuple_75069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 708)
    # Adding element type (line 708)
    # Getting the type of 't' (line 708)
    t_75070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 39), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 39), tuple_75069, t_75070)
    # Adding element type (line 708)
    # Getting the type of 'deriv_l' (line 708)
    deriv_l_75071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 42), 'deriv_l', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 39), tuple_75069, deriv_l_75071)
    # Adding element type (line 708)
    # Getting the type of 'deriv_r' (line 708)
    deriv_r_75072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 51), 'deriv_r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 39), tuple_75069, deriv_r_75072)
    
    comprehension_75073 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_75069)
    # Assigning a type to the variable '_' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 15), '_', comprehension_75073)
    
    # Getting the type of '_' (line 708)
    __75066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 15), '_', False)
    # Getting the type of 'None' (line 708)
    None_75067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 24), 'None', False)
    # Applying the binary operator 'isnot' (line 708)
    result_is_not_75068 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 15), 'isnot', __75066, None_75067)
    
    list_75074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 15), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), list_75074, result_is_not_75068)
    # Processing the call keyword arguments (line 708)
    kwargs_75075 = {}
    # Getting the type of 'any' (line 708)
    any_75065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 11), 'any', False)
    # Calling any(args, kwargs) (line 708)
    any_call_result_75076 = invoke(stypy.reporting.localization.Localization(__file__, 708, 11), any_75065, *[list_75074], **kwargs_75075)
    
    # Testing the type of an if condition (line 708)
    if_condition_75077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 708, 8), any_call_result_75076)
    # Assigning a type to the variable 'if_condition_75077' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'if_condition_75077', if_condition_75077)
    # SSA begins for if statement (line 708)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 709)
    # Processing the call arguments (line 709)
    str_75079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 29), 'str', 'Too much info for k=0: t and bc_type can only be None.')
    # Processing the call keyword arguments (line 709)
    kwargs_75080 = {}
    # Getting the type of 'ValueError' (line 709)
    ValueError_75078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 709)
    ValueError_call_result_75081 = invoke(stypy.reporting.localization.Localization(__file__, 709, 18), ValueError_75078, *[str_75079], **kwargs_75080)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 709, 12), ValueError_call_result_75081, 'raise parameter', BaseException)
    # SSA join for if statement (line 708)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 711):
    
    # Assigning a Call to a Name (line 711):
    
    # Call to _as_float_array(...): (line 711)
    # Processing the call arguments (line 711)
    # Getting the type of 'x' (line 711)
    x_75083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 28), 'x', False)
    # Getting the type of 'check_finite' (line 711)
    check_finite_75084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 31), 'check_finite', False)
    # Processing the call keyword arguments (line 711)
    kwargs_75085 = {}
    # Getting the type of '_as_float_array' (line 711)
    _as_float_array_75082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), '_as_float_array', False)
    # Calling _as_float_array(args, kwargs) (line 711)
    _as_float_array_call_result_75086 = invoke(stypy.reporting.localization.Localization(__file__, 711, 12), _as_float_array_75082, *[x_75083, check_finite_75084], **kwargs_75085)
    
    # Assigning a type to the variable 'x' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'x', _as_float_array_call_result_75086)
    
    # Assigning a Subscript to a Name (line 712):
    
    # Assigning a Subscript to a Name (line 712):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 712)
    tuple_75087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 712)
    # Adding element type (line 712)
    # Getting the type of 'x' (line 712)
    x_75088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 18), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 18), tuple_75087, x_75088)
    # Adding element type (line 712)
    
    # Obtaining the type of the subscript
    int_75089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 23), 'int')
    # Getting the type of 'x' (line 712)
    x_75090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 21), 'x')
    # Obtaining the member '__getitem__' of a type (line 712)
    getitem___75091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 21), x_75090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 712)
    subscript_call_result_75092 = invoke(stypy.reporting.localization.Localization(__file__, 712, 21), getitem___75091, int_75089)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 18), tuple_75087, subscript_call_result_75092)
    
    # Getting the type of 'np' (line 712)
    np_75093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 12), 'np')
    # Obtaining the member 'r_' of a type (line 712)
    r__75094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 12), np_75093, 'r_')
    # Obtaining the member '__getitem__' of a type (line 712)
    getitem___75095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 12), r__75094, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 712)
    subscript_call_result_75096 = invoke(stypy.reporting.localization.Localization(__file__, 712, 12), getitem___75095, tuple_75087)
    
    # Assigning a type to the variable 't' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 't', subscript_call_result_75096)
    
    # Assigning a Call to a Name (line 713):
    
    # Assigning a Call to a Name (line 713):
    
    # Call to asarray(...): (line 713)
    # Processing the call arguments (line 713)
    # Getting the type of 'y' (line 713)
    y_75099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 23), 'y', False)
    # Processing the call keyword arguments (line 713)
    kwargs_75100 = {}
    # Getting the type of 'np' (line 713)
    np_75097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 713)
    asarray_75098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 12), np_75097, 'asarray')
    # Calling asarray(args, kwargs) (line 713)
    asarray_call_result_75101 = invoke(stypy.reporting.localization.Localization(__file__, 713, 12), asarray_75098, *[y_75099], **kwargs_75100)
    
    # Assigning a type to the variable 'c' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'c', asarray_call_result_75101)
    
    # Assigning a Call to a Name (line 714):
    
    # Assigning a Call to a Name (line 714):
    
    # Call to ascontiguousarray(...): (line 714)
    # Processing the call arguments (line 714)
    # Getting the type of 'c' (line 714)
    c_75104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 33), 'c', False)
    # Processing the call keyword arguments (line 714)
    
    # Call to _get_dtype(...): (line 714)
    # Processing the call arguments (line 714)
    # Getting the type of 'c' (line 714)
    c_75106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 53), 'c', False)
    # Obtaining the member 'dtype' of a type (line 714)
    dtype_75107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 53), c_75106, 'dtype')
    # Processing the call keyword arguments (line 714)
    kwargs_75108 = {}
    # Getting the type of '_get_dtype' (line 714)
    _get_dtype_75105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 42), '_get_dtype', False)
    # Calling _get_dtype(args, kwargs) (line 714)
    _get_dtype_call_result_75109 = invoke(stypy.reporting.localization.Localization(__file__, 714, 42), _get_dtype_75105, *[dtype_75107], **kwargs_75108)
    
    keyword_75110 = _get_dtype_call_result_75109
    kwargs_75111 = {'dtype': keyword_75110}
    # Getting the type of 'np' (line 714)
    np_75102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 12), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 714)
    ascontiguousarray_75103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 12), np_75102, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 714)
    ascontiguousarray_call_result_75112 = invoke(stypy.reporting.localization.Localization(__file__, 714, 12), ascontiguousarray_75103, *[c_75104], **kwargs_75111)
    
    # Assigning a type to the variable 'c' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'c', ascontiguousarray_call_result_75112)
    
    # Call to construct_fast(...): (line 715)
    # Processing the call arguments (line 715)
    # Getting the type of 't' (line 715)
    t_75115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 38), 't', False)
    # Getting the type of 'c' (line 715)
    c_75116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 41), 'c', False)
    # Getting the type of 'k' (line 715)
    k_75117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 44), 'k', False)
    # Processing the call keyword arguments (line 715)
    # Getting the type of 'axis' (line 715)
    axis_75118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 52), 'axis', False)
    keyword_75119 = axis_75118
    kwargs_75120 = {'axis': keyword_75119}
    # Getting the type of 'BSpline' (line 715)
    BSpline_75113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 15), 'BSpline', False)
    # Obtaining the member 'construct_fast' of a type (line 715)
    construct_fast_75114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 15), BSpline_75113, 'construct_fast')
    # Calling construct_fast(args, kwargs) (line 715)
    construct_fast_call_result_75121 = invoke(stypy.reporting.localization.Localization(__file__, 715, 15), construct_fast_75114, *[t_75115, c_75116, k_75117], **kwargs_75120)
    
    # Assigning a type to the variable 'stypy_return_type' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'stypy_return_type', construct_fast_call_result_75121)
    # SSA join for if statement (line 707)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 718)
    k_75122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 7), 'k')
    int_75123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 12), 'int')
    # Applying the binary operator '==' (line 718)
    result_eq_75124 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 7), '==', k_75122, int_75123)
    
    
    # Getting the type of 't' (line 718)
    t_75125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 18), 't')
    # Getting the type of 'None' (line 718)
    None_75126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 23), 'None')
    # Applying the binary operator 'is' (line 718)
    result_is__75127 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 18), 'is', t_75125, None_75126)
    
    # Applying the binary operator 'and' (line 718)
    result_and_keyword_75128 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 7), 'and', result_eq_75124, result_is__75127)
    
    # Testing the type of an if condition (line 718)
    if_condition_75129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 4), result_and_keyword_75128)
    # Assigning a type to the variable 'if_condition_75129' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'if_condition_75129', if_condition_75129)
    # SSA begins for if statement (line 718)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'deriv_l' (line 719)
    deriv_l_75130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'deriv_l')
    # Getting the type of 'None' (line 719)
    None_75131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 27), 'None')
    # Applying the binary operator 'is' (line 719)
    result_is__75132 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 16), 'is', deriv_l_75130, None_75131)
    
    
    # Getting the type of 'deriv_r' (line 719)
    deriv_r_75133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 36), 'deriv_r')
    # Getting the type of 'None' (line 719)
    None_75134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 47), 'None')
    # Applying the binary operator 'is' (line 719)
    result_is__75135 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 36), 'is', deriv_r_75133, None_75134)
    
    # Applying the binary operator 'and' (line 719)
    result_and_keyword_75136 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 16), 'and', result_is__75132, result_is__75135)
    
    # Applying the 'not' unary operator (line 719)
    result_not__75137 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 11), 'not', result_and_keyword_75136)
    
    # Testing the type of an if condition (line 719)
    if_condition_75138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 719, 8), result_not__75137)
    # Assigning a type to the variable 'if_condition_75138' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'if_condition_75138', if_condition_75138)
    # SSA begins for if statement (line 719)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 720)
    # Processing the call arguments (line 720)
    str_75140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 29), 'str', 'Too much info for k=1: bc_type can only be None.')
    # Processing the call keyword arguments (line 720)
    kwargs_75141 = {}
    # Getting the type of 'ValueError' (line 720)
    ValueError_75139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 720)
    ValueError_call_result_75142 = invoke(stypy.reporting.localization.Localization(__file__, 720, 18), ValueError_75139, *[str_75140], **kwargs_75141)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 720, 12), ValueError_call_result_75142, 'raise parameter', BaseException)
    # SSA join for if statement (line 719)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 721):
    
    # Assigning a Call to a Name (line 721):
    
    # Call to _as_float_array(...): (line 721)
    # Processing the call arguments (line 721)
    # Getting the type of 'x' (line 721)
    x_75144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 28), 'x', False)
    # Getting the type of 'check_finite' (line 721)
    check_finite_75145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 31), 'check_finite', False)
    # Processing the call keyword arguments (line 721)
    kwargs_75146 = {}
    # Getting the type of '_as_float_array' (line 721)
    _as_float_array_75143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), '_as_float_array', False)
    # Calling _as_float_array(args, kwargs) (line 721)
    _as_float_array_call_result_75147 = invoke(stypy.reporting.localization.Localization(__file__, 721, 12), _as_float_array_75143, *[x_75144, check_finite_75145], **kwargs_75146)
    
    # Assigning a type to the variable 'x' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'x', _as_float_array_call_result_75147)
    
    # Assigning a Subscript to a Name (line 722):
    
    # Assigning a Subscript to a Name (line 722):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 722)
    tuple_75148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 722)
    # Adding element type (line 722)
    
    # Obtaining the type of the subscript
    int_75149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 20), 'int')
    # Getting the type of 'x' (line 722)
    x_75150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 18), 'x')
    # Obtaining the member '__getitem__' of a type (line 722)
    getitem___75151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 18), x_75150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 722)
    subscript_call_result_75152 = invoke(stypy.reporting.localization.Localization(__file__, 722, 18), getitem___75151, int_75149)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 18), tuple_75148, subscript_call_result_75152)
    # Adding element type (line 722)
    # Getting the type of 'x' (line 722)
    x_75153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 24), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 18), tuple_75148, x_75153)
    # Adding element type (line 722)
    
    # Obtaining the type of the subscript
    int_75154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 29), 'int')
    # Getting the type of 'x' (line 722)
    x_75155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 27), 'x')
    # Obtaining the member '__getitem__' of a type (line 722)
    getitem___75156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 27), x_75155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 722)
    subscript_call_result_75157 = invoke(stypy.reporting.localization.Localization(__file__, 722, 27), getitem___75156, int_75154)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 18), tuple_75148, subscript_call_result_75157)
    
    # Getting the type of 'np' (line 722)
    np_75158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 12), 'np')
    # Obtaining the member 'r_' of a type (line 722)
    r__75159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 12), np_75158, 'r_')
    # Obtaining the member '__getitem__' of a type (line 722)
    getitem___75160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 12), r__75159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 722)
    subscript_call_result_75161 = invoke(stypy.reporting.localization.Localization(__file__, 722, 12), getitem___75160, tuple_75148)
    
    # Assigning a type to the variable 't' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 't', subscript_call_result_75161)
    
    # Assigning a Call to a Name (line 723):
    
    # Assigning a Call to a Name (line 723):
    
    # Call to asarray(...): (line 723)
    # Processing the call arguments (line 723)
    # Getting the type of 'y' (line 723)
    y_75164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 23), 'y', False)
    # Processing the call keyword arguments (line 723)
    kwargs_75165 = {}
    # Getting the type of 'np' (line 723)
    np_75162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 723)
    asarray_75163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 12), np_75162, 'asarray')
    # Calling asarray(args, kwargs) (line 723)
    asarray_call_result_75166 = invoke(stypy.reporting.localization.Localization(__file__, 723, 12), asarray_75163, *[y_75164], **kwargs_75165)
    
    # Assigning a type to the variable 'c' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'c', asarray_call_result_75166)
    
    # Assigning a Call to a Name (line 724):
    
    # Assigning a Call to a Name (line 724):
    
    # Call to ascontiguousarray(...): (line 724)
    # Processing the call arguments (line 724)
    # Getting the type of 'c' (line 724)
    c_75169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 33), 'c', False)
    # Processing the call keyword arguments (line 724)
    
    # Call to _get_dtype(...): (line 724)
    # Processing the call arguments (line 724)
    # Getting the type of 'c' (line 724)
    c_75171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 53), 'c', False)
    # Obtaining the member 'dtype' of a type (line 724)
    dtype_75172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 53), c_75171, 'dtype')
    # Processing the call keyword arguments (line 724)
    kwargs_75173 = {}
    # Getting the type of '_get_dtype' (line 724)
    _get_dtype_75170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 42), '_get_dtype', False)
    # Calling _get_dtype(args, kwargs) (line 724)
    _get_dtype_call_result_75174 = invoke(stypy.reporting.localization.Localization(__file__, 724, 42), _get_dtype_75170, *[dtype_75172], **kwargs_75173)
    
    keyword_75175 = _get_dtype_call_result_75174
    kwargs_75176 = {'dtype': keyword_75175}
    # Getting the type of 'np' (line 724)
    np_75167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 724)
    ascontiguousarray_75168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 12), np_75167, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 724)
    ascontiguousarray_call_result_75177 = invoke(stypy.reporting.localization.Localization(__file__, 724, 12), ascontiguousarray_75168, *[c_75169], **kwargs_75176)
    
    # Assigning a type to the variable 'c' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'c', ascontiguousarray_call_result_75177)
    
    # Call to construct_fast(...): (line 725)
    # Processing the call arguments (line 725)
    # Getting the type of 't' (line 725)
    t_75180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 38), 't', False)
    # Getting the type of 'c' (line 725)
    c_75181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 41), 'c', False)
    # Getting the type of 'k' (line 725)
    k_75182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 44), 'k', False)
    # Processing the call keyword arguments (line 725)
    # Getting the type of 'axis' (line 725)
    axis_75183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 52), 'axis', False)
    keyword_75184 = axis_75183
    kwargs_75185 = {'axis': keyword_75184}
    # Getting the type of 'BSpline' (line 725)
    BSpline_75178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 15), 'BSpline', False)
    # Obtaining the member 'construct_fast' of a type (line 725)
    construct_fast_75179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 15), BSpline_75178, 'construct_fast')
    # Calling construct_fast(args, kwargs) (line 725)
    construct_fast_call_result_75186 = invoke(stypy.reporting.localization.Localization(__file__, 725, 15), construct_fast_75179, *[t_75180, c_75181, k_75182], **kwargs_75185)
    
    # Assigning a type to the variable 'stypy_return_type' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'stypy_return_type', construct_fast_call_result_75186)
    # SSA join for if statement (line 718)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 728)
    # Getting the type of 't' (line 728)
    t_75187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 7), 't')
    # Getting the type of 'None' (line 728)
    None_75188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 12), 'None')
    
    (may_be_75189, more_types_in_union_75190) = may_be_none(t_75187, None_75188)

    if may_be_75189:

        if more_types_in_union_75190:
            # Runtime conditional SSA (line 728)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'deriv_l' (line 729)
        deriv_l_75191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 11), 'deriv_l')
        # Getting the type of 'None' (line 729)
        None_75192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 22), 'None')
        # Applying the binary operator 'is' (line 729)
        result_is__75193 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 11), 'is', deriv_l_75191, None_75192)
        
        
        # Getting the type of 'deriv_r' (line 729)
        deriv_r_75194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 31), 'deriv_r')
        # Getting the type of 'None' (line 729)
        None_75195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 42), 'None')
        # Applying the binary operator 'is' (line 729)
        result_is__75196 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 31), 'is', deriv_r_75194, None_75195)
        
        # Applying the binary operator 'and' (line 729)
        result_and_keyword_75197 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 11), 'and', result_is__75193, result_is__75196)
        
        # Testing the type of an if condition (line 729)
        if_condition_75198 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 8), result_and_keyword_75197)
        # Assigning a type to the variable 'if_condition_75198' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'if_condition_75198', if_condition_75198)
        # SSA begins for if statement (line 729)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'k' (line 730)
        k_75199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 15), 'k')
        int_75200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 20), 'int')
        # Applying the binary operator '==' (line 730)
        result_eq_75201 = python_operator(stypy.reporting.localization.Localization(__file__, 730, 15), '==', k_75199, int_75200)
        
        # Testing the type of an if condition (line 730)
        if_condition_75202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 730, 12), result_eq_75201)
        # Assigning a type to the variable 'if_condition_75202' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'if_condition_75202', if_condition_75202)
        # SSA begins for if statement (line 730)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 733):
        
        # Assigning a BinOp to a Name (line 733):
        
        # Obtaining the type of the subscript
        int_75203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 23), 'int')
        slice_75204 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 733, 21), int_75203, None, None)
        # Getting the type of 'x' (line 733)
        x_75205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 21), 'x')
        # Obtaining the member '__getitem__' of a type (line 733)
        getitem___75206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 21), x_75205, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 733)
        subscript_call_result_75207 = invoke(stypy.reporting.localization.Localization(__file__, 733, 21), getitem___75206, slice_75204)
        
        
        # Obtaining the type of the subscript
        int_75208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 32), 'int')
        slice_75209 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 733, 29), None, int_75208, None)
        # Getting the type of 'x' (line 733)
        x_75210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 29), 'x')
        # Obtaining the member '__getitem__' of a type (line 733)
        getitem___75211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 29), x_75210, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 733)
        subscript_call_result_75212 = invoke(stypy.reporting.localization.Localization(__file__, 733, 29), getitem___75211, slice_75209)
        
        # Applying the binary operator '+' (line 733)
        result_add_75213 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 21), '+', subscript_call_result_75207, subscript_call_result_75212)
        
        float_75214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 39), 'float')
        # Applying the binary operator 'div' (line 733)
        result_div_75215 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 20), 'div', result_add_75213, float_75214)
        
        # Assigning a type to the variable 't' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 16), 't', result_div_75215)
        
        # Assigning a Subscript to a Name (line 734):
        
        # Assigning a Subscript to a Name (line 734):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 734)
        tuple_75216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 734)
        # Adding element type (line 734)
        
        # Obtaining an instance of the builtin type 'tuple' (line 734)
        tuple_75217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 734)
        # Adding element type (line 734)
        
        # Obtaining the type of the subscript
        int_75218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 29), 'int')
        # Getting the type of 'x' (line 734)
        x_75219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 27), 'x')
        # Obtaining the member '__getitem__' of a type (line 734)
        getitem___75220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 27), x_75219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 734)
        subscript_call_result_75221 = invoke(stypy.reporting.localization.Localization(__file__, 734, 27), getitem___75220, int_75218)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 734, 27), tuple_75217, subscript_call_result_75221)
        
        # Getting the type of 'k' (line 734)
        k_75222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 35), 'k')
        int_75223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 37), 'int')
        # Applying the binary operator '+' (line 734)
        result_add_75224 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 35), '+', k_75222, int_75223)
        
        # Applying the binary operator '*' (line 734)
        result_mul_75225 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 26), '*', tuple_75217, result_add_75224)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 734, 26), tuple_75216, result_mul_75225)
        # Adding element type (line 734)
        
        # Obtaining the type of the subscript
        int_75226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 29), 'int')
        int_75227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 31), 'int')
        slice_75228 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 735, 27), int_75226, int_75227, None)
        # Getting the type of 't' (line 735)
        t_75229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 27), 't')
        # Obtaining the member '__getitem__' of a type (line 735)
        getitem___75230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 27), t_75229, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 735)
        subscript_call_result_75231 = invoke(stypy.reporting.localization.Localization(__file__, 735, 27), getitem___75230, slice_75228)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 734, 26), tuple_75216, subscript_call_result_75231)
        # Adding element type (line 734)
        
        # Obtaining an instance of the builtin type 'tuple' (line 736)
        tuple_75232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 736)
        # Adding element type (line 736)
        
        # Obtaining the type of the subscript
        int_75233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 30), 'int')
        # Getting the type of 'x' (line 736)
        x_75234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 28), 'x')
        # Obtaining the member '__getitem__' of a type (line 736)
        getitem___75235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 28), x_75234, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 736)
        subscript_call_result_75236 = invoke(stypy.reporting.localization.Localization(__file__, 736, 28), getitem___75235, int_75233)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 28), tuple_75232, subscript_call_result_75236)
        
        # Getting the type of 'k' (line 736)
        k_75237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 37), 'k')
        int_75238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 39), 'int')
        # Applying the binary operator '+' (line 736)
        result_add_75239 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 37), '+', k_75237, int_75238)
        
        # Applying the binary operator '*' (line 736)
        result_mul_75240 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 27), '*', tuple_75232, result_add_75239)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 734, 26), tuple_75216, result_mul_75240)
        
        # Getting the type of 'np' (line 734)
        np_75241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 20), 'np')
        # Obtaining the member 'r_' of a type (line 734)
        r__75242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 20), np_75241, 'r_')
        # Obtaining the member '__getitem__' of a type (line 734)
        getitem___75243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 20), r__75242, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 734)
        subscript_call_result_75244 = invoke(stypy.reporting.localization.Localization(__file__, 734, 20), getitem___75243, tuple_75216)
        
        # Assigning a type to the variable 't' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 16), 't', subscript_call_result_75244)
        # SSA branch for the else part of an if statement (line 730)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 738):
        
        # Assigning a Call to a Name (line 738):
        
        # Call to _not_a_knot(...): (line 738)
        # Processing the call arguments (line 738)
        # Getting the type of 'x' (line 738)
        x_75246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 32), 'x', False)
        # Getting the type of 'k' (line 738)
        k_75247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 35), 'k', False)
        # Processing the call keyword arguments (line 738)
        kwargs_75248 = {}
        # Getting the type of '_not_a_knot' (line 738)
        _not_a_knot_75245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 20), '_not_a_knot', False)
        # Calling _not_a_knot(args, kwargs) (line 738)
        _not_a_knot_call_result_75249 = invoke(stypy.reporting.localization.Localization(__file__, 738, 20), _not_a_knot_75245, *[x_75246, k_75247], **kwargs_75248)
        
        # Assigning a type to the variable 't' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 16), 't', _not_a_knot_call_result_75249)
        # SSA join for if statement (line 730)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 729)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 740):
        
        # Assigning a Call to a Name (line 740):
        
        # Call to _augknt(...): (line 740)
        # Processing the call arguments (line 740)
        # Getting the type of 'x' (line 740)
        x_75251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 24), 'x', False)
        # Getting the type of 'k' (line 740)
        k_75252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 27), 'k', False)
        # Processing the call keyword arguments (line 740)
        kwargs_75253 = {}
        # Getting the type of '_augknt' (line 740)
        _augknt_75250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 16), '_augknt', False)
        # Calling _augknt(args, kwargs) (line 740)
        _augknt_call_result_75254 = invoke(stypy.reporting.localization.Localization(__file__, 740, 16), _augknt_75250, *[x_75251, k_75252], **kwargs_75253)
        
        # Assigning a type to the variable 't' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 't', _augknt_call_result_75254)
        # SSA join for if statement (line 729)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_75190:
            # SSA join for if statement (line 728)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 742):
    
    # Assigning a Call to a Name (line 742):
    
    # Call to _as_float_array(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'x' (line 742)
    x_75256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 24), 'x', False)
    # Getting the type of 'check_finite' (line 742)
    check_finite_75257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 27), 'check_finite', False)
    # Processing the call keyword arguments (line 742)
    kwargs_75258 = {}
    # Getting the type of '_as_float_array' (line 742)
    _as_float_array_75255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), '_as_float_array', False)
    # Calling _as_float_array(args, kwargs) (line 742)
    _as_float_array_call_result_75259 = invoke(stypy.reporting.localization.Localization(__file__, 742, 8), _as_float_array_75255, *[x_75256, check_finite_75257], **kwargs_75258)
    
    # Assigning a type to the variable 'x' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'x', _as_float_array_call_result_75259)
    
    # Assigning a Call to a Name (line 743):
    
    # Assigning a Call to a Name (line 743):
    
    # Call to _as_float_array(...): (line 743)
    # Processing the call arguments (line 743)
    # Getting the type of 'y' (line 743)
    y_75261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 24), 'y', False)
    # Getting the type of 'check_finite' (line 743)
    check_finite_75262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 27), 'check_finite', False)
    # Processing the call keyword arguments (line 743)
    kwargs_75263 = {}
    # Getting the type of '_as_float_array' (line 743)
    _as_float_array_75260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), '_as_float_array', False)
    # Calling _as_float_array(args, kwargs) (line 743)
    _as_float_array_call_result_75264 = invoke(stypy.reporting.localization.Localization(__file__, 743, 8), _as_float_array_75260, *[y_75261, check_finite_75262], **kwargs_75263)
    
    # Assigning a type to the variable 'y' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 4), 'y', _as_float_array_call_result_75264)
    
    # Assigning a Call to a Name (line 744):
    
    # Assigning a Call to a Name (line 744):
    
    # Call to _as_float_array(...): (line 744)
    # Processing the call arguments (line 744)
    # Getting the type of 't' (line 744)
    t_75266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 24), 't', False)
    # Getting the type of 'check_finite' (line 744)
    check_finite_75267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 27), 'check_finite', False)
    # Processing the call keyword arguments (line 744)
    kwargs_75268 = {}
    # Getting the type of '_as_float_array' (line 744)
    _as_float_array_75265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), '_as_float_array', False)
    # Calling _as_float_array(args, kwargs) (line 744)
    _as_float_array_call_result_75269 = invoke(stypy.reporting.localization.Localization(__file__, 744, 8), _as_float_array_75265, *[t_75266, check_finite_75267], **kwargs_75268)
    
    # Assigning a type to the variable 't' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 't', _as_float_array_call_result_75269)
    
    # Assigning a Call to a Name (line 745):
    
    # Assigning a Call to a Name (line 745):
    
    # Call to int(...): (line 745)
    # Processing the call arguments (line 745)
    # Getting the type of 'k' (line 745)
    k_75271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'k', False)
    # Processing the call keyword arguments (line 745)
    kwargs_75272 = {}
    # Getting the type of 'int' (line 745)
    int_75270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'int', False)
    # Calling int(args, kwargs) (line 745)
    int_call_result_75273 = invoke(stypy.reporting.localization.Localization(__file__, 745, 8), int_75270, *[k_75271], **kwargs_75272)
    
    # Assigning a type to the variable 'k' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'k', int_call_result_75273)
    
    # Assigning a BinOp to a Name (line 747):
    
    # Assigning a BinOp to a Name (line 747):
    # Getting the type of 'axis' (line 747)
    axis_75274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 11), 'axis')
    # Getting the type of 'y' (line 747)
    y_75275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 18), 'y')
    # Obtaining the member 'ndim' of a type (line 747)
    ndim_75276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 18), y_75275, 'ndim')
    # Applying the binary operator '%' (line 747)
    result_mod_75277 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 11), '%', axis_75274, ndim_75276)
    
    # Assigning a type to the variable 'axis' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'axis', result_mod_75277)
    
    # Assigning a Call to a Name (line 748):
    
    # Assigning a Call to a Name (line 748):
    
    # Call to rollaxis(...): (line 748)
    # Processing the call arguments (line 748)
    # Getting the type of 'y' (line 748)
    y_75280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 20), 'y', False)
    # Getting the type of 'axis' (line 748)
    axis_75281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 23), 'axis', False)
    # Processing the call keyword arguments (line 748)
    kwargs_75282 = {}
    # Getting the type of 'np' (line 748)
    np_75278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 748)
    rollaxis_75279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 8), np_75278, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 748)
    rollaxis_call_result_75283 = invoke(stypy.reporting.localization.Localization(__file__, 748, 8), rollaxis_75279, *[y_75280, axis_75281], **kwargs_75282)
    
    # Assigning a type to the variable 'y' (line 748)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'y', rollaxis_call_result_75283)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 750)
    x_75284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 750)
    ndim_75285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 7), x_75284, 'ndim')
    int_75286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 17), 'int')
    # Applying the binary operator '!=' (line 750)
    result_ne_75287 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 7), '!=', ndim_75285, int_75286)
    
    
    # Call to any(...): (line 750)
    # Processing the call arguments (line 750)
    
    
    # Obtaining the type of the subscript
    int_75290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 31), 'int')
    slice_75291 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 750, 29), int_75290, None, None)
    # Getting the type of 'x' (line 750)
    x_75292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 29), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 750)
    getitem___75293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 29), x_75292, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 750)
    subscript_call_result_75294 = invoke(stypy.reporting.localization.Localization(__file__, 750, 29), getitem___75293, slice_75291)
    
    
    # Obtaining the type of the subscript
    int_75295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 41), 'int')
    slice_75296 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 750, 38), None, int_75295, None)
    # Getting the type of 'x' (line 750)
    x_75297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 38), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 750)
    getitem___75298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 38), x_75297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 750)
    subscript_call_result_75299 = invoke(stypy.reporting.localization.Localization(__file__, 750, 38), getitem___75298, slice_75296)
    
    # Applying the binary operator '<=' (line 750)
    result_le_75300 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 29), '<=', subscript_call_result_75294, subscript_call_result_75299)
    
    # Processing the call keyword arguments (line 750)
    kwargs_75301 = {}
    # Getting the type of 'np' (line 750)
    np_75288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 22), 'np', False)
    # Obtaining the member 'any' of a type (line 750)
    any_75289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 22), np_75288, 'any')
    # Calling any(args, kwargs) (line 750)
    any_call_result_75302 = invoke(stypy.reporting.localization.Localization(__file__, 750, 22), any_75289, *[result_le_75300], **kwargs_75301)
    
    # Applying the binary operator 'or' (line 750)
    result_or_keyword_75303 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 7), 'or', result_ne_75287, any_call_result_75302)
    
    # Testing the type of an if condition (line 750)
    if_condition_75304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 750, 4), result_or_keyword_75303)
    # Assigning a type to the variable 'if_condition_75304' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'if_condition_75304', if_condition_75304)
    # SSA begins for if statement (line 750)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 751)
    # Processing the call arguments (line 751)
    str_75306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 25), 'str', 'Expect x to be a 1-D sorted array_like.')
    # Processing the call keyword arguments (line 751)
    kwargs_75307 = {}
    # Getting the type of 'ValueError' (line 751)
    ValueError_75305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 751)
    ValueError_call_result_75308 = invoke(stypy.reporting.localization.Localization(__file__, 751, 14), ValueError_75305, *[str_75306], **kwargs_75307)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 751, 8), ValueError_call_result_75308, 'raise parameter', BaseException)
    # SSA join for if statement (line 750)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 752)
    k_75309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 7), 'k')
    int_75310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 11), 'int')
    # Applying the binary operator '<' (line 752)
    result_lt_75311 = python_operator(stypy.reporting.localization.Localization(__file__, 752, 7), '<', k_75309, int_75310)
    
    # Testing the type of an if condition (line 752)
    if_condition_75312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 752, 4), result_lt_75311)
    # Assigning a type to the variable 'if_condition_75312' (line 752)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 4), 'if_condition_75312', if_condition_75312)
    # SSA begins for if statement (line 752)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 753)
    # Processing the call arguments (line 753)
    str_75314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 25), 'str', 'Expect non-negative k.')
    # Processing the call keyword arguments (line 753)
    kwargs_75315 = {}
    # Getting the type of 'ValueError' (line 753)
    ValueError_75313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 753)
    ValueError_call_result_75316 = invoke(stypy.reporting.localization.Localization(__file__, 753, 14), ValueError_75313, *[str_75314], **kwargs_75315)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 753, 8), ValueError_call_result_75316, 'raise parameter', BaseException)
    # SSA join for if statement (line 752)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 't' (line 754)
    t_75317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 7), 't')
    # Obtaining the member 'ndim' of a type (line 754)
    ndim_75318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 7), t_75317, 'ndim')
    int_75319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 17), 'int')
    # Applying the binary operator '!=' (line 754)
    result_ne_75320 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 7), '!=', ndim_75318, int_75319)
    
    
    # Call to any(...): (line 754)
    # Processing the call arguments (line 754)
    
    
    # Obtaining the type of the subscript
    int_75323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 31), 'int')
    slice_75324 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 754, 29), int_75323, None, None)
    # Getting the type of 't' (line 754)
    t_75325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 29), 't', False)
    # Obtaining the member '__getitem__' of a type (line 754)
    getitem___75326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 29), t_75325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 754)
    subscript_call_result_75327 = invoke(stypy.reporting.localization.Localization(__file__, 754, 29), getitem___75326, slice_75324)
    
    
    # Obtaining the type of the subscript
    int_75328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 40), 'int')
    slice_75329 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 754, 37), None, int_75328, None)
    # Getting the type of 't' (line 754)
    t_75330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 37), 't', False)
    # Obtaining the member '__getitem__' of a type (line 754)
    getitem___75331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 37), t_75330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 754)
    subscript_call_result_75332 = invoke(stypy.reporting.localization.Localization(__file__, 754, 37), getitem___75331, slice_75329)
    
    # Applying the binary operator '<' (line 754)
    result_lt_75333 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 29), '<', subscript_call_result_75327, subscript_call_result_75332)
    
    # Processing the call keyword arguments (line 754)
    kwargs_75334 = {}
    # Getting the type of 'np' (line 754)
    np_75321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 22), 'np', False)
    # Obtaining the member 'any' of a type (line 754)
    any_75322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 22), np_75321, 'any')
    # Calling any(args, kwargs) (line 754)
    any_call_result_75335 = invoke(stypy.reporting.localization.Localization(__file__, 754, 22), any_75322, *[result_lt_75333], **kwargs_75334)
    
    # Applying the binary operator 'or' (line 754)
    result_or_keyword_75336 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 7), 'or', result_ne_75320, any_call_result_75335)
    
    # Testing the type of an if condition (line 754)
    if_condition_75337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 754, 4), result_or_keyword_75336)
    # Assigning a type to the variable 'if_condition_75337' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'if_condition_75337', if_condition_75337)
    # SSA begins for if statement (line 754)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 755)
    # Processing the call arguments (line 755)
    str_75339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 25), 'str', 'Expect t to be a 1-D sorted array_like.')
    # Processing the call keyword arguments (line 755)
    kwargs_75340 = {}
    # Getting the type of 'ValueError' (line 755)
    ValueError_75338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 755)
    ValueError_call_result_75341 = invoke(stypy.reporting.localization.Localization(__file__, 755, 14), ValueError_75338, *[str_75339], **kwargs_75340)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 755, 8), ValueError_call_result_75341, 'raise parameter', BaseException)
    # SSA join for if statement (line 754)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 756)
    x_75342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 7), 'x')
    # Obtaining the member 'size' of a type (line 756)
    size_75343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 7), x_75342, 'size')
    
    # Obtaining the type of the subscript
    int_75344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 25), 'int')
    # Getting the type of 'y' (line 756)
    y_75345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 17), 'y')
    # Obtaining the member 'shape' of a type (line 756)
    shape_75346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 17), y_75345, 'shape')
    # Obtaining the member '__getitem__' of a type (line 756)
    getitem___75347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 17), shape_75346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 756)
    subscript_call_result_75348 = invoke(stypy.reporting.localization.Localization(__file__, 756, 17), getitem___75347, int_75344)
    
    # Applying the binary operator '!=' (line 756)
    result_ne_75349 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 7), '!=', size_75343, subscript_call_result_75348)
    
    # Testing the type of an if condition (line 756)
    if_condition_75350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 756, 4), result_ne_75349)
    # Assigning a type to the variable 'if_condition_75350' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'if_condition_75350', if_condition_75350)
    # SSA begins for if statement (line 756)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 757)
    # Processing the call arguments (line 757)
    str_75352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 25), 'str', 'x and y are incompatible.')
    # Processing the call keyword arguments (line 757)
    kwargs_75353 = {}
    # Getting the type of 'ValueError' (line 757)
    ValueError_75351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 757)
    ValueError_call_result_75354 = invoke(stypy.reporting.localization.Localization(__file__, 757, 14), ValueError_75351, *[str_75352], **kwargs_75353)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 757, 8), ValueError_call_result_75354, 'raise parameter', BaseException)
    # SSA join for if statement (line 756)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 't' (line 758)
    t_75355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 7), 't')
    # Obtaining the member 'size' of a type (line 758)
    size_75356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 7), t_75355, 'size')
    # Getting the type of 'x' (line 758)
    x_75357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'x')
    # Obtaining the member 'size' of a type (line 758)
    size_75358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 16), x_75357, 'size')
    # Getting the type of 'k' (line 758)
    k_75359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 25), 'k')
    # Applying the binary operator '+' (line 758)
    result_add_75360 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 16), '+', size_75358, k_75359)
    
    int_75361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 29), 'int')
    # Applying the binary operator '+' (line 758)
    result_add_75362 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 27), '+', result_add_75360, int_75361)
    
    # Applying the binary operator '<' (line 758)
    result_lt_75363 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 7), '<', size_75356, result_add_75362)
    
    # Testing the type of an if condition (line 758)
    if_condition_75364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 758, 4), result_lt_75363)
    # Assigning a type to the variable 'if_condition_75364' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'if_condition_75364', if_condition_75364)
    # SSA begins for if statement (line 758)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 759)
    # Processing the call arguments (line 759)
    str_75366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 25), 'str', 'Got %d knots, need at least %d.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 760)
    tuple_75367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 760)
    # Adding element type (line 760)
    # Getting the type of 't' (line 760)
    t_75368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 26), 't', False)
    # Obtaining the member 'size' of a type (line 760)
    size_75369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 26), t_75368, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 26), tuple_75367, size_75369)
    # Adding element type (line 760)
    # Getting the type of 'x' (line 760)
    x_75370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 34), 'x', False)
    # Obtaining the member 'size' of a type (line 760)
    size_75371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 34), x_75370, 'size')
    # Getting the type of 'k' (line 760)
    k_75372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 43), 'k', False)
    # Applying the binary operator '+' (line 760)
    result_add_75373 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 34), '+', size_75371, k_75372)
    
    int_75374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 47), 'int')
    # Applying the binary operator '+' (line 760)
    result_add_75375 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 45), '+', result_add_75373, int_75374)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 26), tuple_75367, result_add_75375)
    
    # Applying the binary operator '%' (line 759)
    result_mod_75376 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 25), '%', str_75366, tuple_75367)
    
    # Processing the call keyword arguments (line 759)
    kwargs_75377 = {}
    # Getting the type of 'ValueError' (line 759)
    ValueError_75365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 759)
    ValueError_call_result_75378 = invoke(stypy.reporting.localization.Localization(__file__, 759, 14), ValueError_75365, *[result_mod_75376], **kwargs_75377)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 759, 8), ValueError_call_result_75378, 'raise parameter', BaseException)
    # SSA join for if statement (line 758)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_75379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 10), 'int')
    # Getting the type of 'x' (line 761)
    x_75380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___75381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 8), x_75380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_75382 = invoke(stypy.reporting.localization.Localization(__file__, 761, 8), getitem___75381, int_75379)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 761)
    k_75383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 17), 'k')
    # Getting the type of 't' (line 761)
    t_75384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 15), 't')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___75385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 15), t_75384, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_75386 = invoke(stypy.reporting.localization.Localization(__file__, 761, 15), getitem___75385, k_75383)
    
    # Applying the binary operator '<' (line 761)
    result_lt_75387 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 8), '<', subscript_call_result_75382, subscript_call_result_75386)
    
    
    
    # Obtaining the type of the subscript
    int_75388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 27), 'int')
    # Getting the type of 'x' (line 761)
    x_75389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 25), 'x')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___75390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 25), x_75389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_75391 = invoke(stypy.reporting.localization.Localization(__file__, 761, 25), getitem___75390, int_75388)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'k' (line 761)
    k_75392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 36), 'k')
    # Applying the 'usub' unary operator (line 761)
    result___neg___75393 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 35), 'usub', k_75392)
    
    # Getting the type of 't' (line 761)
    t_75394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 33), 't')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___75395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 33), t_75394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_75396 = invoke(stypy.reporting.localization.Localization(__file__, 761, 33), getitem___75395, result___neg___75393)
    
    # Applying the binary operator '>' (line 761)
    result_gt_75397 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 25), '>', subscript_call_result_75391, subscript_call_result_75396)
    
    # Applying the binary operator 'or' (line 761)
    result_or_keyword_75398 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 7), 'or', result_lt_75387, result_gt_75397)
    
    # Testing the type of an if condition (line 761)
    if_condition_75399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 761, 4), result_or_keyword_75398)
    # Assigning a type to the variable 'if_condition_75399' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'if_condition_75399', if_condition_75399)
    # SSA begins for if statement (line 761)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 762)
    # Processing the call arguments (line 762)
    str_75401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 25), 'str', 'Out of bounds w/ x = %s.')
    # Getting the type of 'x' (line 762)
    x_75402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 54), 'x', False)
    # Applying the binary operator '%' (line 762)
    result_mod_75403 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 25), '%', str_75401, x_75402)
    
    # Processing the call keyword arguments (line 762)
    kwargs_75404 = {}
    # Getting the type of 'ValueError' (line 762)
    ValueError_75400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 762)
    ValueError_call_result_75405 = invoke(stypy.reporting.localization.Localization(__file__, 762, 14), ValueError_75400, *[result_mod_75403], **kwargs_75404)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 762, 8), ValueError_call_result_75405, 'raise parameter', BaseException)
    # SSA join for if statement (line 761)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 765)
    # Getting the type of 'deriv_l' (line 765)
    deriv_l_75406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'deriv_l')
    # Getting the type of 'None' (line 765)
    None_75407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 22), 'None')
    
    (may_be_75408, more_types_in_union_75409) = may_not_be_none(deriv_l_75406, None_75407)

    if may_be_75408:

        if more_types_in_union_75409:
            # Runtime conditional SSA (line 765)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Tuple (line 766):
        
        # Assigning a Subscript to a Name (line 766):
        
        # Obtaining the type of the subscript
        int_75410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 8), 'int')
        
        # Call to zip(...): (line 766)
        # Getting the type of 'deriv_l' (line 766)
        deriv_l_75412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 42), 'deriv_l', False)
        # Processing the call keyword arguments (line 766)
        kwargs_75413 = {}
        # Getting the type of 'zip' (line 766)
        zip_75411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 37), 'zip', False)
        # Calling zip(args, kwargs) (line 766)
        zip_call_result_75414 = invoke(stypy.reporting.localization.Localization(__file__, 766, 37), zip_75411, *[deriv_l_75412], **kwargs_75413)
        
        # Obtaining the member '__getitem__' of a type (line 766)
        getitem___75415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 8), zip_call_result_75414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 766)
        subscript_call_result_75416 = invoke(stypy.reporting.localization.Localization(__file__, 766, 8), getitem___75415, int_75410)
        
        # Assigning a type to the variable 'tuple_var_assignment_73692' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'tuple_var_assignment_73692', subscript_call_result_75416)
        
        # Assigning a Subscript to a Name (line 766):
        
        # Obtaining the type of the subscript
        int_75417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 8), 'int')
        
        # Call to zip(...): (line 766)
        # Getting the type of 'deriv_l' (line 766)
        deriv_l_75419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 42), 'deriv_l', False)
        # Processing the call keyword arguments (line 766)
        kwargs_75420 = {}
        # Getting the type of 'zip' (line 766)
        zip_75418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 37), 'zip', False)
        # Calling zip(args, kwargs) (line 766)
        zip_call_result_75421 = invoke(stypy.reporting.localization.Localization(__file__, 766, 37), zip_75418, *[deriv_l_75419], **kwargs_75420)
        
        # Obtaining the member '__getitem__' of a type (line 766)
        getitem___75422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 8), zip_call_result_75421, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 766)
        subscript_call_result_75423 = invoke(stypy.reporting.localization.Localization(__file__, 766, 8), getitem___75422, int_75417)
        
        # Assigning a type to the variable 'tuple_var_assignment_73693' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'tuple_var_assignment_73693', subscript_call_result_75423)
        
        # Assigning a Name to a Name (line 766):
        # Getting the type of 'tuple_var_assignment_73692' (line 766)
        tuple_var_assignment_73692_75424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'tuple_var_assignment_73692')
        # Assigning a type to the variable 'deriv_l_ords' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'deriv_l_ords', tuple_var_assignment_73692_75424)
        
        # Assigning a Name to a Name (line 766):
        # Getting the type of 'tuple_var_assignment_73693' (line 766)
        tuple_var_assignment_73693_75425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'tuple_var_assignment_73693')
        # Assigning a type to the variable 'deriv_l_vals' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 22), 'deriv_l_vals', tuple_var_assignment_73693_75425)

        if more_types_in_union_75409:
            # Runtime conditional SSA for else branch (line 765)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_75408) or more_types_in_union_75409):
        
        # Assigning a Tuple to a Tuple (line 768):
        
        # Assigning a List to a Name (line 768):
        
        # Obtaining an instance of the builtin type 'list' (line 768)
        list_75426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 768)
        
        # Assigning a type to the variable 'tuple_assignment_73694' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'tuple_assignment_73694', list_75426)
        
        # Assigning a List to a Name (line 768):
        
        # Obtaining an instance of the builtin type 'list' (line 768)
        list_75427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 768)
        
        # Assigning a type to the variable 'tuple_assignment_73695' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'tuple_assignment_73695', list_75427)
        
        # Assigning a Name to a Name (line 768):
        # Getting the type of 'tuple_assignment_73694' (line 768)
        tuple_assignment_73694_75428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'tuple_assignment_73694')
        # Assigning a type to the variable 'deriv_l_ords' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'deriv_l_ords', tuple_assignment_73694_75428)
        
        # Assigning a Name to a Name (line 768):
        # Getting the type of 'tuple_assignment_73695' (line 768)
        tuple_assignment_73695_75429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'tuple_assignment_73695')
        # Assigning a type to the variable 'deriv_l_vals' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 22), 'deriv_l_vals', tuple_assignment_73695_75429)

        if (may_be_75408 and more_types_in_union_75409):
            # SSA join for if statement (line 765)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 769):
    
    # Assigning a Subscript to a Name (line 769):
    
    # Obtaining the type of the subscript
    int_75430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 4), 'int')
    
    # Call to atleast_1d(...): (line 769)
    # Processing the call arguments (line 769)
    # Getting the type of 'deriv_l_ords' (line 769)
    deriv_l_ords_75433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 47), 'deriv_l_ords', False)
    # Getting the type of 'deriv_l_vals' (line 769)
    deriv_l_vals_75434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 61), 'deriv_l_vals', False)
    # Processing the call keyword arguments (line 769)
    kwargs_75435 = {}
    # Getting the type of 'np' (line 769)
    np_75431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 33), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 769)
    atleast_1d_75432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 33), np_75431, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 769)
    atleast_1d_call_result_75436 = invoke(stypy.reporting.localization.Localization(__file__, 769, 33), atleast_1d_75432, *[deriv_l_ords_75433, deriv_l_vals_75434], **kwargs_75435)
    
    # Obtaining the member '__getitem__' of a type (line 769)
    getitem___75437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 4), atleast_1d_call_result_75436, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 769)
    subscript_call_result_75438 = invoke(stypy.reporting.localization.Localization(__file__, 769, 4), getitem___75437, int_75430)
    
    # Assigning a type to the variable 'tuple_var_assignment_73696' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'tuple_var_assignment_73696', subscript_call_result_75438)
    
    # Assigning a Subscript to a Name (line 769):
    
    # Obtaining the type of the subscript
    int_75439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 4), 'int')
    
    # Call to atleast_1d(...): (line 769)
    # Processing the call arguments (line 769)
    # Getting the type of 'deriv_l_ords' (line 769)
    deriv_l_ords_75442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 47), 'deriv_l_ords', False)
    # Getting the type of 'deriv_l_vals' (line 769)
    deriv_l_vals_75443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 61), 'deriv_l_vals', False)
    # Processing the call keyword arguments (line 769)
    kwargs_75444 = {}
    # Getting the type of 'np' (line 769)
    np_75440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 33), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 769)
    atleast_1d_75441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 33), np_75440, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 769)
    atleast_1d_call_result_75445 = invoke(stypy.reporting.localization.Localization(__file__, 769, 33), atleast_1d_75441, *[deriv_l_ords_75442, deriv_l_vals_75443], **kwargs_75444)
    
    # Obtaining the member '__getitem__' of a type (line 769)
    getitem___75446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 4), atleast_1d_call_result_75445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 769)
    subscript_call_result_75447 = invoke(stypy.reporting.localization.Localization(__file__, 769, 4), getitem___75446, int_75439)
    
    # Assigning a type to the variable 'tuple_var_assignment_73697' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'tuple_var_assignment_73697', subscript_call_result_75447)
    
    # Assigning a Name to a Name (line 769):
    # Getting the type of 'tuple_var_assignment_73696' (line 769)
    tuple_var_assignment_73696_75448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'tuple_var_assignment_73696')
    # Assigning a type to the variable 'deriv_l_ords' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'deriv_l_ords', tuple_var_assignment_73696_75448)
    
    # Assigning a Name to a Name (line 769):
    # Getting the type of 'tuple_var_assignment_73697' (line 769)
    tuple_var_assignment_73697_75449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'tuple_var_assignment_73697')
    # Assigning a type to the variable 'deriv_l_vals' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 18), 'deriv_l_vals', tuple_var_assignment_73697_75449)
    
    # Assigning a Subscript to a Name (line 770):
    
    # Assigning a Subscript to a Name (line 770):
    
    # Obtaining the type of the subscript
    int_75450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 31), 'int')
    # Getting the type of 'deriv_l_ords' (line 770)
    deriv_l_ords_75451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'deriv_l_ords')
    # Obtaining the member 'shape' of a type (line 770)
    shape_75452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 12), deriv_l_ords_75451, 'shape')
    # Obtaining the member '__getitem__' of a type (line 770)
    getitem___75453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 12), shape_75452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 770)
    subscript_call_result_75454 = invoke(stypy.reporting.localization.Localization(__file__, 770, 12), getitem___75453, int_75450)
    
    # Assigning a type to the variable 'nleft' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 4), 'nleft', subscript_call_result_75454)
    
    # Type idiom detected: calculating its left and rigth part (line 772)
    # Getting the type of 'deriv_r' (line 772)
    deriv_r_75455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'deriv_r')
    # Getting the type of 'None' (line 772)
    None_75456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 22), 'None')
    
    (may_be_75457, more_types_in_union_75458) = may_not_be_none(deriv_r_75455, None_75456)

    if may_be_75457:

        if more_types_in_union_75458:
            # Runtime conditional SSA (line 772)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Tuple (line 773):
        
        # Assigning a Subscript to a Name (line 773):
        
        # Obtaining the type of the subscript
        int_75459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 8), 'int')
        
        # Call to zip(...): (line 773)
        # Getting the type of 'deriv_r' (line 773)
        deriv_r_75461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 42), 'deriv_r', False)
        # Processing the call keyword arguments (line 773)
        kwargs_75462 = {}
        # Getting the type of 'zip' (line 773)
        zip_75460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 37), 'zip', False)
        # Calling zip(args, kwargs) (line 773)
        zip_call_result_75463 = invoke(stypy.reporting.localization.Localization(__file__, 773, 37), zip_75460, *[deriv_r_75461], **kwargs_75462)
        
        # Obtaining the member '__getitem__' of a type (line 773)
        getitem___75464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 8), zip_call_result_75463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 773)
        subscript_call_result_75465 = invoke(stypy.reporting.localization.Localization(__file__, 773, 8), getitem___75464, int_75459)
        
        # Assigning a type to the variable 'tuple_var_assignment_73698' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'tuple_var_assignment_73698', subscript_call_result_75465)
        
        # Assigning a Subscript to a Name (line 773):
        
        # Obtaining the type of the subscript
        int_75466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 8), 'int')
        
        # Call to zip(...): (line 773)
        # Getting the type of 'deriv_r' (line 773)
        deriv_r_75468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 42), 'deriv_r', False)
        # Processing the call keyword arguments (line 773)
        kwargs_75469 = {}
        # Getting the type of 'zip' (line 773)
        zip_75467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 37), 'zip', False)
        # Calling zip(args, kwargs) (line 773)
        zip_call_result_75470 = invoke(stypy.reporting.localization.Localization(__file__, 773, 37), zip_75467, *[deriv_r_75468], **kwargs_75469)
        
        # Obtaining the member '__getitem__' of a type (line 773)
        getitem___75471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 8), zip_call_result_75470, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 773)
        subscript_call_result_75472 = invoke(stypy.reporting.localization.Localization(__file__, 773, 8), getitem___75471, int_75466)
        
        # Assigning a type to the variable 'tuple_var_assignment_73699' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'tuple_var_assignment_73699', subscript_call_result_75472)
        
        # Assigning a Name to a Name (line 773):
        # Getting the type of 'tuple_var_assignment_73698' (line 773)
        tuple_var_assignment_73698_75473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'tuple_var_assignment_73698')
        # Assigning a type to the variable 'deriv_r_ords' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'deriv_r_ords', tuple_var_assignment_73698_75473)
        
        # Assigning a Name to a Name (line 773):
        # Getting the type of 'tuple_var_assignment_73699' (line 773)
        tuple_var_assignment_73699_75474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'tuple_var_assignment_73699')
        # Assigning a type to the variable 'deriv_r_vals' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 22), 'deriv_r_vals', tuple_var_assignment_73699_75474)

        if more_types_in_union_75458:
            # Runtime conditional SSA for else branch (line 772)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_75457) or more_types_in_union_75458):
        
        # Assigning a Tuple to a Tuple (line 775):
        
        # Assigning a List to a Name (line 775):
        
        # Obtaining an instance of the builtin type 'list' (line 775)
        list_75475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 775)
        
        # Assigning a type to the variable 'tuple_assignment_73700' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_73700', list_75475)
        
        # Assigning a List to a Name (line 775):
        
        # Obtaining an instance of the builtin type 'list' (line 775)
        list_75476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 775)
        
        # Assigning a type to the variable 'tuple_assignment_73701' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_73701', list_75476)
        
        # Assigning a Name to a Name (line 775):
        # Getting the type of 'tuple_assignment_73700' (line 775)
        tuple_assignment_73700_75477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_73700')
        # Assigning a type to the variable 'deriv_r_ords' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'deriv_r_ords', tuple_assignment_73700_75477)
        
        # Assigning a Name to a Name (line 775):
        # Getting the type of 'tuple_assignment_73701' (line 775)
        tuple_assignment_73701_75478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_73701')
        # Assigning a type to the variable 'deriv_r_vals' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 22), 'deriv_r_vals', tuple_assignment_73701_75478)

        if (may_be_75457 and more_types_in_union_75458):
            # SSA join for if statement (line 772)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 776):
    
    # Assigning a Subscript to a Name (line 776):
    
    # Obtaining the type of the subscript
    int_75479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 4), 'int')
    
    # Call to atleast_1d(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'deriv_r_ords' (line 776)
    deriv_r_ords_75482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 47), 'deriv_r_ords', False)
    # Getting the type of 'deriv_r_vals' (line 776)
    deriv_r_vals_75483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 61), 'deriv_r_vals', False)
    # Processing the call keyword arguments (line 776)
    kwargs_75484 = {}
    # Getting the type of 'np' (line 776)
    np_75480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 33), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 776)
    atleast_1d_75481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 33), np_75480, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 776)
    atleast_1d_call_result_75485 = invoke(stypy.reporting.localization.Localization(__file__, 776, 33), atleast_1d_75481, *[deriv_r_ords_75482, deriv_r_vals_75483], **kwargs_75484)
    
    # Obtaining the member '__getitem__' of a type (line 776)
    getitem___75486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 4), atleast_1d_call_result_75485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 776)
    subscript_call_result_75487 = invoke(stypy.reporting.localization.Localization(__file__, 776, 4), getitem___75486, int_75479)
    
    # Assigning a type to the variable 'tuple_var_assignment_73702' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'tuple_var_assignment_73702', subscript_call_result_75487)
    
    # Assigning a Subscript to a Name (line 776):
    
    # Obtaining the type of the subscript
    int_75488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 4), 'int')
    
    # Call to atleast_1d(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'deriv_r_ords' (line 776)
    deriv_r_ords_75491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 47), 'deriv_r_ords', False)
    # Getting the type of 'deriv_r_vals' (line 776)
    deriv_r_vals_75492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 61), 'deriv_r_vals', False)
    # Processing the call keyword arguments (line 776)
    kwargs_75493 = {}
    # Getting the type of 'np' (line 776)
    np_75489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 33), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 776)
    atleast_1d_75490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 33), np_75489, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 776)
    atleast_1d_call_result_75494 = invoke(stypy.reporting.localization.Localization(__file__, 776, 33), atleast_1d_75490, *[deriv_r_ords_75491, deriv_r_vals_75492], **kwargs_75493)
    
    # Obtaining the member '__getitem__' of a type (line 776)
    getitem___75495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 4), atleast_1d_call_result_75494, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 776)
    subscript_call_result_75496 = invoke(stypy.reporting.localization.Localization(__file__, 776, 4), getitem___75495, int_75488)
    
    # Assigning a type to the variable 'tuple_var_assignment_73703' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'tuple_var_assignment_73703', subscript_call_result_75496)
    
    # Assigning a Name to a Name (line 776):
    # Getting the type of 'tuple_var_assignment_73702' (line 776)
    tuple_var_assignment_73702_75497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'tuple_var_assignment_73702')
    # Assigning a type to the variable 'deriv_r_ords' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'deriv_r_ords', tuple_var_assignment_73702_75497)
    
    # Assigning a Name to a Name (line 776):
    # Getting the type of 'tuple_var_assignment_73703' (line 776)
    tuple_var_assignment_73703_75498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'tuple_var_assignment_73703')
    # Assigning a type to the variable 'deriv_r_vals' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 18), 'deriv_r_vals', tuple_var_assignment_73703_75498)
    
    # Assigning a Subscript to a Name (line 777):
    
    # Assigning a Subscript to a Name (line 777):
    
    # Obtaining the type of the subscript
    int_75499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 32), 'int')
    # Getting the type of 'deriv_r_ords' (line 777)
    deriv_r_ords_75500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 13), 'deriv_r_ords')
    # Obtaining the member 'shape' of a type (line 777)
    shape_75501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 13), deriv_r_ords_75500, 'shape')
    # Obtaining the member '__getitem__' of a type (line 777)
    getitem___75502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 13), shape_75501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 777)
    subscript_call_result_75503 = invoke(stypy.reporting.localization.Localization(__file__, 777, 13), getitem___75502, int_75499)
    
    # Assigning a type to the variable 'nright' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 4), 'nright', subscript_call_result_75503)
    
    # Assigning a Attribute to a Name (line 780):
    
    # Assigning a Attribute to a Name (line 780):
    # Getting the type of 'x' (line 780)
    x_75504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'x')
    # Obtaining the member 'size' of a type (line 780)
    size_75505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 8), x_75504, 'size')
    # Assigning a type to the variable 'n' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 4), 'n', size_75505)
    
    # Assigning a BinOp to a Name (line 781):
    
    # Assigning a BinOp to a Name (line 781):
    # Getting the type of 't' (line 781)
    t_75506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 9), 't')
    # Obtaining the member 'size' of a type (line 781)
    size_75507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 9), t_75506, 'size')
    # Getting the type of 'k' (line 781)
    k_75508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 18), 'k')
    # Applying the binary operator '-' (line 781)
    result_sub_75509 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 9), '-', size_75507, k_75508)
    
    int_75510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 22), 'int')
    # Applying the binary operator '-' (line 781)
    result_sub_75511 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 20), '-', result_sub_75509, int_75510)
    
    # Assigning a type to the variable 'nt' (line 781)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 4), 'nt', result_sub_75511)
    
    
    # Getting the type of 'nt' (line 783)
    nt_75512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 7), 'nt')
    # Getting the type of 'n' (line 783)
    n_75513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'n')
    # Applying the binary operator '-' (line 783)
    result_sub_75514 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 7), '-', nt_75512, n_75513)
    
    # Getting the type of 'nleft' (line 783)
    nleft_75515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 17), 'nleft')
    # Getting the type of 'nright' (line 783)
    nright_75516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 25), 'nright')
    # Applying the binary operator '+' (line 783)
    result_add_75517 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 17), '+', nleft_75515, nright_75516)
    
    # Applying the binary operator '!=' (line 783)
    result_ne_75518 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 7), '!=', result_sub_75514, result_add_75517)
    
    # Testing the type of an if condition (line 783)
    if_condition_75519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 783, 4), result_ne_75518)
    # Assigning a type to the variable 'if_condition_75519' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'if_condition_75519', if_condition_75519)
    # SSA begins for if statement (line 783)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 784)
    # Processing the call arguments (line 784)
    str_75521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 25), 'str', 'number of derivatives at boundaries.')
    # Processing the call keyword arguments (line 784)
    kwargs_75522 = {}
    # Getting the type of 'ValueError' (line 784)
    ValueError_75520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 784)
    ValueError_call_result_75523 = invoke(stypy.reporting.localization.Localization(__file__, 784, 14), ValueError_75520, *[str_75521], **kwargs_75522)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 784, 8), ValueError_call_result_75523, 'raise parameter', BaseException)
    # SSA join for if statement (line 783)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Name to a Name (line 787):
    # Getting the type of 'k' (line 787)
    k_75524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 14), 'k')
    # Assigning a type to the variable 'ku' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 9), 'ku', k_75524)
    
    # Assigning a Name to a Name (line 787):
    # Getting the type of 'ku' (line 787)
    ku_75525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 9), 'ku')
    # Assigning a type to the variable 'kl' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'kl', ku_75525)
    
    # Assigning a Call to a Name (line 788):
    
    # Assigning a Call to a Name (line 788):
    
    # Call to zeros(...): (line 788)
    # Processing the call arguments (line 788)
    
    # Obtaining an instance of the builtin type 'tuple' (line 788)
    tuple_75528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 788)
    # Adding element type (line 788)
    int_75529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 19), 'int')
    # Getting the type of 'kl' (line 788)
    kl_75530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 21), 'kl', False)
    # Applying the binary operator '*' (line 788)
    result_mul_75531 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 19), '*', int_75529, kl_75530)
    
    # Getting the type of 'ku' (line 788)
    ku_75532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 26), 'ku', False)
    # Applying the binary operator '+' (line 788)
    result_add_75533 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 19), '+', result_mul_75531, ku_75532)
    
    int_75534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 31), 'int')
    # Applying the binary operator '+' (line 788)
    result_add_75535 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 29), '+', result_add_75533, int_75534)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 19), tuple_75528, result_add_75535)
    # Adding element type (line 788)
    # Getting the type of 'nt' (line 788)
    nt_75536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 34), 'nt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 19), tuple_75528, nt_75536)
    
    # Processing the call keyword arguments (line 788)
    # Getting the type of 'np' (line 788)
    np_75537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 45), 'np', False)
    # Obtaining the member 'float_' of a type (line 788)
    float__75538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 45), np_75537, 'float_')
    keyword_75539 = float__75538
    str_75540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 62), 'str', 'F')
    keyword_75541 = str_75540
    kwargs_75542 = {'dtype': keyword_75539, 'order': keyword_75541}
    # Getting the type of 'np' (line 788)
    np_75526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 788)
    zeros_75527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 9), np_75526, 'zeros')
    # Calling zeros(args, kwargs) (line 788)
    zeros_call_result_75543 = invoke(stypy.reporting.localization.Localization(__file__, 788, 9), zeros_75527, *[tuple_75528], **kwargs_75542)
    
    # Assigning a type to the variable 'ab' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 4), 'ab', zeros_call_result_75543)
    
    # Call to _colloc(...): (line 789)
    # Processing the call arguments (line 789)
    # Getting the type of 'x' (line 789)
    x_75546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 18), 'x', False)
    # Getting the type of 't' (line 789)
    t_75547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 21), 't', False)
    # Getting the type of 'k' (line 789)
    k_75548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 24), 'k', False)
    # Getting the type of 'ab' (line 789)
    ab_75549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 27), 'ab', False)
    # Processing the call keyword arguments (line 789)
    # Getting the type of 'nleft' (line 789)
    nleft_75550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 38), 'nleft', False)
    keyword_75551 = nleft_75550
    kwargs_75552 = {'offset': keyword_75551}
    # Getting the type of '_bspl' (line 789)
    _bspl_75544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 4), '_bspl', False)
    # Obtaining the member '_colloc' of a type (line 789)
    _colloc_75545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 4), _bspl_75544, '_colloc')
    # Calling _colloc(args, kwargs) (line 789)
    _colloc_call_result_75553 = invoke(stypy.reporting.localization.Localization(__file__, 789, 4), _colloc_75545, *[x_75546, t_75547, k_75548, ab_75549], **kwargs_75552)
    
    
    
    # Getting the type of 'nleft' (line 790)
    nleft_75554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 7), 'nleft')
    int_75555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 15), 'int')
    # Applying the binary operator '>' (line 790)
    result_gt_75556 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 7), '>', nleft_75554, int_75555)
    
    # Testing the type of an if condition (line 790)
    if_condition_75557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 790, 4), result_gt_75556)
    # Assigning a type to the variable 'if_condition_75557' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 4), 'if_condition_75557', if_condition_75557)
    # SSA begins for if statement (line 790)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _handle_lhs_derivatives(...): (line 791)
    # Processing the call arguments (line 791)
    # Getting the type of 't' (line 791)
    t_75560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 38), 't', False)
    # Getting the type of 'k' (line 791)
    k_75561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 41), 'k', False)
    
    # Obtaining the type of the subscript
    int_75562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 46), 'int')
    # Getting the type of 'x' (line 791)
    x_75563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 44), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 791)
    getitem___75564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 44), x_75563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 791)
    subscript_call_result_75565 = invoke(stypy.reporting.localization.Localization(__file__, 791, 44), getitem___75564, int_75562)
    
    # Getting the type of 'ab' (line 791)
    ab_75566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 50), 'ab', False)
    # Getting the type of 'kl' (line 791)
    kl_75567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 54), 'kl', False)
    # Getting the type of 'ku' (line 791)
    ku_75568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 58), 'ku', False)
    # Getting the type of 'deriv_l_ords' (line 791)
    deriv_l_ords_75569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 62), 'deriv_l_ords', False)
    # Processing the call keyword arguments (line 791)
    kwargs_75570 = {}
    # Getting the type of '_bspl' (line 791)
    _bspl_75558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 8), '_bspl', False)
    # Obtaining the member '_handle_lhs_derivatives' of a type (line 791)
    _handle_lhs_derivatives_75559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 8), _bspl_75558, '_handle_lhs_derivatives')
    # Calling _handle_lhs_derivatives(args, kwargs) (line 791)
    _handle_lhs_derivatives_call_result_75571 = invoke(stypy.reporting.localization.Localization(__file__, 791, 8), _handle_lhs_derivatives_75559, *[t_75560, k_75561, subscript_call_result_75565, ab_75566, kl_75567, ku_75568, deriv_l_ords_75569], **kwargs_75570)
    
    # SSA join for if statement (line 790)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'nright' (line 792)
    nright_75572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 7), 'nright')
    int_75573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 16), 'int')
    # Applying the binary operator '>' (line 792)
    result_gt_75574 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 7), '>', nright_75572, int_75573)
    
    # Testing the type of an if condition (line 792)
    if_condition_75575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 792, 4), result_gt_75574)
    # Assigning a type to the variable 'if_condition_75575' (line 792)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 4), 'if_condition_75575', if_condition_75575)
    # SSA begins for if statement (line 792)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _handle_lhs_derivatives(...): (line 793)
    # Processing the call arguments (line 793)
    # Getting the type of 't' (line 793)
    t_75578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 38), 't', False)
    # Getting the type of 'k' (line 793)
    k_75579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 41), 'k', False)
    
    # Obtaining the type of the subscript
    int_75580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 46), 'int')
    # Getting the type of 'x' (line 793)
    x_75581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 44), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 793)
    getitem___75582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 44), x_75581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 793)
    subscript_call_result_75583 = invoke(stypy.reporting.localization.Localization(__file__, 793, 44), getitem___75582, int_75580)
    
    # Getting the type of 'ab' (line 793)
    ab_75584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 51), 'ab', False)
    # Getting the type of 'kl' (line 793)
    kl_75585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 55), 'kl', False)
    # Getting the type of 'ku' (line 793)
    ku_75586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 59), 'ku', False)
    # Getting the type of 'deriv_r_ords' (line 793)
    deriv_r_ords_75587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 63), 'deriv_r_ords', False)
    # Processing the call keyword arguments (line 793)
    # Getting the type of 'nt' (line 794)
    nt_75588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 39), 'nt', False)
    # Getting the type of 'nright' (line 794)
    nright_75589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 42), 'nright', False)
    # Applying the binary operator '-' (line 794)
    result_sub_75590 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 39), '-', nt_75588, nright_75589)
    
    keyword_75591 = result_sub_75590
    kwargs_75592 = {'offset': keyword_75591}
    # Getting the type of '_bspl' (line 793)
    _bspl_75576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), '_bspl', False)
    # Obtaining the member '_handle_lhs_derivatives' of a type (line 793)
    _handle_lhs_derivatives_75577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), _bspl_75576, '_handle_lhs_derivatives')
    # Calling _handle_lhs_derivatives(args, kwargs) (line 793)
    _handle_lhs_derivatives_call_result_75593 = invoke(stypy.reporting.localization.Localization(__file__, 793, 8), _handle_lhs_derivatives_75577, *[t_75578, k_75579, subscript_call_result_75583, ab_75584, kl_75585, ku_75586, deriv_r_ords_75587], **kwargs_75592)
    
    # SSA join for if statement (line 792)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 797):
    
    # Assigning a Call to a Name (line 797):
    
    # Call to prod(...): (line 797)
    # Processing the call arguments (line 797)
    
    # Obtaining the type of the subscript
    int_75595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 28), 'int')
    slice_75596 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 797, 20), int_75595, None, None)
    # Getting the type of 'y' (line 797)
    y_75597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 20), 'y', False)
    # Obtaining the member 'shape' of a type (line 797)
    shape_75598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 20), y_75597, 'shape')
    # Obtaining the member '__getitem__' of a type (line 797)
    getitem___75599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 20), shape_75598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 797)
    subscript_call_result_75600 = invoke(stypy.reporting.localization.Localization(__file__, 797, 20), getitem___75599, slice_75596)
    
    # Processing the call keyword arguments (line 797)
    kwargs_75601 = {}
    # Getting the type of 'prod' (line 797)
    prod_75594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 15), 'prod', False)
    # Calling prod(args, kwargs) (line 797)
    prod_call_result_75602 = invoke(stypy.reporting.localization.Localization(__file__, 797, 15), prod_75594, *[subscript_call_result_75600], **kwargs_75601)
    
    # Assigning a type to the variable 'extradim' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'extradim', prod_call_result_75602)
    
    # Assigning a Call to a Name (line 798):
    
    # Assigning a Call to a Name (line 798):
    
    # Call to empty(...): (line 798)
    # Processing the call arguments (line 798)
    
    # Obtaining an instance of the builtin type 'tuple' (line 798)
    tuple_75605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 798)
    # Adding element type (line 798)
    # Getting the type of 'nt' (line 798)
    nt_75606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 20), 'nt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 798, 20), tuple_75605, nt_75606)
    # Adding element type (line 798)
    # Getting the type of 'extradim' (line 798)
    extradim_75607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 24), 'extradim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 798, 20), tuple_75605, extradim_75607)
    
    # Processing the call keyword arguments (line 798)
    # Getting the type of 'y' (line 798)
    y_75608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 41), 'y', False)
    # Obtaining the member 'dtype' of a type (line 798)
    dtype_75609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 41), y_75608, 'dtype')
    keyword_75610 = dtype_75609
    kwargs_75611 = {'dtype': keyword_75610}
    # Getting the type of 'np' (line 798)
    np_75603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 798)
    empty_75604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 10), np_75603, 'empty')
    # Calling empty(args, kwargs) (line 798)
    empty_call_result_75612 = invoke(stypy.reporting.localization.Localization(__file__, 798, 10), empty_75604, *[tuple_75605], **kwargs_75611)
    
    # Assigning a type to the variable 'rhs' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'rhs', empty_call_result_75612)
    
    
    # Getting the type of 'nleft' (line 799)
    nleft_75613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 7), 'nleft')
    int_75614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 15), 'int')
    # Applying the binary operator '>' (line 799)
    result_gt_75615 = python_operator(stypy.reporting.localization.Localization(__file__, 799, 7), '>', nleft_75613, int_75614)
    
    # Testing the type of an if condition (line 799)
    if_condition_75616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 799, 4), result_gt_75615)
    # Assigning a type to the variable 'if_condition_75616' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'if_condition_75616', if_condition_75616)
    # SSA begins for if statement (line 799)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 800):
    
    # Assigning a Call to a Subscript (line 800):
    
    # Call to reshape(...): (line 800)
    # Processing the call arguments (line 800)
    int_75619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 43), 'int')
    # Getting the type of 'extradim' (line 800)
    extradim_75620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 47), 'extradim', False)
    # Processing the call keyword arguments (line 800)
    kwargs_75621 = {}
    # Getting the type of 'deriv_l_vals' (line 800)
    deriv_l_vals_75617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 22), 'deriv_l_vals', False)
    # Obtaining the member 'reshape' of a type (line 800)
    reshape_75618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 22), deriv_l_vals_75617, 'reshape')
    # Calling reshape(args, kwargs) (line 800)
    reshape_call_result_75622 = invoke(stypy.reporting.localization.Localization(__file__, 800, 22), reshape_75618, *[int_75619, extradim_75620], **kwargs_75621)
    
    # Getting the type of 'rhs' (line 800)
    rhs_75623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'rhs')
    # Getting the type of 'nleft' (line 800)
    nleft_75624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 13), 'nleft')
    slice_75625 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 800, 8), None, nleft_75624, None)
    # Storing an element on a container (line 800)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 800, 8), rhs_75623, (slice_75625, reshape_call_result_75622))
    # SSA join for if statement (line 799)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 801):
    
    # Assigning a Call to a Subscript (line 801):
    
    # Call to reshape(...): (line 801)
    # Processing the call arguments (line 801)
    int_75628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 39), 'int')
    # Getting the type of 'extradim' (line 801)
    extradim_75629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 43), 'extradim', False)
    # Processing the call keyword arguments (line 801)
    kwargs_75630 = {}
    # Getting the type of 'y' (line 801)
    y_75626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 29), 'y', False)
    # Obtaining the member 'reshape' of a type (line 801)
    reshape_75627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 29), y_75626, 'reshape')
    # Calling reshape(args, kwargs) (line 801)
    reshape_call_result_75631 = invoke(stypy.reporting.localization.Localization(__file__, 801, 29), reshape_75627, *[int_75628, extradim_75629], **kwargs_75630)
    
    # Getting the type of 'rhs' (line 801)
    rhs_75632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 4), 'rhs')
    # Getting the type of 'nleft' (line 801)
    nleft_75633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'nleft')
    # Getting the type of 'nt' (line 801)
    nt_75634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 14), 'nt')
    # Getting the type of 'nright' (line 801)
    nright_75635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 19), 'nright')
    # Applying the binary operator '-' (line 801)
    result_sub_75636 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 14), '-', nt_75634, nright_75635)
    
    slice_75637 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 801, 4), nleft_75633, result_sub_75636, None)
    # Storing an element on a container (line 801)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 801, 4), rhs_75632, (slice_75637, reshape_call_result_75631))
    
    
    # Getting the type of 'nright' (line 802)
    nright_75638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 7), 'nright')
    int_75639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 16), 'int')
    # Applying the binary operator '>' (line 802)
    result_gt_75640 = python_operator(stypy.reporting.localization.Localization(__file__, 802, 7), '>', nright_75638, int_75639)
    
    # Testing the type of an if condition (line 802)
    if_condition_75641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 802, 4), result_gt_75640)
    # Assigning a type to the variable 'if_condition_75641' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 4), 'if_condition_75641', if_condition_75641)
    # SSA begins for if statement (line 802)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 803):
    
    # Assigning a Call to a Subscript (line 803):
    
    # Call to reshape(...): (line 803)
    # Processing the call arguments (line 803)
    int_75644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 49), 'int')
    # Getting the type of 'extradim' (line 803)
    extradim_75645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 53), 'extradim', False)
    # Processing the call keyword arguments (line 803)
    kwargs_75646 = {}
    # Getting the type of 'deriv_r_vals' (line 803)
    deriv_r_vals_75642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 28), 'deriv_r_vals', False)
    # Obtaining the member 'reshape' of a type (line 803)
    reshape_75643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 28), deriv_r_vals_75642, 'reshape')
    # Calling reshape(args, kwargs) (line 803)
    reshape_call_result_75647 = invoke(stypy.reporting.localization.Localization(__file__, 803, 28), reshape_75643, *[int_75644, extradim_75645], **kwargs_75646)
    
    # Getting the type of 'rhs' (line 803)
    rhs_75648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'rhs')
    # Getting the type of 'nt' (line 803)
    nt_75649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 12), 'nt')
    # Getting the type of 'nright' (line 803)
    nright_75650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 17), 'nright')
    # Applying the binary operator '-' (line 803)
    result_sub_75651 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 12), '-', nt_75649, nright_75650)
    
    slice_75652 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 803, 8), result_sub_75651, None, None)
    # Storing an element on a container (line 803)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 803, 8), rhs_75648, (slice_75652, reshape_call_result_75647))
    # SSA join for if statement (line 802)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'check_finite' (line 806)
    check_finite_75653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 7), 'check_finite')
    # Testing the type of an if condition (line 806)
    if_condition_75654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 806, 4), check_finite_75653)
    # Assigning a type to the variable 'if_condition_75654' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 4), 'if_condition_75654', if_condition_75654)
    # SSA begins for if statement (line 806)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 807):
    
    # Assigning a Subscript to a Name (line 807):
    
    # Obtaining the type of the subscript
    int_75655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 8), 'int')
    
    # Call to map(...): (line 807)
    # Processing the call arguments (line 807)
    # Getting the type of 'np' (line 807)
    np_75657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 22), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 807)
    asarray_chkfinite_75658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 22), np_75657, 'asarray_chkfinite')
    
    # Obtaining an instance of the builtin type 'tuple' (line 807)
    tuple_75659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 807)
    # Adding element type (line 807)
    # Getting the type of 'ab' (line 807)
    ab_75660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 45), 'ab', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 45), tuple_75659, ab_75660)
    # Adding element type (line 807)
    # Getting the type of 'rhs' (line 807)
    rhs_75661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 49), 'rhs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 45), tuple_75659, rhs_75661)
    
    # Processing the call keyword arguments (line 807)
    kwargs_75662 = {}
    # Getting the type of 'map' (line 807)
    map_75656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 18), 'map', False)
    # Calling map(args, kwargs) (line 807)
    map_call_result_75663 = invoke(stypy.reporting.localization.Localization(__file__, 807, 18), map_75656, *[asarray_chkfinite_75658, tuple_75659], **kwargs_75662)
    
    # Obtaining the member '__getitem__' of a type (line 807)
    getitem___75664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 8), map_call_result_75663, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 807)
    subscript_call_result_75665 = invoke(stypy.reporting.localization.Localization(__file__, 807, 8), getitem___75664, int_75655)
    
    # Assigning a type to the variable 'tuple_var_assignment_73704' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'tuple_var_assignment_73704', subscript_call_result_75665)
    
    # Assigning a Subscript to a Name (line 807):
    
    # Obtaining the type of the subscript
    int_75666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 8), 'int')
    
    # Call to map(...): (line 807)
    # Processing the call arguments (line 807)
    # Getting the type of 'np' (line 807)
    np_75668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 22), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 807)
    asarray_chkfinite_75669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 22), np_75668, 'asarray_chkfinite')
    
    # Obtaining an instance of the builtin type 'tuple' (line 807)
    tuple_75670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 807)
    # Adding element type (line 807)
    # Getting the type of 'ab' (line 807)
    ab_75671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 45), 'ab', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 45), tuple_75670, ab_75671)
    # Adding element type (line 807)
    # Getting the type of 'rhs' (line 807)
    rhs_75672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 49), 'rhs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 45), tuple_75670, rhs_75672)
    
    # Processing the call keyword arguments (line 807)
    kwargs_75673 = {}
    # Getting the type of 'map' (line 807)
    map_75667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 18), 'map', False)
    # Calling map(args, kwargs) (line 807)
    map_call_result_75674 = invoke(stypy.reporting.localization.Localization(__file__, 807, 18), map_75667, *[asarray_chkfinite_75669, tuple_75670], **kwargs_75673)
    
    # Obtaining the member '__getitem__' of a type (line 807)
    getitem___75675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 8), map_call_result_75674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 807)
    subscript_call_result_75676 = invoke(stypy.reporting.localization.Localization(__file__, 807, 8), getitem___75675, int_75666)
    
    # Assigning a type to the variable 'tuple_var_assignment_73705' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'tuple_var_assignment_73705', subscript_call_result_75676)
    
    # Assigning a Name to a Name (line 807):
    # Getting the type of 'tuple_var_assignment_73704' (line 807)
    tuple_var_assignment_73704_75677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'tuple_var_assignment_73704')
    # Assigning a type to the variable 'ab' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'ab', tuple_var_assignment_73704_75677)
    
    # Assigning a Name to a Name (line 807):
    # Getting the type of 'tuple_var_assignment_73705' (line 807)
    tuple_var_assignment_73705_75678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'tuple_var_assignment_73705')
    # Assigning a type to the variable 'rhs' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 12), 'rhs', tuple_var_assignment_73705_75678)
    # SSA join for if statement (line 806)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 808):
    
    # Assigning a Subscript to a Name (line 808):
    
    # Obtaining the type of the subscript
    int_75679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 808)
    # Processing the call arguments (line 808)
    
    # Obtaining an instance of the builtin type 'tuple' (line 808)
    tuple_75681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 808)
    # Adding element type (line 808)
    str_75682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 30), 'str', 'gbsv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 30), tuple_75681, str_75682)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 808)
    tuple_75683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 808)
    # Adding element type (line 808)
    # Getting the type of 'ab' (line 808)
    ab_75684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 41), 'ab', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 41), tuple_75683, ab_75684)
    # Adding element type (line 808)
    # Getting the type of 'rhs' (line 808)
    rhs_75685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 45), 'rhs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 41), tuple_75683, rhs_75685)
    
    # Processing the call keyword arguments (line 808)
    kwargs_75686 = {}
    # Getting the type of 'get_lapack_funcs' (line 808)
    get_lapack_funcs_75680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 12), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 808)
    get_lapack_funcs_call_result_75687 = invoke(stypy.reporting.localization.Localization(__file__, 808, 12), get_lapack_funcs_75680, *[tuple_75681, tuple_75683], **kwargs_75686)
    
    # Obtaining the member '__getitem__' of a type (line 808)
    getitem___75688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 4), get_lapack_funcs_call_result_75687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 808)
    subscript_call_result_75689 = invoke(stypy.reporting.localization.Localization(__file__, 808, 4), getitem___75688, int_75679)
    
    # Assigning a type to the variable 'tuple_var_assignment_73706' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'tuple_var_assignment_73706', subscript_call_result_75689)
    
    # Assigning a Name to a Name (line 808):
    # Getting the type of 'tuple_var_assignment_73706' (line 808)
    tuple_var_assignment_73706_75690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'tuple_var_assignment_73706')
    # Assigning a type to the variable 'gbsv' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'gbsv', tuple_var_assignment_73706_75690)
    
    # Assigning a Call to a Tuple (line 809):
    
    # Assigning a Subscript to a Name (line 809):
    
    # Obtaining the type of the subscript
    int_75691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 4), 'int')
    
    # Call to gbsv(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'kl' (line 809)
    kl_75693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 28), 'kl', False)
    # Getting the type of 'ku' (line 809)
    ku_75694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 32), 'ku', False)
    # Getting the type of 'ab' (line 809)
    ab_75695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 36), 'ab', False)
    # Getting the type of 'rhs' (line 809)
    rhs_75696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 40), 'rhs', False)
    # Processing the call keyword arguments (line 809)
    # Getting the type of 'True' (line 810)
    True_75697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 25), 'True', False)
    keyword_75698 = True_75697
    # Getting the type of 'True' (line 810)
    True_75699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 43), 'True', False)
    keyword_75700 = True_75699
    kwargs_75701 = {'overwrite_ab': keyword_75698, 'overwrite_b': keyword_75700}
    # Getting the type of 'gbsv' (line 809)
    gbsv_75692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 23), 'gbsv', False)
    # Calling gbsv(args, kwargs) (line 809)
    gbsv_call_result_75702 = invoke(stypy.reporting.localization.Localization(__file__, 809, 23), gbsv_75692, *[kl_75693, ku_75694, ab_75695, rhs_75696], **kwargs_75701)
    
    # Obtaining the member '__getitem__' of a type (line 809)
    getitem___75703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 4), gbsv_call_result_75702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 809)
    subscript_call_result_75704 = invoke(stypy.reporting.localization.Localization(__file__, 809, 4), getitem___75703, int_75691)
    
    # Assigning a type to the variable 'tuple_var_assignment_73707' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_73707', subscript_call_result_75704)
    
    # Assigning a Subscript to a Name (line 809):
    
    # Obtaining the type of the subscript
    int_75705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 4), 'int')
    
    # Call to gbsv(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'kl' (line 809)
    kl_75707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 28), 'kl', False)
    # Getting the type of 'ku' (line 809)
    ku_75708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 32), 'ku', False)
    # Getting the type of 'ab' (line 809)
    ab_75709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 36), 'ab', False)
    # Getting the type of 'rhs' (line 809)
    rhs_75710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 40), 'rhs', False)
    # Processing the call keyword arguments (line 809)
    # Getting the type of 'True' (line 810)
    True_75711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 25), 'True', False)
    keyword_75712 = True_75711
    # Getting the type of 'True' (line 810)
    True_75713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 43), 'True', False)
    keyword_75714 = True_75713
    kwargs_75715 = {'overwrite_ab': keyword_75712, 'overwrite_b': keyword_75714}
    # Getting the type of 'gbsv' (line 809)
    gbsv_75706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 23), 'gbsv', False)
    # Calling gbsv(args, kwargs) (line 809)
    gbsv_call_result_75716 = invoke(stypy.reporting.localization.Localization(__file__, 809, 23), gbsv_75706, *[kl_75707, ku_75708, ab_75709, rhs_75710], **kwargs_75715)
    
    # Obtaining the member '__getitem__' of a type (line 809)
    getitem___75717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 4), gbsv_call_result_75716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 809)
    subscript_call_result_75718 = invoke(stypy.reporting.localization.Localization(__file__, 809, 4), getitem___75717, int_75705)
    
    # Assigning a type to the variable 'tuple_var_assignment_73708' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_73708', subscript_call_result_75718)
    
    # Assigning a Subscript to a Name (line 809):
    
    # Obtaining the type of the subscript
    int_75719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 4), 'int')
    
    # Call to gbsv(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'kl' (line 809)
    kl_75721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 28), 'kl', False)
    # Getting the type of 'ku' (line 809)
    ku_75722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 32), 'ku', False)
    # Getting the type of 'ab' (line 809)
    ab_75723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 36), 'ab', False)
    # Getting the type of 'rhs' (line 809)
    rhs_75724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 40), 'rhs', False)
    # Processing the call keyword arguments (line 809)
    # Getting the type of 'True' (line 810)
    True_75725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 25), 'True', False)
    keyword_75726 = True_75725
    # Getting the type of 'True' (line 810)
    True_75727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 43), 'True', False)
    keyword_75728 = True_75727
    kwargs_75729 = {'overwrite_ab': keyword_75726, 'overwrite_b': keyword_75728}
    # Getting the type of 'gbsv' (line 809)
    gbsv_75720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 23), 'gbsv', False)
    # Calling gbsv(args, kwargs) (line 809)
    gbsv_call_result_75730 = invoke(stypy.reporting.localization.Localization(__file__, 809, 23), gbsv_75720, *[kl_75721, ku_75722, ab_75723, rhs_75724], **kwargs_75729)
    
    # Obtaining the member '__getitem__' of a type (line 809)
    getitem___75731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 4), gbsv_call_result_75730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 809)
    subscript_call_result_75732 = invoke(stypy.reporting.localization.Localization(__file__, 809, 4), getitem___75731, int_75719)
    
    # Assigning a type to the variable 'tuple_var_assignment_73709' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_73709', subscript_call_result_75732)
    
    # Assigning a Subscript to a Name (line 809):
    
    # Obtaining the type of the subscript
    int_75733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 4), 'int')
    
    # Call to gbsv(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'kl' (line 809)
    kl_75735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 28), 'kl', False)
    # Getting the type of 'ku' (line 809)
    ku_75736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 32), 'ku', False)
    # Getting the type of 'ab' (line 809)
    ab_75737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 36), 'ab', False)
    # Getting the type of 'rhs' (line 809)
    rhs_75738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 40), 'rhs', False)
    # Processing the call keyword arguments (line 809)
    # Getting the type of 'True' (line 810)
    True_75739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 25), 'True', False)
    keyword_75740 = True_75739
    # Getting the type of 'True' (line 810)
    True_75741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 43), 'True', False)
    keyword_75742 = True_75741
    kwargs_75743 = {'overwrite_ab': keyword_75740, 'overwrite_b': keyword_75742}
    # Getting the type of 'gbsv' (line 809)
    gbsv_75734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 23), 'gbsv', False)
    # Calling gbsv(args, kwargs) (line 809)
    gbsv_call_result_75744 = invoke(stypy.reporting.localization.Localization(__file__, 809, 23), gbsv_75734, *[kl_75735, ku_75736, ab_75737, rhs_75738], **kwargs_75743)
    
    # Obtaining the member '__getitem__' of a type (line 809)
    getitem___75745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 4), gbsv_call_result_75744, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 809)
    subscript_call_result_75746 = invoke(stypy.reporting.localization.Localization(__file__, 809, 4), getitem___75745, int_75733)
    
    # Assigning a type to the variable 'tuple_var_assignment_73710' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_73710', subscript_call_result_75746)
    
    # Assigning a Name to a Name (line 809):
    # Getting the type of 'tuple_var_assignment_73707' (line 809)
    tuple_var_assignment_73707_75747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_73707')
    # Assigning a type to the variable 'lu' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'lu', tuple_var_assignment_73707_75747)
    
    # Assigning a Name to a Name (line 809):
    # Getting the type of 'tuple_var_assignment_73708' (line 809)
    tuple_var_assignment_73708_75748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_73708')
    # Assigning a type to the variable 'piv' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'piv', tuple_var_assignment_73708_75748)
    
    # Assigning a Name to a Name (line 809):
    # Getting the type of 'tuple_var_assignment_73709' (line 809)
    tuple_var_assignment_73709_75749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_73709')
    # Assigning a type to the variable 'c' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 13), 'c', tuple_var_assignment_73709_75749)
    
    # Assigning a Name to a Name (line 809):
    # Getting the type of 'tuple_var_assignment_73710' (line 809)
    tuple_var_assignment_73710_75750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_73710')
    # Assigning a type to the variable 'info' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 16), 'info', tuple_var_assignment_73710_75750)
    
    
    # Getting the type of 'info' (line 812)
    info_75751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 7), 'info')
    int_75752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 14), 'int')
    # Applying the binary operator '>' (line 812)
    result_gt_75753 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 7), '>', info_75751, int_75752)
    
    # Testing the type of an if condition (line 812)
    if_condition_75754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 812, 4), result_gt_75753)
    # Assigning a type to the variable 'if_condition_75754' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'if_condition_75754', if_condition_75754)
    # SSA begins for if statement (line 812)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 813)
    # Processing the call arguments (line 813)
    str_75756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 26), 'str', 'Collocation matix is singular.')
    # Processing the call keyword arguments (line 813)
    kwargs_75757 = {}
    # Getting the type of 'LinAlgError' (line 813)
    LinAlgError_75755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 813)
    LinAlgError_call_result_75758 = invoke(stypy.reporting.localization.Localization(__file__, 813, 14), LinAlgError_75755, *[str_75756], **kwargs_75757)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 813, 8), LinAlgError_call_result_75758, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 812)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 814)
    info_75759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 9), 'info')
    int_75760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 16), 'int')
    # Applying the binary operator '<' (line 814)
    result_lt_75761 = python_operator(stypy.reporting.localization.Localization(__file__, 814, 9), '<', info_75759, int_75760)
    
    # Testing the type of an if condition (line 814)
    if_condition_75762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 814, 9), result_lt_75761)
    # Assigning a type to the variable 'if_condition_75762' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 9), 'if_condition_75762', if_condition_75762)
    # SSA begins for if statement (line 814)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 815)
    # Processing the call arguments (line 815)
    str_75764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 25), 'str', 'illegal value in %d-th argument of internal gbsv')
    
    # Getting the type of 'info' (line 815)
    info_75765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 79), 'info', False)
    # Applying the 'usub' unary operator (line 815)
    result___neg___75766 = python_operator(stypy.reporting.localization.Localization(__file__, 815, 78), 'usub', info_75765)
    
    # Applying the binary operator '%' (line 815)
    result_mod_75767 = python_operator(stypy.reporting.localization.Localization(__file__, 815, 25), '%', str_75764, result___neg___75766)
    
    # Processing the call keyword arguments (line 815)
    kwargs_75768 = {}
    # Getting the type of 'ValueError' (line 815)
    ValueError_75763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 815)
    ValueError_call_result_75769 = invoke(stypy.reporting.localization.Localization(__file__, 815, 14), ValueError_75763, *[result_mod_75767], **kwargs_75768)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 815, 8), ValueError_call_result_75769, 'raise parameter', BaseException)
    # SSA join for if statement (line 814)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 812)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 817):
    
    # Assigning a Call to a Name (line 817):
    
    # Call to ascontiguousarray(...): (line 817)
    # Processing the call arguments (line 817)
    
    # Call to reshape(...): (line 817)
    # Processing the call arguments (line 817)
    
    # Obtaining an instance of the builtin type 'tuple' (line 817)
    tuple_75774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 817)
    # Adding element type (line 817)
    # Getting the type of 'nt' (line 817)
    nt_75775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 40), 'nt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 817, 40), tuple_75774, nt_75775)
    
    
    # Obtaining the type of the subscript
    int_75776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 55), 'int')
    slice_75777 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 817, 47), int_75776, None, None)
    # Getting the type of 'y' (line 817)
    y_75778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 47), 'y', False)
    # Obtaining the member 'shape' of a type (line 817)
    shape_75779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 47), y_75778, 'shape')
    # Obtaining the member '__getitem__' of a type (line 817)
    getitem___75780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 47), shape_75779, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 817)
    subscript_call_result_75781 = invoke(stypy.reporting.localization.Localization(__file__, 817, 47), getitem___75780, slice_75777)
    
    # Applying the binary operator '+' (line 817)
    result_add_75782 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 39), '+', tuple_75774, subscript_call_result_75781)
    
    # Processing the call keyword arguments (line 817)
    kwargs_75783 = {}
    # Getting the type of 'c' (line 817)
    c_75772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 29), 'c', False)
    # Obtaining the member 'reshape' of a type (line 817)
    reshape_75773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 29), c_75772, 'reshape')
    # Calling reshape(args, kwargs) (line 817)
    reshape_call_result_75784 = invoke(stypy.reporting.localization.Localization(__file__, 817, 29), reshape_75773, *[result_add_75782], **kwargs_75783)
    
    # Processing the call keyword arguments (line 817)
    kwargs_75785 = {}
    # Getting the type of 'np' (line 817)
    np_75770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 817)
    ascontiguousarray_75771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 8), np_75770, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 817)
    ascontiguousarray_call_result_75786 = invoke(stypy.reporting.localization.Localization(__file__, 817, 8), ascontiguousarray_75771, *[reshape_call_result_75784], **kwargs_75785)
    
    # Assigning a type to the variable 'c' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'c', ascontiguousarray_call_result_75786)
    
    # Call to construct_fast(...): (line 818)
    # Processing the call arguments (line 818)
    # Getting the type of 't' (line 818)
    t_75789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 34), 't', False)
    # Getting the type of 'c' (line 818)
    c_75790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 37), 'c', False)
    # Getting the type of 'k' (line 818)
    k_75791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 40), 'k', False)
    # Processing the call keyword arguments (line 818)
    # Getting the type of 'axis' (line 818)
    axis_75792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 48), 'axis', False)
    keyword_75793 = axis_75792
    kwargs_75794 = {'axis': keyword_75793}
    # Getting the type of 'BSpline' (line 818)
    BSpline_75787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 11), 'BSpline', False)
    # Obtaining the member 'construct_fast' of a type (line 818)
    construct_fast_75788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 11), BSpline_75787, 'construct_fast')
    # Calling construct_fast(args, kwargs) (line 818)
    construct_fast_call_result_75795 = invoke(stypy.reporting.localization.Localization(__file__, 818, 11), construct_fast_75788, *[t_75789, c_75790, k_75791], **kwargs_75794)
    
    # Assigning a type to the variable 'stypy_return_type' (line 818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 4), 'stypy_return_type', construct_fast_call_result_75795)
    
    # ################# End of 'make_interp_spline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_interp_spline' in the type store
    # Getting the type of 'stypy_return_type' (line 597)
    stypy_return_type_75796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_75796)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_interp_spline'
    return stypy_return_type_75796

# Assigning a type to the variable 'make_interp_spline' (line 597)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 0), 'make_interp_spline', make_interp_spline)

@norecursion
def make_lsq_spline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_75797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 31), 'int')
    # Getting the type of 'None' (line 821)
    None_75798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 36), 'None')
    int_75799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 47), 'int')
    # Getting the type of 'True' (line 821)
    True_75800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 63), 'True')
    defaults = [int_75797, None_75798, int_75799, True_75800]
    # Create a new context for function 'make_lsq_spline'
    module_type_store = module_type_store.open_function_context('make_lsq_spline', 821, 0, False)
    
    # Passed parameters checking function
    make_lsq_spline.stypy_localization = localization
    make_lsq_spline.stypy_type_of_self = None
    make_lsq_spline.stypy_type_store = module_type_store
    make_lsq_spline.stypy_function_name = 'make_lsq_spline'
    make_lsq_spline.stypy_param_names_list = ['x', 'y', 't', 'k', 'w', 'axis', 'check_finite']
    make_lsq_spline.stypy_varargs_param_name = None
    make_lsq_spline.stypy_kwargs_param_name = None
    make_lsq_spline.stypy_call_defaults = defaults
    make_lsq_spline.stypy_call_varargs = varargs
    make_lsq_spline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_lsq_spline', ['x', 'y', 't', 'k', 'w', 'axis', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_lsq_spline', localization, ['x', 'y', 't', 'k', 'w', 'axis', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_lsq_spline(...)' code ##################

    str_75801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, (-1)), 'str', "Compute the (coefficients of) an LSQ B-spline.\n\n    The result is a linear combination\n\n    .. math::\n\n            S(x) = \\sum_j c_j B_j(x; t)\n\n    of the B-spline basis elements, :math:`B_j(x; t)`, which minimizes\n\n    .. math::\n\n        \\sum_{j} \\left( w_j \\times (S(x_j) - y_j) \\right)^2\n\n    Parameters\n    ----------\n    x : array_like, shape (m,)\n        Abscissas.\n    y : array_like, shape (m, ...)\n        Ordinates.\n    t : array_like, shape (n + k + 1,).\n        Knots.\n        Knots and data points must satisfy Schoenberg-Whitney conditions.\n    k : int, optional\n        B-spline degree. Default is cubic, k=3.\n    w : array_like, shape (n,), optional\n        Weights for spline fitting. Must be positive. If ``None``,\n        then weights are all equal.\n        Default is ``None``.\n    axis : int, optional\n        Interpolation axis. Default is zero.\n    check_finite : bool, optional\n        Whether to check that the input arrays contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default is True.\n\n    Returns\n    -------\n    b : a BSpline object of the degree `k` with knots `t`.\n\n    Notes\n    -----\n\n    The number of data points must be larger than the spline degree `k`.\n\n    Knots `t` must satisfy the Schoenberg-Whitney conditions,\n    i.e., there must be a subset of data points ``x[j]`` such that\n    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.\n\n    Examples\n    --------\n    Generate some noisy data:\n\n    >>> x = np.linspace(-3, 3, 50)\n    >>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)\n\n    Now fit a smoothing cubic spline with a pre-defined internal knots.\n    Here we make the knot vector (k+1)-regular by adding boundary knots:\n\n    >>> from scipy.interpolate import make_lsq_spline, BSpline\n    >>> t = [-1, 0, 1]\n    >>> k = 3\n    >>> t = np.r_[(x[0],)*(k+1),\n    ...           t,\n    ...           (x[-1],)*(k+1)]\n    >>> spl = make_lsq_spline(x, y, t, k)\n\n    For comparison, we also construct an interpolating spline for the same\n    set of data:\n\n    >>> from scipy.interpolate import make_interp_spline\n    >>> spl_i = make_interp_spline(x, y)\n\n    Plot both:\n\n    >>> import matplotlib.pyplot as plt\n    >>> xs = np.linspace(-3, 3, 100)\n    >>> plt.plot(x, y, 'ro', ms=5)\n    >>> plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')\n    >>> plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n\n    **NaN handling**: If the input arrays contain ``nan`` values, the result is\n    not useful since the underlying spline fitting routines cannot deal with\n    ``nan``. A workaround is to use zero weights for not-a-number data points:\n\n    >>> y[8] = np.nan\n    >>> w = np.isnan(y)\n    >>> y[w] = 0.\n    >>> tck = make_lsq_spline(x, y, t, w=~w)\n\n    Notice the need to replace a ``nan`` by a numerical value (precise value\n    does not matter as long as the corresponding weight is zero.)\n\n    See Also\n    --------\n    BSpline : base class representing the B-spline objects\n    make_interp_spline : a similar factory function for interpolating splines\n    LSQUnivariateSpline : a FITPACK-based spline fitting routine\n    splrep : a FITPACK-based fitting routine\n\n    ")
    
    # Assigning a Call to a Name (line 926):
    
    # Assigning a Call to a Name (line 926):
    
    # Call to _as_float_array(...): (line 926)
    # Processing the call arguments (line 926)
    # Getting the type of 'x' (line 926)
    x_75803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 24), 'x', False)
    # Getting the type of 'check_finite' (line 926)
    check_finite_75804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 27), 'check_finite', False)
    # Processing the call keyword arguments (line 926)
    kwargs_75805 = {}
    # Getting the type of '_as_float_array' (line 926)
    _as_float_array_75802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), '_as_float_array', False)
    # Calling _as_float_array(args, kwargs) (line 926)
    _as_float_array_call_result_75806 = invoke(stypy.reporting.localization.Localization(__file__, 926, 8), _as_float_array_75802, *[x_75803, check_finite_75804], **kwargs_75805)
    
    # Assigning a type to the variable 'x' (line 926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 4), 'x', _as_float_array_call_result_75806)
    
    # Assigning a Call to a Name (line 927):
    
    # Assigning a Call to a Name (line 927):
    
    # Call to _as_float_array(...): (line 927)
    # Processing the call arguments (line 927)
    # Getting the type of 'y' (line 927)
    y_75808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 24), 'y', False)
    # Getting the type of 'check_finite' (line 927)
    check_finite_75809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 27), 'check_finite', False)
    # Processing the call keyword arguments (line 927)
    kwargs_75810 = {}
    # Getting the type of '_as_float_array' (line 927)
    _as_float_array_75807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), '_as_float_array', False)
    # Calling _as_float_array(args, kwargs) (line 927)
    _as_float_array_call_result_75811 = invoke(stypy.reporting.localization.Localization(__file__, 927, 8), _as_float_array_75807, *[y_75808, check_finite_75809], **kwargs_75810)
    
    # Assigning a type to the variable 'y' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 4), 'y', _as_float_array_call_result_75811)
    
    # Assigning a Call to a Name (line 928):
    
    # Assigning a Call to a Name (line 928):
    
    # Call to _as_float_array(...): (line 928)
    # Processing the call arguments (line 928)
    # Getting the type of 't' (line 928)
    t_75813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 24), 't', False)
    # Getting the type of 'check_finite' (line 928)
    check_finite_75814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 27), 'check_finite', False)
    # Processing the call keyword arguments (line 928)
    kwargs_75815 = {}
    # Getting the type of '_as_float_array' (line 928)
    _as_float_array_75812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 8), '_as_float_array', False)
    # Calling _as_float_array(args, kwargs) (line 928)
    _as_float_array_call_result_75816 = invoke(stypy.reporting.localization.Localization(__file__, 928, 8), _as_float_array_75812, *[t_75813, check_finite_75814], **kwargs_75815)
    
    # Assigning a type to the variable 't' (line 928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 4), 't', _as_float_array_call_result_75816)
    
    # Type idiom detected: calculating its left and rigth part (line 929)
    # Getting the type of 'w' (line 929)
    w_75817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 4), 'w')
    # Getting the type of 'None' (line 929)
    None_75818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 16), 'None')
    
    (may_be_75819, more_types_in_union_75820) = may_not_be_none(w_75817, None_75818)

    if may_be_75819:

        if more_types_in_union_75820:
            # Runtime conditional SSA (line 929)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 930):
        
        # Assigning a Call to a Name (line 930):
        
        # Call to _as_float_array(...): (line 930)
        # Processing the call arguments (line 930)
        # Getting the type of 'w' (line 930)
        w_75822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 28), 'w', False)
        # Getting the type of 'check_finite' (line 930)
        check_finite_75823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 31), 'check_finite', False)
        # Processing the call keyword arguments (line 930)
        kwargs_75824 = {}
        # Getting the type of '_as_float_array' (line 930)
        _as_float_array_75821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 12), '_as_float_array', False)
        # Calling _as_float_array(args, kwargs) (line 930)
        _as_float_array_call_result_75825 = invoke(stypy.reporting.localization.Localization(__file__, 930, 12), _as_float_array_75821, *[w_75822, check_finite_75823], **kwargs_75824)
        
        # Assigning a type to the variable 'w' (line 930)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 8), 'w', _as_float_array_call_result_75825)

        if more_types_in_union_75820:
            # Runtime conditional SSA for else branch (line 929)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_75819) or more_types_in_union_75820):
        
        # Assigning a Call to a Name (line 932):
        
        # Assigning a Call to a Name (line 932):
        
        # Call to ones_like(...): (line 932)
        # Processing the call arguments (line 932)
        # Getting the type of 'x' (line 932)
        x_75828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 25), 'x', False)
        # Processing the call keyword arguments (line 932)
        kwargs_75829 = {}
        # Getting the type of 'np' (line 932)
        np_75826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 12), 'np', False)
        # Obtaining the member 'ones_like' of a type (line 932)
        ones_like_75827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 12), np_75826, 'ones_like')
        # Calling ones_like(args, kwargs) (line 932)
        ones_like_call_result_75830 = invoke(stypy.reporting.localization.Localization(__file__, 932, 12), ones_like_75827, *[x_75828], **kwargs_75829)
        
        # Assigning a type to the variable 'w' (line 932)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'w', ones_like_call_result_75830)

        if (may_be_75819 and more_types_in_union_75820):
            # SSA join for if statement (line 929)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 933):
    
    # Assigning a Call to a Name (line 933):
    
    # Call to int(...): (line 933)
    # Processing the call arguments (line 933)
    # Getting the type of 'k' (line 933)
    k_75832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 12), 'k', False)
    # Processing the call keyword arguments (line 933)
    kwargs_75833 = {}
    # Getting the type of 'int' (line 933)
    int_75831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 8), 'int', False)
    # Calling int(args, kwargs) (line 933)
    int_call_result_75834 = invoke(stypy.reporting.localization.Localization(__file__, 933, 8), int_75831, *[k_75832], **kwargs_75833)
    
    # Assigning a type to the variable 'k' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'k', int_call_result_75834)
    
    # Assigning a BinOp to a Name (line 935):
    
    # Assigning a BinOp to a Name (line 935):
    # Getting the type of 'axis' (line 935)
    axis_75835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 11), 'axis')
    # Getting the type of 'y' (line 935)
    y_75836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 18), 'y')
    # Obtaining the member 'ndim' of a type (line 935)
    ndim_75837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 18), y_75836, 'ndim')
    # Applying the binary operator '%' (line 935)
    result_mod_75838 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 11), '%', axis_75835, ndim_75837)
    
    # Assigning a type to the variable 'axis' (line 935)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 4), 'axis', result_mod_75838)
    
    # Assigning a Call to a Name (line 936):
    
    # Assigning a Call to a Name (line 936):
    
    # Call to rollaxis(...): (line 936)
    # Processing the call arguments (line 936)
    # Getting the type of 'y' (line 936)
    y_75841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 20), 'y', False)
    # Getting the type of 'axis' (line 936)
    axis_75842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 23), 'axis', False)
    # Processing the call keyword arguments (line 936)
    kwargs_75843 = {}
    # Getting the type of 'np' (line 936)
    np_75839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 936)
    rollaxis_75840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 8), np_75839, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 936)
    rollaxis_call_result_75844 = invoke(stypy.reporting.localization.Localization(__file__, 936, 8), rollaxis_75840, *[y_75841, axis_75842], **kwargs_75843)
    
    # Assigning a type to the variable 'y' (line 936)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 4), 'y', rollaxis_call_result_75844)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 938)
    x_75845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 938)
    ndim_75846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 7), x_75845, 'ndim')
    int_75847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 17), 'int')
    # Applying the binary operator '!=' (line 938)
    result_ne_75848 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 7), '!=', ndim_75846, int_75847)
    
    
    # Call to any(...): (line 938)
    # Processing the call arguments (line 938)
    
    
    # Obtaining the type of the subscript
    int_75851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 31), 'int')
    slice_75852 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 938, 29), int_75851, None, None)
    # Getting the type of 'x' (line 938)
    x_75853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 29), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 938)
    getitem___75854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 29), x_75853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 938)
    subscript_call_result_75855 = invoke(stypy.reporting.localization.Localization(__file__, 938, 29), getitem___75854, slice_75852)
    
    
    # Obtaining the type of the subscript
    int_75856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 40), 'int')
    slice_75857 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 938, 37), None, int_75856, None)
    # Getting the type of 'x' (line 938)
    x_75858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 37), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 938)
    getitem___75859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 37), x_75858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 938)
    subscript_call_result_75860 = invoke(stypy.reporting.localization.Localization(__file__, 938, 37), getitem___75859, slice_75857)
    
    # Applying the binary operator '-' (line 938)
    result_sub_75861 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 29), '-', subscript_call_result_75855, subscript_call_result_75860)
    
    int_75862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 47), 'int')
    # Applying the binary operator '<=' (line 938)
    result_le_75863 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 29), '<=', result_sub_75861, int_75862)
    
    # Processing the call keyword arguments (line 938)
    kwargs_75864 = {}
    # Getting the type of 'np' (line 938)
    np_75849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 22), 'np', False)
    # Obtaining the member 'any' of a type (line 938)
    any_75850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 22), np_75849, 'any')
    # Calling any(args, kwargs) (line 938)
    any_call_result_75865 = invoke(stypy.reporting.localization.Localization(__file__, 938, 22), any_75850, *[result_le_75863], **kwargs_75864)
    
    # Applying the binary operator 'or' (line 938)
    result_or_keyword_75866 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 7), 'or', result_ne_75848, any_call_result_75865)
    
    # Testing the type of an if condition (line 938)
    if_condition_75867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 938, 4), result_or_keyword_75866)
    # Assigning a type to the variable 'if_condition_75867' (line 938)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 938, 4), 'if_condition_75867', if_condition_75867)
    # SSA begins for if statement (line 938)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 939)
    # Processing the call arguments (line 939)
    str_75869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 25), 'str', 'Expect x to be a 1-D sorted array_like.')
    # Processing the call keyword arguments (line 939)
    kwargs_75870 = {}
    # Getting the type of 'ValueError' (line 939)
    ValueError_75868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 939)
    ValueError_call_result_75871 = invoke(stypy.reporting.localization.Localization(__file__, 939, 14), ValueError_75868, *[str_75869], **kwargs_75870)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 939, 8), ValueError_call_result_75871, 'raise parameter', BaseException)
    # SSA join for if statement (line 938)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_75872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 15), 'int')
    # Getting the type of 'x' (line 940)
    x_75873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 7), 'x')
    # Obtaining the member 'shape' of a type (line 940)
    shape_75874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 7), x_75873, 'shape')
    # Obtaining the member '__getitem__' of a type (line 940)
    getitem___75875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 7), shape_75874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 940)
    subscript_call_result_75876 = invoke(stypy.reporting.localization.Localization(__file__, 940, 7), getitem___75875, int_75872)
    
    # Getting the type of 'k' (line 940)
    k_75877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 20), 'k')
    int_75878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 22), 'int')
    # Applying the binary operator '+' (line 940)
    result_add_75879 = python_operator(stypy.reporting.localization.Localization(__file__, 940, 20), '+', k_75877, int_75878)
    
    # Applying the binary operator '<' (line 940)
    result_lt_75880 = python_operator(stypy.reporting.localization.Localization(__file__, 940, 7), '<', subscript_call_result_75876, result_add_75879)
    
    # Testing the type of an if condition (line 940)
    if_condition_75881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 940, 4), result_lt_75880)
    # Assigning a type to the variable 'if_condition_75881' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 4), 'if_condition_75881', if_condition_75881)
    # SSA begins for if statement (line 940)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 941)
    # Processing the call arguments (line 941)
    str_75883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 25), 'str', 'Need more x points.')
    # Processing the call keyword arguments (line 941)
    kwargs_75884 = {}
    # Getting the type of 'ValueError' (line 941)
    ValueError_75882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 941)
    ValueError_call_result_75885 = invoke(stypy.reporting.localization.Localization(__file__, 941, 14), ValueError_75882, *[str_75883], **kwargs_75884)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 941, 8), ValueError_call_result_75885, 'raise parameter', BaseException)
    # SSA join for if statement (line 940)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 942)
    k_75886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 7), 'k')
    int_75887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 11), 'int')
    # Applying the binary operator '<' (line 942)
    result_lt_75888 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 7), '<', k_75886, int_75887)
    
    # Testing the type of an if condition (line 942)
    if_condition_75889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 942, 4), result_lt_75888)
    # Assigning a type to the variable 'if_condition_75889' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 4), 'if_condition_75889', if_condition_75889)
    # SSA begins for if statement (line 942)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 943)
    # Processing the call arguments (line 943)
    str_75891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 25), 'str', 'Expect non-negative k.')
    # Processing the call keyword arguments (line 943)
    kwargs_75892 = {}
    # Getting the type of 'ValueError' (line 943)
    ValueError_75890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 943)
    ValueError_call_result_75893 = invoke(stypy.reporting.localization.Localization(__file__, 943, 14), ValueError_75890, *[str_75891], **kwargs_75892)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 943, 8), ValueError_call_result_75893, 'raise parameter', BaseException)
    # SSA join for if statement (line 942)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 't' (line 944)
    t_75894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 7), 't')
    # Obtaining the member 'ndim' of a type (line 944)
    ndim_75895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 7), t_75894, 'ndim')
    int_75896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 17), 'int')
    # Applying the binary operator '!=' (line 944)
    result_ne_75897 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 7), '!=', ndim_75895, int_75896)
    
    
    # Call to any(...): (line 944)
    # Processing the call arguments (line 944)
    
    
    # Obtaining the type of the subscript
    int_75900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 31), 'int')
    slice_75901 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 944, 29), int_75900, None, None)
    # Getting the type of 't' (line 944)
    t_75902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 29), 't', False)
    # Obtaining the member '__getitem__' of a type (line 944)
    getitem___75903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 29), t_75902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 944)
    subscript_call_result_75904 = invoke(stypy.reporting.localization.Localization(__file__, 944, 29), getitem___75903, slice_75901)
    
    
    # Obtaining the type of the subscript
    int_75905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 40), 'int')
    slice_75906 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 944, 37), None, int_75905, None)
    # Getting the type of 't' (line 944)
    t_75907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 37), 't', False)
    # Obtaining the member '__getitem__' of a type (line 944)
    getitem___75908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 37), t_75907, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 944)
    subscript_call_result_75909 = invoke(stypy.reporting.localization.Localization(__file__, 944, 37), getitem___75908, slice_75906)
    
    # Applying the binary operator '-' (line 944)
    result_sub_75910 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 29), '-', subscript_call_result_75904, subscript_call_result_75909)
    
    int_75911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 46), 'int')
    # Applying the binary operator '<' (line 944)
    result_lt_75912 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 29), '<', result_sub_75910, int_75911)
    
    # Processing the call keyword arguments (line 944)
    kwargs_75913 = {}
    # Getting the type of 'np' (line 944)
    np_75898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 22), 'np', False)
    # Obtaining the member 'any' of a type (line 944)
    any_75899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 22), np_75898, 'any')
    # Calling any(args, kwargs) (line 944)
    any_call_result_75914 = invoke(stypy.reporting.localization.Localization(__file__, 944, 22), any_75899, *[result_lt_75912], **kwargs_75913)
    
    # Applying the binary operator 'or' (line 944)
    result_or_keyword_75915 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 7), 'or', result_ne_75897, any_call_result_75914)
    
    # Testing the type of an if condition (line 944)
    if_condition_75916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 944, 4), result_or_keyword_75915)
    # Assigning a type to the variable 'if_condition_75916' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 4), 'if_condition_75916', if_condition_75916)
    # SSA begins for if statement (line 944)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 945)
    # Processing the call arguments (line 945)
    str_75918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 25), 'str', 'Expect t to be a 1-D sorted array_like.')
    # Processing the call keyword arguments (line 945)
    kwargs_75919 = {}
    # Getting the type of 'ValueError' (line 945)
    ValueError_75917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 945)
    ValueError_call_result_75920 = invoke(stypy.reporting.localization.Localization(__file__, 945, 14), ValueError_75917, *[str_75918], **kwargs_75919)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 945, 8), ValueError_call_result_75920, 'raise parameter', BaseException)
    # SSA join for if statement (line 944)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 946)
    x_75921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 7), 'x')
    # Obtaining the member 'size' of a type (line 946)
    size_75922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 7), x_75921, 'size')
    
    # Obtaining the type of the subscript
    int_75923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 25), 'int')
    # Getting the type of 'y' (line 946)
    y_75924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 17), 'y')
    # Obtaining the member 'shape' of a type (line 946)
    shape_75925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 17), y_75924, 'shape')
    # Obtaining the member '__getitem__' of a type (line 946)
    getitem___75926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 17), shape_75925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 946)
    subscript_call_result_75927 = invoke(stypy.reporting.localization.Localization(__file__, 946, 17), getitem___75926, int_75923)
    
    # Applying the binary operator '!=' (line 946)
    result_ne_75928 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 7), '!=', size_75922, subscript_call_result_75927)
    
    # Testing the type of an if condition (line 946)
    if_condition_75929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 946, 4), result_ne_75928)
    # Assigning a type to the variable 'if_condition_75929' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'if_condition_75929', if_condition_75929)
    # SSA begins for if statement (line 946)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 947)
    # Processing the call arguments (line 947)
    str_75931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 25), 'str', 'x & y are incompatible.')
    # Processing the call keyword arguments (line 947)
    kwargs_75932 = {}
    # Getting the type of 'ValueError' (line 947)
    ValueError_75930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 947)
    ValueError_call_result_75933 = invoke(stypy.reporting.localization.Localization(__file__, 947, 14), ValueError_75930, *[str_75931], **kwargs_75932)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 947, 8), ValueError_call_result_75933, 'raise parameter', BaseException)
    # SSA join for if statement (line 946)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 948)
    k_75934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 7), 'k')
    int_75935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 11), 'int')
    # Applying the binary operator '>' (line 948)
    result_gt_75936 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 7), '>', k_75934, int_75935)
    
    
    # Call to any(...): (line 948)
    # Processing the call arguments (line 948)
    
    # Getting the type of 'x' (line 948)
    x_75939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 25), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 948)
    k_75940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 31), 'k', False)
    # Getting the type of 't' (line 948)
    t_75941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 29), 't', False)
    # Obtaining the member '__getitem__' of a type (line 948)
    getitem___75942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 29), t_75941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 948)
    subscript_call_result_75943 = invoke(stypy.reporting.localization.Localization(__file__, 948, 29), getitem___75942, k_75940)
    
    # Applying the binary operator '<' (line 948)
    result_lt_75944 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 25), '<', x_75939, subscript_call_result_75943)
    
    
    # Getting the type of 'x' (line 948)
    x_75945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 38), 'x', False)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'k' (line 948)
    k_75946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 45), 'k', False)
    # Applying the 'usub' unary operator (line 948)
    result___neg___75947 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 44), 'usub', k_75946)
    
    # Getting the type of 't' (line 948)
    t_75948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 42), 't', False)
    # Obtaining the member '__getitem__' of a type (line 948)
    getitem___75949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 42), t_75948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 948)
    subscript_call_result_75950 = invoke(stypy.reporting.localization.Localization(__file__, 948, 42), getitem___75949, result___neg___75947)
    
    # Applying the binary operator '>' (line 948)
    result_gt_75951 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 38), '>', x_75945, subscript_call_result_75950)
    
    # Applying the binary operator '|' (line 948)
    result_or__75952 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 24), '|', result_lt_75944, result_gt_75951)
    
    # Processing the call keyword arguments (line 948)
    kwargs_75953 = {}
    # Getting the type of 'np' (line 948)
    np_75937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 17), 'np', False)
    # Obtaining the member 'any' of a type (line 948)
    any_75938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 17), np_75937, 'any')
    # Calling any(args, kwargs) (line 948)
    any_call_result_75954 = invoke(stypy.reporting.localization.Localization(__file__, 948, 17), any_75938, *[result_or__75952], **kwargs_75953)
    
    # Applying the binary operator 'and' (line 948)
    result_and_keyword_75955 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 7), 'and', result_gt_75936, any_call_result_75954)
    
    # Testing the type of an if condition (line 948)
    if_condition_75956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 948, 4), result_and_keyword_75955)
    # Assigning a type to the variable 'if_condition_75956' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'if_condition_75956', if_condition_75956)
    # SSA begins for if statement (line 948)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 949)
    # Processing the call arguments (line 949)
    str_75958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 25), 'str', 'Out of bounds w/ x = %s.')
    # Getting the type of 'x' (line 949)
    x_75959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 54), 'x', False)
    # Applying the binary operator '%' (line 949)
    result_mod_75960 = python_operator(stypy.reporting.localization.Localization(__file__, 949, 25), '%', str_75958, x_75959)
    
    # Processing the call keyword arguments (line 949)
    kwargs_75961 = {}
    # Getting the type of 'ValueError' (line 949)
    ValueError_75957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 949)
    ValueError_call_result_75962 = invoke(stypy.reporting.localization.Localization(__file__, 949, 14), ValueError_75957, *[result_mod_75960], **kwargs_75961)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 949, 8), ValueError_call_result_75962, 'raise parameter', BaseException)
    # SSA join for if statement (line 948)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 950)
    x_75963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 7), 'x')
    # Obtaining the member 'size' of a type (line 950)
    size_75964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 7), x_75963, 'size')
    # Getting the type of 'w' (line 950)
    w_75965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 17), 'w')
    # Obtaining the member 'size' of a type (line 950)
    size_75966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 17), w_75965, 'size')
    # Applying the binary operator '!=' (line 950)
    result_ne_75967 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 7), '!=', size_75964, size_75966)
    
    # Testing the type of an if condition (line 950)
    if_condition_75968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 950, 4), result_ne_75967)
    # Assigning a type to the variable 'if_condition_75968' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'if_condition_75968', if_condition_75968)
    # SSA begins for if statement (line 950)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 951)
    # Processing the call arguments (line 951)
    str_75970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 25), 'str', 'Incompatible weights.')
    # Processing the call keyword arguments (line 951)
    kwargs_75971 = {}
    # Getting the type of 'ValueError' (line 951)
    ValueError_75969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 951)
    ValueError_call_result_75972 = invoke(stypy.reporting.localization.Localization(__file__, 951, 14), ValueError_75969, *[str_75970], **kwargs_75971)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 951, 8), ValueError_call_result_75972, 'raise parameter', BaseException)
    # SSA join for if statement (line 950)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 954):
    
    # Assigning a BinOp to a Name (line 954):
    # Getting the type of 't' (line 954)
    t_75973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 8), 't')
    # Obtaining the member 'size' of a type (line 954)
    size_75974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 8), t_75973, 'size')
    # Getting the type of 'k' (line 954)
    k_75975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 17), 'k')
    # Applying the binary operator '-' (line 954)
    result_sub_75976 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 8), '-', size_75974, k_75975)
    
    int_75977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 21), 'int')
    # Applying the binary operator '-' (line 954)
    result_sub_75978 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 19), '-', result_sub_75976, int_75977)
    
    # Assigning a type to the variable 'n' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'n', result_sub_75978)
    
    # Assigning a Name to a Name (line 958):
    
    # Assigning a Name to a Name (line 958):
    # Getting the type of 'True' (line 958)
    True_75979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 12), 'True')
    # Assigning a type to the variable 'lower' (line 958)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 4), 'lower', True_75979)
    
    # Assigning a Call to a Name (line 959):
    
    # Assigning a Call to a Name (line 959):
    
    # Call to prod(...): (line 959)
    # Processing the call arguments (line 959)
    
    # Obtaining the type of the subscript
    int_75981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 28), 'int')
    slice_75982 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 959, 20), int_75981, None, None)
    # Getting the type of 'y' (line 959)
    y_75983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 20), 'y', False)
    # Obtaining the member 'shape' of a type (line 959)
    shape_75984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 20), y_75983, 'shape')
    # Obtaining the member '__getitem__' of a type (line 959)
    getitem___75985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 20), shape_75984, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 959)
    subscript_call_result_75986 = invoke(stypy.reporting.localization.Localization(__file__, 959, 20), getitem___75985, slice_75982)
    
    # Processing the call keyword arguments (line 959)
    kwargs_75987 = {}
    # Getting the type of 'prod' (line 959)
    prod_75980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 15), 'prod', False)
    # Calling prod(args, kwargs) (line 959)
    prod_call_result_75988 = invoke(stypy.reporting.localization.Localization(__file__, 959, 15), prod_75980, *[subscript_call_result_75986], **kwargs_75987)
    
    # Assigning a type to the variable 'extradim' (line 959)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 4), 'extradim', prod_call_result_75988)
    
    # Assigning a Call to a Name (line 960):
    
    # Assigning a Call to a Name (line 960):
    
    # Call to zeros(...): (line 960)
    # Processing the call arguments (line 960)
    
    # Obtaining an instance of the builtin type 'tuple' (line 960)
    tuple_75991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 960)
    # Adding element type (line 960)
    # Getting the type of 'k' (line 960)
    k_75992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 19), 'k', False)
    int_75993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 21), 'int')
    # Applying the binary operator '+' (line 960)
    result_add_75994 = python_operator(stypy.reporting.localization.Localization(__file__, 960, 19), '+', k_75992, int_75993)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 19), tuple_75991, result_add_75994)
    # Adding element type (line 960)
    # Getting the type of 'n' (line 960)
    n_75995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 19), tuple_75991, n_75995)
    
    # Processing the call keyword arguments (line 960)
    # Getting the type of 'np' (line 960)
    np_75996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 34), 'np', False)
    # Obtaining the member 'float_' of a type (line 960)
    float__75997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 34), np_75996, 'float_')
    keyword_75998 = float__75997
    str_75999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 51), 'str', 'F')
    keyword_76000 = str_75999
    kwargs_76001 = {'dtype': keyword_75998, 'order': keyword_76000}
    # Getting the type of 'np' (line 960)
    np_75989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 960)
    zeros_75990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 9), np_75989, 'zeros')
    # Calling zeros(args, kwargs) (line 960)
    zeros_call_result_76002 = invoke(stypy.reporting.localization.Localization(__file__, 960, 9), zeros_75990, *[tuple_75991], **kwargs_76001)
    
    # Assigning a type to the variable 'ab' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'ab', zeros_call_result_76002)
    
    # Assigning a Call to a Name (line 961):
    
    # Assigning a Call to a Name (line 961):
    
    # Call to zeros(...): (line 961)
    # Processing the call arguments (line 961)
    
    # Obtaining an instance of the builtin type 'tuple' (line 961)
    tuple_76005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 961)
    # Adding element type (line 961)
    # Getting the type of 'n' (line 961)
    n_76006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 20), tuple_76005, n_76006)
    # Adding element type (line 961)
    # Getting the type of 'extradim' (line 961)
    extradim_76007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 23), 'extradim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 20), tuple_76005, extradim_76007)
    
    # Processing the call keyword arguments (line 961)
    # Getting the type of 'y' (line 961)
    y_76008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 40), 'y', False)
    # Obtaining the member 'dtype' of a type (line 961)
    dtype_76009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 40), y_76008, 'dtype')
    keyword_76010 = dtype_76009
    str_76011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 55), 'str', 'F')
    keyword_76012 = str_76011
    kwargs_76013 = {'dtype': keyword_76010, 'order': keyword_76012}
    # Getting the type of 'np' (line 961)
    np_76003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 961)
    zeros_76004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 10), np_76003, 'zeros')
    # Calling zeros(args, kwargs) (line 961)
    zeros_call_result_76014 = invoke(stypy.reporting.localization.Localization(__file__, 961, 10), zeros_76004, *[tuple_76005], **kwargs_76013)
    
    # Assigning a type to the variable 'rhs' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'rhs', zeros_call_result_76014)
    
    # Call to _norm_eq_lsq(...): (line 962)
    # Processing the call arguments (line 962)
    # Getting the type of 'x' (line 962)
    x_76017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 23), 'x', False)
    # Getting the type of 't' (line 962)
    t_76018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 26), 't', False)
    # Getting the type of 'k' (line 962)
    k_76019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 29), 'k', False)
    
    # Call to reshape(...): (line 963)
    # Processing the call arguments (line 963)
    int_76022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 32), 'int')
    # Getting the type of 'extradim' (line 963)
    extradim_76023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 36), 'extradim', False)
    # Processing the call keyword arguments (line 963)
    kwargs_76024 = {}
    # Getting the type of 'y' (line 963)
    y_76020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 22), 'y', False)
    # Obtaining the member 'reshape' of a type (line 963)
    reshape_76021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 22), y_76020, 'reshape')
    # Calling reshape(args, kwargs) (line 963)
    reshape_call_result_76025 = invoke(stypy.reporting.localization.Localization(__file__, 963, 22), reshape_76021, *[int_76022, extradim_76023], **kwargs_76024)
    
    # Getting the type of 'w' (line 964)
    w_76026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 22), 'w', False)
    # Getting the type of 'ab' (line 965)
    ab_76027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 22), 'ab', False)
    # Getting the type of 'rhs' (line 965)
    rhs_76028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 26), 'rhs', False)
    # Processing the call keyword arguments (line 962)
    kwargs_76029 = {}
    # Getting the type of '_bspl' (line 962)
    _bspl_76015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 4), '_bspl', False)
    # Obtaining the member '_norm_eq_lsq' of a type (line 962)
    _norm_eq_lsq_76016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 4), _bspl_76015, '_norm_eq_lsq')
    # Calling _norm_eq_lsq(args, kwargs) (line 962)
    _norm_eq_lsq_call_result_76030 = invoke(stypy.reporting.localization.Localization(__file__, 962, 4), _norm_eq_lsq_76016, *[x_76017, t_76018, k_76019, reshape_call_result_76025, w_76026, ab_76027, rhs_76028], **kwargs_76029)
    
    
    # Assigning a Call to a Name (line 966):
    
    # Assigning a Call to a Name (line 966):
    
    # Call to reshape(...): (line 966)
    # Processing the call arguments (line 966)
    
    # Obtaining an instance of the builtin type 'tuple' (line 966)
    tuple_76033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 966)
    # Adding element type (line 966)
    # Getting the type of 'n' (line 966)
    n_76034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 23), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 966, 23), tuple_76033, n_76034)
    
    
    # Obtaining the type of the subscript
    int_76035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 37), 'int')
    slice_76036 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 966, 29), int_76035, None, None)
    # Getting the type of 'y' (line 966)
    y_76037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 29), 'y', False)
    # Obtaining the member 'shape' of a type (line 966)
    shape_76038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 29), y_76037, 'shape')
    # Obtaining the member '__getitem__' of a type (line 966)
    getitem___76039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 29), shape_76038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 966)
    subscript_call_result_76040 = invoke(stypy.reporting.localization.Localization(__file__, 966, 29), getitem___76039, slice_76036)
    
    # Applying the binary operator '+' (line 966)
    result_add_76041 = python_operator(stypy.reporting.localization.Localization(__file__, 966, 22), '+', tuple_76033, subscript_call_result_76040)
    
    # Processing the call keyword arguments (line 966)
    kwargs_76042 = {}
    # Getting the type of 'rhs' (line 966)
    rhs_76031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 10), 'rhs', False)
    # Obtaining the member 'reshape' of a type (line 966)
    reshape_76032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 10), rhs_76031, 'reshape')
    # Calling reshape(args, kwargs) (line 966)
    reshape_call_result_76043 = invoke(stypy.reporting.localization.Localization(__file__, 966, 10), reshape_76032, *[result_add_76041], **kwargs_76042)
    
    # Assigning a type to the variable 'rhs' (line 966)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 4), 'rhs', reshape_call_result_76043)
    
    # Assigning a Call to a Name (line 969):
    
    # Assigning a Call to a Name (line 969):
    
    # Call to cholesky_banded(...): (line 969)
    # Processing the call arguments (line 969)
    # Getting the type of 'ab' (line 969)
    ab_76045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 33), 'ab', False)
    # Processing the call keyword arguments (line 969)
    # Getting the type of 'True' (line 969)
    True_76046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 50), 'True', False)
    keyword_76047 = True_76046
    # Getting the type of 'lower' (line 969)
    lower_76048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 62), 'lower', False)
    keyword_76049 = lower_76048
    # Getting the type of 'check_finite' (line 970)
    check_finite_76050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 46), 'check_finite', False)
    keyword_76051 = check_finite_76050
    kwargs_76052 = {'lower': keyword_76049, 'overwrite_ab': keyword_76047, 'check_finite': keyword_76051}
    # Getting the type of 'cholesky_banded' (line 969)
    cholesky_banded_76044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 17), 'cholesky_banded', False)
    # Calling cholesky_banded(args, kwargs) (line 969)
    cholesky_banded_call_result_76053 = invoke(stypy.reporting.localization.Localization(__file__, 969, 17), cholesky_banded_76044, *[ab_76045], **kwargs_76052)
    
    # Assigning a type to the variable 'cho_decomp' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'cho_decomp', cholesky_banded_call_result_76053)
    
    # Assigning a Call to a Name (line 971):
    
    # Assigning a Call to a Name (line 971):
    
    # Call to cho_solve_banded(...): (line 971)
    # Processing the call arguments (line 971)
    
    # Obtaining an instance of the builtin type 'tuple' (line 971)
    tuple_76055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 971)
    # Adding element type (line 971)
    # Getting the type of 'cho_decomp' (line 971)
    cho_decomp_76056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 26), 'cho_decomp', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 971, 26), tuple_76055, cho_decomp_76056)
    # Adding element type (line 971)
    # Getting the type of 'lower' (line 971)
    lower_76057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 38), 'lower', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 971, 26), tuple_76055, lower_76057)
    
    # Getting the type of 'rhs' (line 971)
    rhs_76058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 46), 'rhs', False)
    # Processing the call keyword arguments (line 971)
    # Getting the type of 'True' (line 971)
    True_76059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 63), 'True', False)
    keyword_76060 = True_76059
    # Getting the type of 'check_finite' (line 972)
    check_finite_76061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 38), 'check_finite', False)
    keyword_76062 = check_finite_76061
    kwargs_76063 = {'check_finite': keyword_76062, 'overwrite_b': keyword_76060}
    # Getting the type of 'cho_solve_banded' (line 971)
    cho_solve_banded_76054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 8), 'cho_solve_banded', False)
    # Calling cho_solve_banded(args, kwargs) (line 971)
    cho_solve_banded_call_result_76064 = invoke(stypy.reporting.localization.Localization(__file__, 971, 8), cho_solve_banded_76054, *[tuple_76055, rhs_76058], **kwargs_76063)
    
    # Assigning a type to the variable 'c' (line 971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 4), 'c', cho_solve_banded_call_result_76064)
    
    # Assigning a Call to a Name (line 974):
    
    # Assigning a Call to a Name (line 974):
    
    # Call to ascontiguousarray(...): (line 974)
    # Processing the call arguments (line 974)
    # Getting the type of 'c' (line 974)
    c_76067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 29), 'c', False)
    # Processing the call keyword arguments (line 974)
    kwargs_76068 = {}
    # Getting the type of 'np' (line 974)
    np_76065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 974)
    ascontiguousarray_76066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 8), np_76065, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 974)
    ascontiguousarray_call_result_76069 = invoke(stypy.reporting.localization.Localization(__file__, 974, 8), ascontiguousarray_76066, *[c_76067], **kwargs_76068)
    
    # Assigning a type to the variable 'c' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 4), 'c', ascontiguousarray_call_result_76069)
    
    # Call to construct_fast(...): (line 975)
    # Processing the call arguments (line 975)
    # Getting the type of 't' (line 975)
    t_76072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 34), 't', False)
    # Getting the type of 'c' (line 975)
    c_76073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 37), 'c', False)
    # Getting the type of 'k' (line 975)
    k_76074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 40), 'k', False)
    # Processing the call keyword arguments (line 975)
    # Getting the type of 'axis' (line 975)
    axis_76075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 48), 'axis', False)
    keyword_76076 = axis_76075
    kwargs_76077 = {'axis': keyword_76076}
    # Getting the type of 'BSpline' (line 975)
    BSpline_76070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 11), 'BSpline', False)
    # Obtaining the member 'construct_fast' of a type (line 975)
    construct_fast_76071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 11), BSpline_76070, 'construct_fast')
    # Calling construct_fast(args, kwargs) (line 975)
    construct_fast_call_result_76078 = invoke(stypy.reporting.localization.Localization(__file__, 975, 11), construct_fast_76071, *[t_76072, c_76073, k_76074], **kwargs_76077)
    
    # Assigning a type to the variable 'stypy_return_type' (line 975)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 4), 'stypy_return_type', construct_fast_call_result_76078)
    
    # ################# End of 'make_lsq_spline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_lsq_spline' in the type store
    # Getting the type of 'stypy_return_type' (line 821)
    stypy_return_type_76079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_76079)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_lsq_spline'
    return stypy_return_type_76079

# Assigning a type to the variable 'make_lsq_spline' (line 821)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 0), 'make_lsq_spline', make_lsq_spline)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
