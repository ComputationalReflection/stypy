
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Interpolation algorithms using piecewise cubic polynomials.'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: import numpy as np
6: 
7: from scipy._lib.six import string_types
8: 
9: from . import BPoly, PPoly
10: from .polyint import _isscalar
11: from scipy._lib._util import _asarray_validated
12: from scipy.linalg import solve_banded, solve
13: 
14: 
15: __all__ = ["PchipInterpolator", "pchip_interpolate", "pchip",
16:            "Akima1DInterpolator", "CubicSpline"]
17: 
18: 
19: class PchipInterpolator(BPoly):
20:     r'''PCHIP 1-d monotonic cubic interpolation.
21: 
22:     `x` and `y` are arrays of values used to approximate some function f,
23:     with ``y = f(x)``. The interpolant uses monotonic cubic splines
24:     to find the value of new points. (PCHIP stands for Piecewise Cubic
25:     Hermite Interpolating Polynomial).
26: 
27:     Parameters
28:     ----------
29:     x : ndarray
30:         A 1-D array of monotonically increasing real values.  `x` cannot
31:         include duplicate values (otherwise f is overspecified)
32:     y : ndarray
33:         A 1-D array of real values. `y`'s length along the interpolation
34:         axis must be equal to the length of `x`. If N-D array, use `axis`
35:         parameter to select correct axis.
36:     axis : int, optional
37:         Axis in the y array corresponding to the x-coordinate values.
38:     extrapolate : bool, optional
39:         Whether to extrapolate to out-of-bounds points based on first
40:         and last intervals, or to return NaNs.
41: 
42:     Methods
43:     -------
44:     __call__
45:     derivative
46:     antiderivative
47:     roots
48: 
49:     See Also
50:     --------
51:     Akima1DInterpolator
52:     CubicSpline
53:     BPoly
54: 
55:     Notes
56:     -----
57:     The interpolator preserves monotonicity in the interpolation data and does
58:     not overshoot if the data is not smooth.
59: 
60:     The first derivatives are guaranteed to be continuous, but the second
61:     derivatives may jump at :math:`x_k`.
62: 
63:     Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,
64:     by using PCHIP algorithm [1]_.
65: 
66:     Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`
67:     are the slopes at internal points :math:`x_k`.
68:     If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of
69:     them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the
70:     weighted harmonic mean
71: 
72:     .. math::
73: 
74:         \frac{w_1 + w_2}{f'_k} = \frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}
75: 
76:     where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.
77: 
78:     The end slopes are set using a one-sided scheme [2]_.
79: 
80: 
81:     References
82:     ----------
83:     .. [1] F. N. Fritsch and R. E. Carlson, Monotone Piecewise Cubic Interpolation,
84:            SIAM J. Numer. Anal., 17(2), 238 (1980).
85:            :doi:`10.1137/0717021`.
86:     .. [2] see, e.g., C. Moler, Numerical Computing with Matlab, 2004.
87:            :doi:`10.1137/1.9780898717952`
88: 
89: 
90:     '''
91:     def __init__(self, x, y, axis=0, extrapolate=None):
92:         x = _asarray_validated(x, check_finite=False, as_inexact=True)
93:         y = _asarray_validated(y, check_finite=False, as_inexact=True)
94: 
95:         axis = axis % y.ndim
96: 
97:         xp = x.reshape((x.shape[0],) + (1,)*(y.ndim-1))
98:         yp = np.rollaxis(y, axis)
99: 
100:         dk = self._find_derivatives(xp, yp)
101:         data = np.hstack((yp[:, None, ...], dk[:, None, ...]))
102: 
103:         _b = BPoly.from_derivatives(x, data, orders=None)
104:         super(PchipInterpolator, self).__init__(_b.c, _b.x,
105:                                                 extrapolate=extrapolate)
106:         self.axis = axis
107: 
108:     def roots(self):
109:         '''
110:         Return the roots of the interpolated function.
111:         '''
112:         return (PPoly.from_bernstein_basis(self)).roots()
113: 
114:     @staticmethod
115:     def _edge_case(h0, h1, m0, m1):
116:         # one-sided three-point estimate for the derivative
117:         d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)
118: 
119:         # try to preserve shape
120:         mask = np.sign(d) != np.sign(m0)
121:         mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.*np.abs(m0))
122:         mmm = (~mask) & mask2
123: 
124:         d[mask] = 0.
125:         d[mmm] = 3.*m0[mmm]
126: 
127:         return d
128: 
129:     @staticmethod
130:     def _find_derivatives(x, y):
131:         # Determine the derivatives at the points y_k, d_k, by using
132:         #  PCHIP algorithm is:
133:         # We choose the derivatives at the point x_k by
134:         # Let m_k be the slope of the kth segment (between k and k+1)
135:         # If m_k=0 or m_{k-1}=0 or sgn(m_k) != sgn(m_{k-1}) then d_k == 0
136:         # else use weighted harmonic mean:
137:         #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
138:         #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
139:         #   where h_k is the spacing between x_k and x_{k+1}
140:         y_shape = y.shape
141:         if y.ndim == 1:
142:             # So that _edge_case doesn't end up assigning to scalars
143:             x = x[:, None]
144:             y = y[:, None]
145: 
146:         hk = x[1:] - x[:-1]
147:         mk = (y[1:] - y[:-1]) / hk
148: 
149:         if y.shape[0] == 2:
150:             # edge case: only have two points, use linear interpolation
151:             dk = np.zeros_like(y)
152:             dk[0] = mk
153:             dk[1] = mk
154:             return dk.reshape(y_shape)
155: 
156:         smk = np.sign(mk)
157:         condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)
158: 
159:         w1 = 2*hk[1:] + hk[:-1]
160:         w2 = hk[1:] + 2*hk[:-1]
161: 
162:         # values where division by zero occurs will be excluded
163:         # by 'condition' afterwards
164:         with np.errstate(divide='ignore'):
165:             whmean = (w1/mk[:-1] + w2/mk[1:]) / (w1 + w2)
166: 
167:         dk = np.zeros_like(y)
168:         dk[1:-1][condition] = 0.0
169:         dk[1:-1][~condition] = 1.0 / whmean[~condition]
170: 
171:         # special case endpoints, as suggested in
172:         # Cleve Moler, Numerical Computing with MATLAB, Chap 3.4
173:         dk[0] = PchipInterpolator._edge_case(hk[0], hk[1], mk[0], mk[1])
174:         dk[-1] = PchipInterpolator._edge_case(hk[-1], hk[-2], mk[-1], mk[-2])
175: 
176:         return dk.reshape(y_shape)
177: 
178: 
179: def pchip_interpolate(xi, yi, x, der=0, axis=0):
180:     '''
181:     Convenience function for pchip interpolation.
182:     xi and yi are arrays of values used to approximate some function f,
183:     with ``yi = f(xi)``.  The interpolant uses monotonic cubic splines
184:     to find the value of new points x and the derivatives there.
185: 
186:     See `PchipInterpolator` for details.
187: 
188:     Parameters
189:     ----------
190:     xi : array_like
191:         A sorted list of x-coordinates, of length N.
192:     yi :  array_like
193:         A 1-D array of real values.  `yi`'s length along the interpolation
194:         axis must be equal to the length of `xi`. If N-D array, use axis
195:         parameter to select correct axis.
196:     x : scalar or array_like
197:         Of length M.
198:     der : int or list, optional
199:         Derivatives to extract.  The 0-th derivative can be included to
200:         return the function value.
201:     axis : int, optional
202:         Axis in the yi array corresponding to the x-coordinate values.
203: 
204:     See Also
205:     --------
206:     PchipInterpolator
207: 
208:     Returns
209:     -------
210:     y : scalar or array_like
211:         The result, of length R or length M or M by R,
212: 
213:     '''
214:     P = PchipInterpolator(xi, yi, axis=axis)
215: 
216:     if der == 0:
217:         return P(x)
218:     elif _isscalar(der):
219:         return P.derivative(der)(x)
220:     else:
221:         return [P.derivative(nu)(x) for nu in der]
222: 
223: 
224: # Backwards compatibility
225: pchip = PchipInterpolator
226: 
227: 
228: class Akima1DInterpolator(PPoly):
229:     '''
230:     Akima interpolator
231: 
232:     Fit piecewise cubic polynomials, given vectors x and y. The interpolation
233:     method by Akima uses a continuously differentiable sub-spline built from
234:     piecewise cubic polynomials. The resultant curve passes through the given
235:     data points and will appear smooth and natural.
236: 
237:     Parameters
238:     ----------
239:     x : ndarray, shape (m, )
240:         1-D array of monotonically increasing real values.
241:     y : ndarray, shape (m, ...)
242:         N-D array of real values. The length of `y` along the first axis must
243:         be equal to the length of `x`.
244:     axis : int, optional
245:         Specifies the axis of `y` along which to interpolate. Interpolation
246:         defaults to the first axis of `y`.
247: 
248:     Methods
249:     -------
250:     __call__
251:     derivative
252:     antiderivative
253:     roots
254: 
255:     See Also
256:     --------
257:     PchipInterpolator
258:     CubicSpline
259:     PPoly
260: 
261:     Notes
262:     -----
263:     .. versionadded:: 0.14
264: 
265:     Use only for precise data, as the fitted curve passes through the given
266:     points exactly. This routine is useful for plotting a pleasingly smooth
267:     curve through a few given points for purposes of plotting.
268: 
269:     References
270:     ----------
271:     [1] A new method of interpolation and smooth curve fitting based
272:         on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
273:         589-602.
274: 
275:     '''
276: 
277:     def __init__(self, x, y, axis=0):
278:         # Original implementation in MATLAB by N. Shamsundar (BSD licensed), see
279:         # http://www.mathworks.de/matlabcentral/fileexchange/1814-akima-interpolation
280:         x, y = map(np.asarray, (x, y))
281:         axis = axis % y.ndim
282: 
283:         if np.any(np.diff(x) < 0.):
284:             raise ValueError("x must be strictly ascending")
285:         if x.ndim != 1:
286:             raise ValueError("x must be 1-dimensional")
287:         if x.size < 2:
288:             raise ValueError("at least 2 breakpoints are needed")
289:         if x.size != y.shape[axis]:
290:             raise ValueError("x.shape must equal y.shape[%s]" % axis)
291: 
292:         # move interpolation axis to front
293:         y = np.rollaxis(y, axis)
294: 
295:         # determine slopes between breakpoints
296:         m = np.empty((x.size + 3, ) + y.shape[1:])
297:         dx = np.diff(x)
298:         dx = dx[(slice(None), ) + (None, ) * (y.ndim - 1)]
299:         m[2:-2] = np.diff(y, axis=0) / dx
300: 
301:         # add two additional points on the left ...
302:         m[1] = 2. * m[2] - m[3]
303:         m[0] = 2. * m[1] - m[2]
304:         # ... and on the right
305:         m[-2] = 2. * m[-3] - m[-4]
306:         m[-1] = 2. * m[-2] - m[-3]
307: 
308:         # if m1 == m2 != m3 == m4, the slope at the breakpoint is not defined.
309:         # This is the fill value:
310:         t = .5 * (m[3:] + m[:-3])
311:         # get the denominator of the slope t
312:         dm = np.abs(np.diff(m, axis=0))
313:         f1 = dm[2:]
314:         f2 = dm[:-2]
315:         f12 = f1 + f2
316:         # These are the mask of where the the slope at breakpoint is defined:
317:         ind = np.nonzero(f12 > 1e-9 * np.max(f12))
318:         x_ind, y_ind = ind[0], ind[1:]
319:         # Set the slope at breakpoint
320:         t[ind] = (f1[ind] * m[(x_ind + 1,) + y_ind] +
321:                   f2[ind] * m[(x_ind + 2,) + y_ind]) / f12[ind]
322:         # calculate the higher order coefficients
323:         c = (3. * m[2:-2] - 2. * t[:-1] - t[1:]) / dx
324:         d = (t[:-1] + t[1:] - 2. * m[2:-2]) / dx ** 2
325: 
326:         coeff = np.zeros((4, x.size - 1) + y.shape[1:])
327:         coeff[3] = y[:-1]
328:         coeff[2] = t[:-1]
329:         coeff[1] = c
330:         coeff[0] = d
331: 
332:         super(Akima1DInterpolator, self).__init__(coeff, x, extrapolate=False)
333:         self.axis = axis
334: 
335:     def extend(self, c, x, right=True):
336:         raise NotImplementedError("Extending a 1D Akima interpolator is not "
337:                                   "yet implemented")
338: 
339:     # These are inherited from PPoly, but they do not produce an Akima
340:     # interpolator. Hence stub them out.
341:     @classmethod
342:     def from_spline(cls, tck, extrapolate=None):
343:         raise NotImplementedError("This method does not make sense for "
344:                                   "an Akima interpolator.")
345: 
346:     @classmethod
347:     def from_bernstein_basis(cls, bp, extrapolate=None):
348:         raise NotImplementedError("This method does not make sense for "
349:                                   "an Akima interpolator.")
350: 
351: 
352: class CubicSpline(PPoly):
353:     '''Cubic spline data interpolator.
354: 
355:     Interpolate data with a piecewise cubic polynomial which is twice
356:     continuously differentiable [1]_. The result is represented as a `PPoly`
357:     instance with breakpoints matching the given data.
358: 
359:     Parameters
360:     ----------
361:     x : array_like, shape (n,)
362:         1-d array containing values of the independent variable.
363:         Values must be real, finite and in strictly increasing order.
364:     y : array_like
365:         Array containing values of the dependent variable. It can have
366:         arbitrary number of dimensions, but the length along `axis` (see below)
367:         must match the length of `x`. Values must be finite.
368:     axis : int, optional
369:         Axis along which `y` is assumed to be varying. Meaning that for
370:         ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
371:         Default is 0.
372:     bc_type : string or 2-tuple, optional
373:         Boundary condition type. Two additional equations, given by the
374:         boundary conditions, are required to determine all coefficients of
375:         polynomials on each segment [2]_.
376: 
377:         If `bc_type` is a string, then the specified condition will be applied
378:         at both ends of a spline. Available conditions are:
379: 
380:         * 'not-a-knot' (default): The first and second segment at a curve end
381:           are the same polynomial. It is a good default when there is no
382:           information on boundary conditions.
383:         * 'periodic': The interpolated functions is assumed to be periodic
384:           of period ``x[-1] - x[0]``. The first and last value of `y` must be
385:           identical: ``y[0] == y[-1]``. This boundary condition will result in
386:           ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
387:         * 'clamped': The first derivative at curves ends are zero. Assuming
388:           a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
389:         * 'natural': The second derivative at curve ends are zero. Assuming
390:           a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.
391: 
392:         If `bc_type` is a 2-tuple, the first and the second value will be
393:         applied at the curve start and end respectively. The tuple values can
394:         be one of the previously mentioned strings (except 'periodic') or a
395:         tuple `(order, deriv_values)` allowing to specify arbitrary
396:         derivatives at curve ends:
397: 
398:         * `order`: the derivative order, 1 or 2.
399:         * `deriv_value`: array_like containing derivative values, shape must
400:           be the same as `y`, excluding `axis` dimension. For example, if `y`
401:           is 1D, then `deriv_value` must be a scalar. If `y` is 3D with the
402:           shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D
403:           and have the shape (n0, n1).
404:     extrapolate : {bool, 'periodic', None}, optional
405:         If bool, determines whether to extrapolate to out-of-bounds points
406:         based on first and last intervals, or to return NaNs. If 'periodic',
407:         periodic extrapolation is used. If None (default), `extrapolate` is
408:         set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.
409: 
410:     Attributes
411:     ----------
412:     x : ndarray, shape (n,)
413:         Breakpoints. The same `x` which was passed to the constructor.
414:     c : ndarray, shape (4, n-1, ...)
415:         Coefficients of the polynomials on each segment. The trailing
416:         dimensions match the dimensions of `y`, excluding `axis`. For example,
417:         if `y` is 1-d, then ``c[k, i]`` is a coefficient for
418:         ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
419:     axis : int
420:         Interpolation axis. The same `axis` which was passed to the
421:         constructor.
422: 
423:     Methods
424:     -------
425:     __call__
426:     derivative
427:     antiderivative
428:     integrate
429:     roots
430: 
431:     See Also
432:     --------
433:     Akima1DInterpolator
434:     PchipInterpolator
435:     PPoly
436: 
437:     Notes
438:     -----
439:     Parameters `bc_type` and `interpolate` work independently, i.e. the former
440:     controls only construction of a spline, and the latter only evaluation.
441: 
442:     When a boundary condition is 'not-a-knot' and n = 2, it is replaced by
443:     a condition that the first derivative is equal to the linear interpolant
444:     slope. When both boundary conditions are 'not-a-knot' and n = 3, the
445:     solution is sought as a parabola passing through given points.
446: 
447:     When 'not-a-knot' boundary conditions is applied to both ends, the
448:     resulting spline will be the same as returned by `splrep` (with ``s=0``)
449:     and `InterpolatedUnivariateSpline`, but these two methods use a
450:     representation in B-spline basis.
451: 
452:     .. versionadded:: 0.18.0
453: 
454:     Examples
455:     --------
456:     In this example the cubic spline is used to interpolate a sampled sinusoid.
457:     You can see that the spline continuity property holds for the first and
458:     second derivatives and violates only for the third derivative.
459: 
460:     >>> from scipy.interpolate import CubicSpline
461:     >>> import matplotlib.pyplot as plt
462:     >>> x = np.arange(10)
463:     >>> y = np.sin(x)
464:     >>> cs = CubicSpline(x, y)
465:     >>> xs = np.arange(-0.5, 9.6, 0.1)
466:     >>> plt.figure(figsize=(6.5, 4))
467:     >>> plt.plot(x, y, 'o', label='data')
468:     >>> plt.plot(xs, np.sin(xs), label='true')
469:     >>> plt.plot(xs, cs(xs), label="S")
470:     >>> plt.plot(xs, cs(xs, 1), label="S'")
471:     >>> plt.plot(xs, cs(xs, 2), label="S''")
472:     >>> plt.plot(xs, cs(xs, 3), label="S'''")
473:     >>> plt.xlim(-0.5, 9.5)
474:     >>> plt.legend(loc='lower left', ncol=2)
475:     >>> plt.show()
476: 
477:     In the second example, the unit circle is interpolated with a spline. A
478:     periodic boundary condition is used. You can see that the first derivative
479:     values, ds/dx=0, ds/dy=1 at the periodic point (1, 0) are correctly
480:     computed. Note that a circle cannot be exactly represented by a cubic
481:     spline. To increase precision, more breakpoints would be required.
482: 
483:     >>> theta = 2 * np.pi * np.linspace(0, 1, 5)
484:     >>> y = np.c_[np.cos(theta), np.sin(theta)]
485:     >>> cs = CubicSpline(theta, y, bc_type='periodic')
486:     >>> print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
487:     ds/dx=0.0 ds/dy=1.0
488:     >>> xs = 2 * np.pi * np.linspace(0, 1, 100)
489:     >>> plt.figure(figsize=(6.5, 4))
490:     >>> plt.plot(y[:, 0], y[:, 1], 'o', label='data')
491:     >>> plt.plot(np.cos(xs), np.sin(xs), label='true')
492:     >>> plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
493:     >>> plt.axes().set_aspect('equal')
494:     >>> plt.legend(loc='center')
495:     >>> plt.show()
496: 
497:     The third example is the interpolation of a polynomial y = x**3 on the
498:     interval 0 <= x<= 1. A cubic spline can represent this function exactly.
499:     To achieve that we need to specify values and first derivatives at
500:     endpoints of the interval. Note that y' = 3 * x**2 and thus y'(0) = 0 and
501:     y'(1) = 3.
502: 
503:     >>> cs = CubicSpline([0, 1], [0, 1], bc_type=((1, 0), (1, 3)))
504:     >>> x = np.linspace(0, 1)
505:     >>> np.allclose(x**3, cs(x))
506:     True
507: 
508:     References
509:     ----------
510:     .. [1] `Cubic Spline Interpolation
511:             <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
512:             on Wikiversity.
513:     .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
514:     '''
515:     def __init__(self, x, y, axis=0, bc_type='not-a-knot', extrapolate=None):
516:         x, y = map(np.asarray, (x, y))
517: 
518:         if np.issubdtype(x.dtype, np.complexfloating):
519:             raise ValueError("`x` must contain real values.")
520: 
521:         if np.issubdtype(y.dtype, np.complexfloating):
522:             dtype = complex
523:         else:
524:             dtype = float
525:         y = y.astype(dtype, copy=False)
526: 
527:         axis = axis % y.ndim
528:         if x.ndim != 1:
529:             raise ValueError("`x` must be 1-dimensional.")
530:         if x.shape[0] < 2:
531:             raise ValueError("`x` must contain at least 2 elements.")
532:         if x.shape[0] != y.shape[axis]:
533:             raise ValueError("The length of `y` along `axis`={0} doesn't "
534:                              "match the length of `x`".format(axis))
535: 
536:         if not np.all(np.isfinite(x)):
537:             raise ValueError("`x` must contain only finite values.")
538:         if not np.all(np.isfinite(y)):
539:             raise ValueError("`y` must contain only finite values.")
540: 
541:         dx = np.diff(x)
542:         if np.any(dx <= 0):
543:             raise ValueError("`x` must be strictly increasing sequence.")
544: 
545:         n = x.shape[0]
546:         y = np.rollaxis(y, axis)
547: 
548:         bc, y = self._validate_bc(bc_type, y, y.shape[1:], axis)
549: 
550:         if extrapolate is None:
551:             if bc[0] == 'periodic':
552:                 extrapolate = 'periodic'
553:             else:
554:                 extrapolate = True
555: 
556:         dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
557:         slope = np.diff(y, axis=0) / dxr
558: 
559:         # If bc is 'not-a-knot' this change is just a convention.
560:         # If bc is 'periodic' then we already checked that y[0] == y[-1],
561:         # and the spline is just a constant, we handle this case in the same
562:         # way by setting the first derivatives to slope, which is 0.
563:         if n == 2:
564:             if bc[0] in ['not-a-knot', 'periodic']:
565:                 bc[0] = (1, slope[0])
566:             if bc[1] in ['not-a-knot', 'periodic']:
567:                 bc[1] = (1, slope[0])
568: 
569:         # This is a very special case, when both conditions are 'not-a-knot'
570:         # and n == 3. In this case 'not-a-knot' can't be handled regularly
571:         # as the both conditions are identical. We handle this case by
572:         # constructing a parabola passing through given points.
573:         if n == 3 and bc[0] == 'not-a-knot' and bc[1] == 'not-a-knot':
574:             A = np.zeros((3, 3))  # This is a standard matrix.
575:             b = np.empty((3,) + y.shape[1:], dtype=y.dtype)
576: 
577:             A[0, 0] = 1
578:             A[0, 1] = 1
579:             A[1, 0] = dx[1]
580:             A[1, 1] = 2 * (dx[0] + dx[1])
581:             A[1, 2] = dx[0]
582:             A[2, 1] = 1
583:             A[2, 2] = 1
584: 
585:             b[0] = 2 * slope[0]
586:             b[1] = 3 * (dxr[0] * slope[1] + dxr[1] * slope[0])
587:             b[2] = 2 * slope[1]
588: 
589:             s = solve(A, b, overwrite_a=True, overwrite_b=True,
590:                       check_finite=False)
591:         else:
592:             # Find derivative values at each x[i] by solving a tridiagonal
593:             # system.
594:             A = np.zeros((3, n))  # This is a banded matrix representation.
595:             b = np.empty((n,) + y.shape[1:], dtype=y.dtype)
596: 
597:             # Filling the system for i=1..n-2
598:             #                         (x[i-1] - x[i]) * s[i-1] +\
599:             # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
600:             #                         (x[i] - x[i-1]) * s[i+1] =\
601:             #       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
602:             #           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))
603: 
604:             A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
605:             A[0, 2:] = dx[:-1]                   # The upper diagonal
606:             A[-1, :-2] = dx[1:]                  # The lower diagonal
607: 
608:             b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])
609: 
610:             bc_start, bc_end = bc
611: 
612:             if bc_start == 'periodic':
613:                 # Due to the periodicity, and because y[-1] = y[0], the linear
614:                 # system has (n-1) unknowns/equations instead of n:
615:                 A = A[:, 0:-1]
616:                 A[1, 0] = 2 * (dx[-1] + dx[0])
617:                 A[0, 1] = dx[-1]
618: 
619:                 b = b[:-1]
620: 
621:                 # Also, due to the periodicity, the system is not tri-diagonal.
622:                 # We need to compute a "condensed" matrix of shape (n-2, n-2).
623:                 # See http://www.cfm.brown.edu/people/gk/chap6/node14.html for
624:                 # more explanations.
625:                 # The condensed matrix is obtained by removing the last column
626:                 # and last row of the (n-1, n-1) system matrix. The removed
627:                 # values are saved in scalar variables with the (n-1, n-1)
628:                 # system matrix indices forming their names:
629:                 a_m1_0 = dx[-2]  # lower left corner value: A[-1, 0]
630:                 a_m1_m2 = dx[-1]
631:                 a_m1_m1 = 2 * (dx[-1] + dx[-2])
632:                 a_m2_m1 = dx[-2]
633:                 a_0_m1 = dx[0]
634: 
635:                 b[0] = 3 * (dxr[0] * slope[-1] + dxr[-1] * slope[0])
636:                 b[-1] = 3 * (dxr[-1] * slope[-2] + dxr[-2] * slope[-1])
637: 
638:                 Ac = A[:, :-1]
639:                 b1 = b[:-1]
640:                 b2 = np.zeros_like(b1)
641:                 b2[0] = -a_0_m1
642:                 b2[-1] = -a_m2_m1
643: 
644:                 # s1 and s2 are the solutions of (n-2, n-2) system
645:                 s1 = solve_banded((1, 1), Ac, b1, overwrite_ab=False,
646:                                   overwrite_b=False, check_finite=False)
647: 
648:                 s2 = solve_banded((1, 1), Ac, b2, overwrite_ab=False,
649:                                   overwrite_b=False, check_finite=False)
650: 
651:                 # computing the s[n-2] solution:
652:                 s_m1 = ((b[-1] - a_m1_0 * s1[0] - a_m1_m2 * s1[-1]) /
653:                         (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[-1]))
654: 
655:                 # s is the solution of the (n, n) system:
656:                 s = np.empty((n,) + y.shape[1:], dtype=y.dtype)
657:                 s[:-2] = s1 + s_m1 * s2
658:                 s[-2] = s_m1
659:                 s[-1] = s[0]
660:             else:
661:                 if bc_start == 'not-a-knot':
662:                     A[1, 0] = dx[1]
663:                     A[0, 1] = x[2] - x[0]
664:                     d = x[2] - x[0]
665:                     b[0] = ((dxr[0] + 2*d) * dxr[1] * slope[0] +
666:                             dxr[0]**2 * slope[1]) / d
667:                 elif bc_start[0] == 1:
668:                     A[1, 0] = 1
669:                     A[0, 1] = 0
670:                     b[0] = bc_start[1]
671:                 elif bc_start[0] == 2:
672:                     A[1, 0] = 2 * dx[0]
673:                     A[0, 1] = dx[0]
674:                     b[0] = -0.5 * bc_start[1] * dx[0]**2 + 3 * (y[1] - y[0])
675: 
676:                 if bc_end == 'not-a-knot':
677:                     A[1, -1] = dx[-2]
678:                     A[-1, -2] = x[-1] - x[-3]
679:                     d = x[-1] - x[-3]
680:                     b[-1] = ((dxr[-1]**2*slope[-2] +
681:                              (2*d + dxr[-1])*dxr[-2]*slope[-1]) / d)
682:                 elif bc_end[0] == 1:
683:                     A[1, -1] = 1
684:                     A[-1, -2] = 0
685:                     b[-1] = bc_end[1]
686:                 elif bc_end[0] == 2:
687:                     A[1, -1] = 2 * dx[-1]
688:                     A[-1, -2] = dx[-1]
689:                     b[-1] = 0.5 * bc_end[1] * dx[-1]**2 + 3 * (y[-1] - y[-2])
690: 
691:                 s = solve_banded((1, 1), A, b, overwrite_ab=True,
692:                                  overwrite_b=True, check_finite=False)
693: 
694:         # Compute coefficients in PPoly form.
695:         t = (s[:-1] + s[1:] - 2 * slope) / dxr
696:         c = np.empty((4, n - 1) + y.shape[1:], dtype=t.dtype)
697:         c[0] = t / dxr
698:         c[1] = (slope - s[:-1]) / dxr - t
699:         c[2] = s[:-1]
700:         c[3] = y[:-1]
701: 
702:         super(CubicSpline, self).__init__(c, x, extrapolate=extrapolate)
703:         self.axis = axis
704: 
705:     @staticmethod
706:     def _validate_bc(bc_type, y, expected_deriv_shape, axis):
707:         '''Validate and prepare boundary conditions.
708: 
709:         Returns
710:         -------
711:         validated_bc : 2-tuple
712:             Boundary conditions for a curve start and end.
713:         y : ndarray
714:             y casted to complex dtype if one of the boundary conditions has
715:             complex dtype.
716:         '''
717:         if isinstance(bc_type, string_types):
718:             if bc_type == 'periodic':
719:                 if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
720:                     raise ValueError(
721:                         "The first and last `y` point along axis {} must "
722:                         "be identical (within machine precision) when "
723:                         "bc_type='periodic'.".format(axis))
724: 
725:             bc_type = (bc_type, bc_type)
726: 
727:         else:
728:             if len(bc_type) != 2:
729:                 raise ValueError("`bc_type` must contain 2 elements to "
730:                                  "specify start and end conditions.")
731: 
732:             if 'periodic' in bc_type:
733:                 raise ValueError("'periodic' `bc_type` is defined for both "
734:                                  "curve ends and cannot be used with other "
735:                                  "boundary conditions.")
736: 
737:         validated_bc = []
738:         for bc in bc_type:
739:             if isinstance(bc, string_types):
740:                 if bc == 'clamped':
741:                     validated_bc.append((1, np.zeros(expected_deriv_shape)))
742:                 elif bc == 'natural':
743:                     validated_bc.append((2, np.zeros(expected_deriv_shape)))
744:                 elif bc in ['not-a-knot', 'periodic']:
745:                     validated_bc.append(bc)
746:                 else:
747:                     raise ValueError("bc_type={} is not allowed.".format(bc))
748:             else:
749:                 try:
750:                     deriv_order, deriv_value = bc
751:                 except Exception:
752:                     raise ValueError("A specified derivative value must be "
753:                                      "given in the form (order, value).")
754: 
755:                 if deriv_order not in [1, 2]:
756:                     raise ValueError("The specified derivative order must "
757:                                      "be 1 or 2.")
758: 
759:                 deriv_value = np.asarray(deriv_value)
760:                 if deriv_value.shape != expected_deriv_shape:
761:                     raise ValueError(
762:                         "`deriv_value` shape {} is not the expected one {}."
763:                         .format(deriv_value.shape, expected_deriv_shape))
764: 
765:                 if np.issubdtype(deriv_value.dtype, np.complexfloating):
766:                     y = y.astype(complex, copy=False)
767: 
768:                 validated_bc.append((deriv_order, deriv_value))
769: 
770:         return validated_bc, y
771: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_76092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Interpolation algorithms using piecewise cubic polynomials.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_76093 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_76093) is not StypyTypeError):

    if (import_76093 != 'pyd_module'):
        __import__(import_76093)
        sys_modules_76094 = sys.modules[import_76093]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_76094.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_76093)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib.six import string_types' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_76095 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib.six')

if (type(import_76095) is not StypyTypeError):

    if (import_76095 != 'pyd_module'):
        __import__(import_76095)
        sys_modules_76096 = sys.modules[import_76095]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib.six', sys_modules_76096.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_76096, sys_modules_76096.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib.six', import_76095)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.interpolate import BPoly, PPoly' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_76097 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.interpolate')

if (type(import_76097) is not StypyTypeError):

    if (import_76097 != 'pyd_module'):
        __import__(import_76097)
        sys_modules_76098 = sys.modules[import_76097]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.interpolate', sys_modules_76098.module_type_store, module_type_store, ['BPoly', 'PPoly'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_76098, sys_modules_76098.module_type_store, module_type_store)
    else:
        from scipy.interpolate import BPoly, PPoly

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.interpolate', None, module_type_store, ['BPoly', 'PPoly'], [BPoly, PPoly])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.interpolate', import_76097)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.interpolate.polyint import _isscalar' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_76099 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.polyint')

if (type(import_76099) is not StypyTypeError):

    if (import_76099 != 'pyd_module'):
        __import__(import_76099)
        sys_modules_76100 = sys.modules[import_76099]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.polyint', sys_modules_76100.module_type_store, module_type_store, ['_isscalar'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_76100, sys_modules_76100.module_type_store, module_type_store)
    else:
        from scipy.interpolate.polyint import _isscalar

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.polyint', None, module_type_store, ['_isscalar'], [_isscalar])

else:
    # Assigning a type to the variable 'scipy.interpolate.polyint' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.polyint', import_76099)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._util import _asarray_validated' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_76101 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._util')

if (type(import_76101) is not StypyTypeError):

    if (import_76101 != 'pyd_module'):
        __import__(import_76101)
        sys_modules_76102 = sys.modules[import_76101]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._util', sys_modules_76102.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_76102, sys_modules_76102.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._util', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._util', import_76101)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.linalg import solve_banded, solve' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_76103 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg')

if (type(import_76103) is not StypyTypeError):

    if (import_76103 != 'pyd_module'):
        __import__(import_76103)
        sys_modules_76104 = sys.modules[import_76103]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg', sys_modules_76104.module_type_store, module_type_store, ['solve_banded', 'solve'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_76104, sys_modules_76104.module_type_store, module_type_store)
    else:
        from scipy.linalg import solve_banded, solve

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg', None, module_type_store, ['solve_banded', 'solve'], [solve_banded, solve])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg', import_76103)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):
__all__ = ['PchipInterpolator', 'pchip_interpolate', 'pchip', 'Akima1DInterpolator', 'CubicSpline']
module_type_store.set_exportable_members(['PchipInterpolator', 'pchip_interpolate', 'pchip', 'Akima1DInterpolator', 'CubicSpline'])

# Obtaining an instance of the builtin type 'list' (line 15)
list_76105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_76106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'PchipInterpolator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_76105, str_76106)
# Adding element type (line 15)
str_76107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', 'pchip_interpolate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_76105, str_76107)
# Adding element type (line 15)
str_76108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 53), 'str', 'pchip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_76105, str_76108)
# Adding element type (line 15)
str_76109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'Akima1DInterpolator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_76105, str_76109)
# Adding element type (line 15)
str_76110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'str', 'CubicSpline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_76105, str_76110)

# Assigning a type to the variable '__all__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__all__', list_76105)
# Declaration of the 'PchipInterpolator' class
# Getting the type of 'BPoly' (line 19)
BPoly_76111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'BPoly')

class PchipInterpolator(BPoly_76111, ):
    str_76112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', "PCHIP 1-d monotonic cubic interpolation.\n\n    `x` and `y` are arrays of values used to approximate some function f,\n    with ``y = f(x)``. The interpolant uses monotonic cubic splines\n    to find the value of new points. (PCHIP stands for Piecewise Cubic\n    Hermite Interpolating Polynomial).\n\n    Parameters\n    ----------\n    x : ndarray\n        A 1-D array of monotonically increasing real values.  `x` cannot\n        include duplicate values (otherwise f is overspecified)\n    y : ndarray\n        A 1-D array of real values. `y`'s length along the interpolation\n        axis must be equal to the length of `x`. If N-D array, use `axis`\n        parameter to select correct axis.\n    axis : int, optional\n        Axis in the y array corresponding to the x-coordinate values.\n    extrapolate : bool, optional\n        Whether to extrapolate to out-of-bounds points based on first\n        and last intervals, or to return NaNs.\n\n    Methods\n    -------\n    __call__\n    derivative\n    antiderivative\n    roots\n\n    See Also\n    --------\n    Akima1DInterpolator\n    CubicSpline\n    BPoly\n\n    Notes\n    -----\n    The interpolator preserves monotonicity in the interpolation data and does\n    not overshoot if the data is not smooth.\n\n    The first derivatives are guaranteed to be continuous, but the second\n    derivatives may jump at :math:`x_k`.\n\n    Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,\n    by using PCHIP algorithm [1]_.\n\n    Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`\n    are the slopes at internal points :math:`x_k`.\n    If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of\n    them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the\n    weighted harmonic mean\n\n    .. math::\n\n        \\frac{w_1 + w_2}{f'_k} = \\frac{w_1}{d_{k-1}} + \\frac{w_2}{d_k}\n\n    where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.\n\n    The end slopes are set using a one-sided scheme [2]_.\n\n\n    References\n    ----------\n    .. [1] F. N. Fritsch and R. E. Carlson, Monotone Piecewise Cubic Interpolation,\n           SIAM J. Numer. Anal., 17(2), 238 (1980).\n           :doi:`10.1137/0717021`.\n    .. [2] see, e.g., C. Moler, Numerical Computing with Matlab, 2004.\n           :doi:`10.1137/1.9780898717952`\n\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_76113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 34), 'int')
        # Getting the type of 'None' (line 91)
        None_76114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'None')
        defaults = [int_76113, None_76114]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PchipInterpolator.__init__', ['x', 'y', 'axis', 'extrapolate'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'axis', 'extrapolate'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to _asarray_validated(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'x' (line 92)
        x_76116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'x', False)
        # Processing the call keyword arguments (line 92)
        # Getting the type of 'False' (line 92)
        False_76117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'False', False)
        keyword_76118 = False_76117
        # Getting the type of 'True' (line 92)
        True_76119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 65), 'True', False)
        keyword_76120 = True_76119
        kwargs_76121 = {'as_inexact': keyword_76120, 'check_finite': keyword_76118}
        # Getting the type of '_asarray_validated' (line 92)
        _asarray_validated_76115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 92)
        _asarray_validated_call_result_76122 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), _asarray_validated_76115, *[x_76116], **kwargs_76121)
        
        # Assigning a type to the variable 'x' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'x', _asarray_validated_call_result_76122)
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to _asarray_validated(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'y' (line 93)
        y_76124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'y', False)
        # Processing the call keyword arguments (line 93)
        # Getting the type of 'False' (line 93)
        False_76125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 47), 'False', False)
        keyword_76126 = False_76125
        # Getting the type of 'True' (line 93)
        True_76127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 65), 'True', False)
        keyword_76128 = True_76127
        kwargs_76129 = {'as_inexact': keyword_76128, 'check_finite': keyword_76126}
        # Getting the type of '_asarray_validated' (line 93)
        _asarray_validated_76123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 93)
        _asarray_validated_call_result_76130 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), _asarray_validated_76123, *[y_76124], **kwargs_76129)
        
        # Assigning a type to the variable 'y' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'y', _asarray_validated_call_result_76130)
        
        # Assigning a BinOp to a Name (line 95):
        
        # Assigning a BinOp to a Name (line 95):
        # Getting the type of 'axis' (line 95)
        axis_76131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'axis')
        # Getting the type of 'y' (line 95)
        y_76132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'y')
        # Obtaining the member 'ndim' of a type (line 95)
        ndim_76133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 22), y_76132, 'ndim')
        # Applying the binary operator '%' (line 95)
        result_mod_76134 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), '%', axis_76131, ndim_76133)
        
        # Assigning a type to the variable 'axis' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'axis', result_mod_76134)
        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to reshape(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_76137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        # Adding element type (line 97)
        
        # Obtaining the type of the subscript
        int_76138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 32), 'int')
        # Getting the type of 'x' (line 97)
        x_76139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'x', False)
        # Obtaining the member 'shape' of a type (line 97)
        shape_76140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), x_76139, 'shape')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___76141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), shape_76140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_76142 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), getitem___76141, int_76138)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 24), tuple_76137, subscript_call_result_76142)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_76143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        # Adding element type (line 97)
        int_76144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 40), tuple_76143, int_76144)
        
        # Getting the type of 'y' (line 97)
        y_76145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 45), 'y', False)
        # Obtaining the member 'ndim' of a type (line 97)
        ndim_76146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 45), y_76145, 'ndim')
        int_76147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 52), 'int')
        # Applying the binary operator '-' (line 97)
        result_sub_76148 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 45), '-', ndim_76146, int_76147)
        
        # Applying the binary operator '*' (line 97)
        result_mul_76149 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 39), '*', tuple_76143, result_sub_76148)
        
        # Applying the binary operator '+' (line 97)
        result_add_76150 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 23), '+', tuple_76137, result_mul_76149)
        
        # Processing the call keyword arguments (line 97)
        kwargs_76151 = {}
        # Getting the type of 'x' (line 97)
        x_76135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'x', False)
        # Obtaining the member 'reshape' of a type (line 97)
        reshape_76136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 13), x_76135, 'reshape')
        # Calling reshape(args, kwargs) (line 97)
        reshape_call_result_76152 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), reshape_76136, *[result_add_76150], **kwargs_76151)
        
        # Assigning a type to the variable 'xp' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'xp', reshape_call_result_76152)
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to rollaxis(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'y' (line 98)
        y_76155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'y', False)
        # Getting the type of 'axis' (line 98)
        axis_76156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'axis', False)
        # Processing the call keyword arguments (line 98)
        kwargs_76157 = {}
        # Getting the type of 'np' (line 98)
        np_76153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'np', False)
        # Obtaining the member 'rollaxis' of a type (line 98)
        rollaxis_76154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 13), np_76153, 'rollaxis')
        # Calling rollaxis(args, kwargs) (line 98)
        rollaxis_call_result_76158 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), rollaxis_76154, *[y_76155, axis_76156], **kwargs_76157)
        
        # Assigning a type to the variable 'yp' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'yp', rollaxis_call_result_76158)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to _find_derivatives(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'xp' (line 100)
        xp_76161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'xp', False)
        # Getting the type of 'yp' (line 100)
        yp_76162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 40), 'yp', False)
        # Processing the call keyword arguments (line 100)
        kwargs_76163 = {}
        # Getting the type of 'self' (line 100)
        self_76159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'self', False)
        # Obtaining the member '_find_derivatives' of a type (line 100)
        _find_derivatives_76160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), self_76159, '_find_derivatives')
        # Calling _find_derivatives(args, kwargs) (line 100)
        _find_derivatives_call_result_76164 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), _find_derivatives_76160, *[xp_76161, yp_76162], **kwargs_76163)
        
        # Assigning a type to the variable 'dk' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'dk', _find_derivatives_call_result_76164)
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to hstack(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_76167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        
        # Obtaining the type of the subscript
        slice_76168 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 26), None, None, None)
        # Getting the type of 'None' (line 101)
        None_76169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'None', False)
        Ellipsis_76170 = Ellipsis
        # Getting the type of 'yp' (line 101)
        yp_76171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'yp', False)
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___76172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 26), yp_76171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_76173 = invoke(stypy.reporting.localization.Localization(__file__, 101, 26), getitem___76172, (slice_76168, None_76169, Ellipsis_76170))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 26), tuple_76167, subscript_call_result_76173)
        # Adding element type (line 101)
        
        # Obtaining the type of the subscript
        slice_76174 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 44), None, None, None)
        # Getting the type of 'None' (line 101)
        None_76175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'None', False)
        Ellipsis_76176 = Ellipsis
        # Getting the type of 'dk' (line 101)
        dk_76177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'dk', False)
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___76178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 44), dk_76177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_76179 = invoke(stypy.reporting.localization.Localization(__file__, 101, 44), getitem___76178, (slice_76174, None_76175, Ellipsis_76176))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 26), tuple_76167, subscript_call_result_76179)
        
        # Processing the call keyword arguments (line 101)
        kwargs_76180 = {}
        # Getting the type of 'np' (line 101)
        np_76165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'np', False)
        # Obtaining the member 'hstack' of a type (line 101)
        hstack_76166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), np_76165, 'hstack')
        # Calling hstack(args, kwargs) (line 101)
        hstack_call_result_76181 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), hstack_76166, *[tuple_76167], **kwargs_76180)
        
        # Assigning a type to the variable 'data' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'data', hstack_call_result_76181)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to from_derivatives(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'x' (line 103)
        x_76184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'x', False)
        # Getting the type of 'data' (line 103)
        data_76185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'data', False)
        # Processing the call keyword arguments (line 103)
        # Getting the type of 'None' (line 103)
        None_76186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 52), 'None', False)
        keyword_76187 = None_76186
        kwargs_76188 = {'orders': keyword_76187}
        # Getting the type of 'BPoly' (line 103)
        BPoly_76182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'BPoly', False)
        # Obtaining the member 'from_derivatives' of a type (line 103)
        from_derivatives_76183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 13), BPoly_76182, 'from_derivatives')
        # Calling from_derivatives(args, kwargs) (line 103)
        from_derivatives_call_result_76189 = invoke(stypy.reporting.localization.Localization(__file__, 103, 13), from_derivatives_76183, *[x_76184, data_76185], **kwargs_76188)
        
        # Assigning a type to the variable '_b' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), '_b', from_derivatives_call_result_76189)
        
        # Call to __init__(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of '_b' (line 104)
        _b_76196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), '_b', False)
        # Obtaining the member 'c' of a type (line 104)
        c_76197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 48), _b_76196, 'c')
        # Getting the type of '_b' (line 104)
        _b_76198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 54), '_b', False)
        # Obtaining the member 'x' of a type (line 104)
        x_76199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 54), _b_76198, 'x')
        # Processing the call keyword arguments (line 104)
        # Getting the type of 'extrapolate' (line 105)
        extrapolate_76200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 60), 'extrapolate', False)
        keyword_76201 = extrapolate_76200
        kwargs_76202 = {'extrapolate': keyword_76201}
        
        # Call to super(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'PchipInterpolator' (line 104)
        PchipInterpolator_76191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'PchipInterpolator', False)
        # Getting the type of 'self' (line 104)
        self_76192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'self', False)
        # Processing the call keyword arguments (line 104)
        kwargs_76193 = {}
        # Getting the type of 'super' (line 104)
        super_76190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'super', False)
        # Calling super(args, kwargs) (line 104)
        super_call_result_76194 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), super_76190, *[PchipInterpolator_76191, self_76192], **kwargs_76193)
        
        # Obtaining the member '__init__' of a type (line 104)
        init___76195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), super_call_result_76194, '__init__')
        # Calling __init__(args, kwargs) (line 104)
        init___call_result_76203 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), init___76195, *[c_76197, x_76199], **kwargs_76202)
        
        
        # Assigning a Name to a Attribute (line 106):
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'axis' (line 106)
        axis_76204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'axis')
        # Getting the type of 'self' (line 106)
        self_76205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_76205, 'axis', axis_76204)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def roots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'roots'
        module_type_store = module_type_store.open_function_context('roots', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PchipInterpolator.roots.__dict__.__setitem__('stypy_localization', localization)
        PchipInterpolator.roots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PchipInterpolator.roots.__dict__.__setitem__('stypy_type_store', module_type_store)
        PchipInterpolator.roots.__dict__.__setitem__('stypy_function_name', 'PchipInterpolator.roots')
        PchipInterpolator.roots.__dict__.__setitem__('stypy_param_names_list', [])
        PchipInterpolator.roots.__dict__.__setitem__('stypy_varargs_param_name', None)
        PchipInterpolator.roots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PchipInterpolator.roots.__dict__.__setitem__('stypy_call_defaults', defaults)
        PchipInterpolator.roots.__dict__.__setitem__('stypy_call_varargs', varargs)
        PchipInterpolator.roots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PchipInterpolator.roots.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PchipInterpolator.roots', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'roots', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'roots(...)' code ##################

        str_76206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, (-1)), 'str', '\n        Return the roots of the interpolated function.\n        ')
        
        # Call to roots(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_76213 = {}
        
        # Call to from_bernstein_basis(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_76209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 'self', False)
        # Processing the call keyword arguments (line 112)
        kwargs_76210 = {}
        # Getting the type of 'PPoly' (line 112)
        PPoly_76207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'PPoly', False)
        # Obtaining the member 'from_bernstein_basis' of a type (line 112)
        from_bernstein_basis_76208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), PPoly_76207, 'from_bernstein_basis')
        # Calling from_bernstein_basis(args, kwargs) (line 112)
        from_bernstein_basis_call_result_76211 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), from_bernstein_basis_76208, *[self_76209], **kwargs_76210)
        
        # Obtaining the member 'roots' of a type (line 112)
        roots_76212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), from_bernstein_basis_call_result_76211, 'roots')
        # Calling roots(args, kwargs) (line 112)
        roots_call_result_76214 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), roots_76212, *[], **kwargs_76213)
        
        # Assigning a type to the variable 'stypy_return_type' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'stypy_return_type', roots_call_result_76214)
        
        # ################# End of 'roots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'roots' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_76215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_76215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'roots'
        return stypy_return_type_76215


    @staticmethod
    @norecursion
    def _edge_case(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_edge_case'
        module_type_store = module_type_store.open_function_context('_edge_case', 114, 4, False)
        
        # Passed parameters checking function
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_localization', localization)
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_type_of_self', None)
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_type_store', module_type_store)
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_function_name', '_edge_case')
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_param_names_list', ['h0', 'h1', 'm0', 'm1'])
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_varargs_param_name', None)
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_call_defaults', defaults)
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_call_varargs', varargs)
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PchipInterpolator._edge_case.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, None, module_type_store, '_edge_case', ['h0', 'h1', 'm0', 'm1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_edge_case', localization, ['h1', 'm0', 'm1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_edge_case(...)' code ##################

        
        # Assigning a BinOp to a Name (line 117):
        
        # Assigning a BinOp to a Name (line 117):
        int_76216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 14), 'int')
        # Getting the type of 'h0' (line 117)
        h0_76217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'h0')
        # Applying the binary operator '*' (line 117)
        result_mul_76218 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 14), '*', int_76216, h0_76217)
        
        # Getting the type of 'h1' (line 117)
        h1_76219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'h1')
        # Applying the binary operator '+' (line 117)
        result_add_76220 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 14), '+', result_mul_76218, h1_76219)
        
        # Getting the type of 'm0' (line 117)
        m0_76221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'm0')
        # Applying the binary operator '*' (line 117)
        result_mul_76222 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 13), '*', result_add_76220, m0_76221)
        
        # Getting the type of 'h0' (line 117)
        h0_76223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'h0')
        # Getting the type of 'm1' (line 117)
        m1_76224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'm1')
        # Applying the binary operator '*' (line 117)
        result_mul_76225 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 30), '*', h0_76223, m1_76224)
        
        # Applying the binary operator '-' (line 117)
        result_sub_76226 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 13), '-', result_mul_76222, result_mul_76225)
        
        # Getting the type of 'h0' (line 117)
        h0_76227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 40), 'h0')
        # Getting the type of 'h1' (line 117)
        h1_76228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 45), 'h1')
        # Applying the binary operator '+' (line 117)
        result_add_76229 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 40), '+', h0_76227, h1_76228)
        
        # Applying the binary operator 'div' (line 117)
        result_div_76230 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), 'div', result_sub_76226, result_add_76229)
        
        # Assigning a type to the variable 'd' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'd', result_div_76230)
        
        # Assigning a Compare to a Name (line 120):
        
        # Assigning a Compare to a Name (line 120):
        
        
        # Call to sign(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'd' (line 120)
        d_76233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'd', False)
        # Processing the call keyword arguments (line 120)
        kwargs_76234 = {}
        # Getting the type of 'np' (line 120)
        np_76231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'np', False)
        # Obtaining the member 'sign' of a type (line 120)
        sign_76232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), np_76231, 'sign')
        # Calling sign(args, kwargs) (line 120)
        sign_call_result_76235 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), sign_76232, *[d_76233], **kwargs_76234)
        
        
        # Call to sign(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'm0' (line 120)
        m0_76238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'm0', False)
        # Processing the call keyword arguments (line 120)
        kwargs_76239 = {}
        # Getting the type of 'np' (line 120)
        np_76236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'np', False)
        # Obtaining the member 'sign' of a type (line 120)
        sign_76237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 29), np_76236, 'sign')
        # Calling sign(args, kwargs) (line 120)
        sign_call_result_76240 = invoke(stypy.reporting.localization.Localization(__file__, 120, 29), sign_76237, *[m0_76238], **kwargs_76239)
        
        # Applying the binary operator '!=' (line 120)
        result_ne_76241 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 15), '!=', sign_call_result_76235, sign_call_result_76240)
        
        # Assigning a type to the variable 'mask' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'mask', result_ne_76241)
        
        # Assigning a BinOp to a Name (line 121):
        
        # Assigning a BinOp to a Name (line 121):
        
        
        # Call to sign(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'm0' (line 121)
        m0_76244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 25), 'm0', False)
        # Processing the call keyword arguments (line 121)
        kwargs_76245 = {}
        # Getting the type of 'np' (line 121)
        np_76242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'np', False)
        # Obtaining the member 'sign' of a type (line 121)
        sign_76243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), np_76242, 'sign')
        # Calling sign(args, kwargs) (line 121)
        sign_call_result_76246 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), sign_76243, *[m0_76244], **kwargs_76245)
        
        
        # Call to sign(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'm1' (line 121)
        m1_76249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 40), 'm1', False)
        # Processing the call keyword arguments (line 121)
        kwargs_76250 = {}
        # Getting the type of 'np' (line 121)
        np_76247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'np', False)
        # Obtaining the member 'sign' of a type (line 121)
        sign_76248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), np_76247, 'sign')
        # Calling sign(args, kwargs) (line 121)
        sign_call_result_76251 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), sign_76248, *[m1_76249], **kwargs_76250)
        
        # Applying the binary operator '!=' (line 121)
        result_ne_76252 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 17), '!=', sign_call_result_76246, sign_call_result_76251)
        
        
        
        # Call to abs(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'd' (line 121)
        d_76255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 'd', False)
        # Processing the call keyword arguments (line 121)
        kwargs_76256 = {}
        # Getting the type of 'np' (line 121)
        np_76253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'np', False)
        # Obtaining the member 'abs' of a type (line 121)
        abs_76254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 48), np_76253, 'abs')
        # Calling abs(args, kwargs) (line 121)
        abs_call_result_76257 = invoke(stypy.reporting.localization.Localization(__file__, 121, 48), abs_76254, *[d_76255], **kwargs_76256)
        
        float_76258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 60), 'float')
        
        # Call to abs(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'm0' (line 121)
        m0_76261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 70), 'm0', False)
        # Processing the call keyword arguments (line 121)
        kwargs_76262 = {}
        # Getting the type of 'np' (line 121)
        np_76259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 63), 'np', False)
        # Obtaining the member 'abs' of a type (line 121)
        abs_76260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 63), np_76259, 'abs')
        # Calling abs(args, kwargs) (line 121)
        abs_call_result_76263 = invoke(stypy.reporting.localization.Localization(__file__, 121, 63), abs_76260, *[m0_76261], **kwargs_76262)
        
        # Applying the binary operator '*' (line 121)
        result_mul_76264 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 60), '*', float_76258, abs_call_result_76263)
        
        # Applying the binary operator '>' (line 121)
        result_gt_76265 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 48), '>', abs_call_result_76257, result_mul_76264)
        
        # Applying the binary operator '&' (line 121)
        result_and__76266 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 16), '&', result_ne_76252, result_gt_76265)
        
        # Assigning a type to the variable 'mask2' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'mask2', result_and__76266)
        
        # Assigning a BinOp to a Name (line 122):
        
        # Assigning a BinOp to a Name (line 122):
        
        # Getting the type of 'mask' (line 122)
        mask_76267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'mask')
        # Applying the '~' unary operator (line 122)
        result_inv_76268 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 15), '~', mask_76267)
        
        # Getting the type of 'mask2' (line 122)
        mask2_76269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'mask2')
        # Applying the binary operator '&' (line 122)
        result_and__76270 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 14), '&', result_inv_76268, mask2_76269)
        
        # Assigning a type to the variable 'mmm' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'mmm', result_and__76270)
        
        # Assigning a Num to a Subscript (line 124):
        
        # Assigning a Num to a Subscript (line 124):
        float_76271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'float')
        # Getting the type of 'd' (line 124)
        d_76272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'd')
        # Getting the type of 'mask' (line 124)
        mask_76273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'mask')
        # Storing an element on a container (line 124)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), d_76272, (mask_76273, float_76271))
        
        # Assigning a BinOp to a Subscript (line 125):
        
        # Assigning a BinOp to a Subscript (line 125):
        float_76274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 17), 'float')
        
        # Obtaining the type of the subscript
        # Getting the type of 'mmm' (line 125)
        mmm_76275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'mmm')
        # Getting the type of 'm0' (line 125)
        m0_76276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'm0')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___76277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 20), m0_76276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_76278 = invoke(stypy.reporting.localization.Localization(__file__, 125, 20), getitem___76277, mmm_76275)
        
        # Applying the binary operator '*' (line 125)
        result_mul_76279 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 17), '*', float_76274, subscript_call_result_76278)
        
        # Getting the type of 'd' (line 125)
        d_76280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'd')
        # Getting the type of 'mmm' (line 125)
        mmm_76281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 10), 'mmm')
        # Storing an element on a container (line 125)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 8), d_76280, (mmm_76281, result_mul_76279))
        # Getting the type of 'd' (line 127)
        d_76282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', d_76282)
        
        # ################# End of '_edge_case(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_edge_case' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_76283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_76283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_edge_case'
        return stypy_return_type_76283


    @staticmethod
    @norecursion
    def _find_derivatives(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_find_derivatives'
        module_type_store = module_type_store.open_function_context('_find_derivatives', 129, 4, False)
        
        # Passed parameters checking function
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_localization', localization)
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_type_of_self', None)
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_type_store', module_type_store)
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_function_name', '_find_derivatives')
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_param_names_list', ['x', 'y'])
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_varargs_param_name', None)
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_call_defaults', defaults)
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_call_varargs', varargs)
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PchipInterpolator._find_derivatives.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '_find_derivatives', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_find_derivatives', localization, ['y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_find_derivatives(...)' code ##################

        
        # Assigning a Attribute to a Name (line 140):
        
        # Assigning a Attribute to a Name (line 140):
        # Getting the type of 'y' (line 140)
        y_76284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'y')
        # Obtaining the member 'shape' of a type (line 140)
        shape_76285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 18), y_76284, 'shape')
        # Assigning a type to the variable 'y_shape' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'y_shape', shape_76285)
        
        
        # Getting the type of 'y' (line 141)
        y_76286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'y')
        # Obtaining the member 'ndim' of a type (line 141)
        ndim_76287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 11), y_76286, 'ndim')
        int_76288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 21), 'int')
        # Applying the binary operator '==' (line 141)
        result_eq_76289 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), '==', ndim_76287, int_76288)
        
        # Testing the type of an if condition (line 141)
        if_condition_76290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_eq_76289)
        # Assigning a type to the variable 'if_condition_76290' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_76290', if_condition_76290)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 143):
        
        # Assigning a Subscript to a Name (line 143):
        
        # Obtaining the type of the subscript
        slice_76291 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 143, 16), None, None, None)
        # Getting the type of 'None' (line 143)
        None_76292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'None')
        # Getting the type of 'x' (line 143)
        x_76293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'x')
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___76294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), x_76293, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_76295 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), getitem___76294, (slice_76291, None_76292))
        
        # Assigning a type to the variable 'x' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'x', subscript_call_result_76295)
        
        # Assigning a Subscript to a Name (line 144):
        
        # Assigning a Subscript to a Name (line 144):
        
        # Obtaining the type of the subscript
        slice_76296 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 144, 16), None, None, None)
        # Getting the type of 'None' (line 144)
        None_76297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'None')
        # Getting the type of 'y' (line 144)
        y_76298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'y')
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___76299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), y_76298, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_76300 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), getitem___76299, (slice_76296, None_76297))
        
        # Assigning a type to the variable 'y' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'y', subscript_call_result_76300)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 146):
        
        # Assigning a BinOp to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_76301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 15), 'int')
        slice_76302 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 13), int_76301, None, None)
        # Getting the type of 'x' (line 146)
        x_76303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 13), 'x')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___76304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 13), x_76303, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_76305 = invoke(stypy.reporting.localization.Localization(__file__, 146, 13), getitem___76304, slice_76302)
        
        
        # Obtaining the type of the subscript
        int_76306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'int')
        slice_76307 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 21), None, int_76306, None)
        # Getting the type of 'x' (line 146)
        x_76308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 21), 'x')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___76309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 21), x_76308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_76310 = invoke(stypy.reporting.localization.Localization(__file__, 146, 21), getitem___76309, slice_76307)
        
        # Applying the binary operator '-' (line 146)
        result_sub_76311 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 13), '-', subscript_call_result_76305, subscript_call_result_76310)
        
        # Assigning a type to the variable 'hk' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'hk', result_sub_76311)
        
        # Assigning a BinOp to a Name (line 147):
        
        # Assigning a BinOp to a Name (line 147):
        
        # Obtaining the type of the subscript
        int_76312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 16), 'int')
        slice_76313 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 147, 14), int_76312, None, None)
        # Getting the type of 'y' (line 147)
        y_76314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'y')
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___76315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 14), y_76314, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_76316 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), getitem___76315, slice_76313)
        
        
        # Obtaining the type of the subscript
        int_76317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'int')
        slice_76318 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 147, 22), None, int_76317, None)
        # Getting the type of 'y' (line 147)
        y_76319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'y')
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___76320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 22), y_76319, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_76321 = invoke(stypy.reporting.localization.Localization(__file__, 147, 22), getitem___76320, slice_76318)
        
        # Applying the binary operator '-' (line 147)
        result_sub_76322 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 14), '-', subscript_call_result_76316, subscript_call_result_76321)
        
        # Getting the type of 'hk' (line 147)
        hk_76323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 32), 'hk')
        # Applying the binary operator 'div' (line 147)
        result_div_76324 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 13), 'div', result_sub_76322, hk_76323)
        
        # Assigning a type to the variable 'mk' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'mk', result_div_76324)
        
        
        
        # Obtaining the type of the subscript
        int_76325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 19), 'int')
        # Getting the type of 'y' (line 149)
        y_76326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'y')
        # Obtaining the member 'shape' of a type (line 149)
        shape_76327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 11), y_76326, 'shape')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___76328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 11), shape_76327, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_76329 = invoke(stypy.reporting.localization.Localization(__file__, 149, 11), getitem___76328, int_76325)
        
        int_76330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 25), 'int')
        # Applying the binary operator '==' (line 149)
        result_eq_76331 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '==', subscript_call_result_76329, int_76330)
        
        # Testing the type of an if condition (line 149)
        if_condition_76332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_eq_76331)
        # Assigning a type to the variable 'if_condition_76332' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_76332', if_condition_76332)
        # SSA begins for if statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to zeros_like(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'y' (line 151)
        y_76335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'y', False)
        # Processing the call keyword arguments (line 151)
        kwargs_76336 = {}
        # Getting the type of 'np' (line 151)
        np_76333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 151)
        zeros_like_76334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 17), np_76333, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 151)
        zeros_like_call_result_76337 = invoke(stypy.reporting.localization.Localization(__file__, 151, 17), zeros_like_76334, *[y_76335], **kwargs_76336)
        
        # Assigning a type to the variable 'dk' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'dk', zeros_like_call_result_76337)
        
        # Assigning a Name to a Subscript (line 152):
        
        # Assigning a Name to a Subscript (line 152):
        # Getting the type of 'mk' (line 152)
        mk_76338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'mk')
        # Getting the type of 'dk' (line 152)
        dk_76339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'dk')
        int_76340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'int')
        # Storing an element on a container (line 152)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 12), dk_76339, (int_76340, mk_76338))
        
        # Assigning a Name to a Subscript (line 153):
        
        # Assigning a Name to a Subscript (line 153):
        # Getting the type of 'mk' (line 153)
        mk_76341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'mk')
        # Getting the type of 'dk' (line 153)
        dk_76342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'dk')
        int_76343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 15), 'int')
        # Storing an element on a container (line 153)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 12), dk_76342, (int_76343, mk_76341))
        
        # Call to reshape(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'y_shape' (line 154)
        y_shape_76346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'y_shape', False)
        # Processing the call keyword arguments (line 154)
        kwargs_76347 = {}
        # Getting the type of 'dk' (line 154)
        dk_76344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'dk', False)
        # Obtaining the member 'reshape' of a type (line 154)
        reshape_76345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 19), dk_76344, 'reshape')
        # Calling reshape(args, kwargs) (line 154)
        reshape_call_result_76348 = invoke(stypy.reporting.localization.Localization(__file__, 154, 19), reshape_76345, *[y_shape_76346], **kwargs_76347)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'stypy_return_type', reshape_call_result_76348)
        # SSA join for if statement (line 149)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to sign(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'mk' (line 156)
        mk_76351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'mk', False)
        # Processing the call keyword arguments (line 156)
        kwargs_76352 = {}
        # Getting the type of 'np' (line 156)
        np_76349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'np', False)
        # Obtaining the member 'sign' of a type (line 156)
        sign_76350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 14), np_76349, 'sign')
        # Calling sign(args, kwargs) (line 156)
        sign_call_result_76353 = invoke(stypy.reporting.localization.Localization(__file__, 156, 14), sign_76350, *[mk_76351], **kwargs_76352)
        
        # Assigning a type to the variable 'smk' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'smk', sign_call_result_76353)
        
        # Assigning a BinOp to a Name (line 157):
        
        # Assigning a BinOp to a Name (line 157):
        
        
        # Obtaining the type of the subscript
        int_76354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'int')
        slice_76355 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 157, 21), int_76354, None, None)
        # Getting the type of 'smk' (line 157)
        smk_76356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'smk')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___76357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 21), smk_76356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_76358 = invoke(stypy.reporting.localization.Localization(__file__, 157, 21), getitem___76357, slice_76355)
        
        
        # Obtaining the type of the subscript
        int_76359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 37), 'int')
        slice_76360 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 157, 32), None, int_76359, None)
        # Getting the type of 'smk' (line 157)
        smk_76361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'smk')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___76362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 32), smk_76361, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_76363 = invoke(stypy.reporting.localization.Localization(__file__, 157, 32), getitem___76362, slice_76360)
        
        # Applying the binary operator '!=' (line 157)
        result_ne_76364 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 21), '!=', subscript_call_result_76358, subscript_call_result_76363)
        
        
        
        # Obtaining the type of the subscript
        int_76365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 48), 'int')
        slice_76366 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 157, 45), int_76365, None, None)
        # Getting the type of 'mk' (line 157)
        mk_76367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 45), 'mk')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___76368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 45), mk_76367, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_76369 = invoke(stypy.reporting.localization.Localization(__file__, 157, 45), getitem___76368, slice_76366)
        
        int_76370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 55), 'int')
        # Applying the binary operator '==' (line 157)
        result_eq_76371 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 45), '==', subscript_call_result_76369, int_76370)
        
        # Applying the binary operator '|' (line 157)
        result_or__76372 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 20), '|', result_ne_76364, result_eq_76371)
        
        
        
        # Obtaining the type of the subscript
        int_76373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 65), 'int')
        slice_76374 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 157, 61), None, int_76373, None)
        # Getting the type of 'mk' (line 157)
        mk_76375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 61), 'mk')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___76376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 61), mk_76375, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_76377 = invoke(stypy.reporting.localization.Localization(__file__, 157, 61), getitem___76376, slice_76374)
        
        int_76378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 72), 'int')
        # Applying the binary operator '==' (line 157)
        result_eq_76379 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 61), '==', subscript_call_result_76377, int_76378)
        
        # Applying the binary operator '|' (line 157)
        result_or__76380 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 58), '|', result_or__76372, result_eq_76379)
        
        # Assigning a type to the variable 'condition' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'condition', result_or__76380)
        
        # Assigning a BinOp to a Name (line 159):
        
        # Assigning a BinOp to a Name (line 159):
        int_76381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 13), 'int')
        
        # Obtaining the type of the subscript
        int_76382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'int')
        slice_76383 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 15), int_76382, None, None)
        # Getting the type of 'hk' (line 159)
        hk_76384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'hk')
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___76385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), hk_76384, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_76386 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), getitem___76385, slice_76383)
        
        # Applying the binary operator '*' (line 159)
        result_mul_76387 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 13), '*', int_76381, subscript_call_result_76386)
        
        
        # Obtaining the type of the subscript
        int_76388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'int')
        slice_76389 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 24), None, int_76388, None)
        # Getting the type of 'hk' (line 159)
        hk_76390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'hk')
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___76391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 24), hk_76390, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_76392 = invoke(stypy.reporting.localization.Localization(__file__, 159, 24), getitem___76391, slice_76389)
        
        # Applying the binary operator '+' (line 159)
        result_add_76393 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 13), '+', result_mul_76387, subscript_call_result_76392)
        
        # Assigning a type to the variable 'w1' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'w1', result_add_76393)
        
        # Assigning a BinOp to a Name (line 160):
        
        # Assigning a BinOp to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_76394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 16), 'int')
        slice_76395 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 13), int_76394, None, None)
        # Getting the type of 'hk' (line 160)
        hk_76396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'hk')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___76397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 13), hk_76396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_76398 = invoke(stypy.reporting.localization.Localization(__file__, 160, 13), getitem___76397, slice_76395)
        
        int_76399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 22), 'int')
        
        # Obtaining the type of the subscript
        int_76400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'int')
        slice_76401 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 24), None, int_76400, None)
        # Getting the type of 'hk' (line 160)
        hk_76402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'hk')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___76403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 24), hk_76402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_76404 = invoke(stypy.reporting.localization.Localization(__file__, 160, 24), getitem___76403, slice_76401)
        
        # Applying the binary operator '*' (line 160)
        result_mul_76405 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 22), '*', int_76399, subscript_call_result_76404)
        
        # Applying the binary operator '+' (line 160)
        result_add_76406 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 13), '+', subscript_call_result_76398, result_mul_76405)
        
        # Assigning a type to the variable 'w2' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'w2', result_add_76406)
        
        # Call to errstate(...): (line 164)
        # Processing the call keyword arguments (line 164)
        str_76409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 32), 'str', 'ignore')
        keyword_76410 = str_76409
        kwargs_76411 = {'divide': keyword_76410}
        # Getting the type of 'np' (line 164)
        np_76407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'np', False)
        # Obtaining the member 'errstate' of a type (line 164)
        errstate_76408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), np_76407, 'errstate')
        # Calling errstate(args, kwargs) (line 164)
        errstate_call_result_76412 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), errstate_76408, *[], **kwargs_76411)
        
        with_76413 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 164, 13), errstate_call_result_76412, 'with parameter', '__enter__', '__exit__')

        if with_76413:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 164)
            enter___76414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), errstate_call_result_76412, '__enter__')
            with_enter_76415 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), enter___76414)
            
            # Assigning a BinOp to a Name (line 165):
            
            # Assigning a BinOp to a Name (line 165):
            # Getting the type of 'w1' (line 165)
            w1_76416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'w1')
            
            # Obtaining the type of the subscript
            int_76417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 29), 'int')
            slice_76418 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 165, 25), None, int_76417, None)
            # Getting the type of 'mk' (line 165)
            mk_76419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'mk')
            # Obtaining the member '__getitem__' of a type (line 165)
            getitem___76420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), mk_76419, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 165)
            subscript_call_result_76421 = invoke(stypy.reporting.localization.Localization(__file__, 165, 25), getitem___76420, slice_76418)
            
            # Applying the binary operator 'div' (line 165)
            result_div_76422 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 22), 'div', w1_76416, subscript_call_result_76421)
            
            # Getting the type of 'w2' (line 165)
            w2_76423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 35), 'w2')
            
            # Obtaining the type of the subscript
            int_76424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 41), 'int')
            slice_76425 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 165, 38), int_76424, None, None)
            # Getting the type of 'mk' (line 165)
            mk_76426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 38), 'mk')
            # Obtaining the member '__getitem__' of a type (line 165)
            getitem___76427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 38), mk_76426, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 165)
            subscript_call_result_76428 = invoke(stypy.reporting.localization.Localization(__file__, 165, 38), getitem___76427, slice_76425)
            
            # Applying the binary operator 'div' (line 165)
            result_div_76429 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 35), 'div', w2_76423, subscript_call_result_76428)
            
            # Applying the binary operator '+' (line 165)
            result_add_76430 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 22), '+', result_div_76422, result_div_76429)
            
            # Getting the type of 'w1' (line 165)
            w1_76431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 49), 'w1')
            # Getting the type of 'w2' (line 165)
            w2_76432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 54), 'w2')
            # Applying the binary operator '+' (line 165)
            result_add_76433 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 49), '+', w1_76431, w2_76432)
            
            # Applying the binary operator 'div' (line 165)
            result_div_76434 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 21), 'div', result_add_76430, result_add_76433)
            
            # Assigning a type to the variable 'whmean' (line 165)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'whmean', result_div_76434)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 164)
            exit___76435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), errstate_call_result_76412, '__exit__')
            with_exit_76436 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), exit___76435, None, None, None)

        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to zeros_like(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'y' (line 167)
        y_76439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'y', False)
        # Processing the call keyword arguments (line 167)
        kwargs_76440 = {}
        # Getting the type of 'np' (line 167)
        np_76437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 13), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 167)
        zeros_like_76438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 13), np_76437, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 167)
        zeros_like_call_result_76441 = invoke(stypy.reporting.localization.Localization(__file__, 167, 13), zeros_like_76438, *[y_76439], **kwargs_76440)
        
        # Assigning a type to the variable 'dk' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'dk', zeros_like_call_result_76441)
        
        # Assigning a Num to a Subscript (line 168):
        
        # Assigning a Num to a Subscript (line 168):
        float_76442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 30), 'float')
        
        # Obtaining the type of the subscript
        int_76443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 11), 'int')
        int_76444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 13), 'int')
        slice_76445 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 168, 8), int_76443, int_76444, None)
        # Getting the type of 'dk' (line 168)
        dk_76446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'dk')
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___76447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), dk_76446, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_76448 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___76447, slice_76445)
        
        # Getting the type of 'condition' (line 168)
        condition_76449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'condition')
        # Storing an element on a container (line 168)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), subscript_call_result_76448, (condition_76449, float_76442))
        
        # Assigning a BinOp to a Subscript (line 169):
        
        # Assigning a BinOp to a Subscript (line 169):
        float_76450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 31), 'float')
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'condition' (line 169)
        condition_76451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 45), 'condition')
        # Applying the '~' unary operator (line 169)
        result_inv_76452 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 44), '~', condition_76451)
        
        # Getting the type of 'whmean' (line 169)
        whmean_76453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'whmean')
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___76454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 37), whmean_76453, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_76455 = invoke(stypy.reporting.localization.Localization(__file__, 169, 37), getitem___76454, result_inv_76452)
        
        # Applying the binary operator 'div' (line 169)
        result_div_76456 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 31), 'div', float_76450, subscript_call_result_76455)
        
        
        # Obtaining the type of the subscript
        int_76457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 11), 'int')
        int_76458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 13), 'int')
        slice_76459 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 8), int_76457, int_76458, None)
        # Getting the type of 'dk' (line 169)
        dk_76460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'dk')
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___76461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), dk_76460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_76462 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___76461, slice_76459)
        
        
        # Getting the type of 'condition' (line 169)
        condition_76463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 18), 'condition')
        # Applying the '~' unary operator (line 169)
        result_inv_76464 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 17), '~', condition_76463)
        
        # Storing an element on a container (line 169)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), subscript_call_result_76462, (result_inv_76464, result_div_76456))
        
        # Assigning a Call to a Subscript (line 173):
        
        # Assigning a Call to a Subscript (line 173):
        
        # Call to _edge_case(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Obtaining the type of the subscript
        int_76467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 48), 'int')
        # Getting the type of 'hk' (line 173)
        hk_76468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 45), 'hk', False)
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___76469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 45), hk_76468, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_76470 = invoke(stypy.reporting.localization.Localization(__file__, 173, 45), getitem___76469, int_76467)
        
        
        # Obtaining the type of the subscript
        int_76471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 55), 'int')
        # Getting the type of 'hk' (line 173)
        hk_76472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 52), 'hk', False)
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___76473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 52), hk_76472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_76474 = invoke(stypy.reporting.localization.Localization(__file__, 173, 52), getitem___76473, int_76471)
        
        
        # Obtaining the type of the subscript
        int_76475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 62), 'int')
        # Getting the type of 'mk' (line 173)
        mk_76476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 59), 'mk', False)
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___76477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 59), mk_76476, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_76478 = invoke(stypy.reporting.localization.Localization(__file__, 173, 59), getitem___76477, int_76475)
        
        
        # Obtaining the type of the subscript
        int_76479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 69), 'int')
        # Getting the type of 'mk' (line 173)
        mk_76480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 66), 'mk', False)
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___76481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 66), mk_76480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_76482 = invoke(stypy.reporting.localization.Localization(__file__, 173, 66), getitem___76481, int_76479)
        
        # Processing the call keyword arguments (line 173)
        kwargs_76483 = {}
        # Getting the type of 'PchipInterpolator' (line 173)
        PchipInterpolator_76465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'PchipInterpolator', False)
        # Obtaining the member '_edge_case' of a type (line 173)
        _edge_case_76466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), PchipInterpolator_76465, '_edge_case')
        # Calling _edge_case(args, kwargs) (line 173)
        _edge_case_call_result_76484 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), _edge_case_76466, *[subscript_call_result_76470, subscript_call_result_76474, subscript_call_result_76478, subscript_call_result_76482], **kwargs_76483)
        
        # Getting the type of 'dk' (line 173)
        dk_76485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'dk')
        int_76486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 11), 'int')
        # Storing an element on a container (line 173)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 8), dk_76485, (int_76486, _edge_case_call_result_76484))
        
        # Assigning a Call to a Subscript (line 174):
        
        # Assigning a Call to a Subscript (line 174):
        
        # Call to _edge_case(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Obtaining the type of the subscript
        int_76489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 49), 'int')
        # Getting the type of 'hk' (line 174)
        hk_76490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 46), 'hk', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___76491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 46), hk_76490, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_76492 = invoke(stypy.reporting.localization.Localization(__file__, 174, 46), getitem___76491, int_76489)
        
        
        # Obtaining the type of the subscript
        int_76493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 57), 'int')
        # Getting the type of 'hk' (line 174)
        hk_76494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 54), 'hk', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___76495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 54), hk_76494, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_76496 = invoke(stypy.reporting.localization.Localization(__file__, 174, 54), getitem___76495, int_76493)
        
        
        # Obtaining the type of the subscript
        int_76497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 65), 'int')
        # Getting the type of 'mk' (line 174)
        mk_76498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 62), 'mk', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___76499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 62), mk_76498, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_76500 = invoke(stypy.reporting.localization.Localization(__file__, 174, 62), getitem___76499, int_76497)
        
        
        # Obtaining the type of the subscript
        int_76501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 73), 'int')
        # Getting the type of 'mk' (line 174)
        mk_76502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 70), 'mk', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___76503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 70), mk_76502, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_76504 = invoke(stypy.reporting.localization.Localization(__file__, 174, 70), getitem___76503, int_76501)
        
        # Processing the call keyword arguments (line 174)
        kwargs_76505 = {}
        # Getting the type of 'PchipInterpolator' (line 174)
        PchipInterpolator_76487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'PchipInterpolator', False)
        # Obtaining the member '_edge_case' of a type (line 174)
        _edge_case_76488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 17), PchipInterpolator_76487, '_edge_case')
        # Calling _edge_case(args, kwargs) (line 174)
        _edge_case_call_result_76506 = invoke(stypy.reporting.localization.Localization(__file__, 174, 17), _edge_case_76488, *[subscript_call_result_76492, subscript_call_result_76496, subscript_call_result_76500, subscript_call_result_76504], **kwargs_76505)
        
        # Getting the type of 'dk' (line 174)
        dk_76507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'dk')
        int_76508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 11), 'int')
        # Storing an element on a container (line 174)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 8), dk_76507, (int_76508, _edge_case_call_result_76506))
        
        # Call to reshape(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'y_shape' (line 176)
        y_shape_76511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'y_shape', False)
        # Processing the call keyword arguments (line 176)
        kwargs_76512 = {}
        # Getting the type of 'dk' (line 176)
        dk_76509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'dk', False)
        # Obtaining the member 'reshape' of a type (line 176)
        reshape_76510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), dk_76509, 'reshape')
        # Calling reshape(args, kwargs) (line 176)
        reshape_call_result_76513 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), reshape_76510, *[y_shape_76511], **kwargs_76512)
        
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', reshape_call_result_76513)
        
        # ################# End of '_find_derivatives(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_find_derivatives' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_76514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_76514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_find_derivatives'
        return stypy_return_type_76514


# Assigning a type to the variable 'PchipInterpolator' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'PchipInterpolator', PchipInterpolator)

@norecursion
def pchip_interpolate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_76515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 37), 'int')
    int_76516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 45), 'int')
    defaults = [int_76515, int_76516]
    # Create a new context for function 'pchip_interpolate'
    module_type_store = module_type_store.open_function_context('pchip_interpolate', 179, 0, False)
    
    # Passed parameters checking function
    pchip_interpolate.stypy_localization = localization
    pchip_interpolate.stypy_type_of_self = None
    pchip_interpolate.stypy_type_store = module_type_store
    pchip_interpolate.stypy_function_name = 'pchip_interpolate'
    pchip_interpolate.stypy_param_names_list = ['xi', 'yi', 'x', 'der', 'axis']
    pchip_interpolate.stypy_varargs_param_name = None
    pchip_interpolate.stypy_kwargs_param_name = None
    pchip_interpolate.stypy_call_defaults = defaults
    pchip_interpolate.stypy_call_varargs = varargs
    pchip_interpolate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pchip_interpolate', ['xi', 'yi', 'x', 'der', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pchip_interpolate', localization, ['xi', 'yi', 'x', 'der', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pchip_interpolate(...)' code ##################

    str_76517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, (-1)), 'str', "\n    Convenience function for pchip interpolation.\n    xi and yi are arrays of values used to approximate some function f,\n    with ``yi = f(xi)``.  The interpolant uses monotonic cubic splines\n    to find the value of new points x and the derivatives there.\n\n    See `PchipInterpolator` for details.\n\n    Parameters\n    ----------\n    xi : array_like\n        A sorted list of x-coordinates, of length N.\n    yi :  array_like\n        A 1-D array of real values.  `yi`'s length along the interpolation\n        axis must be equal to the length of `xi`. If N-D array, use axis\n        parameter to select correct axis.\n    x : scalar or array_like\n        Of length M.\n    der : int or list, optional\n        Derivatives to extract.  The 0-th derivative can be included to\n        return the function value.\n    axis : int, optional\n        Axis in the yi array corresponding to the x-coordinate values.\n\n    See Also\n    --------\n    PchipInterpolator\n\n    Returns\n    -------\n    y : scalar or array_like\n        The result, of length R or length M or M by R,\n\n    ")
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to PchipInterpolator(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'xi' (line 214)
    xi_76519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 26), 'xi', False)
    # Getting the type of 'yi' (line 214)
    yi_76520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'yi', False)
    # Processing the call keyword arguments (line 214)
    # Getting the type of 'axis' (line 214)
    axis_76521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 39), 'axis', False)
    keyword_76522 = axis_76521
    kwargs_76523 = {'axis': keyword_76522}
    # Getting the type of 'PchipInterpolator' (line 214)
    PchipInterpolator_76518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'PchipInterpolator', False)
    # Calling PchipInterpolator(args, kwargs) (line 214)
    PchipInterpolator_call_result_76524 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), PchipInterpolator_76518, *[xi_76519, yi_76520], **kwargs_76523)
    
    # Assigning a type to the variable 'P' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'P', PchipInterpolator_call_result_76524)
    
    
    # Getting the type of 'der' (line 216)
    der_76525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 7), 'der')
    int_76526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 14), 'int')
    # Applying the binary operator '==' (line 216)
    result_eq_76527 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 7), '==', der_76525, int_76526)
    
    # Testing the type of an if condition (line 216)
    if_condition_76528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 4), result_eq_76527)
    # Assigning a type to the variable 'if_condition_76528' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'if_condition_76528', if_condition_76528)
    # SSA begins for if statement (line 216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to P(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'x' (line 217)
    x_76530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), 'x', False)
    # Processing the call keyword arguments (line 217)
    kwargs_76531 = {}
    # Getting the type of 'P' (line 217)
    P_76529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'P', False)
    # Calling P(args, kwargs) (line 217)
    P_call_result_76532 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), P_76529, *[x_76530], **kwargs_76531)
    
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', P_call_result_76532)
    # SSA branch for the else part of an if statement (line 216)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to _isscalar(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'der' (line 218)
    der_76534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'der', False)
    # Processing the call keyword arguments (line 218)
    kwargs_76535 = {}
    # Getting the type of '_isscalar' (line 218)
    _isscalar_76533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 9), '_isscalar', False)
    # Calling _isscalar(args, kwargs) (line 218)
    _isscalar_call_result_76536 = invoke(stypy.reporting.localization.Localization(__file__, 218, 9), _isscalar_76533, *[der_76534], **kwargs_76535)
    
    # Testing the type of an if condition (line 218)
    if_condition_76537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 9), _isscalar_call_result_76536)
    # Assigning a type to the variable 'if_condition_76537' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 9), 'if_condition_76537', if_condition_76537)
    # SSA begins for if statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to (...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'x' (line 219)
    x_76543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 33), 'x', False)
    # Processing the call keyword arguments (line 219)
    kwargs_76544 = {}
    
    # Call to derivative(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'der' (line 219)
    der_76540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'der', False)
    # Processing the call keyword arguments (line 219)
    kwargs_76541 = {}
    # Getting the type of 'P' (line 219)
    P_76538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'P', False)
    # Obtaining the member 'derivative' of a type (line 219)
    derivative_76539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), P_76538, 'derivative')
    # Calling derivative(args, kwargs) (line 219)
    derivative_call_result_76542 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), derivative_76539, *[der_76540], **kwargs_76541)
    
    # Calling (args, kwargs) (line 219)
    _call_result_76545 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), derivative_call_result_76542, *[x_76543], **kwargs_76544)
    
    # Assigning a type to the variable 'stypy_return_type' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'stypy_return_type', _call_result_76545)
    # SSA branch for the else part of an if statement (line 218)
    module_type_store.open_ssa_branch('else')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'der' (line 221)
    der_76554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 46), 'der')
    comprehension_76555 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 16), der_76554)
    # Assigning a type to the variable 'nu' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'nu', comprehension_76555)
    
    # Call to (...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'x' (line 221)
    x_76551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 33), 'x', False)
    # Processing the call keyword arguments (line 221)
    kwargs_76552 = {}
    
    # Call to derivative(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'nu' (line 221)
    nu_76548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'nu', False)
    # Processing the call keyword arguments (line 221)
    kwargs_76549 = {}
    # Getting the type of 'P' (line 221)
    P_76546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'P', False)
    # Obtaining the member 'derivative' of a type (line 221)
    derivative_76547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), P_76546, 'derivative')
    # Calling derivative(args, kwargs) (line 221)
    derivative_call_result_76550 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), derivative_76547, *[nu_76548], **kwargs_76549)
    
    # Calling (args, kwargs) (line 221)
    _call_result_76553 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), derivative_call_result_76550, *[x_76551], **kwargs_76552)
    
    list_76556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 16), list_76556, _call_result_76553)
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', list_76556)
    # SSA join for if statement (line 218)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 216)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'pchip_interpolate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pchip_interpolate' in the type store
    # Getting the type of 'stypy_return_type' (line 179)
    stypy_return_type_76557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_76557)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pchip_interpolate'
    return stypy_return_type_76557

# Assigning a type to the variable 'pchip_interpolate' (line 179)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'pchip_interpolate', pchip_interpolate)

# Assigning a Name to a Name (line 225):

# Assigning a Name to a Name (line 225):
# Getting the type of 'PchipInterpolator' (line 225)
PchipInterpolator_76558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'PchipInterpolator')
# Assigning a type to the variable 'pchip' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'pchip', PchipInterpolator_76558)
# Declaration of the 'Akima1DInterpolator' class
# Getting the type of 'PPoly' (line 228)
PPoly_76559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'PPoly')

class Akima1DInterpolator(PPoly_76559, ):
    str_76560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, (-1)), 'str', '\n    Akima interpolator\n\n    Fit piecewise cubic polynomials, given vectors x and y. The interpolation\n    method by Akima uses a continuously differentiable sub-spline built from\n    piecewise cubic polynomials. The resultant curve passes through the given\n    data points and will appear smooth and natural.\n\n    Parameters\n    ----------\n    x : ndarray, shape (m, )\n        1-D array of monotonically increasing real values.\n    y : ndarray, shape (m, ...)\n        N-D array of real values. The length of `y` along the first axis must\n        be equal to the length of `x`.\n    axis : int, optional\n        Specifies the axis of `y` along which to interpolate. Interpolation\n        defaults to the first axis of `y`.\n\n    Methods\n    -------\n    __call__\n    derivative\n    antiderivative\n    roots\n\n    See Also\n    --------\n    PchipInterpolator\n    CubicSpline\n    PPoly\n\n    Notes\n    -----\n    .. versionadded:: 0.14\n\n    Use only for precise data, as the fitted curve passes through the given\n    points exactly. This routine is useful for plotting a pleasingly smooth\n    curve through a few given points for purposes of plotting.\n\n    References\n    ----------\n    [1] A new method of interpolation and smooth curve fitting based\n        on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),\n        589-602.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_76561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 34), 'int')
        defaults = [int_76561]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 277, 4, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Akima1DInterpolator.__init__', ['x', 'y', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Tuple (line 280):
        
        # Assigning a Subscript to a Name (line 280):
        
        # Obtaining the type of the subscript
        int_76562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 8), 'int')
        
        # Call to map(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'np' (line 280)
        np_76564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 280)
        asarray_76565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), np_76564, 'asarray')
        
        # Obtaining an instance of the builtin type 'tuple' (line 280)
        tuple_76566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 280)
        # Adding element type (line 280)
        # Getting the type of 'x' (line 280)
        x_76567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 32), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 32), tuple_76566, x_76567)
        # Adding element type (line 280)
        # Getting the type of 'y' (line 280)
        y_76568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 32), tuple_76566, y_76568)
        
        # Processing the call keyword arguments (line 280)
        kwargs_76569 = {}
        # Getting the type of 'map' (line 280)
        map_76563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'map', False)
        # Calling map(args, kwargs) (line 280)
        map_call_result_76570 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), map_76563, *[asarray_76565, tuple_76566], **kwargs_76569)
        
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___76571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), map_call_result_76570, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_76572 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), getitem___76571, int_76562)
        
        # Assigning a type to the variable 'tuple_var_assignment_76080' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'tuple_var_assignment_76080', subscript_call_result_76572)
        
        # Assigning a Subscript to a Name (line 280):
        
        # Obtaining the type of the subscript
        int_76573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 8), 'int')
        
        # Call to map(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'np' (line 280)
        np_76575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 280)
        asarray_76576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), np_76575, 'asarray')
        
        # Obtaining an instance of the builtin type 'tuple' (line 280)
        tuple_76577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 280)
        # Adding element type (line 280)
        # Getting the type of 'x' (line 280)
        x_76578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 32), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 32), tuple_76577, x_76578)
        # Adding element type (line 280)
        # Getting the type of 'y' (line 280)
        y_76579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 32), tuple_76577, y_76579)
        
        # Processing the call keyword arguments (line 280)
        kwargs_76580 = {}
        # Getting the type of 'map' (line 280)
        map_76574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'map', False)
        # Calling map(args, kwargs) (line 280)
        map_call_result_76581 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), map_76574, *[asarray_76576, tuple_76577], **kwargs_76580)
        
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___76582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), map_call_result_76581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_76583 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), getitem___76582, int_76573)
        
        # Assigning a type to the variable 'tuple_var_assignment_76081' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'tuple_var_assignment_76081', subscript_call_result_76583)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'tuple_var_assignment_76080' (line 280)
        tuple_var_assignment_76080_76584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'tuple_var_assignment_76080')
        # Assigning a type to the variable 'x' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'x', tuple_var_assignment_76080_76584)
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'tuple_var_assignment_76081' (line 280)
        tuple_var_assignment_76081_76585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'tuple_var_assignment_76081')
        # Assigning a type to the variable 'y' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'y', tuple_var_assignment_76081_76585)
        
        # Assigning a BinOp to a Name (line 281):
        
        # Assigning a BinOp to a Name (line 281):
        # Getting the type of 'axis' (line 281)
        axis_76586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'axis')
        # Getting the type of 'y' (line 281)
        y_76587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'y')
        # Obtaining the member 'ndim' of a type (line 281)
        ndim_76588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 22), y_76587, 'ndim')
        # Applying the binary operator '%' (line 281)
        result_mod_76589 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 15), '%', axis_76586, ndim_76588)
        
        # Assigning a type to the variable 'axis' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'axis', result_mod_76589)
        
        
        # Call to any(...): (line 283)
        # Processing the call arguments (line 283)
        
        
        # Call to diff(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'x' (line 283)
        x_76594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'x', False)
        # Processing the call keyword arguments (line 283)
        kwargs_76595 = {}
        # Getting the type of 'np' (line 283)
        np_76592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 18), 'np', False)
        # Obtaining the member 'diff' of a type (line 283)
        diff_76593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 18), np_76592, 'diff')
        # Calling diff(args, kwargs) (line 283)
        diff_call_result_76596 = invoke(stypy.reporting.localization.Localization(__file__, 283, 18), diff_76593, *[x_76594], **kwargs_76595)
        
        float_76597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 31), 'float')
        # Applying the binary operator '<' (line 283)
        result_lt_76598 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 18), '<', diff_call_result_76596, float_76597)
        
        # Processing the call keyword arguments (line 283)
        kwargs_76599 = {}
        # Getting the type of 'np' (line 283)
        np_76590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 283)
        any_76591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 11), np_76590, 'any')
        # Calling any(args, kwargs) (line 283)
        any_call_result_76600 = invoke(stypy.reporting.localization.Localization(__file__, 283, 11), any_76591, *[result_lt_76598], **kwargs_76599)
        
        # Testing the type of an if condition (line 283)
        if_condition_76601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 8), any_call_result_76600)
        # Assigning a type to the variable 'if_condition_76601' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'if_condition_76601', if_condition_76601)
        # SSA begins for if statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 284)
        # Processing the call arguments (line 284)
        str_76603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 29), 'str', 'x must be strictly ascending')
        # Processing the call keyword arguments (line 284)
        kwargs_76604 = {}
        # Getting the type of 'ValueError' (line 284)
        ValueError_76602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 284)
        ValueError_call_result_76605 = invoke(stypy.reporting.localization.Localization(__file__, 284, 18), ValueError_76602, *[str_76603], **kwargs_76604)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 284, 12), ValueError_call_result_76605, 'raise parameter', BaseException)
        # SSA join for if statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'x' (line 285)
        x_76606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'x')
        # Obtaining the member 'ndim' of a type (line 285)
        ndim_76607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 11), x_76606, 'ndim')
        int_76608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 21), 'int')
        # Applying the binary operator '!=' (line 285)
        result_ne_76609 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), '!=', ndim_76607, int_76608)
        
        # Testing the type of an if condition (line 285)
        if_condition_76610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_ne_76609)
        # Assigning a type to the variable 'if_condition_76610' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_76610', if_condition_76610)
        # SSA begins for if statement (line 285)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 286)
        # Processing the call arguments (line 286)
        str_76612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 29), 'str', 'x must be 1-dimensional')
        # Processing the call keyword arguments (line 286)
        kwargs_76613 = {}
        # Getting the type of 'ValueError' (line 286)
        ValueError_76611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 286)
        ValueError_call_result_76614 = invoke(stypy.reporting.localization.Localization(__file__, 286, 18), ValueError_76611, *[str_76612], **kwargs_76613)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 286, 12), ValueError_call_result_76614, 'raise parameter', BaseException)
        # SSA join for if statement (line 285)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'x' (line 287)
        x_76615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'x')
        # Obtaining the member 'size' of a type (line 287)
        size_76616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 11), x_76615, 'size')
        int_76617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 20), 'int')
        # Applying the binary operator '<' (line 287)
        result_lt_76618 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 11), '<', size_76616, int_76617)
        
        # Testing the type of an if condition (line 287)
        if_condition_76619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), result_lt_76618)
        # Assigning a type to the variable 'if_condition_76619' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_76619', if_condition_76619)
        # SSA begins for if statement (line 287)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 288)
        # Processing the call arguments (line 288)
        str_76621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 29), 'str', 'at least 2 breakpoints are needed')
        # Processing the call keyword arguments (line 288)
        kwargs_76622 = {}
        # Getting the type of 'ValueError' (line 288)
        ValueError_76620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 288)
        ValueError_call_result_76623 = invoke(stypy.reporting.localization.Localization(__file__, 288, 18), ValueError_76620, *[str_76621], **kwargs_76622)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 288, 12), ValueError_call_result_76623, 'raise parameter', BaseException)
        # SSA join for if statement (line 287)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'x' (line 289)
        x_76624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'x')
        # Obtaining the member 'size' of a type (line 289)
        size_76625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 11), x_76624, 'size')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 289)
        axis_76626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 29), 'axis')
        # Getting the type of 'y' (line 289)
        y_76627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 21), 'y')
        # Obtaining the member 'shape' of a type (line 289)
        shape_76628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 21), y_76627, 'shape')
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___76629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 21), shape_76628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_76630 = invoke(stypy.reporting.localization.Localization(__file__, 289, 21), getitem___76629, axis_76626)
        
        # Applying the binary operator '!=' (line 289)
        result_ne_76631 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), '!=', size_76625, subscript_call_result_76630)
        
        # Testing the type of an if condition (line 289)
        if_condition_76632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 8), result_ne_76631)
        # Assigning a type to the variable 'if_condition_76632' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'if_condition_76632', if_condition_76632)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 290)
        # Processing the call arguments (line 290)
        str_76634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 29), 'str', 'x.shape must equal y.shape[%s]')
        # Getting the type of 'axis' (line 290)
        axis_76635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 64), 'axis', False)
        # Applying the binary operator '%' (line 290)
        result_mod_76636 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 29), '%', str_76634, axis_76635)
        
        # Processing the call keyword arguments (line 290)
        kwargs_76637 = {}
        # Getting the type of 'ValueError' (line 290)
        ValueError_76633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 290)
        ValueError_call_result_76638 = invoke(stypy.reporting.localization.Localization(__file__, 290, 18), ValueError_76633, *[result_mod_76636], **kwargs_76637)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 290, 12), ValueError_call_result_76638, 'raise parameter', BaseException)
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to rollaxis(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'y' (line 293)
        y_76641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'y', False)
        # Getting the type of 'axis' (line 293)
        axis_76642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 27), 'axis', False)
        # Processing the call keyword arguments (line 293)
        kwargs_76643 = {}
        # Getting the type of 'np' (line 293)
        np_76639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'np', False)
        # Obtaining the member 'rollaxis' of a type (line 293)
        rollaxis_76640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), np_76639, 'rollaxis')
        # Calling rollaxis(args, kwargs) (line 293)
        rollaxis_call_result_76644 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), rollaxis_76640, *[y_76641, axis_76642], **kwargs_76643)
        
        # Assigning a type to the variable 'y' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'y', rollaxis_call_result_76644)
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to empty(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Obtaining an instance of the builtin type 'tuple' (line 296)
        tuple_76647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 296)
        # Adding element type (line 296)
        # Getting the type of 'x' (line 296)
        x_76648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'x', False)
        # Obtaining the member 'size' of a type (line 296)
        size_76649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 22), x_76648, 'size')
        int_76650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 31), 'int')
        # Applying the binary operator '+' (line 296)
        result_add_76651 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 22), '+', size_76649, int_76650)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 22), tuple_76647, result_add_76651)
        
        
        # Obtaining the type of the subscript
        int_76652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 46), 'int')
        slice_76653 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 38), int_76652, None, None)
        # Getting the type of 'y' (line 296)
        y_76654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'y', False)
        # Obtaining the member 'shape' of a type (line 296)
        shape_76655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 38), y_76654, 'shape')
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___76656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 38), shape_76655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_76657 = invoke(stypy.reporting.localization.Localization(__file__, 296, 38), getitem___76656, slice_76653)
        
        # Applying the binary operator '+' (line 296)
        result_add_76658 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 21), '+', tuple_76647, subscript_call_result_76657)
        
        # Processing the call keyword arguments (line 296)
        kwargs_76659 = {}
        # Getting the type of 'np' (line 296)
        np_76645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'np', False)
        # Obtaining the member 'empty' of a type (line 296)
        empty_76646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), np_76645, 'empty')
        # Calling empty(args, kwargs) (line 296)
        empty_call_result_76660 = invoke(stypy.reporting.localization.Localization(__file__, 296, 12), empty_76646, *[result_add_76658], **kwargs_76659)
        
        # Assigning a type to the variable 'm' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'm', empty_call_result_76660)
        
        # Assigning a Call to a Name (line 297):
        
        # Assigning a Call to a Name (line 297):
        
        # Call to diff(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'x' (line 297)
        x_76663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'x', False)
        # Processing the call keyword arguments (line 297)
        kwargs_76664 = {}
        # Getting the type of 'np' (line 297)
        np_76661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'np', False)
        # Obtaining the member 'diff' of a type (line 297)
        diff_76662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 13), np_76661, 'diff')
        # Calling diff(args, kwargs) (line 297)
        diff_call_result_76665 = invoke(stypy.reporting.localization.Localization(__file__, 297, 13), diff_76662, *[x_76663], **kwargs_76664)
        
        # Assigning a type to the variable 'dx' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'dx', diff_call_result_76665)
        
        # Assigning a Subscript to a Name (line 298):
        
        # Assigning a Subscript to a Name (line 298):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_76666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        
        # Call to slice(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'None' (line 298)
        None_76668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 23), 'None', False)
        # Processing the call keyword arguments (line 298)
        kwargs_76669 = {}
        # Getting the type of 'slice' (line 298)
        slice_76667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'slice', False)
        # Calling slice(args, kwargs) (line 298)
        slice_call_result_76670 = invoke(stypy.reporting.localization.Localization(__file__, 298, 17), slice_76667, *[None_76668], **kwargs_76669)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 17), tuple_76666, slice_call_result_76670)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_76671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        # Getting the type of 'None' (line 298)
        None_76672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 35), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 35), tuple_76671, None_76672)
        
        # Getting the type of 'y' (line 298)
        y_76673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 46), 'y')
        # Obtaining the member 'ndim' of a type (line 298)
        ndim_76674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 46), y_76673, 'ndim')
        int_76675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 55), 'int')
        # Applying the binary operator '-' (line 298)
        result_sub_76676 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 46), '-', ndim_76674, int_76675)
        
        # Applying the binary operator '*' (line 298)
        result_mul_76677 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 34), '*', tuple_76671, result_sub_76676)
        
        # Applying the binary operator '+' (line 298)
        result_add_76678 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 16), '+', tuple_76666, result_mul_76677)
        
        # Getting the type of 'dx' (line 298)
        dx_76679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 13), 'dx')
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___76680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 13), dx_76679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 298)
        subscript_call_result_76681 = invoke(stypy.reporting.localization.Localization(__file__, 298, 13), getitem___76680, result_add_76678)
        
        # Assigning a type to the variable 'dx' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'dx', subscript_call_result_76681)
        
        # Assigning a BinOp to a Subscript (line 299):
        
        # Assigning a BinOp to a Subscript (line 299):
        
        # Call to diff(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'y' (line 299)
        y_76684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'y', False)
        # Processing the call keyword arguments (line 299)
        int_76685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 34), 'int')
        keyword_76686 = int_76685
        kwargs_76687 = {'axis': keyword_76686}
        # Getting the type of 'np' (line 299)
        np_76682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'np', False)
        # Obtaining the member 'diff' of a type (line 299)
        diff_76683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 18), np_76682, 'diff')
        # Calling diff(args, kwargs) (line 299)
        diff_call_result_76688 = invoke(stypy.reporting.localization.Localization(__file__, 299, 18), diff_76683, *[y_76684], **kwargs_76687)
        
        # Getting the type of 'dx' (line 299)
        dx_76689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 39), 'dx')
        # Applying the binary operator 'div' (line 299)
        result_div_76690 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 18), 'div', diff_call_result_76688, dx_76689)
        
        # Getting the type of 'm' (line 299)
        m_76691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'm')
        int_76692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 10), 'int')
        int_76693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 12), 'int')
        slice_76694 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 299, 8), int_76692, int_76693, None)
        # Storing an element on a container (line 299)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 8), m_76691, (slice_76694, result_div_76690))
        
        # Assigning a BinOp to a Subscript (line 302):
        
        # Assigning a BinOp to a Subscript (line 302):
        float_76695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 15), 'float')
        
        # Obtaining the type of the subscript
        int_76696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'int')
        # Getting the type of 'm' (line 302)
        m_76697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'm')
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___76698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 20), m_76697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 302)
        subscript_call_result_76699 = invoke(stypy.reporting.localization.Localization(__file__, 302, 20), getitem___76698, int_76696)
        
        # Applying the binary operator '*' (line 302)
        result_mul_76700 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 15), '*', float_76695, subscript_call_result_76699)
        
        
        # Obtaining the type of the subscript
        int_76701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 29), 'int')
        # Getting the type of 'm' (line 302)
        m_76702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 27), 'm')
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___76703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 27), m_76702, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 302)
        subscript_call_result_76704 = invoke(stypy.reporting.localization.Localization(__file__, 302, 27), getitem___76703, int_76701)
        
        # Applying the binary operator '-' (line 302)
        result_sub_76705 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 15), '-', result_mul_76700, subscript_call_result_76704)
        
        # Getting the type of 'm' (line 302)
        m_76706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'm')
        int_76707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 10), 'int')
        # Storing an element on a container (line 302)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 8), m_76706, (int_76707, result_sub_76705))
        
        # Assigning a BinOp to a Subscript (line 303):
        
        # Assigning a BinOp to a Subscript (line 303):
        float_76708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 15), 'float')
        
        # Obtaining the type of the subscript
        int_76709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 22), 'int')
        # Getting the type of 'm' (line 303)
        m_76710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'm')
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___76711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), m_76710, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_76712 = invoke(stypy.reporting.localization.Localization(__file__, 303, 20), getitem___76711, int_76709)
        
        # Applying the binary operator '*' (line 303)
        result_mul_76713 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 15), '*', float_76708, subscript_call_result_76712)
        
        
        # Obtaining the type of the subscript
        int_76714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 29), 'int')
        # Getting the type of 'm' (line 303)
        m_76715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 27), 'm')
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___76716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 27), m_76715, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_76717 = invoke(stypy.reporting.localization.Localization(__file__, 303, 27), getitem___76716, int_76714)
        
        # Applying the binary operator '-' (line 303)
        result_sub_76718 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 15), '-', result_mul_76713, subscript_call_result_76717)
        
        # Getting the type of 'm' (line 303)
        m_76719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'm')
        int_76720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 10), 'int')
        # Storing an element on a container (line 303)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 8), m_76719, (int_76720, result_sub_76718))
        
        # Assigning a BinOp to a Subscript (line 305):
        
        # Assigning a BinOp to a Subscript (line 305):
        float_76721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 16), 'float')
        
        # Obtaining the type of the subscript
        int_76722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 23), 'int')
        # Getting the type of 'm' (line 305)
        m_76723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 21), 'm')
        # Obtaining the member '__getitem__' of a type (line 305)
        getitem___76724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 21), m_76723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 305)
        subscript_call_result_76725 = invoke(stypy.reporting.localization.Localization(__file__, 305, 21), getitem___76724, int_76722)
        
        # Applying the binary operator '*' (line 305)
        result_mul_76726 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '*', float_76721, subscript_call_result_76725)
        
        
        # Obtaining the type of the subscript
        int_76727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 31), 'int')
        # Getting the type of 'm' (line 305)
        m_76728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 29), 'm')
        # Obtaining the member '__getitem__' of a type (line 305)
        getitem___76729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 29), m_76728, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 305)
        subscript_call_result_76730 = invoke(stypy.reporting.localization.Localization(__file__, 305, 29), getitem___76729, int_76727)
        
        # Applying the binary operator '-' (line 305)
        result_sub_76731 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '-', result_mul_76726, subscript_call_result_76730)
        
        # Getting the type of 'm' (line 305)
        m_76732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'm')
        int_76733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 10), 'int')
        # Storing an element on a container (line 305)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 8), m_76732, (int_76733, result_sub_76731))
        
        # Assigning a BinOp to a Subscript (line 306):
        
        # Assigning a BinOp to a Subscript (line 306):
        float_76734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 16), 'float')
        
        # Obtaining the type of the subscript
        int_76735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 23), 'int')
        # Getting the type of 'm' (line 306)
        m_76736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'm')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___76737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 21), m_76736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_76738 = invoke(stypy.reporting.localization.Localization(__file__, 306, 21), getitem___76737, int_76735)
        
        # Applying the binary operator '*' (line 306)
        result_mul_76739 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 16), '*', float_76734, subscript_call_result_76738)
        
        
        # Obtaining the type of the subscript
        int_76740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 31), 'int')
        # Getting the type of 'm' (line 306)
        m_76741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 29), 'm')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___76742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 29), m_76741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_76743 = invoke(stypy.reporting.localization.Localization(__file__, 306, 29), getitem___76742, int_76740)
        
        # Applying the binary operator '-' (line 306)
        result_sub_76744 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 16), '-', result_mul_76739, subscript_call_result_76743)
        
        # Getting the type of 'm' (line 306)
        m_76745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'm')
        int_76746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 10), 'int')
        # Storing an element on a container (line 306)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 8), m_76745, (int_76746, result_sub_76744))
        
        # Assigning a BinOp to a Name (line 310):
        
        # Assigning a BinOp to a Name (line 310):
        float_76747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 12), 'float')
        
        # Obtaining the type of the subscript
        int_76748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 20), 'int')
        slice_76749 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 310, 18), int_76748, None, None)
        # Getting the type of 'm' (line 310)
        m_76750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 18), 'm')
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___76751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 18), m_76750, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_76752 = invoke(stypy.reporting.localization.Localization(__file__, 310, 18), getitem___76751, slice_76749)
        
        
        # Obtaining the type of the subscript
        int_76753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 29), 'int')
        slice_76754 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 310, 26), None, int_76753, None)
        # Getting the type of 'm' (line 310)
        m_76755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 26), 'm')
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___76756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 26), m_76755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_76757 = invoke(stypy.reporting.localization.Localization(__file__, 310, 26), getitem___76756, slice_76754)
        
        # Applying the binary operator '+' (line 310)
        result_add_76758 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 18), '+', subscript_call_result_76752, subscript_call_result_76757)
        
        # Applying the binary operator '*' (line 310)
        result_mul_76759 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 12), '*', float_76747, result_add_76758)
        
        # Assigning a type to the variable 't' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 't', result_mul_76759)
        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to abs(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Call to diff(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'm' (line 312)
        m_76764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), 'm', False)
        # Processing the call keyword arguments (line 312)
        int_76765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 36), 'int')
        keyword_76766 = int_76765
        kwargs_76767 = {'axis': keyword_76766}
        # Getting the type of 'np' (line 312)
        np_76762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'np', False)
        # Obtaining the member 'diff' of a type (line 312)
        diff_76763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 20), np_76762, 'diff')
        # Calling diff(args, kwargs) (line 312)
        diff_call_result_76768 = invoke(stypy.reporting.localization.Localization(__file__, 312, 20), diff_76763, *[m_76764], **kwargs_76767)
        
        # Processing the call keyword arguments (line 312)
        kwargs_76769 = {}
        # Getting the type of 'np' (line 312)
        np_76760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 13), 'np', False)
        # Obtaining the member 'abs' of a type (line 312)
        abs_76761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 13), np_76760, 'abs')
        # Calling abs(args, kwargs) (line 312)
        abs_call_result_76770 = invoke(stypy.reporting.localization.Localization(__file__, 312, 13), abs_76761, *[diff_call_result_76768], **kwargs_76769)
        
        # Assigning a type to the variable 'dm' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'dm', abs_call_result_76770)
        
        # Assigning a Subscript to a Name (line 313):
        
        # Assigning a Subscript to a Name (line 313):
        
        # Obtaining the type of the subscript
        int_76771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 16), 'int')
        slice_76772 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 313, 13), int_76771, None, None)
        # Getting the type of 'dm' (line 313)
        dm_76773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'dm')
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___76774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 13), dm_76773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_76775 = invoke(stypy.reporting.localization.Localization(__file__, 313, 13), getitem___76774, slice_76772)
        
        # Assigning a type to the variable 'f1' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'f1', subscript_call_result_76775)
        
        # Assigning a Subscript to a Name (line 314):
        
        # Assigning a Subscript to a Name (line 314):
        
        # Obtaining the type of the subscript
        int_76776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 17), 'int')
        slice_76777 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 314, 13), None, int_76776, None)
        # Getting the type of 'dm' (line 314)
        dm_76778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 13), 'dm')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___76779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 13), dm_76778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_76780 = invoke(stypy.reporting.localization.Localization(__file__, 314, 13), getitem___76779, slice_76777)
        
        # Assigning a type to the variable 'f2' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'f2', subscript_call_result_76780)
        
        # Assigning a BinOp to a Name (line 315):
        
        # Assigning a BinOp to a Name (line 315):
        # Getting the type of 'f1' (line 315)
        f1_76781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 14), 'f1')
        # Getting the type of 'f2' (line 315)
        f2_76782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'f2')
        # Applying the binary operator '+' (line 315)
        result_add_76783 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 14), '+', f1_76781, f2_76782)
        
        # Assigning a type to the variable 'f12' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'f12', result_add_76783)
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to nonzero(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Getting the type of 'f12' (line 317)
        f12_76786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 25), 'f12', False)
        float_76787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 31), 'float')
        
        # Call to max(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'f12' (line 317)
        f12_76790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 45), 'f12', False)
        # Processing the call keyword arguments (line 317)
        kwargs_76791 = {}
        # Getting the type of 'np' (line 317)
        np_76788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 38), 'np', False)
        # Obtaining the member 'max' of a type (line 317)
        max_76789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 38), np_76788, 'max')
        # Calling max(args, kwargs) (line 317)
        max_call_result_76792 = invoke(stypy.reporting.localization.Localization(__file__, 317, 38), max_76789, *[f12_76790], **kwargs_76791)
        
        # Applying the binary operator '*' (line 317)
        result_mul_76793 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 31), '*', float_76787, max_call_result_76792)
        
        # Applying the binary operator '>' (line 317)
        result_gt_76794 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 25), '>', f12_76786, result_mul_76793)
        
        # Processing the call keyword arguments (line 317)
        kwargs_76795 = {}
        # Getting the type of 'np' (line 317)
        np_76784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 14), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 317)
        nonzero_76785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 14), np_76784, 'nonzero')
        # Calling nonzero(args, kwargs) (line 317)
        nonzero_call_result_76796 = invoke(stypy.reporting.localization.Localization(__file__, 317, 14), nonzero_76785, *[result_gt_76794], **kwargs_76795)
        
        # Assigning a type to the variable 'ind' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'ind', nonzero_call_result_76796)
        
        # Assigning a Tuple to a Tuple (line 318):
        
        # Assigning a Subscript to a Name (line 318):
        
        # Obtaining the type of the subscript
        int_76797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'int')
        # Getting the type of 'ind' (line 318)
        ind_76798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'ind')
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___76799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 23), ind_76798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_76800 = invoke(stypy.reporting.localization.Localization(__file__, 318, 23), getitem___76799, int_76797)
        
        # Assigning a type to the variable 'tuple_assignment_76082' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_assignment_76082', subscript_call_result_76800)
        
        # Assigning a Subscript to a Name (line 318):
        
        # Obtaining the type of the subscript
        int_76801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 35), 'int')
        slice_76802 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 318, 31), int_76801, None, None)
        # Getting the type of 'ind' (line 318)
        ind_76803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 31), 'ind')
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___76804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 31), ind_76803, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_76805 = invoke(stypy.reporting.localization.Localization(__file__, 318, 31), getitem___76804, slice_76802)
        
        # Assigning a type to the variable 'tuple_assignment_76083' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_assignment_76083', subscript_call_result_76805)
        
        # Assigning a Name to a Name (line 318):
        # Getting the type of 'tuple_assignment_76082' (line 318)
        tuple_assignment_76082_76806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_assignment_76082')
        # Assigning a type to the variable 'x_ind' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'x_ind', tuple_assignment_76082_76806)
        
        # Assigning a Name to a Name (line 318):
        # Getting the type of 'tuple_assignment_76083' (line 318)
        tuple_assignment_76083_76807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_assignment_76083')
        # Assigning a type to the variable 'y_ind' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 15), 'y_ind', tuple_assignment_76083_76807)
        
        # Assigning a BinOp to a Subscript (line 320):
        
        # Assigning a BinOp to a Subscript (line 320):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 320)
        ind_76808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 21), 'ind')
        # Getting the type of 'f1' (line 320)
        f1_76809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 18), 'f1')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___76810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 18), f1_76809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_76811 = invoke(stypy.reporting.localization.Localization(__file__, 320, 18), getitem___76810, ind_76808)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 320)
        tuple_76812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 320)
        # Adding element type (line 320)
        # Getting the type of 'x_ind' (line 320)
        x_ind_76813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 31), 'x_ind')
        int_76814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 39), 'int')
        # Applying the binary operator '+' (line 320)
        result_add_76815 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 31), '+', x_ind_76813, int_76814)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 31), tuple_76812, result_add_76815)
        
        # Getting the type of 'y_ind' (line 320)
        y_ind_76816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 45), 'y_ind')
        # Applying the binary operator '+' (line 320)
        result_add_76817 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 30), '+', tuple_76812, y_ind_76816)
        
        # Getting the type of 'm' (line 320)
        m_76818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 28), 'm')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___76819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 28), m_76818, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_76820 = invoke(stypy.reporting.localization.Localization(__file__, 320, 28), getitem___76819, result_add_76817)
        
        # Applying the binary operator '*' (line 320)
        result_mul_76821 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 18), '*', subscript_call_result_76811, subscript_call_result_76820)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 321)
        ind_76822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 21), 'ind')
        # Getting the type of 'f2' (line 321)
        f2_76823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'f2')
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___76824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 18), f2_76823, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 321)
        subscript_call_result_76825 = invoke(stypy.reporting.localization.Localization(__file__, 321, 18), getitem___76824, ind_76822)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 321)
        tuple_76826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 321)
        # Adding element type (line 321)
        # Getting the type of 'x_ind' (line 321)
        x_ind_76827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 'x_ind')
        int_76828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 39), 'int')
        # Applying the binary operator '+' (line 321)
        result_add_76829 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 31), '+', x_ind_76827, int_76828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 31), tuple_76826, result_add_76829)
        
        # Getting the type of 'y_ind' (line 321)
        y_ind_76830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 45), 'y_ind')
        # Applying the binary operator '+' (line 321)
        result_add_76831 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 30), '+', tuple_76826, y_ind_76830)
        
        # Getting the type of 'm' (line 321)
        m_76832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 28), 'm')
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___76833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 28), m_76832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 321)
        subscript_call_result_76834 = invoke(stypy.reporting.localization.Localization(__file__, 321, 28), getitem___76833, result_add_76831)
        
        # Applying the binary operator '*' (line 321)
        result_mul_76835 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 18), '*', subscript_call_result_76825, subscript_call_result_76834)
        
        # Applying the binary operator '+' (line 320)
        result_add_76836 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 18), '+', result_mul_76821, result_mul_76835)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 321)
        ind_76837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 59), 'ind')
        # Getting the type of 'f12' (line 321)
        f12_76838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 55), 'f12')
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___76839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 55), f12_76838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 321)
        subscript_call_result_76840 = invoke(stypy.reporting.localization.Localization(__file__, 321, 55), getitem___76839, ind_76837)
        
        # Applying the binary operator 'div' (line 320)
        result_div_76841 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 17), 'div', result_add_76836, subscript_call_result_76840)
        
        # Getting the type of 't' (line 320)
        t_76842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 't')
        # Getting the type of 'ind' (line 320)
        ind_76843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 10), 'ind')
        # Storing an element on a container (line 320)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 8), t_76842, (ind_76843, result_div_76841))
        
        # Assigning a BinOp to a Name (line 323):
        
        # Assigning a BinOp to a Name (line 323):
        float_76844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 13), 'float')
        
        # Obtaining the type of the subscript
        int_76845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'int')
        int_76846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 22), 'int')
        slice_76847 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 323, 18), int_76845, int_76846, None)
        # Getting the type of 'm' (line 323)
        m_76848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 18), 'm')
        # Obtaining the member '__getitem__' of a type (line 323)
        getitem___76849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 18), m_76848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 323)
        subscript_call_result_76850 = invoke(stypy.reporting.localization.Localization(__file__, 323, 18), getitem___76849, slice_76847)
        
        # Applying the binary operator '*' (line 323)
        result_mul_76851 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 13), '*', float_76844, subscript_call_result_76850)
        
        float_76852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 28), 'float')
        
        # Obtaining the type of the subscript
        int_76853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 36), 'int')
        slice_76854 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 323, 33), None, int_76853, None)
        # Getting the type of 't' (line 323)
        t_76855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 33), 't')
        # Obtaining the member '__getitem__' of a type (line 323)
        getitem___76856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 33), t_76855, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 323)
        subscript_call_result_76857 = invoke(stypy.reporting.localization.Localization(__file__, 323, 33), getitem___76856, slice_76854)
        
        # Applying the binary operator '*' (line 323)
        result_mul_76858 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 28), '*', float_76852, subscript_call_result_76857)
        
        # Applying the binary operator '-' (line 323)
        result_sub_76859 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 13), '-', result_mul_76851, result_mul_76858)
        
        
        # Obtaining the type of the subscript
        int_76860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 44), 'int')
        slice_76861 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 323, 42), int_76860, None, None)
        # Getting the type of 't' (line 323)
        t_76862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 42), 't')
        # Obtaining the member '__getitem__' of a type (line 323)
        getitem___76863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 42), t_76862, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 323)
        subscript_call_result_76864 = invoke(stypy.reporting.localization.Localization(__file__, 323, 42), getitem___76863, slice_76861)
        
        # Applying the binary operator '-' (line 323)
        result_sub_76865 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 40), '-', result_sub_76859, subscript_call_result_76864)
        
        # Getting the type of 'dx' (line 323)
        dx_76866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 51), 'dx')
        # Applying the binary operator 'div' (line 323)
        result_div_76867 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 12), 'div', result_sub_76865, dx_76866)
        
        # Assigning a type to the variable 'c' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'c', result_div_76867)
        
        # Assigning a BinOp to a Name (line 324):
        
        # Assigning a BinOp to a Name (line 324):
        
        # Obtaining the type of the subscript
        int_76868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 16), 'int')
        slice_76869 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 324, 13), None, int_76868, None)
        # Getting the type of 't' (line 324)
        t_76870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 13), 't')
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___76871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 13), t_76870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 324)
        subscript_call_result_76872 = invoke(stypy.reporting.localization.Localization(__file__, 324, 13), getitem___76871, slice_76869)
        
        
        # Obtaining the type of the subscript
        int_76873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 24), 'int')
        slice_76874 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 324, 22), int_76873, None, None)
        # Getting the type of 't' (line 324)
        t_76875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 22), 't')
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___76876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 22), t_76875, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 324)
        subscript_call_result_76877 = invoke(stypy.reporting.localization.Localization(__file__, 324, 22), getitem___76876, slice_76874)
        
        # Applying the binary operator '+' (line 324)
        result_add_76878 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 13), '+', subscript_call_result_76872, subscript_call_result_76877)
        
        float_76879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 30), 'float')
        
        # Obtaining the type of the subscript
        int_76880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 37), 'int')
        int_76881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 39), 'int')
        slice_76882 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 324, 35), int_76880, int_76881, None)
        # Getting the type of 'm' (line 324)
        m_76883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 35), 'm')
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___76884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 35), m_76883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 324)
        subscript_call_result_76885 = invoke(stypy.reporting.localization.Localization(__file__, 324, 35), getitem___76884, slice_76882)
        
        # Applying the binary operator '*' (line 324)
        result_mul_76886 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 30), '*', float_76879, subscript_call_result_76885)
        
        # Applying the binary operator '-' (line 324)
        result_sub_76887 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 28), '-', result_add_76878, result_mul_76886)
        
        # Getting the type of 'dx' (line 324)
        dx_76888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 46), 'dx')
        int_76889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 52), 'int')
        # Applying the binary operator '**' (line 324)
        result_pow_76890 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 46), '**', dx_76888, int_76889)
        
        # Applying the binary operator 'div' (line 324)
        result_div_76891 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 12), 'div', result_sub_76887, result_pow_76890)
        
        # Assigning a type to the variable 'd' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'd', result_div_76891)
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to zeros(...): (line 326)
        # Processing the call arguments (line 326)
        
        # Obtaining an instance of the builtin type 'tuple' (line 326)
        tuple_76894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 326)
        # Adding element type (line 326)
        int_76895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 26), tuple_76894, int_76895)
        # Adding element type (line 326)
        # Getting the type of 'x' (line 326)
        x_76896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 29), 'x', False)
        # Obtaining the member 'size' of a type (line 326)
        size_76897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 29), x_76896, 'size')
        int_76898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 38), 'int')
        # Applying the binary operator '-' (line 326)
        result_sub_76899 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 29), '-', size_76897, int_76898)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 26), tuple_76894, result_sub_76899)
        
        
        # Obtaining the type of the subscript
        int_76900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 51), 'int')
        slice_76901 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 326, 43), int_76900, None, None)
        # Getting the type of 'y' (line 326)
        y_76902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 43), 'y', False)
        # Obtaining the member 'shape' of a type (line 326)
        shape_76903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 43), y_76902, 'shape')
        # Obtaining the member '__getitem__' of a type (line 326)
        getitem___76904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 43), shape_76903, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 326)
        subscript_call_result_76905 = invoke(stypy.reporting.localization.Localization(__file__, 326, 43), getitem___76904, slice_76901)
        
        # Applying the binary operator '+' (line 326)
        result_add_76906 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 25), '+', tuple_76894, subscript_call_result_76905)
        
        # Processing the call keyword arguments (line 326)
        kwargs_76907 = {}
        # Getting the type of 'np' (line 326)
        np_76892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'np', False)
        # Obtaining the member 'zeros' of a type (line 326)
        zeros_76893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), np_76892, 'zeros')
        # Calling zeros(args, kwargs) (line 326)
        zeros_call_result_76908 = invoke(stypy.reporting.localization.Localization(__file__, 326, 16), zeros_76893, *[result_add_76906], **kwargs_76907)
        
        # Assigning a type to the variable 'coeff' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'coeff', zeros_call_result_76908)
        
        # Assigning a Subscript to a Subscript (line 327):
        
        # Assigning a Subscript to a Subscript (line 327):
        
        # Obtaining the type of the subscript
        int_76909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 22), 'int')
        slice_76910 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 327, 19), None, int_76909, None)
        # Getting the type of 'y' (line 327)
        y_76911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'y')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___76912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 19), y_76911, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_76913 = invoke(stypy.reporting.localization.Localization(__file__, 327, 19), getitem___76912, slice_76910)
        
        # Getting the type of 'coeff' (line 327)
        coeff_76914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'coeff')
        int_76915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 14), 'int')
        # Storing an element on a container (line 327)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 8), coeff_76914, (int_76915, subscript_call_result_76913))
        
        # Assigning a Subscript to a Subscript (line 328):
        
        # Assigning a Subscript to a Subscript (line 328):
        
        # Obtaining the type of the subscript
        int_76916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 22), 'int')
        slice_76917 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 328, 19), None, int_76916, None)
        # Getting the type of 't' (line 328)
        t_76918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 't')
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___76919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 19), t_76918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_76920 = invoke(stypy.reporting.localization.Localization(__file__, 328, 19), getitem___76919, slice_76917)
        
        # Getting the type of 'coeff' (line 328)
        coeff_76921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'coeff')
        int_76922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 14), 'int')
        # Storing an element on a container (line 328)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 8), coeff_76921, (int_76922, subscript_call_result_76920))
        
        # Assigning a Name to a Subscript (line 329):
        
        # Assigning a Name to a Subscript (line 329):
        # Getting the type of 'c' (line 329)
        c_76923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 19), 'c')
        # Getting the type of 'coeff' (line 329)
        coeff_76924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'coeff')
        int_76925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 14), 'int')
        # Storing an element on a container (line 329)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 8), coeff_76924, (int_76925, c_76923))
        
        # Assigning a Name to a Subscript (line 330):
        
        # Assigning a Name to a Subscript (line 330):
        # Getting the type of 'd' (line 330)
        d_76926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'd')
        # Getting the type of 'coeff' (line 330)
        coeff_76927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'coeff')
        int_76928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 14), 'int')
        # Storing an element on a container (line 330)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 8), coeff_76927, (int_76928, d_76926))
        
        # Call to __init__(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'coeff' (line 332)
        coeff_76935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 50), 'coeff', False)
        # Getting the type of 'x' (line 332)
        x_76936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 57), 'x', False)
        # Processing the call keyword arguments (line 332)
        # Getting the type of 'False' (line 332)
        False_76937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 72), 'False', False)
        keyword_76938 = False_76937
        kwargs_76939 = {'extrapolate': keyword_76938}
        
        # Call to super(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'Akima1DInterpolator' (line 332)
        Akima1DInterpolator_76930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'Akima1DInterpolator', False)
        # Getting the type of 'self' (line 332)
        self_76931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'self', False)
        # Processing the call keyword arguments (line 332)
        kwargs_76932 = {}
        # Getting the type of 'super' (line 332)
        super_76929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'super', False)
        # Calling super(args, kwargs) (line 332)
        super_call_result_76933 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), super_76929, *[Akima1DInterpolator_76930, self_76931], **kwargs_76932)
        
        # Obtaining the member '__init__' of a type (line 332)
        init___76934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), super_call_result_76933, '__init__')
        # Calling __init__(args, kwargs) (line 332)
        init___call_result_76940 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), init___76934, *[coeff_76935, x_76936], **kwargs_76939)
        
        
        # Assigning a Name to a Attribute (line 333):
        
        # Assigning a Name to a Attribute (line 333):
        # Getting the type of 'axis' (line 333)
        axis_76941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'axis')
        # Getting the type of 'self' (line 333)
        self_76942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 333)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_76942, 'axis', axis_76941)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def extend(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 335)
        True_76943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 33), 'True')
        defaults = [True_76943]
        # Create a new context for function 'extend'
        module_type_store = module_type_store.open_function_context('extend', 335, 4, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_localization', localization)
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_type_store', module_type_store)
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_function_name', 'Akima1DInterpolator.extend')
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_param_names_list', ['c', 'x', 'right'])
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_varargs_param_name', None)
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_call_defaults', defaults)
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_call_varargs', varargs)
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Akima1DInterpolator.extend.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Akima1DInterpolator.extend', ['c', 'x', 'right'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'extend', localization, ['c', 'x', 'right'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'extend(...)' code ##################

        
        # Call to NotImplementedError(...): (line 336)
        # Processing the call arguments (line 336)
        str_76945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 34), 'str', 'Extending a 1D Akima interpolator is not yet implemented')
        # Processing the call keyword arguments (line 336)
        kwargs_76946 = {}
        # Getting the type of 'NotImplementedError' (line 336)
        NotImplementedError_76944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 336)
        NotImplementedError_call_result_76947 = invoke(stypy.reporting.localization.Localization(__file__, 336, 14), NotImplementedError_76944, *[str_76945], **kwargs_76946)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 336, 8), NotImplementedError_call_result_76947, 'raise parameter', BaseException)
        
        # ################# End of 'extend(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'extend' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_76948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_76948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'extend'
        return stypy_return_type_76948


    @norecursion
    def from_spline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 342)
        None_76949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 42), 'None')
        defaults = [None_76949]
        # Create a new context for function 'from_spline'
        module_type_store = module_type_store.open_function_context('from_spline', 341, 4, False)
        # Assigning a type to the variable 'self' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_localization', localization)
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_type_store', module_type_store)
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_function_name', 'Akima1DInterpolator.from_spline')
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_param_names_list', ['tck', 'extrapolate'])
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_varargs_param_name', None)
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_call_defaults', defaults)
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_call_varargs', varargs)
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Akima1DInterpolator.from_spline.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Akima1DInterpolator.from_spline', ['tck', 'extrapolate'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_spline', localization, ['tck', 'extrapolate'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_spline(...)' code ##################

        
        # Call to NotImplementedError(...): (line 343)
        # Processing the call arguments (line 343)
        str_76951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 34), 'str', 'This method does not make sense for an Akima interpolator.')
        # Processing the call keyword arguments (line 343)
        kwargs_76952 = {}
        # Getting the type of 'NotImplementedError' (line 343)
        NotImplementedError_76950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 343)
        NotImplementedError_call_result_76953 = invoke(stypy.reporting.localization.Localization(__file__, 343, 14), NotImplementedError_76950, *[str_76951], **kwargs_76952)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 343, 8), NotImplementedError_call_result_76953, 'raise parameter', BaseException)
        
        # ################# End of 'from_spline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_spline' in the type store
        # Getting the type of 'stypy_return_type' (line 341)
        stypy_return_type_76954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_76954)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_spline'
        return stypy_return_type_76954


    @norecursion
    def from_bernstein_basis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 347)
        None_76955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 50), 'None')
        defaults = [None_76955]
        # Create a new context for function 'from_bernstein_basis'
        module_type_store = module_type_store.open_function_context('from_bernstein_basis', 346, 4, False)
        # Assigning a type to the variable 'self' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_localization', localization)
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_type_store', module_type_store)
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_function_name', 'Akima1DInterpolator.from_bernstein_basis')
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_param_names_list', ['bp', 'extrapolate'])
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_varargs_param_name', None)
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_call_defaults', defaults)
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_call_varargs', varargs)
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Akima1DInterpolator.from_bernstein_basis.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Akima1DInterpolator.from_bernstein_basis', ['bp', 'extrapolate'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_bernstein_basis', localization, ['bp', 'extrapolate'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_bernstein_basis(...)' code ##################

        
        # Call to NotImplementedError(...): (line 348)
        # Processing the call arguments (line 348)
        str_76957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 34), 'str', 'This method does not make sense for an Akima interpolator.')
        # Processing the call keyword arguments (line 348)
        kwargs_76958 = {}
        # Getting the type of 'NotImplementedError' (line 348)
        NotImplementedError_76956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 348)
        NotImplementedError_call_result_76959 = invoke(stypy.reporting.localization.Localization(__file__, 348, 14), NotImplementedError_76956, *[str_76957], **kwargs_76958)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 348, 8), NotImplementedError_call_result_76959, 'raise parameter', BaseException)
        
        # ################# End of 'from_bernstein_basis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_bernstein_basis' in the type store
        # Getting the type of 'stypy_return_type' (line 346)
        stypy_return_type_76960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_76960)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_bernstein_basis'
        return stypy_return_type_76960


# Assigning a type to the variable 'Akima1DInterpolator' (line 228)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'Akima1DInterpolator', Akima1DInterpolator)
# Declaration of the 'CubicSpline' class
# Getting the type of 'PPoly' (line 352)
PPoly_76961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 18), 'PPoly')

class CubicSpline(PPoly_76961, ):
    str_76962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, (-1)), 'str', 'Cubic spline data interpolator.\n\n    Interpolate data with a piecewise cubic polynomial which is twice\n    continuously differentiable [1]_. The result is represented as a `PPoly`\n    instance with breakpoints matching the given data.\n\n    Parameters\n    ----------\n    x : array_like, shape (n,)\n        1-d array containing values of the independent variable.\n        Values must be real, finite and in strictly increasing order.\n    y : array_like\n        Array containing values of the dependent variable. It can have\n        arbitrary number of dimensions, but the length along `axis` (see below)\n        must match the length of `x`. Values must be finite.\n    axis : int, optional\n        Axis along which `y` is assumed to be varying. Meaning that for\n        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.\n        Default is 0.\n    bc_type : string or 2-tuple, optional\n        Boundary condition type. Two additional equations, given by the\n        boundary conditions, are required to determine all coefficients of\n        polynomials on each segment [2]_.\n\n        If `bc_type` is a string, then the specified condition will be applied\n        at both ends of a spline. Available conditions are:\n\n        * \'not-a-knot\' (default): The first and second segment at a curve end\n          are the same polynomial. It is a good default when there is no\n          information on boundary conditions.\n        * \'periodic\': The interpolated functions is assumed to be periodic\n          of period ``x[-1] - x[0]``. The first and last value of `y` must be\n          identical: ``y[0] == y[-1]``. This boundary condition will result in\n          ``y\'[0] == y\'[-1]`` and ``y\'\'[0] == y\'\'[-1]``.\n        * \'clamped\': The first derivative at curves ends are zero. Assuming\n          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.\n        * \'natural\': The second derivative at curve ends are zero. Assuming\n          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.\n\n        If `bc_type` is a 2-tuple, the first and the second value will be\n        applied at the curve start and end respectively. The tuple values can\n        be one of the previously mentioned strings (except \'periodic\') or a\n        tuple `(order, deriv_values)` allowing to specify arbitrary\n        derivatives at curve ends:\n\n        * `order`: the derivative order, 1 or 2.\n        * `deriv_value`: array_like containing derivative values, shape must\n          be the same as `y`, excluding `axis` dimension. For example, if `y`\n          is 1D, then `deriv_value` must be a scalar. If `y` is 3D with the\n          shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D\n          and have the shape (n0, n1).\n    extrapolate : {bool, \'periodic\', None}, optional\n        If bool, determines whether to extrapolate to out-of-bounds points\n        based on first and last intervals, or to return NaNs. If \'periodic\',\n        periodic extrapolation is used. If None (default), `extrapolate` is\n        set to \'periodic\' for ``bc_type=\'periodic\'`` and to True otherwise.\n\n    Attributes\n    ----------\n    x : ndarray, shape (n,)\n        Breakpoints. The same `x` which was passed to the constructor.\n    c : ndarray, shape (4, n-1, ...)\n        Coefficients of the polynomials on each segment. The trailing\n        dimensions match the dimensions of `y`, excluding `axis`. For example,\n        if `y` is 1-d, then ``c[k, i]`` is a coefficient for\n        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.\n    axis : int\n        Interpolation axis. The same `axis` which was passed to the\n        constructor.\n\n    Methods\n    -------\n    __call__\n    derivative\n    antiderivative\n    integrate\n    roots\n\n    See Also\n    --------\n    Akima1DInterpolator\n    PchipInterpolator\n    PPoly\n\n    Notes\n    -----\n    Parameters `bc_type` and `interpolate` work independently, i.e. the former\n    controls only construction of a spline, and the latter only evaluation.\n\n    When a boundary condition is \'not-a-knot\' and n = 2, it is replaced by\n    a condition that the first derivative is equal to the linear interpolant\n    slope. When both boundary conditions are \'not-a-knot\' and n = 3, the\n    solution is sought as a parabola passing through given points.\n\n    When \'not-a-knot\' boundary conditions is applied to both ends, the\n    resulting spline will be the same as returned by `splrep` (with ``s=0``)\n    and `InterpolatedUnivariateSpline`, but these two methods use a\n    representation in B-spline basis.\n\n    .. versionadded:: 0.18.0\n\n    Examples\n    --------\n    In this example the cubic spline is used to interpolate a sampled sinusoid.\n    You can see that the spline continuity property holds for the first and\n    second derivatives and violates only for the third derivative.\n\n    >>> from scipy.interpolate import CubicSpline\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.arange(10)\n    >>> y = np.sin(x)\n    >>> cs = CubicSpline(x, y)\n    >>> xs = np.arange(-0.5, 9.6, 0.1)\n    >>> plt.figure(figsize=(6.5, 4))\n    >>> plt.plot(x, y, \'o\', label=\'data\')\n    >>> plt.plot(xs, np.sin(xs), label=\'true\')\n    >>> plt.plot(xs, cs(xs), label="S")\n    >>> plt.plot(xs, cs(xs, 1), label="S\'")\n    >>> plt.plot(xs, cs(xs, 2), label="S\'\'")\n    >>> plt.plot(xs, cs(xs, 3), label="S\'\'\'")\n    >>> plt.xlim(-0.5, 9.5)\n    >>> plt.legend(loc=\'lower left\', ncol=2)\n    >>> plt.show()\n\n    In the second example, the unit circle is interpolated with a spline. A\n    periodic boundary condition is used. You can see that the first derivative\n    values, ds/dx=0, ds/dy=1 at the periodic point (1, 0) are correctly\n    computed. Note that a circle cannot be exactly represented by a cubic\n    spline. To increase precision, more breakpoints would be required.\n\n    >>> theta = 2 * np.pi * np.linspace(0, 1, 5)\n    >>> y = np.c_[np.cos(theta), np.sin(theta)]\n    >>> cs = CubicSpline(theta, y, bc_type=\'periodic\')\n    >>> print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))\n    ds/dx=0.0 ds/dy=1.0\n    >>> xs = 2 * np.pi * np.linspace(0, 1, 100)\n    >>> plt.figure(figsize=(6.5, 4))\n    >>> plt.plot(y[:, 0], y[:, 1], \'o\', label=\'data\')\n    >>> plt.plot(np.cos(xs), np.sin(xs), label=\'true\')\n    >>> plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label=\'spline\')\n    >>> plt.axes().set_aspect(\'equal\')\n    >>> plt.legend(loc=\'center\')\n    >>> plt.show()\n\n    The third example is the interpolation of a polynomial y = x**3 on the\n    interval 0 <= x<= 1. A cubic spline can represent this function exactly.\n    To achieve that we need to specify values and first derivatives at\n    endpoints of the interval. Note that y\' = 3 * x**2 and thus y\'(0) = 0 and\n    y\'(1) = 3.\n\n    >>> cs = CubicSpline([0, 1], [0, 1], bc_type=((1, 0), (1, 3)))\n    >>> x = np.linspace(0, 1)\n    >>> np.allclose(x**3, cs(x))\n    True\n\n    References\n    ----------\n    .. [1] `Cubic Spline Interpolation\n            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_\n            on Wikiversity.\n    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_76963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 34), 'int')
        str_76964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 45), 'str', 'not-a-knot')
        # Getting the type of 'None' (line 515)
        None_76965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 71), 'None')
        defaults = [int_76963, str_76964, None_76965]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CubicSpline.__init__', ['x', 'y', 'axis', 'bc_type', 'extrapolate'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'axis', 'bc_type', 'extrapolate'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Tuple (line 516):
        
        # Assigning a Subscript to a Name (line 516):
        
        # Obtaining the type of the subscript
        int_76966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 8), 'int')
        
        # Call to map(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'np' (line 516)
        np_76968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 516)
        asarray_76969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 19), np_76968, 'asarray')
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_76970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        # Getting the type of 'x' (line 516)
        x_76971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 32), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 32), tuple_76970, x_76971)
        # Adding element type (line 516)
        # Getting the type of 'y' (line 516)
        y_76972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 35), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 32), tuple_76970, y_76972)
        
        # Processing the call keyword arguments (line 516)
        kwargs_76973 = {}
        # Getting the type of 'map' (line 516)
        map_76967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'map', False)
        # Calling map(args, kwargs) (line 516)
        map_call_result_76974 = invoke(stypy.reporting.localization.Localization(__file__, 516, 15), map_76967, *[asarray_76969, tuple_76970], **kwargs_76973)
        
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___76975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), map_call_result_76974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 516)
        subscript_call_result_76976 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), getitem___76975, int_76966)
        
        # Assigning a type to the variable 'tuple_var_assignment_76084' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_76084', subscript_call_result_76976)
        
        # Assigning a Subscript to a Name (line 516):
        
        # Obtaining the type of the subscript
        int_76977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 8), 'int')
        
        # Call to map(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'np' (line 516)
        np_76979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 516)
        asarray_76980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 19), np_76979, 'asarray')
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_76981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        # Getting the type of 'x' (line 516)
        x_76982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 32), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 32), tuple_76981, x_76982)
        # Adding element type (line 516)
        # Getting the type of 'y' (line 516)
        y_76983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 35), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 32), tuple_76981, y_76983)
        
        # Processing the call keyword arguments (line 516)
        kwargs_76984 = {}
        # Getting the type of 'map' (line 516)
        map_76978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'map', False)
        # Calling map(args, kwargs) (line 516)
        map_call_result_76985 = invoke(stypy.reporting.localization.Localization(__file__, 516, 15), map_76978, *[asarray_76980, tuple_76981], **kwargs_76984)
        
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___76986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), map_call_result_76985, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 516)
        subscript_call_result_76987 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), getitem___76986, int_76977)
        
        # Assigning a type to the variable 'tuple_var_assignment_76085' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_76085', subscript_call_result_76987)
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'tuple_var_assignment_76084' (line 516)
        tuple_var_assignment_76084_76988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_76084')
        # Assigning a type to the variable 'x' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'x', tuple_var_assignment_76084_76988)
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'tuple_var_assignment_76085' (line 516)
        tuple_var_assignment_76085_76989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_76085')
        # Assigning a type to the variable 'y' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'y', tuple_var_assignment_76085_76989)
        
        
        # Call to issubdtype(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'x' (line 518)
        x_76992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 25), 'x', False)
        # Obtaining the member 'dtype' of a type (line 518)
        dtype_76993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 25), x_76992, 'dtype')
        # Getting the type of 'np' (line 518)
        np_76994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 34), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 518)
        complexfloating_76995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 34), np_76994, 'complexfloating')
        # Processing the call keyword arguments (line 518)
        kwargs_76996 = {}
        # Getting the type of 'np' (line 518)
        np_76990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 11), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 518)
        issubdtype_76991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 11), np_76990, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 518)
        issubdtype_call_result_76997 = invoke(stypy.reporting.localization.Localization(__file__, 518, 11), issubdtype_76991, *[dtype_76993, complexfloating_76995], **kwargs_76996)
        
        # Testing the type of an if condition (line 518)
        if_condition_76998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 518, 8), issubdtype_call_result_76997)
        # Assigning a type to the variable 'if_condition_76998' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'if_condition_76998', if_condition_76998)
        # SSA begins for if statement (line 518)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 519)
        # Processing the call arguments (line 519)
        str_77000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 29), 'str', '`x` must contain real values.')
        # Processing the call keyword arguments (line 519)
        kwargs_77001 = {}
        # Getting the type of 'ValueError' (line 519)
        ValueError_76999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 519)
        ValueError_call_result_77002 = invoke(stypy.reporting.localization.Localization(__file__, 519, 18), ValueError_76999, *[str_77000], **kwargs_77001)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 519, 12), ValueError_call_result_77002, 'raise parameter', BaseException)
        # SSA join for if statement (line 518)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to issubdtype(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'y' (line 521)
        y_77005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 521)
        dtype_77006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 25), y_77005, 'dtype')
        # Getting the type of 'np' (line 521)
        np_77007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 34), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 521)
        complexfloating_77008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 34), np_77007, 'complexfloating')
        # Processing the call keyword arguments (line 521)
        kwargs_77009 = {}
        # Getting the type of 'np' (line 521)
        np_77003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 11), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 521)
        issubdtype_77004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 11), np_77003, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 521)
        issubdtype_call_result_77010 = invoke(stypy.reporting.localization.Localization(__file__, 521, 11), issubdtype_77004, *[dtype_77006, complexfloating_77008], **kwargs_77009)
        
        # Testing the type of an if condition (line 521)
        if_condition_77011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 8), issubdtype_call_result_77010)
        # Assigning a type to the variable 'if_condition_77011' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'if_condition_77011', if_condition_77011)
        # SSA begins for if statement (line 521)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 522):
        
        # Assigning a Name to a Name (line 522):
        # Getting the type of 'complex' (line 522)
        complex_77012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 20), 'complex')
        # Assigning a type to the variable 'dtype' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'dtype', complex_77012)
        # SSA branch for the else part of an if statement (line 521)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 524):
        
        # Assigning a Name to a Name (line 524):
        # Getting the type of 'float' (line 524)
        float_77013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 20), 'float')
        # Assigning a type to the variable 'dtype' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'dtype', float_77013)
        # SSA join for if statement (line 521)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 525):
        
        # Assigning a Call to a Name (line 525):
        
        # Call to astype(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'dtype' (line 525)
        dtype_77016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 21), 'dtype', False)
        # Processing the call keyword arguments (line 525)
        # Getting the type of 'False' (line 525)
        False_77017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 33), 'False', False)
        keyword_77018 = False_77017
        kwargs_77019 = {'copy': keyword_77018}
        # Getting the type of 'y' (line 525)
        y_77014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'y', False)
        # Obtaining the member 'astype' of a type (line 525)
        astype_77015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 12), y_77014, 'astype')
        # Calling astype(args, kwargs) (line 525)
        astype_call_result_77020 = invoke(stypy.reporting.localization.Localization(__file__, 525, 12), astype_77015, *[dtype_77016], **kwargs_77019)
        
        # Assigning a type to the variable 'y' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'y', astype_call_result_77020)
        
        # Assigning a BinOp to a Name (line 527):
        
        # Assigning a BinOp to a Name (line 527):
        # Getting the type of 'axis' (line 527)
        axis_77021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 15), 'axis')
        # Getting the type of 'y' (line 527)
        y_77022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 22), 'y')
        # Obtaining the member 'ndim' of a type (line 527)
        ndim_77023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 22), y_77022, 'ndim')
        # Applying the binary operator '%' (line 527)
        result_mod_77024 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 15), '%', axis_77021, ndim_77023)
        
        # Assigning a type to the variable 'axis' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'axis', result_mod_77024)
        
        
        # Getting the type of 'x' (line 528)
        x_77025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 11), 'x')
        # Obtaining the member 'ndim' of a type (line 528)
        ndim_77026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 11), x_77025, 'ndim')
        int_77027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 21), 'int')
        # Applying the binary operator '!=' (line 528)
        result_ne_77028 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 11), '!=', ndim_77026, int_77027)
        
        # Testing the type of an if condition (line 528)
        if_condition_77029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 8), result_ne_77028)
        # Assigning a type to the variable 'if_condition_77029' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'if_condition_77029', if_condition_77029)
        # SSA begins for if statement (line 528)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 529)
        # Processing the call arguments (line 529)
        str_77031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 29), 'str', '`x` must be 1-dimensional.')
        # Processing the call keyword arguments (line 529)
        kwargs_77032 = {}
        # Getting the type of 'ValueError' (line 529)
        ValueError_77030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 529)
        ValueError_call_result_77033 = invoke(stypy.reporting.localization.Localization(__file__, 529, 18), ValueError_77030, *[str_77031], **kwargs_77032)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 529, 12), ValueError_call_result_77033, 'raise parameter', BaseException)
        # SSA join for if statement (line 528)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_77034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 19), 'int')
        # Getting the type of 'x' (line 530)
        x_77035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'x')
        # Obtaining the member 'shape' of a type (line 530)
        shape_77036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 11), x_77035, 'shape')
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___77037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 11), shape_77036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 530)
        subscript_call_result_77038 = invoke(stypy.reporting.localization.Localization(__file__, 530, 11), getitem___77037, int_77034)
        
        int_77039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 24), 'int')
        # Applying the binary operator '<' (line 530)
        result_lt_77040 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 11), '<', subscript_call_result_77038, int_77039)
        
        # Testing the type of an if condition (line 530)
        if_condition_77041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 8), result_lt_77040)
        # Assigning a type to the variable 'if_condition_77041' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'if_condition_77041', if_condition_77041)
        # SSA begins for if statement (line 530)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 531)
        # Processing the call arguments (line 531)
        str_77043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 29), 'str', '`x` must contain at least 2 elements.')
        # Processing the call keyword arguments (line 531)
        kwargs_77044 = {}
        # Getting the type of 'ValueError' (line 531)
        ValueError_77042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 531)
        ValueError_call_result_77045 = invoke(stypy.reporting.localization.Localization(__file__, 531, 18), ValueError_77042, *[str_77043], **kwargs_77044)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 531, 12), ValueError_call_result_77045, 'raise parameter', BaseException)
        # SSA join for if statement (line 530)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_77046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 19), 'int')
        # Getting the type of 'x' (line 532)
        x_77047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 11), 'x')
        # Obtaining the member 'shape' of a type (line 532)
        shape_77048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 11), x_77047, 'shape')
        # Obtaining the member '__getitem__' of a type (line 532)
        getitem___77049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 11), shape_77048, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 532)
        subscript_call_result_77050 = invoke(stypy.reporting.localization.Localization(__file__, 532, 11), getitem___77049, int_77046)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 532)
        axis_77051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 33), 'axis')
        # Getting the type of 'y' (line 532)
        y_77052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 25), 'y')
        # Obtaining the member 'shape' of a type (line 532)
        shape_77053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 25), y_77052, 'shape')
        # Obtaining the member '__getitem__' of a type (line 532)
        getitem___77054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 25), shape_77053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 532)
        subscript_call_result_77055 = invoke(stypy.reporting.localization.Localization(__file__, 532, 25), getitem___77054, axis_77051)
        
        # Applying the binary operator '!=' (line 532)
        result_ne_77056 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 11), '!=', subscript_call_result_77050, subscript_call_result_77055)
        
        # Testing the type of an if condition (line 532)
        if_condition_77057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 8), result_ne_77056)
        # Assigning a type to the variable 'if_condition_77057' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'if_condition_77057', if_condition_77057)
        # SSA begins for if statement (line 532)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 533)
        # Processing the call arguments (line 533)
        
        # Call to format(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'axis' (line 534)
        axis_77061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 62), 'axis', False)
        # Processing the call keyword arguments (line 533)
        kwargs_77062 = {}
        str_77059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 29), 'str', "The length of `y` along `axis`={0} doesn't match the length of `x`")
        # Obtaining the member 'format' of a type (line 533)
        format_77060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 29), str_77059, 'format')
        # Calling format(args, kwargs) (line 533)
        format_call_result_77063 = invoke(stypy.reporting.localization.Localization(__file__, 533, 29), format_77060, *[axis_77061], **kwargs_77062)
        
        # Processing the call keyword arguments (line 533)
        kwargs_77064 = {}
        # Getting the type of 'ValueError' (line 533)
        ValueError_77058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 533)
        ValueError_call_result_77065 = invoke(stypy.reporting.localization.Localization(__file__, 533, 18), ValueError_77058, *[format_call_result_77063], **kwargs_77064)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 533, 12), ValueError_call_result_77065, 'raise parameter', BaseException)
        # SSA join for if statement (line 532)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to all(...): (line 536)
        # Processing the call arguments (line 536)
        
        # Call to isfinite(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'x' (line 536)
        x_77070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 34), 'x', False)
        # Processing the call keyword arguments (line 536)
        kwargs_77071 = {}
        # Getting the type of 'np' (line 536)
        np_77068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 22), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 536)
        isfinite_77069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 22), np_77068, 'isfinite')
        # Calling isfinite(args, kwargs) (line 536)
        isfinite_call_result_77072 = invoke(stypy.reporting.localization.Localization(__file__, 536, 22), isfinite_77069, *[x_77070], **kwargs_77071)
        
        # Processing the call keyword arguments (line 536)
        kwargs_77073 = {}
        # Getting the type of 'np' (line 536)
        np_77066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 536)
        all_77067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 15), np_77066, 'all')
        # Calling all(args, kwargs) (line 536)
        all_call_result_77074 = invoke(stypy.reporting.localization.Localization(__file__, 536, 15), all_77067, *[isfinite_call_result_77072], **kwargs_77073)
        
        # Applying the 'not' unary operator (line 536)
        result_not__77075 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), 'not', all_call_result_77074)
        
        # Testing the type of an if condition (line 536)
        if_condition_77076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), result_not__77075)
        # Assigning a type to the variable 'if_condition_77076' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_77076', if_condition_77076)
        # SSA begins for if statement (line 536)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 537)
        # Processing the call arguments (line 537)
        str_77078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 29), 'str', '`x` must contain only finite values.')
        # Processing the call keyword arguments (line 537)
        kwargs_77079 = {}
        # Getting the type of 'ValueError' (line 537)
        ValueError_77077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 537)
        ValueError_call_result_77080 = invoke(stypy.reporting.localization.Localization(__file__, 537, 18), ValueError_77077, *[str_77078], **kwargs_77079)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 537, 12), ValueError_call_result_77080, 'raise parameter', BaseException)
        # SSA join for if statement (line 536)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to all(...): (line 538)
        # Processing the call arguments (line 538)
        
        # Call to isfinite(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'y' (line 538)
        y_77085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 34), 'y', False)
        # Processing the call keyword arguments (line 538)
        kwargs_77086 = {}
        # Getting the type of 'np' (line 538)
        np_77083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 22), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 538)
        isfinite_77084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 22), np_77083, 'isfinite')
        # Calling isfinite(args, kwargs) (line 538)
        isfinite_call_result_77087 = invoke(stypy.reporting.localization.Localization(__file__, 538, 22), isfinite_77084, *[y_77085], **kwargs_77086)
        
        # Processing the call keyword arguments (line 538)
        kwargs_77088 = {}
        # Getting the type of 'np' (line 538)
        np_77081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 538)
        all_77082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 15), np_77081, 'all')
        # Calling all(args, kwargs) (line 538)
        all_call_result_77089 = invoke(stypy.reporting.localization.Localization(__file__, 538, 15), all_77082, *[isfinite_call_result_77087], **kwargs_77088)
        
        # Applying the 'not' unary operator (line 538)
        result_not__77090 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 11), 'not', all_call_result_77089)
        
        # Testing the type of an if condition (line 538)
        if_condition_77091 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 538, 8), result_not__77090)
        # Assigning a type to the variable 'if_condition_77091' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'if_condition_77091', if_condition_77091)
        # SSA begins for if statement (line 538)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 539)
        # Processing the call arguments (line 539)
        str_77093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 29), 'str', '`y` must contain only finite values.')
        # Processing the call keyword arguments (line 539)
        kwargs_77094 = {}
        # Getting the type of 'ValueError' (line 539)
        ValueError_77092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 539)
        ValueError_call_result_77095 = invoke(stypy.reporting.localization.Localization(__file__, 539, 18), ValueError_77092, *[str_77093], **kwargs_77094)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 539, 12), ValueError_call_result_77095, 'raise parameter', BaseException)
        # SSA join for if statement (line 538)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 541):
        
        # Assigning a Call to a Name (line 541):
        
        # Call to diff(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'x' (line 541)
        x_77098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 21), 'x', False)
        # Processing the call keyword arguments (line 541)
        kwargs_77099 = {}
        # Getting the type of 'np' (line 541)
        np_77096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 13), 'np', False)
        # Obtaining the member 'diff' of a type (line 541)
        diff_77097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 13), np_77096, 'diff')
        # Calling diff(args, kwargs) (line 541)
        diff_call_result_77100 = invoke(stypy.reporting.localization.Localization(__file__, 541, 13), diff_77097, *[x_77098], **kwargs_77099)
        
        # Assigning a type to the variable 'dx' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'dx', diff_call_result_77100)
        
        
        # Call to any(...): (line 542)
        # Processing the call arguments (line 542)
        
        # Getting the type of 'dx' (line 542)
        dx_77103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 18), 'dx', False)
        int_77104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 24), 'int')
        # Applying the binary operator '<=' (line 542)
        result_le_77105 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 18), '<=', dx_77103, int_77104)
        
        # Processing the call keyword arguments (line 542)
        kwargs_77106 = {}
        # Getting the type of 'np' (line 542)
        np_77101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 542)
        any_77102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 11), np_77101, 'any')
        # Calling any(args, kwargs) (line 542)
        any_call_result_77107 = invoke(stypy.reporting.localization.Localization(__file__, 542, 11), any_77102, *[result_le_77105], **kwargs_77106)
        
        # Testing the type of an if condition (line 542)
        if_condition_77108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 542, 8), any_call_result_77107)
        # Assigning a type to the variable 'if_condition_77108' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'if_condition_77108', if_condition_77108)
        # SSA begins for if statement (line 542)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 543)
        # Processing the call arguments (line 543)
        str_77110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 29), 'str', '`x` must be strictly increasing sequence.')
        # Processing the call keyword arguments (line 543)
        kwargs_77111 = {}
        # Getting the type of 'ValueError' (line 543)
        ValueError_77109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 543)
        ValueError_call_result_77112 = invoke(stypy.reporting.localization.Localization(__file__, 543, 18), ValueError_77109, *[str_77110], **kwargs_77111)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 543, 12), ValueError_call_result_77112, 'raise parameter', BaseException)
        # SSA join for if statement (line 542)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 545):
        
        # Assigning a Subscript to a Name (line 545):
        
        # Obtaining the type of the subscript
        int_77113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 20), 'int')
        # Getting the type of 'x' (line 545)
        x_77114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'x')
        # Obtaining the member 'shape' of a type (line 545)
        shape_77115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 12), x_77114, 'shape')
        # Obtaining the member '__getitem__' of a type (line 545)
        getitem___77116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 12), shape_77115, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 545)
        subscript_call_result_77117 = invoke(stypy.reporting.localization.Localization(__file__, 545, 12), getitem___77116, int_77113)
        
        # Assigning a type to the variable 'n' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'n', subscript_call_result_77117)
        
        # Assigning a Call to a Name (line 546):
        
        # Assigning a Call to a Name (line 546):
        
        # Call to rollaxis(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'y' (line 546)
        y_77120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 24), 'y', False)
        # Getting the type of 'axis' (line 546)
        axis_77121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 27), 'axis', False)
        # Processing the call keyword arguments (line 546)
        kwargs_77122 = {}
        # Getting the type of 'np' (line 546)
        np_77118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'np', False)
        # Obtaining the member 'rollaxis' of a type (line 546)
        rollaxis_77119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 12), np_77118, 'rollaxis')
        # Calling rollaxis(args, kwargs) (line 546)
        rollaxis_call_result_77123 = invoke(stypy.reporting.localization.Localization(__file__, 546, 12), rollaxis_77119, *[y_77120, axis_77121], **kwargs_77122)
        
        # Assigning a type to the variable 'y' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'y', rollaxis_call_result_77123)
        
        # Assigning a Call to a Tuple (line 548):
        
        # Assigning a Subscript to a Name (line 548):
        
        # Obtaining the type of the subscript
        int_77124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 8), 'int')
        
        # Call to _validate_bc(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'bc_type' (line 548)
        bc_type_77127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 34), 'bc_type', False)
        # Getting the type of 'y' (line 548)
        y_77128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 43), 'y', False)
        
        # Obtaining the type of the subscript
        int_77129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 54), 'int')
        slice_77130 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 548, 46), int_77129, None, None)
        # Getting the type of 'y' (line 548)
        y_77131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 46), 'y', False)
        # Obtaining the member 'shape' of a type (line 548)
        shape_77132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 46), y_77131, 'shape')
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___77133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 46), shape_77132, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_77134 = invoke(stypy.reporting.localization.Localization(__file__, 548, 46), getitem___77133, slice_77130)
        
        # Getting the type of 'axis' (line 548)
        axis_77135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 59), 'axis', False)
        # Processing the call keyword arguments (line 548)
        kwargs_77136 = {}
        # Getting the type of 'self' (line 548)
        self_77125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'self', False)
        # Obtaining the member '_validate_bc' of a type (line 548)
        _validate_bc_77126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 16), self_77125, '_validate_bc')
        # Calling _validate_bc(args, kwargs) (line 548)
        _validate_bc_call_result_77137 = invoke(stypy.reporting.localization.Localization(__file__, 548, 16), _validate_bc_77126, *[bc_type_77127, y_77128, subscript_call_result_77134, axis_77135], **kwargs_77136)
        
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___77138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 8), _validate_bc_call_result_77137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_77139 = invoke(stypy.reporting.localization.Localization(__file__, 548, 8), getitem___77138, int_77124)
        
        # Assigning a type to the variable 'tuple_var_assignment_76086' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'tuple_var_assignment_76086', subscript_call_result_77139)
        
        # Assigning a Subscript to a Name (line 548):
        
        # Obtaining the type of the subscript
        int_77140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 8), 'int')
        
        # Call to _validate_bc(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'bc_type' (line 548)
        bc_type_77143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 34), 'bc_type', False)
        # Getting the type of 'y' (line 548)
        y_77144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 43), 'y', False)
        
        # Obtaining the type of the subscript
        int_77145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 54), 'int')
        slice_77146 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 548, 46), int_77145, None, None)
        # Getting the type of 'y' (line 548)
        y_77147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 46), 'y', False)
        # Obtaining the member 'shape' of a type (line 548)
        shape_77148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 46), y_77147, 'shape')
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___77149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 46), shape_77148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_77150 = invoke(stypy.reporting.localization.Localization(__file__, 548, 46), getitem___77149, slice_77146)
        
        # Getting the type of 'axis' (line 548)
        axis_77151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 59), 'axis', False)
        # Processing the call keyword arguments (line 548)
        kwargs_77152 = {}
        # Getting the type of 'self' (line 548)
        self_77141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'self', False)
        # Obtaining the member '_validate_bc' of a type (line 548)
        _validate_bc_77142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 16), self_77141, '_validate_bc')
        # Calling _validate_bc(args, kwargs) (line 548)
        _validate_bc_call_result_77153 = invoke(stypy.reporting.localization.Localization(__file__, 548, 16), _validate_bc_77142, *[bc_type_77143, y_77144, subscript_call_result_77150, axis_77151], **kwargs_77152)
        
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___77154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 8), _validate_bc_call_result_77153, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_77155 = invoke(stypy.reporting.localization.Localization(__file__, 548, 8), getitem___77154, int_77140)
        
        # Assigning a type to the variable 'tuple_var_assignment_76087' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'tuple_var_assignment_76087', subscript_call_result_77155)
        
        # Assigning a Name to a Name (line 548):
        # Getting the type of 'tuple_var_assignment_76086' (line 548)
        tuple_var_assignment_76086_77156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'tuple_var_assignment_76086')
        # Assigning a type to the variable 'bc' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'bc', tuple_var_assignment_76086_77156)
        
        # Assigning a Name to a Name (line 548):
        # Getting the type of 'tuple_var_assignment_76087' (line 548)
        tuple_var_assignment_76087_77157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'tuple_var_assignment_76087')
        # Assigning a type to the variable 'y' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'y', tuple_var_assignment_76087_77157)
        
        # Type idiom detected: calculating its left and rigth part (line 550)
        # Getting the type of 'extrapolate' (line 550)
        extrapolate_77158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 11), 'extrapolate')
        # Getting the type of 'None' (line 550)
        None_77159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 26), 'None')
        
        (may_be_77160, more_types_in_union_77161) = may_be_none(extrapolate_77158, None_77159)

        if may_be_77160:

            if more_types_in_union_77161:
                # Runtime conditional SSA (line 550)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            
            # Obtaining the type of the subscript
            int_77162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 18), 'int')
            # Getting the type of 'bc' (line 551)
            bc_77163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'bc')
            # Obtaining the member '__getitem__' of a type (line 551)
            getitem___77164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 15), bc_77163, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 551)
            subscript_call_result_77165 = invoke(stypy.reporting.localization.Localization(__file__, 551, 15), getitem___77164, int_77162)
            
            str_77166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 24), 'str', 'periodic')
            # Applying the binary operator '==' (line 551)
            result_eq_77167 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 15), '==', subscript_call_result_77165, str_77166)
            
            # Testing the type of an if condition (line 551)
            if_condition_77168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 551, 12), result_eq_77167)
            # Assigning a type to the variable 'if_condition_77168' (line 551)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'if_condition_77168', if_condition_77168)
            # SSA begins for if statement (line 551)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 552):
            
            # Assigning a Str to a Name (line 552):
            str_77169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 30), 'str', 'periodic')
            # Assigning a type to the variable 'extrapolate' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'extrapolate', str_77169)
            # SSA branch for the else part of an if statement (line 551)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 554):
            
            # Assigning a Name to a Name (line 554):
            # Getting the type of 'True' (line 554)
            True_77170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 30), 'True')
            # Assigning a type to the variable 'extrapolate' (line 554)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'extrapolate', True_77170)
            # SSA join for if statement (line 551)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_77161:
                # SSA join for if statement (line 550)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 556):
        
        # Assigning a Call to a Name (line 556):
        
        # Call to reshape(...): (line 556)
        # Processing the call arguments (line 556)
        
        # Obtaining an instance of the builtin type 'list' (line 556)
        list_77173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 556)
        # Adding element type (line 556)
        
        # Obtaining the type of the subscript
        int_77174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 35), 'int')
        # Getting the type of 'dx' (line 556)
        dx_77175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 26), 'dx', False)
        # Obtaining the member 'shape' of a type (line 556)
        shape_77176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 26), dx_77175, 'shape')
        # Obtaining the member '__getitem__' of a type (line 556)
        getitem___77177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 26), shape_77176, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 556)
        subscript_call_result_77178 = invoke(stypy.reporting.localization.Localization(__file__, 556, 26), getitem___77177, int_77174)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 25), list_77173, subscript_call_result_77178)
        
        
        # Obtaining an instance of the builtin type 'list' (line 556)
        list_77179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 556)
        # Adding element type (line 556)
        int_77180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 41), list_77179, int_77180)
        
        # Getting the type of 'y' (line 556)
        y_77181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 48), 'y', False)
        # Obtaining the member 'ndim' of a type (line 556)
        ndim_77182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 48), y_77181, 'ndim')
        int_77183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 57), 'int')
        # Applying the binary operator '-' (line 556)
        result_sub_77184 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 48), '-', ndim_77182, int_77183)
        
        # Applying the binary operator '*' (line 556)
        result_mul_77185 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 41), '*', list_77179, result_sub_77184)
        
        # Applying the binary operator '+' (line 556)
        result_add_77186 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 25), '+', list_77173, result_mul_77185)
        
        # Processing the call keyword arguments (line 556)
        kwargs_77187 = {}
        # Getting the type of 'dx' (line 556)
        dx_77171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 14), 'dx', False)
        # Obtaining the member 'reshape' of a type (line 556)
        reshape_77172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 14), dx_77171, 'reshape')
        # Calling reshape(args, kwargs) (line 556)
        reshape_call_result_77188 = invoke(stypy.reporting.localization.Localization(__file__, 556, 14), reshape_77172, *[result_add_77186], **kwargs_77187)
        
        # Assigning a type to the variable 'dxr' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'dxr', reshape_call_result_77188)
        
        # Assigning a BinOp to a Name (line 557):
        
        # Assigning a BinOp to a Name (line 557):
        
        # Call to diff(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'y' (line 557)
        y_77191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 24), 'y', False)
        # Processing the call keyword arguments (line 557)
        int_77192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 32), 'int')
        keyword_77193 = int_77192
        kwargs_77194 = {'axis': keyword_77193}
        # Getting the type of 'np' (line 557)
        np_77189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'np', False)
        # Obtaining the member 'diff' of a type (line 557)
        diff_77190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 16), np_77189, 'diff')
        # Calling diff(args, kwargs) (line 557)
        diff_call_result_77195 = invoke(stypy.reporting.localization.Localization(__file__, 557, 16), diff_77190, *[y_77191], **kwargs_77194)
        
        # Getting the type of 'dxr' (line 557)
        dxr_77196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 37), 'dxr')
        # Applying the binary operator 'div' (line 557)
        result_div_77197 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 16), 'div', diff_call_result_77195, dxr_77196)
        
        # Assigning a type to the variable 'slope' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'slope', result_div_77197)
        
        
        # Getting the type of 'n' (line 563)
        n_77198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 11), 'n')
        int_77199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 16), 'int')
        # Applying the binary operator '==' (line 563)
        result_eq_77200 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 11), '==', n_77198, int_77199)
        
        # Testing the type of an if condition (line 563)
        if_condition_77201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 563, 8), result_eq_77200)
        # Assigning a type to the variable 'if_condition_77201' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'if_condition_77201', if_condition_77201)
        # SSA begins for if statement (line 563)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_77202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 18), 'int')
        # Getting the type of 'bc' (line 564)
        bc_77203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'bc')
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___77204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 15), bc_77203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 564)
        subscript_call_result_77205 = invoke(stypy.reporting.localization.Localization(__file__, 564, 15), getitem___77204, int_77202)
        
        
        # Obtaining an instance of the builtin type 'list' (line 564)
        list_77206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 564)
        # Adding element type (line 564)
        str_77207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 25), 'str', 'not-a-knot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 24), list_77206, str_77207)
        # Adding element type (line 564)
        str_77208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 39), 'str', 'periodic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 24), list_77206, str_77208)
        
        # Applying the binary operator 'in' (line 564)
        result_contains_77209 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 15), 'in', subscript_call_result_77205, list_77206)
        
        # Testing the type of an if condition (line 564)
        if_condition_77210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 564, 12), result_contains_77209)
        # Assigning a type to the variable 'if_condition_77210' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'if_condition_77210', if_condition_77210)
        # SSA begins for if statement (line 564)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Subscript (line 565):
        
        # Assigning a Tuple to a Subscript (line 565):
        
        # Obtaining an instance of the builtin type 'tuple' (line 565)
        tuple_77211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 565)
        # Adding element type (line 565)
        int_77212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 25), tuple_77211, int_77212)
        # Adding element type (line 565)
        
        # Obtaining the type of the subscript
        int_77213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 34), 'int')
        # Getting the type of 'slope' (line 565)
        slope_77214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 28), 'slope')
        # Obtaining the member '__getitem__' of a type (line 565)
        getitem___77215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 28), slope_77214, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 565)
        subscript_call_result_77216 = invoke(stypy.reporting.localization.Localization(__file__, 565, 28), getitem___77215, int_77213)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 25), tuple_77211, subscript_call_result_77216)
        
        # Getting the type of 'bc' (line 565)
        bc_77217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'bc')
        int_77218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 19), 'int')
        # Storing an element on a container (line 565)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 16), bc_77217, (int_77218, tuple_77211))
        # SSA join for if statement (line 564)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_77219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 18), 'int')
        # Getting the type of 'bc' (line 566)
        bc_77220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 15), 'bc')
        # Obtaining the member '__getitem__' of a type (line 566)
        getitem___77221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 15), bc_77220, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 566)
        subscript_call_result_77222 = invoke(stypy.reporting.localization.Localization(__file__, 566, 15), getitem___77221, int_77219)
        
        
        # Obtaining an instance of the builtin type 'list' (line 566)
        list_77223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 566)
        # Adding element type (line 566)
        str_77224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 25), 'str', 'not-a-knot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 24), list_77223, str_77224)
        # Adding element type (line 566)
        str_77225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 39), 'str', 'periodic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 24), list_77223, str_77225)
        
        # Applying the binary operator 'in' (line 566)
        result_contains_77226 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 15), 'in', subscript_call_result_77222, list_77223)
        
        # Testing the type of an if condition (line 566)
        if_condition_77227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 566, 12), result_contains_77226)
        # Assigning a type to the variable 'if_condition_77227' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'if_condition_77227', if_condition_77227)
        # SSA begins for if statement (line 566)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Subscript (line 567):
        
        # Assigning a Tuple to a Subscript (line 567):
        
        # Obtaining an instance of the builtin type 'tuple' (line 567)
        tuple_77228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 567)
        # Adding element type (line 567)
        int_77229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 25), tuple_77228, int_77229)
        # Adding element type (line 567)
        
        # Obtaining the type of the subscript
        int_77230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 34), 'int')
        # Getting the type of 'slope' (line 567)
        slope_77231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 28), 'slope')
        # Obtaining the member '__getitem__' of a type (line 567)
        getitem___77232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 28), slope_77231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 567)
        subscript_call_result_77233 = invoke(stypy.reporting.localization.Localization(__file__, 567, 28), getitem___77232, int_77230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 25), tuple_77228, subscript_call_result_77233)
        
        # Getting the type of 'bc' (line 567)
        bc_77234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 16), 'bc')
        int_77235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 19), 'int')
        # Storing an element on a container (line 567)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 16), bc_77234, (int_77235, tuple_77228))
        # SSA join for if statement (line 566)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 563)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'n' (line 573)
        n_77236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 11), 'n')
        int_77237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 16), 'int')
        # Applying the binary operator '==' (line 573)
        result_eq_77238 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 11), '==', n_77236, int_77237)
        
        
        
        # Obtaining the type of the subscript
        int_77239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 25), 'int')
        # Getting the type of 'bc' (line 573)
        bc_77240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 22), 'bc')
        # Obtaining the member '__getitem__' of a type (line 573)
        getitem___77241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 22), bc_77240, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 573)
        subscript_call_result_77242 = invoke(stypy.reporting.localization.Localization(__file__, 573, 22), getitem___77241, int_77239)
        
        str_77243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 31), 'str', 'not-a-knot')
        # Applying the binary operator '==' (line 573)
        result_eq_77244 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 22), '==', subscript_call_result_77242, str_77243)
        
        # Applying the binary operator 'and' (line 573)
        result_and_keyword_77245 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 11), 'and', result_eq_77238, result_eq_77244)
        
        
        # Obtaining the type of the subscript
        int_77246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 51), 'int')
        # Getting the type of 'bc' (line 573)
        bc_77247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 48), 'bc')
        # Obtaining the member '__getitem__' of a type (line 573)
        getitem___77248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 48), bc_77247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 573)
        subscript_call_result_77249 = invoke(stypy.reporting.localization.Localization(__file__, 573, 48), getitem___77248, int_77246)
        
        str_77250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 57), 'str', 'not-a-knot')
        # Applying the binary operator '==' (line 573)
        result_eq_77251 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 48), '==', subscript_call_result_77249, str_77250)
        
        # Applying the binary operator 'and' (line 573)
        result_and_keyword_77252 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 11), 'and', result_and_keyword_77245, result_eq_77251)
        
        # Testing the type of an if condition (line 573)
        if_condition_77253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 8), result_and_keyword_77252)
        # Assigning a type to the variable 'if_condition_77253' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'if_condition_77253', if_condition_77253)
        # SSA begins for if statement (line 573)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 574):
        
        # Assigning a Call to a Name (line 574):
        
        # Call to zeros(...): (line 574)
        # Processing the call arguments (line 574)
        
        # Obtaining an instance of the builtin type 'tuple' (line 574)
        tuple_77256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 574)
        # Adding element type (line 574)
        int_77257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 26), tuple_77256, int_77257)
        # Adding element type (line 574)
        int_77258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 26), tuple_77256, int_77258)
        
        # Processing the call keyword arguments (line 574)
        kwargs_77259 = {}
        # Getting the type of 'np' (line 574)
        np_77254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 16), 'np', False)
        # Obtaining the member 'zeros' of a type (line 574)
        zeros_77255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 16), np_77254, 'zeros')
        # Calling zeros(args, kwargs) (line 574)
        zeros_call_result_77260 = invoke(stypy.reporting.localization.Localization(__file__, 574, 16), zeros_77255, *[tuple_77256], **kwargs_77259)
        
        # Assigning a type to the variable 'A' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'A', zeros_call_result_77260)
        
        # Assigning a Call to a Name (line 575):
        
        # Assigning a Call to a Name (line 575):
        
        # Call to empty(...): (line 575)
        # Processing the call arguments (line 575)
        
        # Obtaining an instance of the builtin type 'tuple' (line 575)
        tuple_77263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 575)
        # Adding element type (line 575)
        int_77264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 26), tuple_77263, int_77264)
        
        
        # Obtaining the type of the subscript
        int_77265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 40), 'int')
        slice_77266 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 575, 32), int_77265, None, None)
        # Getting the type of 'y' (line 575)
        y_77267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 32), 'y', False)
        # Obtaining the member 'shape' of a type (line 575)
        shape_77268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 32), y_77267, 'shape')
        # Obtaining the member '__getitem__' of a type (line 575)
        getitem___77269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 32), shape_77268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 575)
        subscript_call_result_77270 = invoke(stypy.reporting.localization.Localization(__file__, 575, 32), getitem___77269, slice_77266)
        
        # Applying the binary operator '+' (line 575)
        result_add_77271 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 25), '+', tuple_77263, subscript_call_result_77270)
        
        # Processing the call keyword arguments (line 575)
        # Getting the type of 'y' (line 575)
        y_77272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 51), 'y', False)
        # Obtaining the member 'dtype' of a type (line 575)
        dtype_77273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 51), y_77272, 'dtype')
        keyword_77274 = dtype_77273
        kwargs_77275 = {'dtype': keyword_77274}
        # Getting the type of 'np' (line 575)
        np_77261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'np', False)
        # Obtaining the member 'empty' of a type (line 575)
        empty_77262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 16), np_77261, 'empty')
        # Calling empty(args, kwargs) (line 575)
        empty_call_result_77276 = invoke(stypy.reporting.localization.Localization(__file__, 575, 16), empty_77262, *[result_add_77271], **kwargs_77275)
        
        # Assigning a type to the variable 'b' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'b', empty_call_result_77276)
        
        # Assigning a Num to a Subscript (line 577):
        
        # Assigning a Num to a Subscript (line 577):
        int_77277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 22), 'int')
        # Getting the type of 'A' (line 577)
        A_77278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 577)
        tuple_77279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 577)
        # Adding element type (line 577)
        int_77280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 14), tuple_77279, int_77280)
        # Adding element type (line 577)
        int_77281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 14), tuple_77279, int_77281)
        
        # Storing an element on a container (line 577)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 12), A_77278, (tuple_77279, int_77277))
        
        # Assigning a Num to a Subscript (line 578):
        
        # Assigning a Num to a Subscript (line 578):
        int_77282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 22), 'int')
        # Getting the type of 'A' (line 578)
        A_77283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 578)
        tuple_77284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 578)
        # Adding element type (line 578)
        int_77285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 14), tuple_77284, int_77285)
        # Adding element type (line 578)
        int_77286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 14), tuple_77284, int_77286)
        
        # Storing an element on a container (line 578)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 12), A_77283, (tuple_77284, int_77282))
        
        # Assigning a Subscript to a Subscript (line 579):
        
        # Assigning a Subscript to a Subscript (line 579):
        
        # Obtaining the type of the subscript
        int_77287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 25), 'int')
        # Getting the type of 'dx' (line 579)
        dx_77288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 22), 'dx')
        # Obtaining the member '__getitem__' of a type (line 579)
        getitem___77289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 22), dx_77288, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 579)
        subscript_call_result_77290 = invoke(stypy.reporting.localization.Localization(__file__, 579, 22), getitem___77289, int_77287)
        
        # Getting the type of 'A' (line 579)
        A_77291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 579)
        tuple_77292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 579)
        # Adding element type (line 579)
        int_77293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 14), tuple_77292, int_77293)
        # Adding element type (line 579)
        int_77294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 14), tuple_77292, int_77294)
        
        # Storing an element on a container (line 579)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 12), A_77291, (tuple_77292, subscript_call_result_77290))
        
        # Assigning a BinOp to a Subscript (line 580):
        
        # Assigning a BinOp to a Subscript (line 580):
        int_77295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 22), 'int')
        
        # Obtaining the type of the subscript
        int_77296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 30), 'int')
        # Getting the type of 'dx' (line 580)
        dx_77297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 27), 'dx')
        # Obtaining the member '__getitem__' of a type (line 580)
        getitem___77298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 27), dx_77297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 580)
        subscript_call_result_77299 = invoke(stypy.reporting.localization.Localization(__file__, 580, 27), getitem___77298, int_77296)
        
        
        # Obtaining the type of the subscript
        int_77300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 38), 'int')
        # Getting the type of 'dx' (line 580)
        dx_77301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 35), 'dx')
        # Obtaining the member '__getitem__' of a type (line 580)
        getitem___77302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 35), dx_77301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 580)
        subscript_call_result_77303 = invoke(stypy.reporting.localization.Localization(__file__, 580, 35), getitem___77302, int_77300)
        
        # Applying the binary operator '+' (line 580)
        result_add_77304 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 27), '+', subscript_call_result_77299, subscript_call_result_77303)
        
        # Applying the binary operator '*' (line 580)
        result_mul_77305 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 22), '*', int_77295, result_add_77304)
        
        # Getting the type of 'A' (line 580)
        A_77306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 580)
        tuple_77307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 580)
        # Adding element type (line 580)
        int_77308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 14), tuple_77307, int_77308)
        # Adding element type (line 580)
        int_77309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 14), tuple_77307, int_77309)
        
        # Storing an element on a container (line 580)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 12), A_77306, (tuple_77307, result_mul_77305))
        
        # Assigning a Subscript to a Subscript (line 581):
        
        # Assigning a Subscript to a Subscript (line 581):
        
        # Obtaining the type of the subscript
        int_77310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 25), 'int')
        # Getting the type of 'dx' (line 581)
        dx_77311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 22), 'dx')
        # Obtaining the member '__getitem__' of a type (line 581)
        getitem___77312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 22), dx_77311, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 581)
        subscript_call_result_77313 = invoke(stypy.reporting.localization.Localization(__file__, 581, 22), getitem___77312, int_77310)
        
        # Getting the type of 'A' (line 581)
        A_77314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 581)
        tuple_77315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 581)
        # Adding element type (line 581)
        int_77316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 14), tuple_77315, int_77316)
        # Adding element type (line 581)
        int_77317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 14), tuple_77315, int_77317)
        
        # Storing an element on a container (line 581)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 12), A_77314, (tuple_77315, subscript_call_result_77313))
        
        # Assigning a Num to a Subscript (line 582):
        
        # Assigning a Num to a Subscript (line 582):
        int_77318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 22), 'int')
        # Getting the type of 'A' (line 582)
        A_77319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 582)
        tuple_77320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 582)
        # Adding element type (line 582)
        int_77321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 14), tuple_77320, int_77321)
        # Adding element type (line 582)
        int_77322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 14), tuple_77320, int_77322)
        
        # Storing an element on a container (line 582)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 12), A_77319, (tuple_77320, int_77318))
        
        # Assigning a Num to a Subscript (line 583):
        
        # Assigning a Num to a Subscript (line 583):
        int_77323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 22), 'int')
        # Getting the type of 'A' (line 583)
        A_77324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 583)
        tuple_77325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 583)
        # Adding element type (line 583)
        int_77326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 14), tuple_77325, int_77326)
        # Adding element type (line 583)
        int_77327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 14), tuple_77325, int_77327)
        
        # Storing an element on a container (line 583)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 12), A_77324, (tuple_77325, int_77323))
        
        # Assigning a BinOp to a Subscript (line 585):
        
        # Assigning a BinOp to a Subscript (line 585):
        int_77328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 19), 'int')
        
        # Obtaining the type of the subscript
        int_77329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 29), 'int')
        # Getting the type of 'slope' (line 585)
        slope_77330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 23), 'slope')
        # Obtaining the member '__getitem__' of a type (line 585)
        getitem___77331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 23), slope_77330, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 585)
        subscript_call_result_77332 = invoke(stypy.reporting.localization.Localization(__file__, 585, 23), getitem___77331, int_77329)
        
        # Applying the binary operator '*' (line 585)
        result_mul_77333 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 19), '*', int_77328, subscript_call_result_77332)
        
        # Getting the type of 'b' (line 585)
        b_77334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'b')
        int_77335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 14), 'int')
        # Storing an element on a container (line 585)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 12), b_77334, (int_77335, result_mul_77333))
        
        # Assigning a BinOp to a Subscript (line 586):
        
        # Assigning a BinOp to a Subscript (line 586):
        int_77336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 19), 'int')
        
        # Obtaining the type of the subscript
        int_77337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 28), 'int')
        # Getting the type of 'dxr' (line 586)
        dxr_77338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 24), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 586)
        getitem___77339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 24), dxr_77338, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 586)
        subscript_call_result_77340 = invoke(stypy.reporting.localization.Localization(__file__, 586, 24), getitem___77339, int_77337)
        
        
        # Obtaining the type of the subscript
        int_77341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 39), 'int')
        # Getting the type of 'slope' (line 586)
        slope_77342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 33), 'slope')
        # Obtaining the member '__getitem__' of a type (line 586)
        getitem___77343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 33), slope_77342, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 586)
        subscript_call_result_77344 = invoke(stypy.reporting.localization.Localization(__file__, 586, 33), getitem___77343, int_77341)
        
        # Applying the binary operator '*' (line 586)
        result_mul_77345 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 24), '*', subscript_call_result_77340, subscript_call_result_77344)
        
        
        # Obtaining the type of the subscript
        int_77346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 48), 'int')
        # Getting the type of 'dxr' (line 586)
        dxr_77347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 44), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 586)
        getitem___77348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 44), dxr_77347, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 586)
        subscript_call_result_77349 = invoke(stypy.reporting.localization.Localization(__file__, 586, 44), getitem___77348, int_77346)
        
        
        # Obtaining the type of the subscript
        int_77350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 59), 'int')
        # Getting the type of 'slope' (line 586)
        slope_77351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 53), 'slope')
        # Obtaining the member '__getitem__' of a type (line 586)
        getitem___77352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 53), slope_77351, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 586)
        subscript_call_result_77353 = invoke(stypy.reporting.localization.Localization(__file__, 586, 53), getitem___77352, int_77350)
        
        # Applying the binary operator '*' (line 586)
        result_mul_77354 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 44), '*', subscript_call_result_77349, subscript_call_result_77353)
        
        # Applying the binary operator '+' (line 586)
        result_add_77355 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 24), '+', result_mul_77345, result_mul_77354)
        
        # Applying the binary operator '*' (line 586)
        result_mul_77356 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 19), '*', int_77336, result_add_77355)
        
        # Getting the type of 'b' (line 586)
        b_77357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'b')
        int_77358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 14), 'int')
        # Storing an element on a container (line 586)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 12), b_77357, (int_77358, result_mul_77356))
        
        # Assigning a BinOp to a Subscript (line 587):
        
        # Assigning a BinOp to a Subscript (line 587):
        int_77359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 19), 'int')
        
        # Obtaining the type of the subscript
        int_77360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 29), 'int')
        # Getting the type of 'slope' (line 587)
        slope_77361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 23), 'slope')
        # Obtaining the member '__getitem__' of a type (line 587)
        getitem___77362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 23), slope_77361, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 587)
        subscript_call_result_77363 = invoke(stypy.reporting.localization.Localization(__file__, 587, 23), getitem___77362, int_77360)
        
        # Applying the binary operator '*' (line 587)
        result_mul_77364 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 19), '*', int_77359, subscript_call_result_77363)
        
        # Getting the type of 'b' (line 587)
        b_77365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'b')
        int_77366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 14), 'int')
        # Storing an element on a container (line 587)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 12), b_77365, (int_77366, result_mul_77364))
        
        # Assigning a Call to a Name (line 589):
        
        # Assigning a Call to a Name (line 589):
        
        # Call to solve(...): (line 589)
        # Processing the call arguments (line 589)
        # Getting the type of 'A' (line 589)
        A_77368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 22), 'A', False)
        # Getting the type of 'b' (line 589)
        b_77369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 25), 'b', False)
        # Processing the call keyword arguments (line 589)
        # Getting the type of 'True' (line 589)
        True_77370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 40), 'True', False)
        keyword_77371 = True_77370
        # Getting the type of 'True' (line 589)
        True_77372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 58), 'True', False)
        keyword_77373 = True_77372
        # Getting the type of 'False' (line 590)
        False_77374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 35), 'False', False)
        keyword_77375 = False_77374
        kwargs_77376 = {'overwrite_a': keyword_77371, 'check_finite': keyword_77375, 'overwrite_b': keyword_77373}
        # Getting the type of 'solve' (line 589)
        solve_77367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'solve', False)
        # Calling solve(args, kwargs) (line 589)
        solve_call_result_77377 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), solve_77367, *[A_77368, b_77369], **kwargs_77376)
        
        # Assigning a type to the variable 's' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 's', solve_call_result_77377)
        # SSA branch for the else part of an if statement (line 573)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 594):
        
        # Assigning a Call to a Name (line 594):
        
        # Call to zeros(...): (line 594)
        # Processing the call arguments (line 594)
        
        # Obtaining an instance of the builtin type 'tuple' (line 594)
        tuple_77380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 594)
        # Adding element type (line 594)
        int_77381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 26), tuple_77380, int_77381)
        # Adding element type (line 594)
        # Getting the type of 'n' (line 594)
        n_77382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 29), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 26), tuple_77380, n_77382)
        
        # Processing the call keyword arguments (line 594)
        kwargs_77383 = {}
        # Getting the type of 'np' (line 594)
        np_77378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 16), 'np', False)
        # Obtaining the member 'zeros' of a type (line 594)
        zeros_77379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 16), np_77378, 'zeros')
        # Calling zeros(args, kwargs) (line 594)
        zeros_call_result_77384 = invoke(stypy.reporting.localization.Localization(__file__, 594, 16), zeros_77379, *[tuple_77380], **kwargs_77383)
        
        # Assigning a type to the variable 'A' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'A', zeros_call_result_77384)
        
        # Assigning a Call to a Name (line 595):
        
        # Assigning a Call to a Name (line 595):
        
        # Call to empty(...): (line 595)
        # Processing the call arguments (line 595)
        
        # Obtaining an instance of the builtin type 'tuple' (line 595)
        tuple_77387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 595)
        # Adding element type (line 595)
        # Getting the type of 'n' (line 595)
        n_77388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 26), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 26), tuple_77387, n_77388)
        
        
        # Obtaining the type of the subscript
        int_77389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 40), 'int')
        slice_77390 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 595, 32), int_77389, None, None)
        # Getting the type of 'y' (line 595)
        y_77391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 32), 'y', False)
        # Obtaining the member 'shape' of a type (line 595)
        shape_77392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 32), y_77391, 'shape')
        # Obtaining the member '__getitem__' of a type (line 595)
        getitem___77393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 32), shape_77392, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 595)
        subscript_call_result_77394 = invoke(stypy.reporting.localization.Localization(__file__, 595, 32), getitem___77393, slice_77390)
        
        # Applying the binary operator '+' (line 595)
        result_add_77395 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 25), '+', tuple_77387, subscript_call_result_77394)
        
        # Processing the call keyword arguments (line 595)
        # Getting the type of 'y' (line 595)
        y_77396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 51), 'y', False)
        # Obtaining the member 'dtype' of a type (line 595)
        dtype_77397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 51), y_77396, 'dtype')
        keyword_77398 = dtype_77397
        kwargs_77399 = {'dtype': keyword_77398}
        # Getting the type of 'np' (line 595)
        np_77385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 16), 'np', False)
        # Obtaining the member 'empty' of a type (line 595)
        empty_77386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 16), np_77385, 'empty')
        # Calling empty(args, kwargs) (line 595)
        empty_call_result_77400 = invoke(stypy.reporting.localization.Localization(__file__, 595, 16), empty_77386, *[result_add_77395], **kwargs_77399)
        
        # Assigning a type to the variable 'b' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'b', empty_call_result_77400)
        
        # Assigning a BinOp to a Subscript (line 604):
        
        # Assigning a BinOp to a Subscript (line 604):
        int_77401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 25), 'int')
        
        # Obtaining the type of the subscript
        int_77402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 34), 'int')
        slice_77403 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 604, 30), None, int_77402, None)
        # Getting the type of 'dx' (line 604)
        dx_77404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 30), 'dx')
        # Obtaining the member '__getitem__' of a type (line 604)
        getitem___77405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 30), dx_77404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 604)
        subscript_call_result_77406 = invoke(stypy.reporting.localization.Localization(__file__, 604, 30), getitem___77405, slice_77403)
        
        
        # Obtaining the type of the subscript
        int_77407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 43), 'int')
        slice_77408 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 604, 40), int_77407, None, None)
        # Getting the type of 'dx' (line 604)
        dx_77409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 40), 'dx')
        # Obtaining the member '__getitem__' of a type (line 604)
        getitem___77410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 40), dx_77409, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 604)
        subscript_call_result_77411 = invoke(stypy.reporting.localization.Localization(__file__, 604, 40), getitem___77410, slice_77408)
        
        # Applying the binary operator '+' (line 604)
        result_add_77412 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 30), '+', subscript_call_result_77406, subscript_call_result_77411)
        
        # Applying the binary operator '*' (line 604)
        result_mul_77413 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 25), '*', int_77401, result_add_77412)
        
        # Getting the type of 'A' (line 604)
        A_77414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'A')
        int_77415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 14), 'int')
        int_77416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 17), 'int')
        int_77417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 19), 'int')
        slice_77418 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 604, 12), int_77416, int_77417, None)
        # Storing an element on a container (line 604)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 12), A_77414, ((int_77415, slice_77418), result_mul_77413))
        
        # Assigning a Subscript to a Subscript (line 605):
        
        # Assigning a Subscript to a Subscript (line 605):
        
        # Obtaining the type of the subscript
        int_77419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 27), 'int')
        slice_77420 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 605, 23), None, int_77419, None)
        # Getting the type of 'dx' (line 605)
        dx_77421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 23), 'dx')
        # Obtaining the member '__getitem__' of a type (line 605)
        getitem___77422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 23), dx_77421, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 605)
        subscript_call_result_77423 = invoke(stypy.reporting.localization.Localization(__file__, 605, 23), getitem___77422, slice_77420)
        
        # Getting the type of 'A' (line 605)
        A_77424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'A')
        int_77425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 14), 'int')
        int_77426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 17), 'int')
        slice_77427 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 605, 12), int_77426, None, None)
        # Storing an element on a container (line 605)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 12), A_77424, ((int_77425, slice_77427), subscript_call_result_77423))
        
        # Assigning a Subscript to a Subscript (line 606):
        
        # Assigning a Subscript to a Subscript (line 606):
        
        # Obtaining the type of the subscript
        int_77428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 28), 'int')
        slice_77429 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 606, 25), int_77428, None, None)
        # Getting the type of 'dx' (line 606)
        dx_77430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 25), 'dx')
        # Obtaining the member '__getitem__' of a type (line 606)
        getitem___77431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 25), dx_77430, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 606)
        subscript_call_result_77432 = invoke(stypy.reporting.localization.Localization(__file__, 606, 25), getitem___77431, slice_77429)
        
        # Getting the type of 'A' (line 606)
        A_77433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'A')
        int_77434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 14), 'int')
        int_77435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 19), 'int')
        slice_77436 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 606, 12), None, int_77435, None)
        # Storing an element on a container (line 606)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 12), A_77433, ((int_77434, slice_77436), subscript_call_result_77432))
        
        # Assigning a BinOp to a Subscript (line 608):
        
        # Assigning a BinOp to a Subscript (line 608):
        int_77437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 22), 'int')
        
        # Obtaining the type of the subscript
        int_77438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 31), 'int')
        slice_77439 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 608, 27), int_77438, None, None)
        # Getting the type of 'dxr' (line 608)
        dxr_77440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 27), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 608)
        getitem___77441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 27), dxr_77440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 608)
        subscript_call_result_77442 = invoke(stypy.reporting.localization.Localization(__file__, 608, 27), getitem___77441, slice_77439)
        
        
        # Obtaining the type of the subscript
        int_77443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 44), 'int')
        slice_77444 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 608, 37), None, int_77443, None)
        # Getting the type of 'slope' (line 608)
        slope_77445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 37), 'slope')
        # Obtaining the member '__getitem__' of a type (line 608)
        getitem___77446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 37), slope_77445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 608)
        subscript_call_result_77447 = invoke(stypy.reporting.localization.Localization(__file__, 608, 37), getitem___77446, slice_77444)
        
        # Applying the binary operator '*' (line 608)
        result_mul_77448 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 27), '*', subscript_call_result_77442, subscript_call_result_77447)
        
        
        # Obtaining the type of the subscript
        int_77449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 55), 'int')
        slice_77450 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 608, 50), None, int_77449, None)
        # Getting the type of 'dxr' (line 608)
        dxr_77451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 50), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 608)
        getitem___77452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 50), dxr_77451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 608)
        subscript_call_result_77453 = invoke(stypy.reporting.localization.Localization(__file__, 608, 50), getitem___77452, slice_77450)
        
        
        # Obtaining the type of the subscript
        int_77454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 67), 'int')
        slice_77455 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 608, 61), int_77454, None, None)
        # Getting the type of 'slope' (line 608)
        slope_77456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 61), 'slope')
        # Obtaining the member '__getitem__' of a type (line 608)
        getitem___77457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 61), slope_77456, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 608)
        subscript_call_result_77458 = invoke(stypy.reporting.localization.Localization(__file__, 608, 61), getitem___77457, slice_77455)
        
        # Applying the binary operator '*' (line 608)
        result_mul_77459 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 50), '*', subscript_call_result_77453, subscript_call_result_77458)
        
        # Applying the binary operator '+' (line 608)
        result_add_77460 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 27), '+', result_mul_77448, result_mul_77459)
        
        # Applying the binary operator '*' (line 608)
        result_mul_77461 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 22), '*', int_77437, result_add_77460)
        
        # Getting the type of 'b' (line 608)
        b_77462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'b')
        int_77463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 14), 'int')
        int_77464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 16), 'int')
        slice_77465 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 608, 12), int_77463, int_77464, None)
        # Storing an element on a container (line 608)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 12), b_77462, (slice_77465, result_mul_77461))
        
        # Assigning a Name to a Tuple (line 610):
        
        # Assigning a Subscript to a Name (line 610):
        
        # Obtaining the type of the subscript
        int_77466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 12), 'int')
        # Getting the type of 'bc' (line 610)
        bc_77467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 31), 'bc')
        # Obtaining the member '__getitem__' of a type (line 610)
        getitem___77468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 12), bc_77467, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 610)
        subscript_call_result_77469 = invoke(stypy.reporting.localization.Localization(__file__, 610, 12), getitem___77468, int_77466)
        
        # Assigning a type to the variable 'tuple_var_assignment_76088' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'tuple_var_assignment_76088', subscript_call_result_77469)
        
        # Assigning a Subscript to a Name (line 610):
        
        # Obtaining the type of the subscript
        int_77470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 12), 'int')
        # Getting the type of 'bc' (line 610)
        bc_77471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 31), 'bc')
        # Obtaining the member '__getitem__' of a type (line 610)
        getitem___77472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 12), bc_77471, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 610)
        subscript_call_result_77473 = invoke(stypy.reporting.localization.Localization(__file__, 610, 12), getitem___77472, int_77470)
        
        # Assigning a type to the variable 'tuple_var_assignment_76089' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'tuple_var_assignment_76089', subscript_call_result_77473)
        
        # Assigning a Name to a Name (line 610):
        # Getting the type of 'tuple_var_assignment_76088' (line 610)
        tuple_var_assignment_76088_77474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'tuple_var_assignment_76088')
        # Assigning a type to the variable 'bc_start' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'bc_start', tuple_var_assignment_76088_77474)
        
        # Assigning a Name to a Name (line 610):
        # Getting the type of 'tuple_var_assignment_76089' (line 610)
        tuple_var_assignment_76089_77475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'tuple_var_assignment_76089')
        # Assigning a type to the variable 'bc_end' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 22), 'bc_end', tuple_var_assignment_76089_77475)
        
        
        # Getting the type of 'bc_start' (line 612)
        bc_start_77476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 15), 'bc_start')
        str_77477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 27), 'str', 'periodic')
        # Applying the binary operator '==' (line 612)
        result_eq_77478 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 15), '==', bc_start_77476, str_77477)
        
        # Testing the type of an if condition (line 612)
        if_condition_77479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 12), result_eq_77478)
        # Assigning a type to the variable 'if_condition_77479' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'if_condition_77479', if_condition_77479)
        # SSA begins for if statement (line 612)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 615):
        
        # Assigning a Subscript to a Name (line 615):
        
        # Obtaining the type of the subscript
        slice_77480 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 615, 20), None, None, None)
        int_77481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 25), 'int')
        int_77482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 27), 'int')
        slice_77483 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 615, 20), int_77481, int_77482, None)
        # Getting the type of 'A' (line 615)
        A_77484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 20), 'A')
        # Obtaining the member '__getitem__' of a type (line 615)
        getitem___77485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 20), A_77484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 615)
        subscript_call_result_77486 = invoke(stypy.reporting.localization.Localization(__file__, 615, 20), getitem___77485, (slice_77480, slice_77483))
        
        # Assigning a type to the variable 'A' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 16), 'A', subscript_call_result_77486)
        
        # Assigning a BinOp to a Subscript (line 616):
        
        # Assigning a BinOp to a Subscript (line 616):
        int_77487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 26), 'int')
        
        # Obtaining the type of the subscript
        int_77488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 34), 'int')
        # Getting the type of 'dx' (line 616)
        dx_77489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 31), 'dx')
        # Obtaining the member '__getitem__' of a type (line 616)
        getitem___77490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 31), dx_77489, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 616)
        subscript_call_result_77491 = invoke(stypy.reporting.localization.Localization(__file__, 616, 31), getitem___77490, int_77488)
        
        
        # Obtaining the type of the subscript
        int_77492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 43), 'int')
        # Getting the type of 'dx' (line 616)
        dx_77493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 40), 'dx')
        # Obtaining the member '__getitem__' of a type (line 616)
        getitem___77494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 40), dx_77493, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 616)
        subscript_call_result_77495 = invoke(stypy.reporting.localization.Localization(__file__, 616, 40), getitem___77494, int_77492)
        
        # Applying the binary operator '+' (line 616)
        result_add_77496 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 31), '+', subscript_call_result_77491, subscript_call_result_77495)
        
        # Applying the binary operator '*' (line 616)
        result_mul_77497 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 26), '*', int_77487, result_add_77496)
        
        # Getting the type of 'A' (line 616)
        A_77498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 616)
        tuple_77499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 616)
        # Adding element type (line 616)
        int_77500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 18), tuple_77499, int_77500)
        # Adding element type (line 616)
        int_77501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 18), tuple_77499, int_77501)
        
        # Storing an element on a container (line 616)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 16), A_77498, (tuple_77499, result_mul_77497))
        
        # Assigning a Subscript to a Subscript (line 617):
        
        # Assigning a Subscript to a Subscript (line 617):
        
        # Obtaining the type of the subscript
        int_77502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 29), 'int')
        # Getting the type of 'dx' (line 617)
        dx_77503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 26), 'dx')
        # Obtaining the member '__getitem__' of a type (line 617)
        getitem___77504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 26), dx_77503, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 617)
        subscript_call_result_77505 = invoke(stypy.reporting.localization.Localization(__file__, 617, 26), getitem___77504, int_77502)
        
        # Getting the type of 'A' (line 617)
        A_77506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 617)
        tuple_77507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 617)
        # Adding element type (line 617)
        int_77508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 18), tuple_77507, int_77508)
        # Adding element type (line 617)
        int_77509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 18), tuple_77507, int_77509)
        
        # Storing an element on a container (line 617)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 16), A_77506, (tuple_77507, subscript_call_result_77505))
        
        # Assigning a Subscript to a Name (line 619):
        
        # Assigning a Subscript to a Name (line 619):
        
        # Obtaining the type of the subscript
        int_77510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 23), 'int')
        slice_77511 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 619, 20), None, int_77510, None)
        # Getting the type of 'b' (line 619)
        b_77512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'b')
        # Obtaining the member '__getitem__' of a type (line 619)
        getitem___77513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 20), b_77512, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 619)
        subscript_call_result_77514 = invoke(stypy.reporting.localization.Localization(__file__, 619, 20), getitem___77513, slice_77511)
        
        # Assigning a type to the variable 'b' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'b', subscript_call_result_77514)
        
        # Assigning a Subscript to a Name (line 629):
        
        # Assigning a Subscript to a Name (line 629):
        
        # Obtaining the type of the subscript
        int_77515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 28), 'int')
        # Getting the type of 'dx' (line 629)
        dx_77516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 25), 'dx')
        # Obtaining the member '__getitem__' of a type (line 629)
        getitem___77517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 25), dx_77516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 629)
        subscript_call_result_77518 = invoke(stypy.reporting.localization.Localization(__file__, 629, 25), getitem___77517, int_77515)
        
        # Assigning a type to the variable 'a_m1_0' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 16), 'a_m1_0', subscript_call_result_77518)
        
        # Assigning a Subscript to a Name (line 630):
        
        # Assigning a Subscript to a Name (line 630):
        
        # Obtaining the type of the subscript
        int_77519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 29), 'int')
        # Getting the type of 'dx' (line 630)
        dx_77520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 26), 'dx')
        # Obtaining the member '__getitem__' of a type (line 630)
        getitem___77521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 26), dx_77520, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 630)
        subscript_call_result_77522 = invoke(stypy.reporting.localization.Localization(__file__, 630, 26), getitem___77521, int_77519)
        
        # Assigning a type to the variable 'a_m1_m2' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 16), 'a_m1_m2', subscript_call_result_77522)
        
        # Assigning a BinOp to a Name (line 631):
        
        # Assigning a BinOp to a Name (line 631):
        int_77523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 26), 'int')
        
        # Obtaining the type of the subscript
        int_77524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 34), 'int')
        # Getting the type of 'dx' (line 631)
        dx_77525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 31), 'dx')
        # Obtaining the member '__getitem__' of a type (line 631)
        getitem___77526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 31), dx_77525, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 631)
        subscript_call_result_77527 = invoke(stypy.reporting.localization.Localization(__file__, 631, 31), getitem___77526, int_77524)
        
        
        # Obtaining the type of the subscript
        int_77528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 43), 'int')
        # Getting the type of 'dx' (line 631)
        dx_77529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 40), 'dx')
        # Obtaining the member '__getitem__' of a type (line 631)
        getitem___77530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 40), dx_77529, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 631)
        subscript_call_result_77531 = invoke(stypy.reporting.localization.Localization(__file__, 631, 40), getitem___77530, int_77528)
        
        # Applying the binary operator '+' (line 631)
        result_add_77532 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 31), '+', subscript_call_result_77527, subscript_call_result_77531)
        
        # Applying the binary operator '*' (line 631)
        result_mul_77533 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 26), '*', int_77523, result_add_77532)
        
        # Assigning a type to the variable 'a_m1_m1' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'a_m1_m1', result_mul_77533)
        
        # Assigning a Subscript to a Name (line 632):
        
        # Assigning a Subscript to a Name (line 632):
        
        # Obtaining the type of the subscript
        int_77534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 29), 'int')
        # Getting the type of 'dx' (line 632)
        dx_77535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 26), 'dx')
        # Obtaining the member '__getitem__' of a type (line 632)
        getitem___77536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 26), dx_77535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 632)
        subscript_call_result_77537 = invoke(stypy.reporting.localization.Localization(__file__, 632, 26), getitem___77536, int_77534)
        
        # Assigning a type to the variable 'a_m2_m1' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 'a_m2_m1', subscript_call_result_77537)
        
        # Assigning a Subscript to a Name (line 633):
        
        # Assigning a Subscript to a Name (line 633):
        
        # Obtaining the type of the subscript
        int_77538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 28), 'int')
        # Getting the type of 'dx' (line 633)
        dx_77539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 25), 'dx')
        # Obtaining the member '__getitem__' of a type (line 633)
        getitem___77540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 25), dx_77539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 633)
        subscript_call_result_77541 = invoke(stypy.reporting.localization.Localization(__file__, 633, 25), getitem___77540, int_77538)
        
        # Assigning a type to the variable 'a_0_m1' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'a_0_m1', subscript_call_result_77541)
        
        # Assigning a BinOp to a Subscript (line 635):
        
        # Assigning a BinOp to a Subscript (line 635):
        int_77542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 23), 'int')
        
        # Obtaining the type of the subscript
        int_77543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 32), 'int')
        # Getting the type of 'dxr' (line 635)
        dxr_77544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 28), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 635)
        getitem___77545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 28), dxr_77544, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 635)
        subscript_call_result_77546 = invoke(stypy.reporting.localization.Localization(__file__, 635, 28), getitem___77545, int_77543)
        
        
        # Obtaining the type of the subscript
        int_77547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 43), 'int')
        # Getting the type of 'slope' (line 635)
        slope_77548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 37), 'slope')
        # Obtaining the member '__getitem__' of a type (line 635)
        getitem___77549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 37), slope_77548, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 635)
        subscript_call_result_77550 = invoke(stypy.reporting.localization.Localization(__file__, 635, 37), getitem___77549, int_77547)
        
        # Applying the binary operator '*' (line 635)
        result_mul_77551 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 28), '*', subscript_call_result_77546, subscript_call_result_77550)
        
        
        # Obtaining the type of the subscript
        int_77552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 53), 'int')
        # Getting the type of 'dxr' (line 635)
        dxr_77553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 49), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 635)
        getitem___77554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 49), dxr_77553, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 635)
        subscript_call_result_77555 = invoke(stypy.reporting.localization.Localization(__file__, 635, 49), getitem___77554, int_77552)
        
        
        # Obtaining the type of the subscript
        int_77556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 65), 'int')
        # Getting the type of 'slope' (line 635)
        slope_77557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 59), 'slope')
        # Obtaining the member '__getitem__' of a type (line 635)
        getitem___77558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 59), slope_77557, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 635)
        subscript_call_result_77559 = invoke(stypy.reporting.localization.Localization(__file__, 635, 59), getitem___77558, int_77556)
        
        # Applying the binary operator '*' (line 635)
        result_mul_77560 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 49), '*', subscript_call_result_77555, subscript_call_result_77559)
        
        # Applying the binary operator '+' (line 635)
        result_add_77561 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 28), '+', result_mul_77551, result_mul_77560)
        
        # Applying the binary operator '*' (line 635)
        result_mul_77562 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 23), '*', int_77542, result_add_77561)
        
        # Getting the type of 'b' (line 635)
        b_77563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 16), 'b')
        int_77564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 18), 'int')
        # Storing an element on a container (line 635)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 16), b_77563, (int_77564, result_mul_77562))
        
        # Assigning a BinOp to a Subscript (line 636):
        
        # Assigning a BinOp to a Subscript (line 636):
        int_77565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 24), 'int')
        
        # Obtaining the type of the subscript
        int_77566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 33), 'int')
        # Getting the type of 'dxr' (line 636)
        dxr_77567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 29), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___77568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 29), dxr_77567, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_77569 = invoke(stypy.reporting.localization.Localization(__file__, 636, 29), getitem___77568, int_77566)
        
        
        # Obtaining the type of the subscript
        int_77570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 45), 'int')
        # Getting the type of 'slope' (line 636)
        slope_77571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 39), 'slope')
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___77572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 39), slope_77571, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_77573 = invoke(stypy.reporting.localization.Localization(__file__, 636, 39), getitem___77572, int_77570)
        
        # Applying the binary operator '*' (line 636)
        result_mul_77574 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 29), '*', subscript_call_result_77569, subscript_call_result_77573)
        
        
        # Obtaining the type of the subscript
        int_77575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 55), 'int')
        # Getting the type of 'dxr' (line 636)
        dxr_77576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 51), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___77577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 51), dxr_77576, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_77578 = invoke(stypy.reporting.localization.Localization(__file__, 636, 51), getitem___77577, int_77575)
        
        
        # Obtaining the type of the subscript
        int_77579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 67), 'int')
        # Getting the type of 'slope' (line 636)
        slope_77580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 61), 'slope')
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___77581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 61), slope_77580, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_77582 = invoke(stypy.reporting.localization.Localization(__file__, 636, 61), getitem___77581, int_77579)
        
        # Applying the binary operator '*' (line 636)
        result_mul_77583 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 51), '*', subscript_call_result_77578, subscript_call_result_77582)
        
        # Applying the binary operator '+' (line 636)
        result_add_77584 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 29), '+', result_mul_77574, result_mul_77583)
        
        # Applying the binary operator '*' (line 636)
        result_mul_77585 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 24), '*', int_77565, result_add_77584)
        
        # Getting the type of 'b' (line 636)
        b_77586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'b')
        int_77587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 18), 'int')
        # Storing an element on a container (line 636)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 16), b_77586, (int_77587, result_mul_77585))
        
        # Assigning a Subscript to a Name (line 638):
        
        # Assigning a Subscript to a Name (line 638):
        
        # Obtaining the type of the subscript
        slice_77588 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 638, 21), None, None, None)
        int_77589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 27), 'int')
        slice_77590 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 638, 21), None, int_77589, None)
        # Getting the type of 'A' (line 638)
        A_77591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 21), 'A')
        # Obtaining the member '__getitem__' of a type (line 638)
        getitem___77592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 21), A_77591, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 638)
        subscript_call_result_77593 = invoke(stypy.reporting.localization.Localization(__file__, 638, 21), getitem___77592, (slice_77588, slice_77590))
        
        # Assigning a type to the variable 'Ac' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 16), 'Ac', subscript_call_result_77593)
        
        # Assigning a Subscript to a Name (line 639):
        
        # Assigning a Subscript to a Name (line 639):
        
        # Obtaining the type of the subscript
        int_77594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 24), 'int')
        slice_77595 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 639, 21), None, int_77594, None)
        # Getting the type of 'b' (line 639)
        b_77596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 21), 'b')
        # Obtaining the member '__getitem__' of a type (line 639)
        getitem___77597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 21), b_77596, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 639)
        subscript_call_result_77598 = invoke(stypy.reporting.localization.Localization(__file__, 639, 21), getitem___77597, slice_77595)
        
        # Assigning a type to the variable 'b1' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 16), 'b1', subscript_call_result_77598)
        
        # Assigning a Call to a Name (line 640):
        
        # Assigning a Call to a Name (line 640):
        
        # Call to zeros_like(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'b1' (line 640)
        b1_77601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 35), 'b1', False)
        # Processing the call keyword arguments (line 640)
        kwargs_77602 = {}
        # Getting the type of 'np' (line 640)
        np_77599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 21), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 640)
        zeros_like_77600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 21), np_77599, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 640)
        zeros_like_call_result_77603 = invoke(stypy.reporting.localization.Localization(__file__, 640, 21), zeros_like_77600, *[b1_77601], **kwargs_77602)
        
        # Assigning a type to the variable 'b2' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'b2', zeros_like_call_result_77603)
        
        # Assigning a UnaryOp to a Subscript (line 641):
        
        # Assigning a UnaryOp to a Subscript (line 641):
        
        # Getting the type of 'a_0_m1' (line 641)
        a_0_m1_77604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 25), 'a_0_m1')
        # Applying the 'usub' unary operator (line 641)
        result___neg___77605 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 24), 'usub', a_0_m1_77604)
        
        # Getting the type of 'b2' (line 641)
        b2_77606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 16), 'b2')
        int_77607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 19), 'int')
        # Storing an element on a container (line 641)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 16), b2_77606, (int_77607, result___neg___77605))
        
        # Assigning a UnaryOp to a Subscript (line 642):
        
        # Assigning a UnaryOp to a Subscript (line 642):
        
        # Getting the type of 'a_m2_m1' (line 642)
        a_m2_m1_77608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 26), 'a_m2_m1')
        # Applying the 'usub' unary operator (line 642)
        result___neg___77609 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 25), 'usub', a_m2_m1_77608)
        
        # Getting the type of 'b2' (line 642)
        b2_77610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 16), 'b2')
        int_77611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 19), 'int')
        # Storing an element on a container (line 642)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 16), b2_77610, (int_77611, result___neg___77609))
        
        # Assigning a Call to a Name (line 645):
        
        # Assigning a Call to a Name (line 645):
        
        # Call to solve_banded(...): (line 645)
        # Processing the call arguments (line 645)
        
        # Obtaining an instance of the builtin type 'tuple' (line 645)
        tuple_77613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 645)
        # Adding element type (line 645)
        int_77614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 35), tuple_77613, int_77614)
        # Adding element type (line 645)
        int_77615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 35), tuple_77613, int_77615)
        
        # Getting the type of 'Ac' (line 645)
        Ac_77616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 42), 'Ac', False)
        # Getting the type of 'b1' (line 645)
        b1_77617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 46), 'b1', False)
        # Processing the call keyword arguments (line 645)
        # Getting the type of 'False' (line 645)
        False_77618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 63), 'False', False)
        keyword_77619 = False_77618
        # Getting the type of 'False' (line 646)
        False_77620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 46), 'False', False)
        keyword_77621 = False_77620
        # Getting the type of 'False' (line 646)
        False_77622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 66), 'False', False)
        keyword_77623 = False_77622
        kwargs_77624 = {'overwrite_ab': keyword_77619, 'check_finite': keyword_77623, 'overwrite_b': keyword_77621}
        # Getting the type of 'solve_banded' (line 645)
        solve_banded_77612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 21), 'solve_banded', False)
        # Calling solve_banded(args, kwargs) (line 645)
        solve_banded_call_result_77625 = invoke(stypy.reporting.localization.Localization(__file__, 645, 21), solve_banded_77612, *[tuple_77613, Ac_77616, b1_77617], **kwargs_77624)
        
        # Assigning a type to the variable 's1' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 16), 's1', solve_banded_call_result_77625)
        
        # Assigning a Call to a Name (line 648):
        
        # Assigning a Call to a Name (line 648):
        
        # Call to solve_banded(...): (line 648)
        # Processing the call arguments (line 648)
        
        # Obtaining an instance of the builtin type 'tuple' (line 648)
        tuple_77627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 648)
        # Adding element type (line 648)
        int_77628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 648, 35), tuple_77627, int_77628)
        # Adding element type (line 648)
        int_77629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 648, 35), tuple_77627, int_77629)
        
        # Getting the type of 'Ac' (line 648)
        Ac_77630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 42), 'Ac', False)
        # Getting the type of 'b2' (line 648)
        b2_77631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 46), 'b2', False)
        # Processing the call keyword arguments (line 648)
        # Getting the type of 'False' (line 648)
        False_77632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 63), 'False', False)
        keyword_77633 = False_77632
        # Getting the type of 'False' (line 649)
        False_77634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), 'False', False)
        keyword_77635 = False_77634
        # Getting the type of 'False' (line 649)
        False_77636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 66), 'False', False)
        keyword_77637 = False_77636
        kwargs_77638 = {'overwrite_ab': keyword_77633, 'check_finite': keyword_77637, 'overwrite_b': keyword_77635}
        # Getting the type of 'solve_banded' (line 648)
        solve_banded_77626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 21), 'solve_banded', False)
        # Calling solve_banded(args, kwargs) (line 648)
        solve_banded_call_result_77639 = invoke(stypy.reporting.localization.Localization(__file__, 648, 21), solve_banded_77626, *[tuple_77627, Ac_77630, b2_77631], **kwargs_77638)
        
        # Assigning a type to the variable 's2' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 's2', solve_banded_call_result_77639)
        
        # Assigning a BinOp to a Name (line 652):
        
        # Assigning a BinOp to a Name (line 652):
        
        # Obtaining the type of the subscript
        int_77640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 27), 'int')
        # Getting the type of 'b' (line 652)
        b_77641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 25), 'b')
        # Obtaining the member '__getitem__' of a type (line 652)
        getitem___77642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 25), b_77641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 652)
        subscript_call_result_77643 = invoke(stypy.reporting.localization.Localization(__file__, 652, 25), getitem___77642, int_77640)
        
        # Getting the type of 'a_m1_0' (line 652)
        a_m1_0_77644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 33), 'a_m1_0')
        
        # Obtaining the type of the subscript
        int_77645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 45), 'int')
        # Getting the type of 's1' (line 652)
        s1_77646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 42), 's1')
        # Obtaining the member '__getitem__' of a type (line 652)
        getitem___77647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 42), s1_77646, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 652)
        subscript_call_result_77648 = invoke(stypy.reporting.localization.Localization(__file__, 652, 42), getitem___77647, int_77645)
        
        # Applying the binary operator '*' (line 652)
        result_mul_77649 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 33), '*', a_m1_0_77644, subscript_call_result_77648)
        
        # Applying the binary operator '-' (line 652)
        result_sub_77650 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 25), '-', subscript_call_result_77643, result_mul_77649)
        
        # Getting the type of 'a_m1_m2' (line 652)
        a_m1_m2_77651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 50), 'a_m1_m2')
        
        # Obtaining the type of the subscript
        int_77652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 63), 'int')
        # Getting the type of 's1' (line 652)
        s1_77653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 60), 's1')
        # Obtaining the member '__getitem__' of a type (line 652)
        getitem___77654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 60), s1_77653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 652)
        subscript_call_result_77655 = invoke(stypy.reporting.localization.Localization(__file__, 652, 60), getitem___77654, int_77652)
        
        # Applying the binary operator '*' (line 652)
        result_mul_77656 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 50), '*', a_m1_m2_77651, subscript_call_result_77655)
        
        # Applying the binary operator '-' (line 652)
        result_sub_77657 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 48), '-', result_sub_77650, result_mul_77656)
        
        # Getting the type of 'a_m1_m1' (line 653)
        a_m1_m1_77658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 25), 'a_m1_m1')
        # Getting the type of 'a_m1_0' (line 653)
        a_m1_0_77659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 35), 'a_m1_0')
        
        # Obtaining the type of the subscript
        int_77660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 47), 'int')
        # Getting the type of 's2' (line 653)
        s2_77661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 44), 's2')
        # Obtaining the member '__getitem__' of a type (line 653)
        getitem___77662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 44), s2_77661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 653)
        subscript_call_result_77663 = invoke(stypy.reporting.localization.Localization(__file__, 653, 44), getitem___77662, int_77660)
        
        # Applying the binary operator '*' (line 653)
        result_mul_77664 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 35), '*', a_m1_0_77659, subscript_call_result_77663)
        
        # Applying the binary operator '+' (line 653)
        result_add_77665 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 25), '+', a_m1_m1_77658, result_mul_77664)
        
        # Getting the type of 'a_m1_m2' (line 653)
        a_m1_m2_77666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 52), 'a_m1_m2')
        
        # Obtaining the type of the subscript
        int_77667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 65), 'int')
        # Getting the type of 's2' (line 653)
        s2_77668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 62), 's2')
        # Obtaining the member '__getitem__' of a type (line 653)
        getitem___77669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 62), s2_77668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 653)
        subscript_call_result_77670 = invoke(stypy.reporting.localization.Localization(__file__, 653, 62), getitem___77669, int_77667)
        
        # Applying the binary operator '*' (line 653)
        result_mul_77671 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 52), '*', a_m1_m2_77666, subscript_call_result_77670)
        
        # Applying the binary operator '+' (line 653)
        result_add_77672 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 50), '+', result_add_77665, result_mul_77671)
        
        # Applying the binary operator 'div' (line 652)
        result_div_77673 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 24), 'div', result_sub_77657, result_add_77672)
        
        # Assigning a type to the variable 's_m1' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 16), 's_m1', result_div_77673)
        
        # Assigning a Call to a Name (line 656):
        
        # Assigning a Call to a Name (line 656):
        
        # Call to empty(...): (line 656)
        # Processing the call arguments (line 656)
        
        # Obtaining an instance of the builtin type 'tuple' (line 656)
        tuple_77676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 656)
        # Adding element type (line 656)
        # Getting the type of 'n' (line 656)
        n_77677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 30), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 30), tuple_77676, n_77677)
        
        
        # Obtaining the type of the subscript
        int_77678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 44), 'int')
        slice_77679 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 656, 36), int_77678, None, None)
        # Getting the type of 'y' (line 656)
        y_77680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 36), 'y', False)
        # Obtaining the member 'shape' of a type (line 656)
        shape_77681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 36), y_77680, 'shape')
        # Obtaining the member '__getitem__' of a type (line 656)
        getitem___77682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 36), shape_77681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 656)
        subscript_call_result_77683 = invoke(stypy.reporting.localization.Localization(__file__, 656, 36), getitem___77682, slice_77679)
        
        # Applying the binary operator '+' (line 656)
        result_add_77684 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 29), '+', tuple_77676, subscript_call_result_77683)
        
        # Processing the call keyword arguments (line 656)
        # Getting the type of 'y' (line 656)
        y_77685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 55), 'y', False)
        # Obtaining the member 'dtype' of a type (line 656)
        dtype_77686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 55), y_77685, 'dtype')
        keyword_77687 = dtype_77686
        kwargs_77688 = {'dtype': keyword_77687}
        # Getting the type of 'np' (line 656)
        np_77674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 20), 'np', False)
        # Obtaining the member 'empty' of a type (line 656)
        empty_77675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 20), np_77674, 'empty')
        # Calling empty(args, kwargs) (line 656)
        empty_call_result_77689 = invoke(stypy.reporting.localization.Localization(__file__, 656, 20), empty_77675, *[result_add_77684], **kwargs_77688)
        
        # Assigning a type to the variable 's' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 's', empty_call_result_77689)
        
        # Assigning a BinOp to a Subscript (line 657):
        
        # Assigning a BinOp to a Subscript (line 657):
        # Getting the type of 's1' (line 657)
        s1_77690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 25), 's1')
        # Getting the type of 's_m1' (line 657)
        s_m1_77691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 30), 's_m1')
        # Getting the type of 's2' (line 657)
        s2_77692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 37), 's2')
        # Applying the binary operator '*' (line 657)
        result_mul_77693 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 30), '*', s_m1_77691, s2_77692)
        
        # Applying the binary operator '+' (line 657)
        result_add_77694 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 25), '+', s1_77690, result_mul_77693)
        
        # Getting the type of 's' (line 657)
        s_77695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 's')
        int_77696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 19), 'int')
        slice_77697 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 657, 16), None, int_77696, None)
        # Storing an element on a container (line 657)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 16), s_77695, (slice_77697, result_add_77694))
        
        # Assigning a Name to a Subscript (line 658):
        
        # Assigning a Name to a Subscript (line 658):
        # Getting the type of 's_m1' (line 658)
        s_m1_77698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 24), 's_m1')
        # Getting the type of 's' (line 658)
        s_77699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 's')
        int_77700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 18), 'int')
        # Storing an element on a container (line 658)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 16), s_77699, (int_77700, s_m1_77698))
        
        # Assigning a Subscript to a Subscript (line 659):
        
        # Assigning a Subscript to a Subscript (line 659):
        
        # Obtaining the type of the subscript
        int_77701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 26), 'int')
        # Getting the type of 's' (line 659)
        s_77702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 24), 's')
        # Obtaining the member '__getitem__' of a type (line 659)
        getitem___77703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 24), s_77702, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 659)
        subscript_call_result_77704 = invoke(stypy.reporting.localization.Localization(__file__, 659, 24), getitem___77703, int_77701)
        
        # Getting the type of 's' (line 659)
        s_77705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 's')
        int_77706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 18), 'int')
        # Storing an element on a container (line 659)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 16), s_77705, (int_77706, subscript_call_result_77704))
        # SSA branch for the else part of an if statement (line 612)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'bc_start' (line 661)
        bc_start_77707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 19), 'bc_start')
        str_77708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 31), 'str', 'not-a-knot')
        # Applying the binary operator '==' (line 661)
        result_eq_77709 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 19), '==', bc_start_77707, str_77708)
        
        # Testing the type of an if condition (line 661)
        if_condition_77710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 16), result_eq_77709)
        # Assigning a type to the variable 'if_condition_77710' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'if_condition_77710', if_condition_77710)
        # SSA begins for if statement (line 661)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 662):
        
        # Assigning a Subscript to a Subscript (line 662):
        
        # Obtaining the type of the subscript
        int_77711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 33), 'int')
        # Getting the type of 'dx' (line 662)
        dx_77712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 30), 'dx')
        # Obtaining the member '__getitem__' of a type (line 662)
        getitem___77713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 30), dx_77712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 662)
        subscript_call_result_77714 = invoke(stypy.reporting.localization.Localization(__file__, 662, 30), getitem___77713, int_77711)
        
        # Getting the type of 'A' (line 662)
        A_77715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 662)
        tuple_77716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 662)
        # Adding element type (line 662)
        int_77717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 22), tuple_77716, int_77717)
        # Adding element type (line 662)
        int_77718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 22), tuple_77716, int_77718)
        
        # Storing an element on a container (line 662)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 20), A_77715, (tuple_77716, subscript_call_result_77714))
        
        # Assigning a BinOp to a Subscript (line 663):
        
        # Assigning a BinOp to a Subscript (line 663):
        
        # Obtaining the type of the subscript
        int_77719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 32), 'int')
        # Getting the type of 'x' (line 663)
        x_77720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 30), 'x')
        # Obtaining the member '__getitem__' of a type (line 663)
        getitem___77721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 30), x_77720, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 663)
        subscript_call_result_77722 = invoke(stypy.reporting.localization.Localization(__file__, 663, 30), getitem___77721, int_77719)
        
        
        # Obtaining the type of the subscript
        int_77723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 39), 'int')
        # Getting the type of 'x' (line 663)
        x_77724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 37), 'x')
        # Obtaining the member '__getitem__' of a type (line 663)
        getitem___77725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), x_77724, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 663)
        subscript_call_result_77726 = invoke(stypy.reporting.localization.Localization(__file__, 663, 37), getitem___77725, int_77723)
        
        # Applying the binary operator '-' (line 663)
        result_sub_77727 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 30), '-', subscript_call_result_77722, subscript_call_result_77726)
        
        # Getting the type of 'A' (line 663)
        A_77728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 663)
        tuple_77729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 663)
        # Adding element type (line 663)
        int_77730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 22), tuple_77729, int_77730)
        # Adding element type (line 663)
        int_77731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 22), tuple_77729, int_77731)
        
        # Storing an element on a container (line 663)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 20), A_77728, (tuple_77729, result_sub_77727))
        
        # Assigning a BinOp to a Name (line 664):
        
        # Assigning a BinOp to a Name (line 664):
        
        # Obtaining the type of the subscript
        int_77732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 26), 'int')
        # Getting the type of 'x' (line 664)
        x_77733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 24), 'x')
        # Obtaining the member '__getitem__' of a type (line 664)
        getitem___77734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 24), x_77733, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 664)
        subscript_call_result_77735 = invoke(stypy.reporting.localization.Localization(__file__, 664, 24), getitem___77734, int_77732)
        
        
        # Obtaining the type of the subscript
        int_77736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 33), 'int')
        # Getting the type of 'x' (line 664)
        x_77737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 31), 'x')
        # Obtaining the member '__getitem__' of a type (line 664)
        getitem___77738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 31), x_77737, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 664)
        subscript_call_result_77739 = invoke(stypy.reporting.localization.Localization(__file__, 664, 31), getitem___77738, int_77736)
        
        # Applying the binary operator '-' (line 664)
        result_sub_77740 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 24), '-', subscript_call_result_77735, subscript_call_result_77739)
        
        # Assigning a type to the variable 'd' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 20), 'd', result_sub_77740)
        
        # Assigning a BinOp to a Subscript (line 665):
        
        # Assigning a BinOp to a Subscript (line 665):
        
        # Obtaining the type of the subscript
        int_77741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 33), 'int')
        # Getting the type of 'dxr' (line 665)
        dxr_77742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 29), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 665)
        getitem___77743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 29), dxr_77742, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 665)
        subscript_call_result_77744 = invoke(stypy.reporting.localization.Localization(__file__, 665, 29), getitem___77743, int_77741)
        
        int_77745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 38), 'int')
        # Getting the type of 'd' (line 665)
        d_77746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 40), 'd')
        # Applying the binary operator '*' (line 665)
        result_mul_77747 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 38), '*', int_77745, d_77746)
        
        # Applying the binary operator '+' (line 665)
        result_add_77748 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 29), '+', subscript_call_result_77744, result_mul_77747)
        
        
        # Obtaining the type of the subscript
        int_77749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 49), 'int')
        # Getting the type of 'dxr' (line 665)
        dxr_77750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 45), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 665)
        getitem___77751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 45), dxr_77750, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 665)
        subscript_call_result_77752 = invoke(stypy.reporting.localization.Localization(__file__, 665, 45), getitem___77751, int_77749)
        
        # Applying the binary operator '*' (line 665)
        result_mul_77753 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 28), '*', result_add_77748, subscript_call_result_77752)
        
        
        # Obtaining the type of the subscript
        int_77754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 60), 'int')
        # Getting the type of 'slope' (line 665)
        slope_77755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 54), 'slope')
        # Obtaining the member '__getitem__' of a type (line 665)
        getitem___77756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 54), slope_77755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 665)
        subscript_call_result_77757 = invoke(stypy.reporting.localization.Localization(__file__, 665, 54), getitem___77756, int_77754)
        
        # Applying the binary operator '*' (line 665)
        result_mul_77758 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 52), '*', result_mul_77753, subscript_call_result_77757)
        
        
        # Obtaining the type of the subscript
        int_77759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 32), 'int')
        # Getting the type of 'dxr' (line 666)
        dxr_77760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 28), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 666)
        getitem___77761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 28), dxr_77760, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 666)
        subscript_call_result_77762 = invoke(stypy.reporting.localization.Localization(__file__, 666, 28), getitem___77761, int_77759)
        
        int_77763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 36), 'int')
        # Applying the binary operator '**' (line 666)
        result_pow_77764 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 28), '**', subscript_call_result_77762, int_77763)
        
        
        # Obtaining the type of the subscript
        int_77765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 46), 'int')
        # Getting the type of 'slope' (line 666)
        slope_77766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 40), 'slope')
        # Obtaining the member '__getitem__' of a type (line 666)
        getitem___77767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 40), slope_77766, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 666)
        subscript_call_result_77768 = invoke(stypy.reporting.localization.Localization(__file__, 666, 40), getitem___77767, int_77765)
        
        # Applying the binary operator '*' (line 666)
        result_mul_77769 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 28), '*', result_pow_77764, subscript_call_result_77768)
        
        # Applying the binary operator '+' (line 665)
        result_add_77770 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 28), '+', result_mul_77758, result_mul_77769)
        
        # Getting the type of 'd' (line 666)
        d_77771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 52), 'd')
        # Applying the binary operator 'div' (line 665)
        result_div_77772 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 27), 'div', result_add_77770, d_77771)
        
        # Getting the type of 'b' (line 665)
        b_77773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 20), 'b')
        int_77774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 22), 'int')
        # Storing an element on a container (line 665)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 20), b_77773, (int_77774, result_div_77772))
        # SSA branch for the else part of an if statement (line 661)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_77775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 30), 'int')
        # Getting the type of 'bc_start' (line 667)
        bc_start_77776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 21), 'bc_start')
        # Obtaining the member '__getitem__' of a type (line 667)
        getitem___77777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 21), bc_start_77776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 667)
        subscript_call_result_77778 = invoke(stypy.reporting.localization.Localization(__file__, 667, 21), getitem___77777, int_77775)
        
        int_77779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 36), 'int')
        # Applying the binary operator '==' (line 667)
        result_eq_77780 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 21), '==', subscript_call_result_77778, int_77779)
        
        # Testing the type of an if condition (line 667)
        if_condition_77781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 667, 21), result_eq_77780)
        # Assigning a type to the variable 'if_condition_77781' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 21), 'if_condition_77781', if_condition_77781)
        # SSA begins for if statement (line 667)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 668):
        
        # Assigning a Num to a Subscript (line 668):
        int_77782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 30), 'int')
        # Getting the type of 'A' (line 668)
        A_77783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 668)
        tuple_77784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 668)
        # Adding element type (line 668)
        int_77785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 22), tuple_77784, int_77785)
        # Adding element type (line 668)
        int_77786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 22), tuple_77784, int_77786)
        
        # Storing an element on a container (line 668)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 20), A_77783, (tuple_77784, int_77782))
        
        # Assigning a Num to a Subscript (line 669):
        
        # Assigning a Num to a Subscript (line 669):
        int_77787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 30), 'int')
        # Getting the type of 'A' (line 669)
        A_77788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 669)
        tuple_77789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 669)
        # Adding element type (line 669)
        int_77790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 22), tuple_77789, int_77790)
        # Adding element type (line 669)
        int_77791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 22), tuple_77789, int_77791)
        
        # Storing an element on a container (line 669)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 20), A_77788, (tuple_77789, int_77787))
        
        # Assigning a Subscript to a Subscript (line 670):
        
        # Assigning a Subscript to a Subscript (line 670):
        
        # Obtaining the type of the subscript
        int_77792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 36), 'int')
        # Getting the type of 'bc_start' (line 670)
        bc_start_77793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 27), 'bc_start')
        # Obtaining the member '__getitem__' of a type (line 670)
        getitem___77794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 27), bc_start_77793, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 670)
        subscript_call_result_77795 = invoke(stypy.reporting.localization.Localization(__file__, 670, 27), getitem___77794, int_77792)
        
        # Getting the type of 'b' (line 670)
        b_77796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 20), 'b')
        int_77797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 22), 'int')
        # Storing an element on a container (line 670)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 20), b_77796, (int_77797, subscript_call_result_77795))
        # SSA branch for the else part of an if statement (line 667)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_77798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 30), 'int')
        # Getting the type of 'bc_start' (line 671)
        bc_start_77799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 21), 'bc_start')
        # Obtaining the member '__getitem__' of a type (line 671)
        getitem___77800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 21), bc_start_77799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 671)
        subscript_call_result_77801 = invoke(stypy.reporting.localization.Localization(__file__, 671, 21), getitem___77800, int_77798)
        
        int_77802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 36), 'int')
        # Applying the binary operator '==' (line 671)
        result_eq_77803 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 21), '==', subscript_call_result_77801, int_77802)
        
        # Testing the type of an if condition (line 671)
        if_condition_77804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 671, 21), result_eq_77803)
        # Assigning a type to the variable 'if_condition_77804' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 21), 'if_condition_77804', if_condition_77804)
        # SSA begins for if statement (line 671)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 672):
        
        # Assigning a BinOp to a Subscript (line 672):
        int_77805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 30), 'int')
        
        # Obtaining the type of the subscript
        int_77806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 37), 'int')
        # Getting the type of 'dx' (line 672)
        dx_77807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 34), 'dx')
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___77808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 34), dx_77807, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_77809 = invoke(stypy.reporting.localization.Localization(__file__, 672, 34), getitem___77808, int_77806)
        
        # Applying the binary operator '*' (line 672)
        result_mul_77810 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 30), '*', int_77805, subscript_call_result_77809)
        
        # Getting the type of 'A' (line 672)
        A_77811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 672)
        tuple_77812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 672)
        # Adding element type (line 672)
        int_77813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 22), tuple_77812, int_77813)
        # Adding element type (line 672)
        int_77814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 22), tuple_77812, int_77814)
        
        # Storing an element on a container (line 672)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 20), A_77811, (tuple_77812, result_mul_77810))
        
        # Assigning a Subscript to a Subscript (line 673):
        
        # Assigning a Subscript to a Subscript (line 673):
        
        # Obtaining the type of the subscript
        int_77815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 33), 'int')
        # Getting the type of 'dx' (line 673)
        dx_77816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 30), 'dx')
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___77817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 30), dx_77816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 673)
        subscript_call_result_77818 = invoke(stypy.reporting.localization.Localization(__file__, 673, 30), getitem___77817, int_77815)
        
        # Getting the type of 'A' (line 673)
        A_77819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 673)
        tuple_77820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 673)
        # Adding element type (line 673)
        int_77821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 22), tuple_77820, int_77821)
        # Adding element type (line 673)
        int_77822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 22), tuple_77820, int_77822)
        
        # Storing an element on a container (line 673)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 20), A_77819, (tuple_77820, subscript_call_result_77818))
        
        # Assigning a BinOp to a Subscript (line 674):
        
        # Assigning a BinOp to a Subscript (line 674):
        float_77823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 27), 'float')
        
        # Obtaining the type of the subscript
        int_77824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 43), 'int')
        # Getting the type of 'bc_start' (line 674)
        bc_start_77825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 34), 'bc_start')
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___77826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 34), bc_start_77825, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_77827 = invoke(stypy.reporting.localization.Localization(__file__, 674, 34), getitem___77826, int_77824)
        
        # Applying the binary operator '*' (line 674)
        result_mul_77828 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 27), '*', float_77823, subscript_call_result_77827)
        
        
        # Obtaining the type of the subscript
        int_77829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 51), 'int')
        # Getting the type of 'dx' (line 674)
        dx_77830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 48), 'dx')
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___77831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 48), dx_77830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_77832 = invoke(stypy.reporting.localization.Localization(__file__, 674, 48), getitem___77831, int_77829)
        
        int_77833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 55), 'int')
        # Applying the binary operator '**' (line 674)
        result_pow_77834 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 48), '**', subscript_call_result_77832, int_77833)
        
        # Applying the binary operator '*' (line 674)
        result_mul_77835 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 46), '*', result_mul_77828, result_pow_77834)
        
        int_77836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 59), 'int')
        
        # Obtaining the type of the subscript
        int_77837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 66), 'int')
        # Getting the type of 'y' (line 674)
        y_77838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 64), 'y')
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___77839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 64), y_77838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_77840 = invoke(stypy.reporting.localization.Localization(__file__, 674, 64), getitem___77839, int_77837)
        
        
        # Obtaining the type of the subscript
        int_77841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 73), 'int')
        # Getting the type of 'y' (line 674)
        y_77842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 71), 'y')
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___77843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 71), y_77842, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_77844 = invoke(stypy.reporting.localization.Localization(__file__, 674, 71), getitem___77843, int_77841)
        
        # Applying the binary operator '-' (line 674)
        result_sub_77845 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 64), '-', subscript_call_result_77840, subscript_call_result_77844)
        
        # Applying the binary operator '*' (line 674)
        result_mul_77846 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 59), '*', int_77836, result_sub_77845)
        
        # Applying the binary operator '+' (line 674)
        result_add_77847 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 27), '+', result_mul_77835, result_mul_77846)
        
        # Getting the type of 'b' (line 674)
        b_77848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 20), 'b')
        int_77849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 22), 'int')
        # Storing an element on a container (line 674)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 20), b_77848, (int_77849, result_add_77847))
        # SSA join for if statement (line 671)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 667)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 661)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'bc_end' (line 676)
        bc_end_77850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 19), 'bc_end')
        str_77851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 29), 'str', 'not-a-knot')
        # Applying the binary operator '==' (line 676)
        result_eq_77852 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 19), '==', bc_end_77850, str_77851)
        
        # Testing the type of an if condition (line 676)
        if_condition_77853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 676, 16), result_eq_77852)
        # Assigning a type to the variable 'if_condition_77853' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'if_condition_77853', if_condition_77853)
        # SSA begins for if statement (line 676)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 677):
        
        # Assigning a Subscript to a Subscript (line 677):
        
        # Obtaining the type of the subscript
        int_77854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 34), 'int')
        # Getting the type of 'dx' (line 677)
        dx_77855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 31), 'dx')
        # Obtaining the member '__getitem__' of a type (line 677)
        getitem___77856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 31), dx_77855, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 677)
        subscript_call_result_77857 = invoke(stypy.reporting.localization.Localization(__file__, 677, 31), getitem___77856, int_77854)
        
        # Getting the type of 'A' (line 677)
        A_77858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 677)
        tuple_77859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 677)
        # Adding element type (line 677)
        int_77860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 22), tuple_77859, int_77860)
        # Adding element type (line 677)
        int_77861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 22), tuple_77859, int_77861)
        
        # Storing an element on a container (line 677)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 20), A_77858, (tuple_77859, subscript_call_result_77857))
        
        # Assigning a BinOp to a Subscript (line 678):
        
        # Assigning a BinOp to a Subscript (line 678):
        
        # Obtaining the type of the subscript
        int_77862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 34), 'int')
        # Getting the type of 'x' (line 678)
        x_77863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 32), 'x')
        # Obtaining the member '__getitem__' of a type (line 678)
        getitem___77864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 32), x_77863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 678)
        subscript_call_result_77865 = invoke(stypy.reporting.localization.Localization(__file__, 678, 32), getitem___77864, int_77862)
        
        
        # Obtaining the type of the subscript
        int_77866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 42), 'int')
        # Getting the type of 'x' (line 678)
        x_77867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 40), 'x')
        # Obtaining the member '__getitem__' of a type (line 678)
        getitem___77868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 40), x_77867, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 678)
        subscript_call_result_77869 = invoke(stypy.reporting.localization.Localization(__file__, 678, 40), getitem___77868, int_77866)
        
        # Applying the binary operator '-' (line 678)
        result_sub_77870 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 32), '-', subscript_call_result_77865, subscript_call_result_77869)
        
        # Getting the type of 'A' (line 678)
        A_77871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 678)
        tuple_77872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 678)
        # Adding element type (line 678)
        int_77873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 22), tuple_77872, int_77873)
        # Adding element type (line 678)
        int_77874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 22), tuple_77872, int_77874)
        
        # Storing an element on a container (line 678)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 20), A_77871, (tuple_77872, result_sub_77870))
        
        # Assigning a BinOp to a Name (line 679):
        
        # Assigning a BinOp to a Name (line 679):
        
        # Obtaining the type of the subscript
        int_77875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 26), 'int')
        # Getting the type of 'x' (line 679)
        x_77876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 24), 'x')
        # Obtaining the member '__getitem__' of a type (line 679)
        getitem___77877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 24), x_77876, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 679)
        subscript_call_result_77878 = invoke(stypy.reporting.localization.Localization(__file__, 679, 24), getitem___77877, int_77875)
        
        
        # Obtaining the type of the subscript
        int_77879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 34), 'int')
        # Getting the type of 'x' (line 679)
        x_77880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 32), 'x')
        # Obtaining the member '__getitem__' of a type (line 679)
        getitem___77881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 32), x_77880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 679)
        subscript_call_result_77882 = invoke(stypy.reporting.localization.Localization(__file__, 679, 32), getitem___77881, int_77879)
        
        # Applying the binary operator '-' (line 679)
        result_sub_77883 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 24), '-', subscript_call_result_77878, subscript_call_result_77882)
        
        # Assigning a type to the variable 'd' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 20), 'd', result_sub_77883)
        
        # Assigning a BinOp to a Subscript (line 680):
        
        # Assigning a BinOp to a Subscript (line 680):
        
        # Obtaining the type of the subscript
        int_77884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 34), 'int')
        # Getting the type of 'dxr' (line 680)
        dxr_77885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 30), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 680)
        getitem___77886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 30), dxr_77885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 680)
        subscript_call_result_77887 = invoke(stypy.reporting.localization.Localization(__file__, 680, 30), getitem___77886, int_77884)
        
        int_77888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 39), 'int')
        # Applying the binary operator '**' (line 680)
        result_pow_77889 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 30), '**', subscript_call_result_77887, int_77888)
        
        
        # Obtaining the type of the subscript
        int_77890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 47), 'int')
        # Getting the type of 'slope' (line 680)
        slope_77891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 41), 'slope')
        # Obtaining the member '__getitem__' of a type (line 680)
        getitem___77892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 41), slope_77891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 680)
        subscript_call_result_77893 = invoke(stypy.reporting.localization.Localization(__file__, 680, 41), getitem___77892, int_77890)
        
        # Applying the binary operator '*' (line 680)
        result_mul_77894 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 30), '*', result_pow_77889, subscript_call_result_77893)
        
        int_77895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 30), 'int')
        # Getting the type of 'd' (line 681)
        d_77896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 32), 'd')
        # Applying the binary operator '*' (line 681)
        result_mul_77897 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 30), '*', int_77895, d_77896)
        
        
        # Obtaining the type of the subscript
        int_77898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 40), 'int')
        # Getting the type of 'dxr' (line 681)
        dxr_77899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 36), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 681)
        getitem___77900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 36), dxr_77899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 681)
        subscript_call_result_77901 = invoke(stypy.reporting.localization.Localization(__file__, 681, 36), getitem___77900, int_77898)
        
        # Applying the binary operator '+' (line 681)
        result_add_77902 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 30), '+', result_mul_77897, subscript_call_result_77901)
        
        
        # Obtaining the type of the subscript
        int_77903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 49), 'int')
        # Getting the type of 'dxr' (line 681)
        dxr_77904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 45), 'dxr')
        # Obtaining the member '__getitem__' of a type (line 681)
        getitem___77905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 45), dxr_77904, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 681)
        subscript_call_result_77906 = invoke(stypy.reporting.localization.Localization(__file__, 681, 45), getitem___77905, int_77903)
        
        # Applying the binary operator '*' (line 681)
        result_mul_77907 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 29), '*', result_add_77902, subscript_call_result_77906)
        
        
        # Obtaining the type of the subscript
        int_77908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 59), 'int')
        # Getting the type of 'slope' (line 681)
        slope_77909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 53), 'slope')
        # Obtaining the member '__getitem__' of a type (line 681)
        getitem___77910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 53), slope_77909, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 681)
        subscript_call_result_77911 = invoke(stypy.reporting.localization.Localization(__file__, 681, 53), getitem___77910, int_77908)
        
        # Applying the binary operator '*' (line 681)
        result_mul_77912 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 52), '*', result_mul_77907, subscript_call_result_77911)
        
        # Applying the binary operator '+' (line 680)
        result_add_77913 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 30), '+', result_mul_77894, result_mul_77912)
        
        # Getting the type of 'd' (line 681)
        d_77914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 66), 'd')
        # Applying the binary operator 'div' (line 680)
        result_div_77915 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 29), 'div', result_add_77913, d_77914)
        
        # Getting the type of 'b' (line 680)
        b_77916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 20), 'b')
        int_77917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 22), 'int')
        # Storing an element on a container (line 680)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 20), b_77916, (int_77917, result_div_77915))
        # SSA branch for the else part of an if statement (line 676)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_77918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 28), 'int')
        # Getting the type of 'bc_end' (line 682)
        bc_end_77919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 21), 'bc_end')
        # Obtaining the member '__getitem__' of a type (line 682)
        getitem___77920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 21), bc_end_77919, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 682)
        subscript_call_result_77921 = invoke(stypy.reporting.localization.Localization(__file__, 682, 21), getitem___77920, int_77918)
        
        int_77922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 34), 'int')
        # Applying the binary operator '==' (line 682)
        result_eq_77923 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 21), '==', subscript_call_result_77921, int_77922)
        
        # Testing the type of an if condition (line 682)
        if_condition_77924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 682, 21), result_eq_77923)
        # Assigning a type to the variable 'if_condition_77924' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 21), 'if_condition_77924', if_condition_77924)
        # SSA begins for if statement (line 682)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 683):
        
        # Assigning a Num to a Subscript (line 683):
        int_77925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 31), 'int')
        # Getting the type of 'A' (line 683)
        A_77926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 683)
        tuple_77927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 683)
        # Adding element type (line 683)
        int_77928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 22), tuple_77927, int_77928)
        # Adding element type (line 683)
        int_77929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 22), tuple_77927, int_77929)
        
        # Storing an element on a container (line 683)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 20), A_77926, (tuple_77927, int_77925))
        
        # Assigning a Num to a Subscript (line 684):
        
        # Assigning a Num to a Subscript (line 684):
        int_77930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 32), 'int')
        # Getting the type of 'A' (line 684)
        A_77931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 684)
        tuple_77932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 684)
        # Adding element type (line 684)
        int_77933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 22), tuple_77932, int_77933)
        # Adding element type (line 684)
        int_77934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 22), tuple_77932, int_77934)
        
        # Storing an element on a container (line 684)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 20), A_77931, (tuple_77932, int_77930))
        
        # Assigning a Subscript to a Subscript (line 685):
        
        # Assigning a Subscript to a Subscript (line 685):
        
        # Obtaining the type of the subscript
        int_77935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 35), 'int')
        # Getting the type of 'bc_end' (line 685)
        bc_end_77936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 28), 'bc_end')
        # Obtaining the member '__getitem__' of a type (line 685)
        getitem___77937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 28), bc_end_77936, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 685)
        subscript_call_result_77938 = invoke(stypy.reporting.localization.Localization(__file__, 685, 28), getitem___77937, int_77935)
        
        # Getting the type of 'b' (line 685)
        b_77939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 20), 'b')
        int_77940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 22), 'int')
        # Storing an element on a container (line 685)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 20), b_77939, (int_77940, subscript_call_result_77938))
        # SSA branch for the else part of an if statement (line 682)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_77941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 28), 'int')
        # Getting the type of 'bc_end' (line 686)
        bc_end_77942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 21), 'bc_end')
        # Obtaining the member '__getitem__' of a type (line 686)
        getitem___77943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 21), bc_end_77942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 686)
        subscript_call_result_77944 = invoke(stypy.reporting.localization.Localization(__file__, 686, 21), getitem___77943, int_77941)
        
        int_77945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 34), 'int')
        # Applying the binary operator '==' (line 686)
        result_eq_77946 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 21), '==', subscript_call_result_77944, int_77945)
        
        # Testing the type of an if condition (line 686)
        if_condition_77947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 686, 21), result_eq_77946)
        # Assigning a type to the variable 'if_condition_77947' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 21), 'if_condition_77947', if_condition_77947)
        # SSA begins for if statement (line 686)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 687):
        
        # Assigning a BinOp to a Subscript (line 687):
        int_77948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 31), 'int')
        
        # Obtaining the type of the subscript
        int_77949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 38), 'int')
        # Getting the type of 'dx' (line 687)
        dx_77950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 35), 'dx')
        # Obtaining the member '__getitem__' of a type (line 687)
        getitem___77951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 35), dx_77950, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 687)
        subscript_call_result_77952 = invoke(stypy.reporting.localization.Localization(__file__, 687, 35), getitem___77951, int_77949)
        
        # Applying the binary operator '*' (line 687)
        result_mul_77953 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 31), '*', int_77948, subscript_call_result_77952)
        
        # Getting the type of 'A' (line 687)
        A_77954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 687)
        tuple_77955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 687)
        # Adding element type (line 687)
        int_77956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 22), tuple_77955, int_77956)
        # Adding element type (line 687)
        int_77957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 22), tuple_77955, int_77957)
        
        # Storing an element on a container (line 687)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 20), A_77954, (tuple_77955, result_mul_77953))
        
        # Assigning a Subscript to a Subscript (line 688):
        
        # Assigning a Subscript to a Subscript (line 688):
        
        # Obtaining the type of the subscript
        int_77958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 35), 'int')
        # Getting the type of 'dx' (line 688)
        dx_77959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 32), 'dx')
        # Obtaining the member '__getitem__' of a type (line 688)
        getitem___77960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 32), dx_77959, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 688)
        subscript_call_result_77961 = invoke(stypy.reporting.localization.Localization(__file__, 688, 32), getitem___77960, int_77958)
        
        # Getting the type of 'A' (line 688)
        A_77962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 20), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 688)
        tuple_77963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 688)
        # Adding element type (line 688)
        int_77964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 22), tuple_77963, int_77964)
        # Adding element type (line 688)
        int_77965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 22), tuple_77963, int_77965)
        
        # Storing an element on a container (line 688)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 20), A_77962, (tuple_77963, subscript_call_result_77961))
        
        # Assigning a BinOp to a Subscript (line 689):
        
        # Assigning a BinOp to a Subscript (line 689):
        float_77966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 28), 'float')
        
        # Obtaining the type of the subscript
        int_77967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 41), 'int')
        # Getting the type of 'bc_end' (line 689)
        bc_end_77968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 34), 'bc_end')
        # Obtaining the member '__getitem__' of a type (line 689)
        getitem___77969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 34), bc_end_77968, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 689)
        subscript_call_result_77970 = invoke(stypy.reporting.localization.Localization(__file__, 689, 34), getitem___77969, int_77967)
        
        # Applying the binary operator '*' (line 689)
        result_mul_77971 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 28), '*', float_77966, subscript_call_result_77970)
        
        
        # Obtaining the type of the subscript
        int_77972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 49), 'int')
        # Getting the type of 'dx' (line 689)
        dx_77973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 46), 'dx')
        # Obtaining the member '__getitem__' of a type (line 689)
        getitem___77974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 46), dx_77973, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 689)
        subscript_call_result_77975 = invoke(stypy.reporting.localization.Localization(__file__, 689, 46), getitem___77974, int_77972)
        
        int_77976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 54), 'int')
        # Applying the binary operator '**' (line 689)
        result_pow_77977 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 46), '**', subscript_call_result_77975, int_77976)
        
        # Applying the binary operator '*' (line 689)
        result_mul_77978 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 44), '*', result_mul_77971, result_pow_77977)
        
        int_77979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 58), 'int')
        
        # Obtaining the type of the subscript
        int_77980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 65), 'int')
        # Getting the type of 'y' (line 689)
        y_77981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 63), 'y')
        # Obtaining the member '__getitem__' of a type (line 689)
        getitem___77982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 63), y_77981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 689)
        subscript_call_result_77983 = invoke(stypy.reporting.localization.Localization(__file__, 689, 63), getitem___77982, int_77980)
        
        
        # Obtaining the type of the subscript
        int_77984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 73), 'int')
        # Getting the type of 'y' (line 689)
        y_77985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 71), 'y')
        # Obtaining the member '__getitem__' of a type (line 689)
        getitem___77986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 71), y_77985, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 689)
        subscript_call_result_77987 = invoke(stypy.reporting.localization.Localization(__file__, 689, 71), getitem___77986, int_77984)
        
        # Applying the binary operator '-' (line 689)
        result_sub_77988 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 63), '-', subscript_call_result_77983, subscript_call_result_77987)
        
        # Applying the binary operator '*' (line 689)
        result_mul_77989 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 58), '*', int_77979, result_sub_77988)
        
        # Applying the binary operator '+' (line 689)
        result_add_77990 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 28), '+', result_mul_77978, result_mul_77989)
        
        # Getting the type of 'b' (line 689)
        b_77991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 20), 'b')
        int_77992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 22), 'int')
        # Storing an element on a container (line 689)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 20), b_77991, (int_77992, result_add_77990))
        # SSA join for if statement (line 686)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 682)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 676)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 691):
        
        # Assigning a Call to a Name (line 691):
        
        # Call to solve_banded(...): (line 691)
        # Processing the call arguments (line 691)
        
        # Obtaining an instance of the builtin type 'tuple' (line 691)
        tuple_77994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 691)
        # Adding element type (line 691)
        int_77995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 691, 34), tuple_77994, int_77995)
        # Adding element type (line 691)
        int_77996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 691, 34), tuple_77994, int_77996)
        
        # Getting the type of 'A' (line 691)
        A_77997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 41), 'A', False)
        # Getting the type of 'b' (line 691)
        b_77998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 44), 'b', False)
        # Processing the call keyword arguments (line 691)
        # Getting the type of 'True' (line 691)
        True_77999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 60), 'True', False)
        keyword_78000 = True_77999
        # Getting the type of 'True' (line 692)
        True_78001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 45), 'True', False)
        keyword_78002 = True_78001
        # Getting the type of 'False' (line 692)
        False_78003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 64), 'False', False)
        keyword_78004 = False_78003
        kwargs_78005 = {'overwrite_ab': keyword_78000, 'check_finite': keyword_78004, 'overwrite_b': keyword_78002}
        # Getting the type of 'solve_banded' (line 691)
        solve_banded_77993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 20), 'solve_banded', False)
        # Calling solve_banded(args, kwargs) (line 691)
        solve_banded_call_result_78006 = invoke(stypy.reporting.localization.Localization(__file__, 691, 20), solve_banded_77993, *[tuple_77994, A_77997, b_77998], **kwargs_78005)
        
        # Assigning a type to the variable 's' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 's', solve_banded_call_result_78006)
        # SSA join for if statement (line 612)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 573)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 695):
        
        # Assigning a BinOp to a Name (line 695):
        
        # Obtaining the type of the subscript
        int_78007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 16), 'int')
        slice_78008 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 695, 13), None, int_78007, None)
        # Getting the type of 's' (line 695)
        s_78009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 13), 's')
        # Obtaining the member '__getitem__' of a type (line 695)
        getitem___78010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 13), s_78009, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 695)
        subscript_call_result_78011 = invoke(stypy.reporting.localization.Localization(__file__, 695, 13), getitem___78010, slice_78008)
        
        
        # Obtaining the type of the subscript
        int_78012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 24), 'int')
        slice_78013 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 695, 22), int_78012, None, None)
        # Getting the type of 's' (line 695)
        s_78014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 22), 's')
        # Obtaining the member '__getitem__' of a type (line 695)
        getitem___78015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 22), s_78014, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 695)
        subscript_call_result_78016 = invoke(stypy.reporting.localization.Localization(__file__, 695, 22), getitem___78015, slice_78013)
        
        # Applying the binary operator '+' (line 695)
        result_add_78017 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 13), '+', subscript_call_result_78011, subscript_call_result_78016)
        
        int_78018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 30), 'int')
        # Getting the type of 'slope' (line 695)
        slope_78019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 34), 'slope')
        # Applying the binary operator '*' (line 695)
        result_mul_78020 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 30), '*', int_78018, slope_78019)
        
        # Applying the binary operator '-' (line 695)
        result_sub_78021 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 28), '-', result_add_78017, result_mul_78020)
        
        # Getting the type of 'dxr' (line 695)
        dxr_78022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 43), 'dxr')
        # Applying the binary operator 'div' (line 695)
        result_div_78023 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 12), 'div', result_sub_78021, dxr_78022)
        
        # Assigning a type to the variable 't' (line 695)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 't', result_div_78023)
        
        # Assigning a Call to a Name (line 696):
        
        # Assigning a Call to a Name (line 696):
        
        # Call to empty(...): (line 696)
        # Processing the call arguments (line 696)
        
        # Obtaining an instance of the builtin type 'tuple' (line 696)
        tuple_78026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 696)
        # Adding element type (line 696)
        int_78027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 696, 22), tuple_78026, int_78027)
        # Adding element type (line 696)
        # Getting the type of 'n' (line 696)
        n_78028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 25), 'n', False)
        int_78029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 29), 'int')
        # Applying the binary operator '-' (line 696)
        result_sub_78030 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 25), '-', n_78028, int_78029)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 696, 22), tuple_78026, result_sub_78030)
        
        
        # Obtaining the type of the subscript
        int_78031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 42), 'int')
        slice_78032 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 696, 34), int_78031, None, None)
        # Getting the type of 'y' (line 696)
        y_78033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 34), 'y', False)
        # Obtaining the member 'shape' of a type (line 696)
        shape_78034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 34), y_78033, 'shape')
        # Obtaining the member '__getitem__' of a type (line 696)
        getitem___78035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 34), shape_78034, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 696)
        subscript_call_result_78036 = invoke(stypy.reporting.localization.Localization(__file__, 696, 34), getitem___78035, slice_78032)
        
        # Applying the binary operator '+' (line 696)
        result_add_78037 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 21), '+', tuple_78026, subscript_call_result_78036)
        
        # Processing the call keyword arguments (line 696)
        # Getting the type of 't' (line 696)
        t_78038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 53), 't', False)
        # Obtaining the member 'dtype' of a type (line 696)
        dtype_78039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 53), t_78038, 'dtype')
        keyword_78040 = dtype_78039
        kwargs_78041 = {'dtype': keyword_78040}
        # Getting the type of 'np' (line 696)
        np_78024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 12), 'np', False)
        # Obtaining the member 'empty' of a type (line 696)
        empty_78025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 12), np_78024, 'empty')
        # Calling empty(args, kwargs) (line 696)
        empty_call_result_78042 = invoke(stypy.reporting.localization.Localization(__file__, 696, 12), empty_78025, *[result_add_78037], **kwargs_78041)
        
        # Assigning a type to the variable 'c' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'c', empty_call_result_78042)
        
        # Assigning a BinOp to a Subscript (line 697):
        
        # Assigning a BinOp to a Subscript (line 697):
        # Getting the type of 't' (line 697)
        t_78043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 15), 't')
        # Getting the type of 'dxr' (line 697)
        dxr_78044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 19), 'dxr')
        # Applying the binary operator 'div' (line 697)
        result_div_78045 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 15), 'div', t_78043, dxr_78044)
        
        # Getting the type of 'c' (line 697)
        c_78046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'c')
        int_78047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 10), 'int')
        # Storing an element on a container (line 697)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 8), c_78046, (int_78047, result_div_78045))
        
        # Assigning a BinOp to a Subscript (line 698):
        
        # Assigning a BinOp to a Subscript (line 698):
        # Getting the type of 'slope' (line 698)
        slope_78048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 16), 'slope')
        
        # Obtaining the type of the subscript
        int_78049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 27), 'int')
        slice_78050 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 698, 24), None, int_78049, None)
        # Getting the type of 's' (line 698)
        s_78051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 24), 's')
        # Obtaining the member '__getitem__' of a type (line 698)
        getitem___78052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 24), s_78051, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 698)
        subscript_call_result_78053 = invoke(stypy.reporting.localization.Localization(__file__, 698, 24), getitem___78052, slice_78050)
        
        # Applying the binary operator '-' (line 698)
        result_sub_78054 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 16), '-', slope_78048, subscript_call_result_78053)
        
        # Getting the type of 'dxr' (line 698)
        dxr_78055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 34), 'dxr')
        # Applying the binary operator 'div' (line 698)
        result_div_78056 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 15), 'div', result_sub_78054, dxr_78055)
        
        # Getting the type of 't' (line 698)
        t_78057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 40), 't')
        # Applying the binary operator '-' (line 698)
        result_sub_78058 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 15), '-', result_div_78056, t_78057)
        
        # Getting the type of 'c' (line 698)
        c_78059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'c')
        int_78060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 10), 'int')
        # Storing an element on a container (line 698)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 8), c_78059, (int_78060, result_sub_78058))
        
        # Assigning a Subscript to a Subscript (line 699):
        
        # Assigning a Subscript to a Subscript (line 699):
        
        # Obtaining the type of the subscript
        int_78061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 18), 'int')
        slice_78062 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 699, 15), None, int_78061, None)
        # Getting the type of 's' (line 699)
        s_78063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 15), 's')
        # Obtaining the member '__getitem__' of a type (line 699)
        getitem___78064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 15), s_78063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 699)
        subscript_call_result_78065 = invoke(stypy.reporting.localization.Localization(__file__, 699, 15), getitem___78064, slice_78062)
        
        # Getting the type of 'c' (line 699)
        c_78066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'c')
        int_78067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 10), 'int')
        # Storing an element on a container (line 699)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 8), c_78066, (int_78067, subscript_call_result_78065))
        
        # Assigning a Subscript to a Subscript (line 700):
        
        # Assigning a Subscript to a Subscript (line 700):
        
        # Obtaining the type of the subscript
        int_78068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 18), 'int')
        slice_78069 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 700, 15), None, int_78068, None)
        # Getting the type of 'y' (line 700)
        y_78070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 15), 'y')
        # Obtaining the member '__getitem__' of a type (line 700)
        getitem___78071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 15), y_78070, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 700)
        subscript_call_result_78072 = invoke(stypy.reporting.localization.Localization(__file__, 700, 15), getitem___78071, slice_78069)
        
        # Getting the type of 'c' (line 700)
        c_78073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'c')
        int_78074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 10), 'int')
        # Storing an element on a container (line 700)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 700, 8), c_78073, (int_78074, subscript_call_result_78072))
        
        # Call to __init__(...): (line 702)
        # Processing the call arguments (line 702)
        # Getting the type of 'c' (line 702)
        c_78081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 42), 'c', False)
        # Getting the type of 'x' (line 702)
        x_78082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 45), 'x', False)
        # Processing the call keyword arguments (line 702)
        # Getting the type of 'extrapolate' (line 702)
        extrapolate_78083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 60), 'extrapolate', False)
        keyword_78084 = extrapolate_78083
        kwargs_78085 = {'extrapolate': keyword_78084}
        
        # Call to super(...): (line 702)
        # Processing the call arguments (line 702)
        # Getting the type of 'CubicSpline' (line 702)
        CubicSpline_78076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 14), 'CubicSpline', False)
        # Getting the type of 'self' (line 702)
        self_78077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 27), 'self', False)
        # Processing the call keyword arguments (line 702)
        kwargs_78078 = {}
        # Getting the type of 'super' (line 702)
        super_78075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'super', False)
        # Calling super(args, kwargs) (line 702)
        super_call_result_78079 = invoke(stypy.reporting.localization.Localization(__file__, 702, 8), super_78075, *[CubicSpline_78076, self_78077], **kwargs_78078)
        
        # Obtaining the member '__init__' of a type (line 702)
        init___78080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), super_call_result_78079, '__init__')
        # Calling __init__(args, kwargs) (line 702)
        init___call_result_78086 = invoke(stypy.reporting.localization.Localization(__file__, 702, 8), init___78080, *[c_78081, x_78082], **kwargs_78085)
        
        
        # Assigning a Name to a Attribute (line 703):
        
        # Assigning a Name to a Attribute (line 703):
        # Getting the type of 'axis' (line 703)
        axis_78087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 20), 'axis')
        # Getting the type of 'self' (line 703)
        self_78088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'self')
        # Setting the type of the member 'axis' of a type (line 703)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 8), self_78088, 'axis', axis_78087)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @staticmethod
    @norecursion
    def _validate_bc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_validate_bc'
        module_type_store = module_type_store.open_function_context('_validate_bc', 705, 4, False)
        
        # Passed parameters checking function
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_localization', localization)
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_type_of_self', None)
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_type_store', module_type_store)
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_function_name', '_validate_bc')
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_param_names_list', ['bc_type', 'y', 'expected_deriv_shape', 'axis'])
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_varargs_param_name', None)
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_call_defaults', defaults)
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_call_varargs', varargs)
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CubicSpline._validate_bc.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, None, module_type_store, '_validate_bc', ['bc_type', 'y', 'expected_deriv_shape', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_validate_bc', localization, ['y', 'expected_deriv_shape', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_validate_bc(...)' code ##################

        str_78089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, (-1)), 'str', 'Validate and prepare boundary conditions.\n\n        Returns\n        -------\n        validated_bc : 2-tuple\n            Boundary conditions for a curve start and end.\n        y : ndarray\n            y casted to complex dtype if one of the boundary conditions has\n            complex dtype.\n        ')
        
        
        # Call to isinstance(...): (line 717)
        # Processing the call arguments (line 717)
        # Getting the type of 'bc_type' (line 717)
        bc_type_78091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 22), 'bc_type', False)
        # Getting the type of 'string_types' (line 717)
        string_types_78092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 31), 'string_types', False)
        # Processing the call keyword arguments (line 717)
        kwargs_78093 = {}
        # Getting the type of 'isinstance' (line 717)
        isinstance_78090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 717)
        isinstance_call_result_78094 = invoke(stypy.reporting.localization.Localization(__file__, 717, 11), isinstance_78090, *[bc_type_78091, string_types_78092], **kwargs_78093)
        
        # Testing the type of an if condition (line 717)
        if_condition_78095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 717, 8), isinstance_call_result_78094)
        # Assigning a type to the variable 'if_condition_78095' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'if_condition_78095', if_condition_78095)
        # SSA begins for if statement (line 717)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'bc_type' (line 718)
        bc_type_78096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 15), 'bc_type')
        str_78097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 26), 'str', 'periodic')
        # Applying the binary operator '==' (line 718)
        result_eq_78098 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 15), '==', bc_type_78096, str_78097)
        
        # Testing the type of an if condition (line 718)
        if_condition_78099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 12), result_eq_78098)
        # Assigning a type to the variable 'if_condition_78099' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 12), 'if_condition_78099', if_condition_78099)
        # SSA begins for if statement (line 718)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to allclose(...): (line 719)
        # Processing the call arguments (line 719)
        
        # Obtaining the type of the subscript
        int_78102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 37), 'int')
        # Getting the type of 'y' (line 719)
        y_78103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 35), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 719)
        getitem___78104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 35), y_78103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 719)
        subscript_call_result_78105 = invoke(stypy.reporting.localization.Localization(__file__, 719, 35), getitem___78104, int_78102)
        
        
        # Obtaining the type of the subscript
        int_78106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 43), 'int')
        # Getting the type of 'y' (line 719)
        y_78107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 41), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 719)
        getitem___78108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 41), y_78107, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 719)
        subscript_call_result_78109 = invoke(stypy.reporting.localization.Localization(__file__, 719, 41), getitem___78108, int_78106)
        
        # Processing the call keyword arguments (line 719)
        float_78110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 53), 'float')
        keyword_78111 = float_78110
        float_78112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 65), 'float')
        keyword_78113 = float_78112
        kwargs_78114 = {'rtol': keyword_78111, 'atol': keyword_78113}
        # Getting the type of 'np' (line 719)
        np_78100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 23), 'np', False)
        # Obtaining the member 'allclose' of a type (line 719)
        allclose_78101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 23), np_78100, 'allclose')
        # Calling allclose(args, kwargs) (line 719)
        allclose_call_result_78115 = invoke(stypy.reporting.localization.Localization(__file__, 719, 23), allclose_78101, *[subscript_call_result_78105, subscript_call_result_78109], **kwargs_78114)
        
        # Applying the 'not' unary operator (line 719)
        result_not__78116 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 19), 'not', allclose_call_result_78115)
        
        # Testing the type of an if condition (line 719)
        if_condition_78117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 719, 16), result_not__78116)
        # Assigning a type to the variable 'if_condition_78117' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'if_condition_78117', if_condition_78117)
        # SSA begins for if statement (line 719)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 720)
        # Processing the call arguments (line 720)
        
        # Call to format(...): (line 721)
        # Processing the call arguments (line 721)
        # Getting the type of 'axis' (line 723)
        axis_78121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 53), 'axis', False)
        # Processing the call keyword arguments (line 721)
        kwargs_78122 = {}
        str_78119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 24), 'str', "The first and last `y` point along axis {} must be identical (within machine precision) when bc_type='periodic'.")
        # Obtaining the member 'format' of a type (line 721)
        format_78120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 24), str_78119, 'format')
        # Calling format(args, kwargs) (line 721)
        format_call_result_78123 = invoke(stypy.reporting.localization.Localization(__file__, 721, 24), format_78120, *[axis_78121], **kwargs_78122)
        
        # Processing the call keyword arguments (line 720)
        kwargs_78124 = {}
        # Getting the type of 'ValueError' (line 720)
        ValueError_78118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 720)
        ValueError_call_result_78125 = invoke(stypy.reporting.localization.Localization(__file__, 720, 26), ValueError_78118, *[format_call_result_78123], **kwargs_78124)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 720, 20), ValueError_call_result_78125, 'raise parameter', BaseException)
        # SSA join for if statement (line 719)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 718)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 725):
        
        # Assigning a Tuple to a Name (line 725):
        
        # Obtaining an instance of the builtin type 'tuple' (line 725)
        tuple_78126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 725)
        # Adding element type (line 725)
        # Getting the type of 'bc_type' (line 725)
        bc_type_78127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 23), 'bc_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 23), tuple_78126, bc_type_78127)
        # Adding element type (line 725)
        # Getting the type of 'bc_type' (line 725)
        bc_type_78128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 32), 'bc_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 23), tuple_78126, bc_type_78128)
        
        # Assigning a type to the variable 'bc_type' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 12), 'bc_type', tuple_78126)
        # SSA branch for the else part of an if statement (line 717)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 728)
        # Processing the call arguments (line 728)
        # Getting the type of 'bc_type' (line 728)
        bc_type_78130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 19), 'bc_type', False)
        # Processing the call keyword arguments (line 728)
        kwargs_78131 = {}
        # Getting the type of 'len' (line 728)
        len_78129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 15), 'len', False)
        # Calling len(args, kwargs) (line 728)
        len_call_result_78132 = invoke(stypy.reporting.localization.Localization(__file__, 728, 15), len_78129, *[bc_type_78130], **kwargs_78131)
        
        int_78133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 31), 'int')
        # Applying the binary operator '!=' (line 728)
        result_ne_78134 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 15), '!=', len_call_result_78132, int_78133)
        
        # Testing the type of an if condition (line 728)
        if_condition_78135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 728, 12), result_ne_78134)
        # Assigning a type to the variable 'if_condition_78135' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 12), 'if_condition_78135', if_condition_78135)
        # SSA begins for if statement (line 728)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 729)
        # Processing the call arguments (line 729)
        str_78137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 33), 'str', '`bc_type` must contain 2 elements to specify start and end conditions.')
        # Processing the call keyword arguments (line 729)
        kwargs_78138 = {}
        # Getting the type of 'ValueError' (line 729)
        ValueError_78136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 729)
        ValueError_call_result_78139 = invoke(stypy.reporting.localization.Localization(__file__, 729, 22), ValueError_78136, *[str_78137], **kwargs_78138)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 729, 16), ValueError_call_result_78139, 'raise parameter', BaseException)
        # SSA join for if statement (line 728)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_78140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 15), 'str', 'periodic')
        # Getting the type of 'bc_type' (line 732)
        bc_type_78141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 29), 'bc_type')
        # Applying the binary operator 'in' (line 732)
        result_contains_78142 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 15), 'in', str_78140, bc_type_78141)
        
        # Testing the type of an if condition (line 732)
        if_condition_78143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 732, 12), result_contains_78142)
        # Assigning a type to the variable 'if_condition_78143' (line 732)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'if_condition_78143', if_condition_78143)
        # SSA begins for if statement (line 732)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 733)
        # Processing the call arguments (line 733)
        str_78145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 33), 'str', "'periodic' `bc_type` is defined for both curve ends and cannot be used with other boundary conditions.")
        # Processing the call keyword arguments (line 733)
        kwargs_78146 = {}
        # Getting the type of 'ValueError' (line 733)
        ValueError_78144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 733)
        ValueError_call_result_78147 = invoke(stypy.reporting.localization.Localization(__file__, 733, 22), ValueError_78144, *[str_78145], **kwargs_78146)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 733, 16), ValueError_call_result_78147, 'raise parameter', BaseException)
        # SSA join for if statement (line 732)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 717)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 737):
        
        # Assigning a List to a Name (line 737):
        
        # Obtaining an instance of the builtin type 'list' (line 737)
        list_78148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 737)
        
        # Assigning a type to the variable 'validated_bc' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'validated_bc', list_78148)
        
        # Getting the type of 'bc_type' (line 738)
        bc_type_78149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 18), 'bc_type')
        # Testing the type of a for loop iterable (line 738)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 738, 8), bc_type_78149)
        # Getting the type of the for loop variable (line 738)
        for_loop_var_78150 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 738, 8), bc_type_78149)
        # Assigning a type to the variable 'bc' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'bc', for_loop_var_78150)
        # SSA begins for a for statement (line 738)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to isinstance(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'bc' (line 739)
        bc_78152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 26), 'bc', False)
        # Getting the type of 'string_types' (line 739)
        string_types_78153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 30), 'string_types', False)
        # Processing the call keyword arguments (line 739)
        kwargs_78154 = {}
        # Getting the type of 'isinstance' (line 739)
        isinstance_78151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 739)
        isinstance_call_result_78155 = invoke(stypy.reporting.localization.Localization(__file__, 739, 15), isinstance_78151, *[bc_78152, string_types_78153], **kwargs_78154)
        
        # Testing the type of an if condition (line 739)
        if_condition_78156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 739, 12), isinstance_call_result_78155)
        # Assigning a type to the variable 'if_condition_78156' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'if_condition_78156', if_condition_78156)
        # SSA begins for if statement (line 739)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'bc' (line 740)
        bc_78157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 19), 'bc')
        str_78158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 25), 'str', 'clamped')
        # Applying the binary operator '==' (line 740)
        result_eq_78159 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 19), '==', bc_78157, str_78158)
        
        # Testing the type of an if condition (line 740)
        if_condition_78160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 740, 16), result_eq_78159)
        # Assigning a type to the variable 'if_condition_78160' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 16), 'if_condition_78160', if_condition_78160)
        # SSA begins for if statement (line 740)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 741)
        # Processing the call arguments (line 741)
        
        # Obtaining an instance of the builtin type 'tuple' (line 741)
        tuple_78163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 741)
        # Adding element type (line 741)
        int_78164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 741, 41), tuple_78163, int_78164)
        # Adding element type (line 741)
        
        # Call to zeros(...): (line 741)
        # Processing the call arguments (line 741)
        # Getting the type of 'expected_deriv_shape' (line 741)
        expected_deriv_shape_78167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 53), 'expected_deriv_shape', False)
        # Processing the call keyword arguments (line 741)
        kwargs_78168 = {}
        # Getting the type of 'np' (line 741)
        np_78165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 44), 'np', False)
        # Obtaining the member 'zeros' of a type (line 741)
        zeros_78166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 44), np_78165, 'zeros')
        # Calling zeros(args, kwargs) (line 741)
        zeros_call_result_78169 = invoke(stypy.reporting.localization.Localization(__file__, 741, 44), zeros_78166, *[expected_deriv_shape_78167], **kwargs_78168)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 741, 41), tuple_78163, zeros_call_result_78169)
        
        # Processing the call keyword arguments (line 741)
        kwargs_78170 = {}
        # Getting the type of 'validated_bc' (line 741)
        validated_bc_78161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 20), 'validated_bc', False)
        # Obtaining the member 'append' of a type (line 741)
        append_78162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 20), validated_bc_78161, 'append')
        # Calling append(args, kwargs) (line 741)
        append_call_result_78171 = invoke(stypy.reporting.localization.Localization(__file__, 741, 20), append_78162, *[tuple_78163], **kwargs_78170)
        
        # SSA branch for the else part of an if statement (line 740)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'bc' (line 742)
        bc_78172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 21), 'bc')
        str_78173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 27), 'str', 'natural')
        # Applying the binary operator '==' (line 742)
        result_eq_78174 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 21), '==', bc_78172, str_78173)
        
        # Testing the type of an if condition (line 742)
        if_condition_78175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 742, 21), result_eq_78174)
        # Assigning a type to the variable 'if_condition_78175' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 21), 'if_condition_78175', if_condition_78175)
        # SSA begins for if statement (line 742)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 743)
        # Processing the call arguments (line 743)
        
        # Obtaining an instance of the builtin type 'tuple' (line 743)
        tuple_78178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 743)
        # Adding element type (line 743)
        int_78179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 41), tuple_78178, int_78179)
        # Adding element type (line 743)
        
        # Call to zeros(...): (line 743)
        # Processing the call arguments (line 743)
        # Getting the type of 'expected_deriv_shape' (line 743)
        expected_deriv_shape_78182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 53), 'expected_deriv_shape', False)
        # Processing the call keyword arguments (line 743)
        kwargs_78183 = {}
        # Getting the type of 'np' (line 743)
        np_78180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'np', False)
        # Obtaining the member 'zeros' of a type (line 743)
        zeros_78181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 44), np_78180, 'zeros')
        # Calling zeros(args, kwargs) (line 743)
        zeros_call_result_78184 = invoke(stypy.reporting.localization.Localization(__file__, 743, 44), zeros_78181, *[expected_deriv_shape_78182], **kwargs_78183)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 41), tuple_78178, zeros_call_result_78184)
        
        # Processing the call keyword arguments (line 743)
        kwargs_78185 = {}
        # Getting the type of 'validated_bc' (line 743)
        validated_bc_78176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 20), 'validated_bc', False)
        # Obtaining the member 'append' of a type (line 743)
        append_78177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 20), validated_bc_78176, 'append')
        # Calling append(args, kwargs) (line 743)
        append_call_result_78186 = invoke(stypy.reporting.localization.Localization(__file__, 743, 20), append_78177, *[tuple_78178], **kwargs_78185)
        
        # SSA branch for the else part of an if statement (line 742)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'bc' (line 744)
        bc_78187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 21), 'bc')
        
        # Obtaining an instance of the builtin type 'list' (line 744)
        list_78188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 744)
        # Adding element type (line 744)
        str_78189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 28), 'str', 'not-a-knot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 27), list_78188, str_78189)
        # Adding element type (line 744)
        str_78190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 42), 'str', 'periodic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 27), list_78188, str_78190)
        
        # Applying the binary operator 'in' (line 744)
        result_contains_78191 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 21), 'in', bc_78187, list_78188)
        
        # Testing the type of an if condition (line 744)
        if_condition_78192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 744, 21), result_contains_78191)
        # Assigning a type to the variable 'if_condition_78192' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 21), 'if_condition_78192', if_condition_78192)
        # SSA begins for if statement (line 744)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 745)
        # Processing the call arguments (line 745)
        # Getting the type of 'bc' (line 745)
        bc_78195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 40), 'bc', False)
        # Processing the call keyword arguments (line 745)
        kwargs_78196 = {}
        # Getting the type of 'validated_bc' (line 745)
        validated_bc_78193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 20), 'validated_bc', False)
        # Obtaining the member 'append' of a type (line 745)
        append_78194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 20), validated_bc_78193, 'append')
        # Calling append(args, kwargs) (line 745)
        append_call_result_78197 = invoke(stypy.reporting.localization.Localization(__file__, 745, 20), append_78194, *[bc_78195], **kwargs_78196)
        
        # SSA branch for the else part of an if statement (line 744)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 747)
        # Processing the call arguments (line 747)
        
        # Call to format(...): (line 747)
        # Processing the call arguments (line 747)
        # Getting the type of 'bc' (line 747)
        bc_78201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 73), 'bc', False)
        # Processing the call keyword arguments (line 747)
        kwargs_78202 = {}
        str_78199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 37), 'str', 'bc_type={} is not allowed.')
        # Obtaining the member 'format' of a type (line 747)
        format_78200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 37), str_78199, 'format')
        # Calling format(args, kwargs) (line 747)
        format_call_result_78203 = invoke(stypy.reporting.localization.Localization(__file__, 747, 37), format_78200, *[bc_78201], **kwargs_78202)
        
        # Processing the call keyword arguments (line 747)
        kwargs_78204 = {}
        # Getting the type of 'ValueError' (line 747)
        ValueError_78198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 747)
        ValueError_call_result_78205 = invoke(stypy.reporting.localization.Localization(__file__, 747, 26), ValueError_78198, *[format_call_result_78203], **kwargs_78204)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 747, 20), ValueError_call_result_78205, 'raise parameter', BaseException)
        # SSA join for if statement (line 744)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 742)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 740)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 739)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 749)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Tuple (line 750):
        
        # Assigning a Subscript to a Name (line 750):
        
        # Obtaining the type of the subscript
        int_78206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 20), 'int')
        # Getting the type of 'bc' (line 750)
        bc_78207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 47), 'bc')
        # Obtaining the member '__getitem__' of a type (line 750)
        getitem___78208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 20), bc_78207, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 750)
        subscript_call_result_78209 = invoke(stypy.reporting.localization.Localization(__file__, 750, 20), getitem___78208, int_78206)
        
        # Assigning a type to the variable 'tuple_var_assignment_76090' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 20), 'tuple_var_assignment_76090', subscript_call_result_78209)
        
        # Assigning a Subscript to a Name (line 750):
        
        # Obtaining the type of the subscript
        int_78210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 20), 'int')
        # Getting the type of 'bc' (line 750)
        bc_78211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 47), 'bc')
        # Obtaining the member '__getitem__' of a type (line 750)
        getitem___78212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 20), bc_78211, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 750)
        subscript_call_result_78213 = invoke(stypy.reporting.localization.Localization(__file__, 750, 20), getitem___78212, int_78210)
        
        # Assigning a type to the variable 'tuple_var_assignment_76091' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 20), 'tuple_var_assignment_76091', subscript_call_result_78213)
        
        # Assigning a Name to a Name (line 750):
        # Getting the type of 'tuple_var_assignment_76090' (line 750)
        tuple_var_assignment_76090_78214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 20), 'tuple_var_assignment_76090')
        # Assigning a type to the variable 'deriv_order' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 20), 'deriv_order', tuple_var_assignment_76090_78214)
        
        # Assigning a Name to a Name (line 750):
        # Getting the type of 'tuple_var_assignment_76091' (line 750)
        tuple_var_assignment_76091_78215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 20), 'tuple_var_assignment_76091')
        # Assigning a type to the variable 'deriv_value' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 33), 'deriv_value', tuple_var_assignment_76091_78215)
        # SSA branch for the except part of a try statement (line 749)
        # SSA branch for the except 'Exception' branch of a try statement (line 749)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 752)
        # Processing the call arguments (line 752)
        str_78217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 37), 'str', 'A specified derivative value must be given in the form (order, value).')
        # Processing the call keyword arguments (line 752)
        kwargs_78218 = {}
        # Getting the type of 'ValueError' (line 752)
        ValueError_78216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 752)
        ValueError_call_result_78219 = invoke(stypy.reporting.localization.Localization(__file__, 752, 26), ValueError_78216, *[str_78217], **kwargs_78218)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 752, 20), ValueError_call_result_78219, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 749)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'deriv_order' (line 755)
        deriv_order_78220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 19), 'deriv_order')
        
        # Obtaining an instance of the builtin type 'list' (line 755)
        list_78221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 755)
        # Adding element type (line 755)
        int_78222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 38), list_78221, int_78222)
        # Adding element type (line 755)
        int_78223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 38), list_78221, int_78223)
        
        # Applying the binary operator 'notin' (line 755)
        result_contains_78224 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 19), 'notin', deriv_order_78220, list_78221)
        
        # Testing the type of an if condition (line 755)
        if_condition_78225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 755, 16), result_contains_78224)
        # Assigning a type to the variable 'if_condition_78225' (line 755)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 16), 'if_condition_78225', if_condition_78225)
        # SSA begins for if statement (line 755)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 756)
        # Processing the call arguments (line 756)
        str_78227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 37), 'str', 'The specified derivative order must be 1 or 2.')
        # Processing the call keyword arguments (line 756)
        kwargs_78228 = {}
        # Getting the type of 'ValueError' (line 756)
        ValueError_78226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 756)
        ValueError_call_result_78229 = invoke(stypy.reporting.localization.Localization(__file__, 756, 26), ValueError_78226, *[str_78227], **kwargs_78228)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 756, 20), ValueError_call_result_78229, 'raise parameter', BaseException)
        # SSA join for if statement (line 755)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 759):
        
        # Assigning a Call to a Name (line 759):
        
        # Call to asarray(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'deriv_value' (line 759)
        deriv_value_78232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 41), 'deriv_value', False)
        # Processing the call keyword arguments (line 759)
        kwargs_78233 = {}
        # Getting the type of 'np' (line 759)
        np_78230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 30), 'np', False)
        # Obtaining the member 'asarray' of a type (line 759)
        asarray_78231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 30), np_78230, 'asarray')
        # Calling asarray(args, kwargs) (line 759)
        asarray_call_result_78234 = invoke(stypy.reporting.localization.Localization(__file__, 759, 30), asarray_78231, *[deriv_value_78232], **kwargs_78233)
        
        # Assigning a type to the variable 'deriv_value' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 16), 'deriv_value', asarray_call_result_78234)
        
        
        # Getting the type of 'deriv_value' (line 760)
        deriv_value_78235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 19), 'deriv_value')
        # Obtaining the member 'shape' of a type (line 760)
        shape_78236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 19), deriv_value_78235, 'shape')
        # Getting the type of 'expected_deriv_shape' (line 760)
        expected_deriv_shape_78237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 40), 'expected_deriv_shape')
        # Applying the binary operator '!=' (line 760)
        result_ne_78238 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 19), '!=', shape_78236, expected_deriv_shape_78237)
        
        # Testing the type of an if condition (line 760)
        if_condition_78239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 760, 16), result_ne_78238)
        # Assigning a type to the variable 'if_condition_78239' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 16), 'if_condition_78239', if_condition_78239)
        # SSA begins for if statement (line 760)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 761)
        # Processing the call arguments (line 761)
        
        # Call to format(...): (line 762)
        # Processing the call arguments (line 762)
        # Getting the type of 'deriv_value' (line 763)
        deriv_value_78243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 32), 'deriv_value', False)
        # Obtaining the member 'shape' of a type (line 763)
        shape_78244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 32), deriv_value_78243, 'shape')
        # Getting the type of 'expected_deriv_shape' (line 763)
        expected_deriv_shape_78245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 51), 'expected_deriv_shape', False)
        # Processing the call keyword arguments (line 762)
        kwargs_78246 = {}
        str_78241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 24), 'str', '`deriv_value` shape {} is not the expected one {}.')
        # Obtaining the member 'format' of a type (line 762)
        format_78242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 24), str_78241, 'format')
        # Calling format(args, kwargs) (line 762)
        format_call_result_78247 = invoke(stypy.reporting.localization.Localization(__file__, 762, 24), format_78242, *[shape_78244, expected_deriv_shape_78245], **kwargs_78246)
        
        # Processing the call keyword arguments (line 761)
        kwargs_78248 = {}
        # Getting the type of 'ValueError' (line 761)
        ValueError_78240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 761)
        ValueError_call_result_78249 = invoke(stypy.reporting.localization.Localization(__file__, 761, 26), ValueError_78240, *[format_call_result_78247], **kwargs_78248)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 761, 20), ValueError_call_result_78249, 'raise parameter', BaseException)
        # SSA join for if statement (line 760)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to issubdtype(...): (line 765)
        # Processing the call arguments (line 765)
        # Getting the type of 'deriv_value' (line 765)
        deriv_value_78252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 33), 'deriv_value', False)
        # Obtaining the member 'dtype' of a type (line 765)
        dtype_78253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 33), deriv_value_78252, 'dtype')
        # Getting the type of 'np' (line 765)
        np_78254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 52), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 765)
        complexfloating_78255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 52), np_78254, 'complexfloating')
        # Processing the call keyword arguments (line 765)
        kwargs_78256 = {}
        # Getting the type of 'np' (line 765)
        np_78250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 19), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 765)
        issubdtype_78251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 19), np_78250, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 765)
        issubdtype_call_result_78257 = invoke(stypy.reporting.localization.Localization(__file__, 765, 19), issubdtype_78251, *[dtype_78253, complexfloating_78255], **kwargs_78256)
        
        # Testing the type of an if condition (line 765)
        if_condition_78258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 765, 16), issubdtype_call_result_78257)
        # Assigning a type to the variable 'if_condition_78258' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'if_condition_78258', if_condition_78258)
        # SSA begins for if statement (line 765)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 766):
        
        # Assigning a Call to a Name (line 766):
        
        # Call to astype(...): (line 766)
        # Processing the call arguments (line 766)
        # Getting the type of 'complex' (line 766)
        complex_78261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 33), 'complex', False)
        # Processing the call keyword arguments (line 766)
        # Getting the type of 'False' (line 766)
        False_78262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 47), 'False', False)
        keyword_78263 = False_78262
        kwargs_78264 = {'copy': keyword_78263}
        # Getting the type of 'y' (line 766)
        y_78259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 24), 'y', False)
        # Obtaining the member 'astype' of a type (line 766)
        astype_78260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 24), y_78259, 'astype')
        # Calling astype(args, kwargs) (line 766)
        astype_call_result_78265 = invoke(stypy.reporting.localization.Localization(__file__, 766, 24), astype_78260, *[complex_78261], **kwargs_78264)
        
        # Assigning a type to the variable 'y' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 20), 'y', astype_call_result_78265)
        # SSA join for if statement (line 765)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 768)
        # Processing the call arguments (line 768)
        
        # Obtaining an instance of the builtin type 'tuple' (line 768)
        tuple_78268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 768)
        # Adding element type (line 768)
        # Getting the type of 'deriv_order' (line 768)
        deriv_order_78269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 37), 'deriv_order', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 37), tuple_78268, deriv_order_78269)
        # Adding element type (line 768)
        # Getting the type of 'deriv_value' (line 768)
        deriv_value_78270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 50), 'deriv_value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 37), tuple_78268, deriv_value_78270)
        
        # Processing the call keyword arguments (line 768)
        kwargs_78271 = {}
        # Getting the type of 'validated_bc' (line 768)
        validated_bc_78266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 16), 'validated_bc', False)
        # Obtaining the member 'append' of a type (line 768)
        append_78267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 16), validated_bc_78266, 'append')
        # Calling append(args, kwargs) (line 768)
        append_call_result_78272 = invoke(stypy.reporting.localization.Localization(__file__, 768, 16), append_78267, *[tuple_78268], **kwargs_78271)
        
        # SSA join for if statement (line 739)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 770)
        tuple_78273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 770)
        # Adding element type (line 770)
        # Getting the type of 'validated_bc' (line 770)
        validated_bc_78274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 15), 'validated_bc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 770, 15), tuple_78273, validated_bc_78274)
        # Adding element type (line 770)
        # Getting the type of 'y' (line 770)
        y_78275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 29), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 770, 15), tuple_78273, y_78275)
        
        # Assigning a type to the variable 'stypy_return_type' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'stypy_return_type', tuple_78273)
        
        # ################# End of '_validate_bc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_validate_bc' in the type store
        # Getting the type of 'stypy_return_type' (line 705)
        stypy_return_type_78276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_78276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_validate_bc'
        return stypy_return_type_78276


# Assigning a type to the variable 'CubicSpline' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'CubicSpline', CubicSpline)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
