
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import print_function, division, absolute_import
2: 
3: __all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde',
4:            'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']
5: 
6: import warnings
7: 
8: import numpy as np
9: 
10: from ._fitpack_impl import bisplrep, bisplev, dblint
11: from . import _fitpack_impl as _impl
12: from ._bsplines import BSpline
13: 
14: 
15: def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
16:             full_output=0, nest=None, per=0, quiet=1):
17:     '''
18:     Find the B-spline representation of an N-dimensional curve.
19: 
20:     Given a list of N rank-1 arrays, `x`, which represent a curve in
21:     N-dimensional space parametrized by `u`, find a smooth approximating
22:     spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.
23: 
24:     Parameters
25:     ----------
26:     x : array_like
27:         A list of sample vector arrays representing the curve.
28:     w : array_like, optional
29:         Strictly positive rank-1 array of weights the same length as `x[0]`.
30:         The weights are used in computing the weighted least-squares spline
31:         fit. If the errors in the `x` values have standard-deviation given by
32:         the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
33:     u : array_like, optional
34:         An array of parameter values. If not given, these values are
35:         calculated automatically as ``M = len(x[0])``, where
36: 
37:             v[0] = 0
38: 
39:             v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)
40: 
41:             u[i] = v[i] / v[M-1]
42: 
43:     ub, ue : int, optional
44:         The end-points of the parameters interval.  Defaults to
45:         u[0] and u[-1].
46:     k : int, optional
47:         Degree of the spline. Cubic splines are recommended.
48:         Even values of `k` should be avoided especially with a small s-value.
49:         ``1 <= k <= 5``, default is 3.
50:     task : int, optional
51:         If task==0 (default), find t and c for a given smoothing factor, s.
52:         If task==1, find t and c for another value of the smoothing factor, s.
53:         There must have been a previous call with task=0 or task=1
54:         for the same set of data.
55:         If task=-1 find the weighted least square spline for a given set of
56:         knots, t.
57:     s : float, optional
58:         A smoothing condition.  The amount of smoothness is determined by
59:         satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
60:         where g(x) is the smoothed interpolation of (x,y).  The user can
61:         use `s` to control the trade-off between closeness and smoothness
62:         of fit.  Larger `s` means more smoothing while smaller values of `s`
63:         indicate less smoothing. Recommended values of `s` depend on the
64:         weights, w.  If the weights represent the inverse of the
65:         standard-deviation of y, then a good `s` value should be found in
66:         the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
67:         data points in x, y, and w.
68:     t : int, optional
69:         The knots needed for task=-1.
70:     full_output : int, optional
71:         If non-zero, then return optional outputs.
72:     nest : int, optional
73:         An over-estimate of the total number of knots of the spline to
74:         help in determining the storage space.  By default nest=m/2.
75:         Always large enough is nest=m+k+1.
76:     per : int, optional
77:        If non-zero, data points are considered periodic with period
78:        ``x[m-1] - x[0]`` and a smooth periodic spline approximation is
79:        returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.
80:     quiet : int, optional
81:          Non-zero to suppress messages.
82:          This parameter is deprecated; use standard Python warning filters
83:          instead.
84: 
85:     Returns
86:     -------
87:     tck : tuple
88:         (t,c,k) a tuple containing the vector of knots, the B-spline
89:         coefficients, and the degree of the spline.
90:     u : array
91:         An array of the values of the parameter.
92:     fp : float
93:         The weighted sum of squared residuals of the spline approximation.
94:     ier : int
95:         An integer flag about splrep success.  Success is indicated
96:         if ier<=0. If ier in [1,2,3] an error occurred but was not raised.
97:         Otherwise an error is raised.
98:     msg : str
99:         A message corresponding to the integer flag, ier.
100: 
101:     See Also
102:     --------
103:     splrep, splev, sproot, spalde, splint,
104:     bisplrep, bisplev
105:     UnivariateSpline, BivariateSpline
106:     BSpline
107:     make_interp_spline
108: 
109:     Notes
110:     -----
111:     See `splev` for evaluation of the spline and its derivatives.
112:     The number of dimensions N must be smaller than 11.
113: 
114:     The number of coefficients in the `c` array is ``k+1`` less then the number
115:     of knots, ``len(t)``. This is in contrast with `splrep`, which zero-pads
116:     the array of coefficients to have the same length as the array of knots.
117:     These additional coefficients are ignored by evaluation routines, `splev`
118:     and `BSpline`.
119: 
120:     References
121:     ----------
122:     .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
123:         parametric splines, Computer Graphics and Image Processing",
124:         20 (1982) 171-184.
125:     .. [2] P. Dierckx, "Algorithms for smoothing data with periodic and
126:         parametric splines", report tw55, Dept. Computer Science,
127:         K.U.Leuven, 1981.
128:     .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs on
129:         Numerical Analysis, Oxford University Press, 1993.
130: 
131:     Examples
132:     --------
133:     Generate a discretization of a limacon curve in the polar coordinates:
134: 
135:     >>> phi = np.linspace(0, 2.*np.pi, 40)
136:     >>> r = 0.5 + np.cos(phi)         # polar coords
137:     >>> x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian
138: 
139:     And interpolate:
140: 
141:     >>> from scipy.interpolate import splprep, splev
142:     >>> tck, u = splprep([x, y], s=0)
143:     >>> new_points = splev(u, tck)
144: 
145:     Notice that (i) we force interpolation by using `s=0`,
146:     (ii) the parameterization, ``u``, is generated automatically.
147:     Now plot the result:
148: 
149:     >>> import matplotlib.pyplot as plt
150:     >>> fig, ax = plt.subplots()
151:     >>> ax.plot(x, y, 'ro')
152:     >>> ax.plot(new_points[0], new_points[1], 'r-')
153:     >>> plt.show()
154: 
155:     '''
156:     res = _impl.splprep(x, w, u, ub, ue, k, task, s, t, full_output, nest, per,
157:                         quiet)
158:     return res
159: 
160: 
161: def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
162:            full_output=0, per=0, quiet=1):
163:     '''
164:     Find the B-spline representation of 1-D curve.
165: 
166:     Given the set of data points ``(x[i], y[i])`` determine a smooth spline
167:     approximation of degree k on the interval ``xb <= x <= xe``.
168: 
169:     Parameters
170:     ----------
171:     x, y : array_like
172:         The data points defining a curve y = f(x).
173:     w : array_like, optional
174:         Strictly positive rank-1 array of weights the same length as x and y.
175:         The weights are used in computing the weighted least-squares spline
176:         fit. If the errors in the y values have standard-deviation given by the
177:         vector d, then w should be 1/d. Default is ones(len(x)).
178:     xb, xe : float, optional
179:         The interval to fit.  If None, these default to x[0] and x[-1]
180:         respectively.
181:     k : int, optional
182:         The degree of the spline fit. It is recommended to use cubic splines.
183:         Even values of k should be avoided especially with small s values.
184:         1 <= k <= 5
185:     task : {1, 0, -1}, optional
186:         If task==0 find t and c for a given smoothing factor, s.
187: 
188:         If task==1 find t and c for another value of the smoothing factor, s.
189:         There must have been a previous call with task=0 or task=1 for the same
190:         set of data (t will be stored an used internally)
191: 
192:         If task=-1 find the weighted least square spline for a given set of
193:         knots, t. These should be interior knots as knots on the ends will be
194:         added automatically.
195:     s : float, optional
196:         A smoothing condition. The amount of smoothness is determined by
197:         satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x)
198:         is the smoothed interpolation of (x,y). The user can use s to control
199:         the tradeoff between closeness and smoothness of fit. Larger s means
200:         more smoothing while smaller values of s indicate less smoothing.
201:         Recommended values of s depend on the weights, w. If the weights
202:         represent the inverse of the standard-deviation of y, then a good s
203:         value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is
204:         the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if
205:         weights are supplied. s = 0.0 (interpolating) if no weights are
206:         supplied.
207:     t : array_like, optional
208:         The knots needed for task=-1. If given then task is automatically set
209:         to -1.
210:     full_output : bool, optional
211:         If non-zero, then return optional outputs.
212:     per : bool, optional
213:         If non-zero, data points are considered periodic with period x[m-1] -
214:         x[0] and a smooth periodic spline approximation is returned. Values of
215:         y[m-1] and w[m-1] are not used.
216:     quiet : bool, optional
217:         Non-zero to suppress messages.
218:         This parameter is deprecated; use standard Python warning filters
219:         instead.
220: 
221:     Returns
222:     -------
223:     tck : tuple
224:         A tuple (t,c,k) containing the vector of knots, the B-spline
225:         coefficients, and the degree of the spline.
226:     fp : array, optional
227:         The weighted sum of squared residuals of the spline approximation.
228:     ier : int, optional
229:         An integer flag about splrep success. Success is indicated if ier<=0.
230:         If ier in [1,2,3] an error occurred but was not raised. Otherwise an
231:         error is raised.
232:     msg : str, optional
233:         A message corresponding to the integer flag, ier.
234: 
235:     See Also
236:     --------
237:     UnivariateSpline, BivariateSpline
238:     splprep, splev, sproot, spalde, splint
239:     bisplrep, bisplev
240:     BSpline
241:     make_interp_spline
242: 
243:     Notes
244:     -----
245:     See `splev` for evaluation of the spline and its derivatives. Uses the
246:     FORTRAN routine ``curfit`` from FITPACK.
247: 
248:     The user is responsible for assuring that the values of `x` are unique.
249:     Otherwise, `splrep` will not return sensible results.
250: 
251:     If provided, knots `t` must satisfy the Schoenberg-Whitney conditions,
252:     i.e., there must be a subset of data points ``x[j]`` such that
253:     ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.
254: 
255:     This routine zero-pads the coefficients array ``c`` to have the same length
256:     as the array of knots ``t`` (the trailing ``k + 1`` coefficients are ignored
257:     by the evaluation routines, `splev` and `BSpline`.) This is in contrast with
258:     `splprep`, which does not zero-pad the coefficients.
259: 
260:     References
261:     ----------
262:     Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:
263: 
264:     .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and
265:        integration of experimental data using spline functions",
266:        J.Comp.Appl.Maths 1 (1975) 165-184.
267:     .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular
268:        grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)
269:        1286-1304.
270:     .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline
271:        functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.
272:     .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on
273:        Numerical Analysis, Oxford University Press, 1993.
274: 
275:     Examples
276:     --------
277: 
278:     >>> import matplotlib.pyplot as plt
279:     >>> from scipy.interpolate import splev, splrep
280:     >>> x = np.linspace(0, 10, 10)
281:     >>> y = np.sin(x)
282:     >>> spl = splrep(x, y)
283:     >>> x2 = np.linspace(0, 10, 200)
284:     >>> y2 = splev(x2, spl)
285:     >>> plt.plot(x, y, 'o', x2, y2)
286:     >>> plt.show()
287: 
288:     '''
289:     res = _impl.splrep(x, y, w, xb, xe, k, task, s, t, full_output, per, quiet)
290:     return res
291: 
292: 
293: def splev(x, tck, der=0, ext=0):
294:     '''
295:     Evaluate a B-spline or its derivatives.
296: 
297:     Given the knots and coefficients of a B-spline representation, evaluate
298:     the value of the smoothing polynomial and its derivatives.  This is a
299:     wrapper around the FORTRAN routines splev and splder of FITPACK.
300: 
301:     Parameters
302:     ----------
303:     x : array_like
304:         An array of points at which to return the value of the smoothed
305:         spline or its derivatives.  If `tck` was returned from `splprep`,
306:         then the parameter values, u should be given.
307:     tck : 3-tuple or a BSpline object
308:         If a tuple, then it should be a sequence of length 3 returned by
309:         `splrep` or `splprep` containing the knots, coefficients, and degree
310:         of the spline. (Also see Notes.)
311:     der : int, optional
312:         The order of derivative of the spline to compute (must be less than
313:         or equal to k).
314:     ext : int, optional
315:         Controls the value returned for elements of ``x`` not in the
316:         interval defined by the knot sequence.
317: 
318:         * if ext=0, return the extrapolated value.
319:         * if ext=1, return 0
320:         * if ext=2, raise a ValueError
321:         * if ext=3, return the boundary value.
322: 
323:         The default value is 0.
324: 
325:     Returns
326:     -------
327:     y : ndarray or list of ndarrays
328:         An array of values representing the spline function evaluated at
329:         the points in `x`.  If `tck` was returned from `splprep`, then this
330:         is a list of arrays representing the curve in N-dimensional space.
331: 
332:     Notes
333:     -----
334:     Manipulating the tck-tuples directly is not recommended. In new code,
335:     prefer using `BSpline` objects.
336: 
337:     See Also
338:     --------
339:     splprep, splrep, sproot, spalde, splint
340:     bisplrep, bisplev
341:     BSpline
342: 
343:     References
344:     ----------
345:     .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
346:         Theory, 6, p.50-62, 1972.
347:     .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
348:         Applics, 10, p.134-149, 1972.
349:     .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
350:         on Numerical Analysis, Oxford University Press, 1993.
351: 
352:     '''
353:     if isinstance(tck, BSpline):
354:         if tck.c.ndim > 1:
355:             mesg = ("Calling splev() with BSpline objects with c.ndim > 1 is "
356:                    "not recommended. Use BSpline.__call__(x) instead.")
357:             warnings.warn(mesg, DeprecationWarning)
358: 
359:         # remap the out-of-bounds behavior
360:         try:
361:             extrapolate = {0: True, }[ext]
362:         except KeyError:
363:             raise ValueError("Extrapolation mode %s is not supported "
364:                              "by BSpline." % ext)
365: 
366:         return tck(x, der, extrapolate=extrapolate)
367:     else:
368:         return _impl.splev(x, tck, der, ext)
369: 
370: 
371: def splint(a, b, tck, full_output=0):
372:     '''
373:     Evaluate the definite integral of a B-spline between two given points.
374: 
375:     Parameters
376:     ----------
377:     a, b : float
378:         The end-points of the integration interval.
379:     tck : tuple or a BSpline instance
380:         If a tuple, then it should be a sequence of length 3, containing the
381:         vector of knots, the B-spline coefficients, and the degree of the
382:         spline (see `splev`).
383:     full_output : int, optional
384:         Non-zero to return optional output.
385: 
386:     Returns
387:     -------
388:     integral : float
389:         The resulting integral.
390:     wrk : ndarray
391:         An array containing the integrals of the normalized B-splines
392:         defined on the set of knots.
393:         (Only returned if `full_output` is non-zero)
394: 
395:     Notes
396:     -----
397:     `splint` silently assumes that the spline function is zero outside the data
398:     interval (`a`, `b`).
399: 
400:     Manipulating the tck-tuples directly is not recommended. In new code,
401:     prefer using the `BSpline` objects.
402: 
403:     See Also
404:     --------
405:     splprep, splrep, sproot, spalde, splev
406:     bisplrep, bisplev
407:     BSpline
408: 
409:     References
410:     ----------
411:     .. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines",
412:         J. Inst. Maths Applics, 17, p.37-41, 1976.
413:     .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs
414:         on Numerical Analysis, Oxford University Press, 1993.
415: 
416:     '''
417:     if isinstance(tck, BSpline):
418:         if tck.c.ndim > 1:
419:             mesg = ("Calling splint() with BSpline objects with c.ndim > 1 is "
420:                    "not recommended. Use BSpline.integrate() instead.")
421:             warnings.warn(mesg, DeprecationWarning)
422: 
423:         if full_output != 0:
424:             mesg = ("full_output = %s is not supported. Proceeding as if "
425:                     "full_output = 0" % full_output)
426: 
427:         return tck.integrate(a, b, extrapolate=False)
428:     else:
429:         return _impl.splint(a, b, tck, full_output)
430: 
431: 
432: def sproot(tck, mest=10):
433:     '''
434:     Find the roots of a cubic B-spline.
435: 
436:     Given the knots (>=8) and coefficients of a cubic B-spline return the
437:     roots of the spline.
438: 
439:     Parameters
440:     ----------
441:     tck : tuple or a BSpline object
442:         If a tuple, then it should be a sequence of length 3, containing the
443:         vector of knots, the B-spline coefficients, and the degree of the
444:         spline.
445:         The number of knots must be >= 8, and the degree must be 3.
446:         The knots must be a montonically increasing sequence.
447:     mest : int, optional
448:         An estimate of the number of zeros (Default is 10).
449: 
450:     Returns
451:     -------
452:     zeros : ndarray
453:         An array giving the roots of the spline.
454: 
455:     Notes
456:     -----
457:     Manipulating the tck-tuples directly is not recommended. In new code,
458:     prefer using the `BSpline` objects.
459: 
460:     See also
461:     --------
462:     splprep, splrep, splint, spalde, splev
463:     bisplrep, bisplev
464:     BSpline
465: 
466: 
467:     References
468:     ----------
469:     .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
470:         Theory, 6, p.50-62, 1972.
471:     .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
472:         Applics, 10, p.134-149, 1972.
473:     .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
474:         on Numerical Analysis, Oxford University Press, 1993.
475: 
476:     '''
477:     if isinstance(tck, BSpline):
478:         if tck.c.ndim > 1:
479:             mesg = ("Calling sproot() with BSpline objects with c.ndim > 1 is "
480:                     "not recommended.")
481:             warnings.warn(mesg, DeprecationWarning)
482: 
483:         t, c, k = tck.tck
484: 
485:         # _impl.sproot expects the interpolation axis to be last, so roll it.
486:         # NB: This transpose is a no-op if c is 1D.
487:         sh = tuple(range(c.ndim))
488:         c = c.transpose(sh[1:] + (0,))
489:         return _impl.sproot((t, c, k), mest)
490:     else:
491:         return _impl.sproot(tck, mest)
492: 
493: 
494: def spalde(x, tck):
495:     '''
496:     Evaluate all derivatives of a B-spline.
497: 
498:     Given the knots and coefficients of a cubic B-spline compute all
499:     derivatives up to order k at a point (or set of points).
500: 
501:     Parameters
502:     ----------
503:     x : array_like
504:         A point or a set of points at which to evaluate the derivatives.
505:         Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.
506:     tck : tuple
507:         A tuple ``(t, c, k)``, containing the vector of knots, the B-spline
508:         coefficients, and the degree of the spline (see `splev`).
509: 
510:     Returns
511:     -------
512:     results : {ndarray, list of ndarrays}
513:         An array (or a list of arrays) containing all derivatives
514:         up to order k inclusive for each point `x`.
515: 
516:     See Also
517:     --------
518:     splprep, splrep, splint, sproot, splev, bisplrep, bisplev,
519:     BSpline
520: 
521:     References
522:     ----------
523:     .. [1] C. de Boor: On calculating with b-splines, J. Approximation Theory
524:        6 (1972) 50-62.
525:     .. [2] M. G. Cox : The numerical evaluation of b-splines, J. Inst. Maths
526:        applics 10 (1972) 134-149.
527:     .. [3] P. Dierckx : Curve and surface fitting with splines, Monographs on
528:        Numerical Analysis, Oxford University Press, 1993.
529: 
530:     '''
531:     if isinstance(tck, BSpline):
532:         raise TypeError("spalde does not accept BSpline instances.")
533:     else:
534:         return _impl.spalde(x, tck)
535: 
536: 
537: def insert(x, tck, m=1, per=0):
538:     '''
539:     Insert knots into a B-spline.
540: 
541:     Given the knots and coefficients of a B-spline representation, create a
542:     new B-spline with a knot inserted `m` times at point `x`.
543:     This is a wrapper around the FORTRAN routine insert of FITPACK.
544: 
545:     Parameters
546:     ----------
547:     x (u) : array_like
548:         A 1-D point at which to insert a new knot(s).  If `tck` was returned
549:         from ``splprep``, then the parameter values, u should be given.
550:     tck : a `BSpline` instance or a tuple
551:         If tuple, then it is expected to be a tuple (t,c,k) containing
552:         the vector of knots, the B-spline coefficients, and the degree of
553:         the spline.
554:     m : int, optional
555:         The number of times to insert the given knot (its multiplicity).
556:         Default is 1.
557:     per : int, optional
558:         If non-zero, the input spline is considered periodic.
559: 
560:     Returns
561:     -------
562:     BSpline instance or a tuple
563:         A new B-spline with knots t, coefficients c, and degree k.
564:         ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.
565:         In case of a periodic spline (``per != 0``) there must be
566:         either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``
567:         or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.
568:         A tuple is returned iff the input argument `tck` is a tuple, otherwise
569:         a BSpline object is constructed and returned.
570: 
571:     Notes
572:     -----
573:     Based on algorithms from [1]_ and [2]_.
574: 
575:     Manipulating the tck-tuples directly is not recommended. In new code,
576:     prefer using the `BSpline` objects.
577: 
578:     References
579:     ----------
580:     .. [1] W. Boehm, "Inserting new knots into b-spline curves.",
581:         Computer Aided Design, 12, p.199-201, 1980.
582:     .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on
583:         Numerical Analysis", Oxford University Press, 1993.
584: 
585:     '''
586:     if isinstance(tck, BSpline):
587: 
588:         t, c, k = tck.tck
589: 
590:         # FITPACK expects the interpolation axis to be last, so roll it over
591:         # NB: if c array is 1D, transposes are no-ops
592:         sh = tuple(range(c.ndim))
593:         c = c.transpose(sh[1:] + (0,))
594:         t_, c_, k_ = _impl.insert(x, (t, c, k), m, per)
595: 
596:         # and roll the last axis back
597:         c_ = np.asarray(c_)
598:         c_ = c_.transpose((sh[-1],) + sh[:-1])
599:         return BSpline(t_, c_, k_)
600:     else:
601:         return _impl.insert(x, tck, m, per)
602: 
603: 
604: def splder(tck, n=1):
605:     '''
606:     Compute the spline representation of the derivative of a given spline
607: 
608:     Parameters
609:     ----------
610:     tck : BSpline instance or a tuple of (t, c, k)
611:         Spline whose derivative to compute
612:     n : int, optional
613:         Order of derivative to evaluate. Default: 1
614: 
615:     Returns
616:     -------
617:     `BSpline` instance or tuple
618:         Spline of order k2=k-n representing the derivative
619:         of the input spline.
620:         A tuple is returned iff the input argument `tck` is a tuple, otherwise
621:         a BSpline object is constructed and returned.
622: 
623:     Notes
624:     -----
625: 
626:     .. versionadded:: 0.13.0
627: 
628:     See Also
629:     --------
630:     splantider, splev, spalde
631:     BSpline
632: 
633:     Examples
634:     --------
635:     This can be used for finding maxima of a curve:
636: 
637:     >>> from scipy.interpolate import splrep, splder, sproot
638:     >>> x = np.linspace(0, 10, 70)
639:     >>> y = np.sin(x)
640:     >>> spl = splrep(x, y, k=4)
641: 
642:     Now, differentiate the spline and find the zeros of the
643:     derivative. (NB: `sproot` only works for order 3 splines, so we
644:     fit an order 4 spline):
645: 
646:     >>> dspl = splder(spl)
647:     >>> sproot(dspl) / np.pi
648:     array([ 0.50000001,  1.5       ,  2.49999998])
649: 
650:     This agrees well with roots :math:`\\pi/2 + n\\pi` of
651:     :math:`\\cos(x) = \\sin'(x)`.
652: 
653:     '''
654:     if isinstance(tck, BSpline):
655:         return tck.derivative(n)
656:     else:
657:         return _impl.splder(tck, n)
658: 
659: 
660: def splantider(tck, n=1):
661:     '''
662:     Compute the spline for the antiderivative (integral) of a given spline.
663: 
664:     Parameters
665:     ----------
666:     tck : BSpline instance or a tuple of (t, c, k)
667:         Spline whose antiderivative to compute
668:     n : int, optional
669:         Order of antiderivative to evaluate. Default: 1
670: 
671:     Returns
672:     -------
673:     BSpline instance or a tuple of (t2, c2, k2)
674:         Spline of order k2=k+n representing the antiderivative of the input
675:         spline.
676:         A tuple is returned iff the input argument `tck` is a tuple, otherwise
677:         a BSpline object is constructed and returned.
678: 
679:     See Also
680:     --------
681:     splder, splev, spalde
682:     BSpline
683: 
684:     Notes
685:     -----
686:     The `splder` function is the inverse operation of this function.
687:     Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
688:     rounding error.
689: 
690:     .. versionadded:: 0.13.0
691: 
692:     Examples
693:     --------
694:     >>> from scipy.interpolate import splrep, splder, splantider, splev
695:     >>> x = np.linspace(0, np.pi/2, 70)
696:     >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
697:     >>> spl = splrep(x, y)
698: 
699:     The derivative is the inverse operation of the antiderivative,
700:     although some floating point error accumulates:
701: 
702:     >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
703:     (array(2.1565429877197317), array(2.1565429877201865))
704: 
705:     Antiderivative can be used to evaluate definite integrals:
706: 
707:     >>> ispl = splantider(spl)
708:     >>> splev(np.pi/2, ispl) - splev(0, ispl)
709:     2.2572053588768486
710: 
711:     This is indeed an approximation to the complete elliptic integral
712:     :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:
713: 
714:     >>> from scipy.special import ellipk
715:     >>> ellipk(0.8)
716:     2.2572053268208538
717: 
718:     '''
719:     if isinstance(tck, BSpline):
720:         return tck.antiderivative(n)
721:     else:
722:         return _impl.splantider(tck, n)
723: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 3):

# Assigning a List to a Name (line 3):
__all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde', 'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']
module_type_store.set_exportable_members(['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde', 'bisplrep', 'bisplev', 'insert', 'splder', 'splantider'])

# Obtaining an instance of the builtin type 'list' (line 3)
list_59085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_59086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'splrep')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59086)
# Adding element type (line 3)
str_59087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 21), 'str', 'splprep')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59087)
# Adding element type (line 3)
str_59088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 32), 'str', 'splev')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59088)
# Adding element type (line 3)
str_59089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 41), 'str', 'splint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59089)
# Adding element type (line 3)
str_59090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 51), 'str', 'sproot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59090)
# Adding element type (line 3)
str_59091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 61), 'str', 'spalde')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59091)
# Adding element type (line 3)
str_59092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'str', 'bisplrep')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59092)
# Adding element type (line 3)
str_59093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 23), 'str', 'bisplev')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59093)
# Adding element type (line 3)
str_59094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 34), 'str', 'insert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59094)
# Adding element type (line 3)
str_59095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 44), 'str', 'splder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59095)
# Adding element type (line 3)
str_59096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 54), 'str', 'splantider')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_59085, str_59096)

# Assigning a type to the variable '__all__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__all__', list_59085)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_59097 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_59097) is not StypyTypeError):

    if (import_59097 != 'pyd_module'):
        __import__(import_59097)
        sys_modules_59098 = sys.modules[import_59097]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_59098.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_59097)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.interpolate._fitpack_impl import bisplrep, bisplev, dblint' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_59099 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate._fitpack_impl')

if (type(import_59099) is not StypyTypeError):

    if (import_59099 != 'pyd_module'):
        __import__(import_59099)
        sys_modules_59100 = sys.modules[import_59099]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate._fitpack_impl', sys_modules_59100.module_type_store, module_type_store, ['bisplrep', 'bisplev', 'dblint'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_59100, sys_modules_59100.module_type_store, module_type_store)
    else:
        from scipy.interpolate._fitpack_impl import bisplrep, bisplev, dblint

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate._fitpack_impl', None, module_type_store, ['bisplrep', 'bisplev', 'dblint'], [bisplrep, bisplev, dblint])

else:
    # Assigning a type to the variable 'scipy.interpolate._fitpack_impl' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate._fitpack_impl', import_59099)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.interpolate import _impl' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_59101 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate')

if (type(import_59101) is not StypyTypeError):

    if (import_59101 != 'pyd_module'):
        __import__(import_59101)
        sys_modules_59102 = sys.modules[import_59101]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate', sys_modules_59102.module_type_store, module_type_store, ['_fitpack_impl'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_59102, sys_modules_59102.module_type_store, module_type_store)
    else:
        from scipy.interpolate import _fitpack_impl as _impl

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate', None, module_type_store, ['_fitpack_impl'], [_impl])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.interpolate', import_59101)

# Adding an alias
module_type_store.add_alias('_impl', '_fitpack_impl')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.interpolate._bsplines import BSpline' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_59103 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.interpolate._bsplines')

if (type(import_59103) is not StypyTypeError):

    if (import_59103 != 'pyd_module'):
        __import__(import_59103)
        sys_modules_59104 = sys.modules[import_59103]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.interpolate._bsplines', sys_modules_59104.module_type_store, module_type_store, ['BSpline'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_59104, sys_modules_59104.module_type_store, module_type_store)
    else:
        from scipy.interpolate._bsplines import BSpline

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.interpolate._bsplines', None, module_type_store, ['BSpline'], [BSpline])

else:
    # Assigning a type to the variable 'scipy.interpolate._bsplines' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.interpolate._bsplines', import_59103)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


@norecursion
def splprep(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 15)
    None_59105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'None')
    # Getting the type of 'None' (line 15)
    None_59106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'None')
    # Getting the type of 'None' (line 15)
    None_59107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 34), 'None')
    # Getting the type of 'None' (line 15)
    None_59108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 43), 'None')
    int_59109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 51), 'int')
    int_59110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 59), 'int')
    # Getting the type of 'None' (line 15)
    None_59111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 64), 'None')
    # Getting the type of 'None' (line 15)
    None_59112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 72), 'None')
    int_59113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'int')
    # Getting the type of 'None' (line 16)
    None_59114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 32), 'None')
    int_59115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 42), 'int')
    int_59116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 51), 'int')
    defaults = [None_59105, None_59106, None_59107, None_59108, int_59109, int_59110, None_59111, None_59112, int_59113, None_59114, int_59115, int_59116]
    # Create a new context for function 'splprep'
    module_type_store = module_type_store.open_function_context('splprep', 15, 0, False)
    
    # Passed parameters checking function
    splprep.stypy_localization = localization
    splprep.stypy_type_of_self = None
    splprep.stypy_type_store = module_type_store
    splprep.stypy_function_name = 'splprep'
    splprep.stypy_param_names_list = ['x', 'w', 'u', 'ub', 'ue', 'k', 'task', 's', 't', 'full_output', 'nest', 'per', 'quiet']
    splprep.stypy_varargs_param_name = None
    splprep.stypy_kwargs_param_name = None
    splprep.stypy_call_defaults = defaults
    splprep.stypy_call_varargs = varargs
    splprep.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splprep', ['x', 'w', 'u', 'ub', 'ue', 'k', 'task', 's', 't', 'full_output', 'nest', 'per', 'quiet'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splprep', localization, ['x', 'w', 'u', 'ub', 'ue', 'k', 'task', 's', 't', 'full_output', 'nest', 'per', 'quiet'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splprep(...)' code ##################

    str_59117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, (-1)), 'str', '\n    Find the B-spline representation of an N-dimensional curve.\n\n    Given a list of N rank-1 arrays, `x`, which represent a curve in\n    N-dimensional space parametrized by `u`, find a smooth approximating\n    spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.\n\n    Parameters\n    ----------\n    x : array_like\n        A list of sample vector arrays representing the curve.\n    w : array_like, optional\n        Strictly positive rank-1 array of weights the same length as `x[0]`.\n        The weights are used in computing the weighted least-squares spline\n        fit. If the errors in the `x` values have standard-deviation given by\n        the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.\n    u : array_like, optional\n        An array of parameter values. If not given, these values are\n        calculated automatically as ``M = len(x[0])``, where\n\n            v[0] = 0\n\n            v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)\n\n            u[i] = v[i] / v[M-1]\n\n    ub, ue : int, optional\n        The end-points of the parameters interval.  Defaults to\n        u[0] and u[-1].\n    k : int, optional\n        Degree of the spline. Cubic splines are recommended.\n        Even values of `k` should be avoided especially with a small s-value.\n        ``1 <= k <= 5``, default is 3.\n    task : int, optional\n        If task==0 (default), find t and c for a given smoothing factor, s.\n        If task==1, find t and c for another value of the smoothing factor, s.\n        There must have been a previous call with task=0 or task=1\n        for the same set of data.\n        If task=-1 find the weighted least square spline for a given set of\n        knots, t.\n    s : float, optional\n        A smoothing condition.  The amount of smoothness is determined by\n        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,\n        where g(x) is the smoothed interpolation of (x,y).  The user can\n        use `s` to control the trade-off between closeness and smoothness\n        of fit.  Larger `s` means more smoothing while smaller values of `s`\n        indicate less smoothing. Recommended values of `s` depend on the\n        weights, w.  If the weights represent the inverse of the\n        standard-deviation of y, then a good `s` value should be found in\n        the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of\n        data points in x, y, and w.\n    t : int, optional\n        The knots needed for task=-1.\n    full_output : int, optional\n        If non-zero, then return optional outputs.\n    nest : int, optional\n        An over-estimate of the total number of knots of the spline to\n        help in determining the storage space.  By default nest=m/2.\n        Always large enough is nest=m+k+1.\n    per : int, optional\n       If non-zero, data points are considered periodic with period\n       ``x[m-1] - x[0]`` and a smooth periodic spline approximation is\n       returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.\n    quiet : int, optional\n         Non-zero to suppress messages.\n         This parameter is deprecated; use standard Python warning filters\n         instead.\n\n    Returns\n    -------\n    tck : tuple\n        (t,c,k) a tuple containing the vector of knots, the B-spline\n        coefficients, and the degree of the spline.\n    u : array\n        An array of the values of the parameter.\n    fp : float\n        The weighted sum of squared residuals of the spline approximation.\n    ier : int\n        An integer flag about splrep success.  Success is indicated\n        if ier<=0. If ier in [1,2,3] an error occurred but was not raised.\n        Otherwise an error is raised.\n    msg : str\n        A message corresponding to the integer flag, ier.\n\n    See Also\n    --------\n    splrep, splev, sproot, spalde, splint,\n    bisplrep, bisplev\n    UnivariateSpline, BivariateSpline\n    BSpline\n    make_interp_spline\n\n    Notes\n    -----\n    See `splev` for evaluation of the spline and its derivatives.\n    The number of dimensions N must be smaller than 11.\n\n    The number of coefficients in the `c` array is ``k+1`` less then the number\n    of knots, ``len(t)``. This is in contrast with `splrep`, which zero-pads\n    the array of coefficients to have the same length as the array of knots.\n    These additional coefficients are ignored by evaluation routines, `splev`\n    and `BSpline`.\n\n    References\n    ----------\n    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and\n        parametric splines, Computer Graphics and Image Processing",\n        20 (1982) 171-184.\n    .. [2] P. Dierckx, "Algorithms for smoothing data with periodic and\n        parametric splines", report tw55, Dept. Computer Science,\n        K.U.Leuven, 1981.\n    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs on\n        Numerical Analysis, Oxford University Press, 1993.\n\n    Examples\n    --------\n    Generate a discretization of a limacon curve in the polar coordinates:\n\n    >>> phi = np.linspace(0, 2.*np.pi, 40)\n    >>> r = 0.5 + np.cos(phi)         # polar coords\n    >>> x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian\n\n    And interpolate:\n\n    >>> from scipy.interpolate import splprep, splev\n    >>> tck, u = splprep([x, y], s=0)\n    >>> new_points = splev(u, tck)\n\n    Notice that (i) we force interpolation by using `s=0`,\n    (ii) the parameterization, ``u``, is generated automatically.\n    Now plot the result:\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> ax.plot(x, y, \'ro\')\n    >>> ax.plot(new_points[0], new_points[1], \'r-\')\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 156):
    
    # Assigning a Call to a Name (line 156):
    
    # Call to splprep(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'x' (line 156)
    x_59120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'x', False)
    # Getting the type of 'w' (line 156)
    w_59121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'w', False)
    # Getting the type of 'u' (line 156)
    u_59122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'u', False)
    # Getting the type of 'ub' (line 156)
    ub_59123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 33), 'ub', False)
    # Getting the type of 'ue' (line 156)
    ue_59124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'ue', False)
    # Getting the type of 'k' (line 156)
    k_59125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 41), 'k', False)
    # Getting the type of 'task' (line 156)
    task_59126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 44), 'task', False)
    # Getting the type of 's' (line 156)
    s_59127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 50), 's', False)
    # Getting the type of 't' (line 156)
    t_59128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 't', False)
    # Getting the type of 'full_output' (line 156)
    full_output_59129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 56), 'full_output', False)
    # Getting the type of 'nest' (line 156)
    nest_59130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 69), 'nest', False)
    # Getting the type of 'per' (line 156)
    per_59131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 75), 'per', False)
    # Getting the type of 'quiet' (line 157)
    quiet_59132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'quiet', False)
    # Processing the call keyword arguments (line 156)
    kwargs_59133 = {}
    # Getting the type of '_impl' (line 156)
    _impl_59118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 10), '_impl', False)
    # Obtaining the member 'splprep' of a type (line 156)
    splprep_59119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 10), _impl_59118, 'splprep')
    # Calling splprep(args, kwargs) (line 156)
    splprep_call_result_59134 = invoke(stypy.reporting.localization.Localization(__file__, 156, 10), splprep_59119, *[x_59120, w_59121, u_59122, ub_59123, ue_59124, k_59125, task_59126, s_59127, t_59128, full_output_59129, nest_59130, per_59131, quiet_59132], **kwargs_59133)
    
    # Assigning a type to the variable 'res' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'res', splprep_call_result_59134)
    # Getting the type of 'res' (line 158)
    res_59135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type', res_59135)
    
    # ################# End of 'splprep(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splprep' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_59136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59136)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splprep'
    return stypy_return_type_59136

# Assigning a type to the variable 'splprep' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'splprep', splprep)

@norecursion
def splrep(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 161)
    None_59137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'None')
    # Getting the type of 'None' (line 161)
    None_59138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'None')
    # Getting the type of 'None' (line 161)
    None_59139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'None')
    int_59140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 45), 'int')
    int_59141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 53), 'int')
    # Getting the type of 'None' (line 161)
    None_59142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 58), 'None')
    # Getting the type of 'None' (line 161)
    None_59143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 66), 'None')
    int_59144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'int')
    int_59145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 30), 'int')
    int_59146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 39), 'int')
    defaults = [None_59137, None_59138, None_59139, int_59140, int_59141, None_59142, None_59143, int_59144, int_59145, int_59146]
    # Create a new context for function 'splrep'
    module_type_store = module_type_store.open_function_context('splrep', 161, 0, False)
    
    # Passed parameters checking function
    splrep.stypy_localization = localization
    splrep.stypy_type_of_self = None
    splrep.stypy_type_store = module_type_store
    splrep.stypy_function_name = 'splrep'
    splrep.stypy_param_names_list = ['x', 'y', 'w', 'xb', 'xe', 'k', 'task', 's', 't', 'full_output', 'per', 'quiet']
    splrep.stypy_varargs_param_name = None
    splrep.stypy_kwargs_param_name = None
    splrep.stypy_call_defaults = defaults
    splrep.stypy_call_varargs = varargs
    splrep.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splrep', ['x', 'y', 'w', 'xb', 'xe', 'k', 'task', 's', 't', 'full_output', 'per', 'quiet'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splrep', localization, ['x', 'y', 'w', 'xb', 'xe', 'k', 'task', 's', 't', 'full_output', 'per', 'quiet'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splrep(...)' code ##################

    str_59147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'str', '\n    Find the B-spline representation of 1-D curve.\n\n    Given the set of data points ``(x[i], y[i])`` determine a smooth spline\n    approximation of degree k on the interval ``xb <= x <= xe``.\n\n    Parameters\n    ----------\n    x, y : array_like\n        The data points defining a curve y = f(x).\n    w : array_like, optional\n        Strictly positive rank-1 array of weights the same length as x and y.\n        The weights are used in computing the weighted least-squares spline\n        fit. If the errors in the y values have standard-deviation given by the\n        vector d, then w should be 1/d. Default is ones(len(x)).\n    xb, xe : float, optional\n        The interval to fit.  If None, these default to x[0] and x[-1]\n        respectively.\n    k : int, optional\n        The degree of the spline fit. It is recommended to use cubic splines.\n        Even values of k should be avoided especially with small s values.\n        1 <= k <= 5\n    task : {1, 0, -1}, optional\n        If task==0 find t and c for a given smoothing factor, s.\n\n        If task==1 find t and c for another value of the smoothing factor, s.\n        There must have been a previous call with task=0 or task=1 for the same\n        set of data (t will be stored an used internally)\n\n        If task=-1 find the weighted least square spline for a given set of\n        knots, t. These should be interior knots as knots on the ends will be\n        added automatically.\n    s : float, optional\n        A smoothing condition. The amount of smoothness is determined by\n        satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x)\n        is the smoothed interpolation of (x,y). The user can use s to control\n        the tradeoff between closeness and smoothness of fit. Larger s means\n        more smoothing while smaller values of s indicate less smoothing.\n        Recommended values of s depend on the weights, w. If the weights\n        represent the inverse of the standard-deviation of y, then a good s\n        value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is\n        the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if\n        weights are supplied. s = 0.0 (interpolating) if no weights are\n        supplied.\n    t : array_like, optional\n        The knots needed for task=-1. If given then task is automatically set\n        to -1.\n    full_output : bool, optional\n        If non-zero, then return optional outputs.\n    per : bool, optional\n        If non-zero, data points are considered periodic with period x[m-1] -\n        x[0] and a smooth periodic spline approximation is returned. Values of\n        y[m-1] and w[m-1] are not used.\n    quiet : bool, optional\n        Non-zero to suppress messages.\n        This parameter is deprecated; use standard Python warning filters\n        instead.\n\n    Returns\n    -------\n    tck : tuple\n        A tuple (t,c,k) containing the vector of knots, the B-spline\n        coefficients, and the degree of the spline.\n    fp : array, optional\n        The weighted sum of squared residuals of the spline approximation.\n    ier : int, optional\n        An integer flag about splrep success. Success is indicated if ier<=0.\n        If ier in [1,2,3] an error occurred but was not raised. Otherwise an\n        error is raised.\n    msg : str, optional\n        A message corresponding to the integer flag, ier.\n\n    See Also\n    --------\n    UnivariateSpline, BivariateSpline\n    splprep, splev, sproot, spalde, splint\n    bisplrep, bisplev\n    BSpline\n    make_interp_spline\n\n    Notes\n    -----\n    See `splev` for evaluation of the spline and its derivatives. Uses the\n    FORTRAN routine ``curfit`` from FITPACK.\n\n    The user is responsible for assuring that the values of `x` are unique.\n    Otherwise, `splrep` will not return sensible results.\n\n    If provided, knots `t` must satisfy the Schoenberg-Whitney conditions,\n    i.e., there must be a subset of data points ``x[j]`` such that\n    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.\n\n    This routine zero-pads the coefficients array ``c`` to have the same length\n    as the array of knots ``t`` (the trailing ``k + 1`` coefficients are ignored\n    by the evaluation routines, `splev` and `BSpline`.) This is in contrast with\n    `splprep`, which does not zero-pad the coefficients.\n\n    References\n    ----------\n    Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:\n\n    .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and\n       integration of experimental data using spline functions",\n       J.Comp.Appl.Maths 1 (1975) 165-184.\n    .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular\n       grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)\n       1286-1304.\n    .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline\n       functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.\n    .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on\n       Numerical Analysis, Oxford University Press, 1993.\n\n    Examples\n    --------\n\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.interpolate import splev, splrep\n    >>> x = np.linspace(0, 10, 10)\n    >>> y = np.sin(x)\n    >>> spl = splrep(x, y)\n    >>> x2 = np.linspace(0, 10, 200)\n    >>> y2 = splev(x2, spl)\n    >>> plt.plot(x, y, \'o\', x2, y2)\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to splrep(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'x' (line 289)
    x_59150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'x', False)
    # Getting the type of 'y' (line 289)
    y_59151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'y', False)
    # Getting the type of 'w' (line 289)
    w_59152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 29), 'w', False)
    # Getting the type of 'xb' (line 289)
    xb_59153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'xb', False)
    # Getting the type of 'xe' (line 289)
    xe_59154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 36), 'xe', False)
    # Getting the type of 'k' (line 289)
    k_59155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 40), 'k', False)
    # Getting the type of 'task' (line 289)
    task_59156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 43), 'task', False)
    # Getting the type of 's' (line 289)
    s_59157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 49), 's', False)
    # Getting the type of 't' (line 289)
    t_59158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 52), 't', False)
    # Getting the type of 'full_output' (line 289)
    full_output_59159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 55), 'full_output', False)
    # Getting the type of 'per' (line 289)
    per_59160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'per', False)
    # Getting the type of 'quiet' (line 289)
    quiet_59161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 73), 'quiet', False)
    # Processing the call keyword arguments (line 289)
    kwargs_59162 = {}
    # Getting the type of '_impl' (line 289)
    _impl_59148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 10), '_impl', False)
    # Obtaining the member 'splrep' of a type (line 289)
    splrep_59149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 10), _impl_59148, 'splrep')
    # Calling splrep(args, kwargs) (line 289)
    splrep_call_result_59163 = invoke(stypy.reporting.localization.Localization(__file__, 289, 10), splrep_59149, *[x_59150, y_59151, w_59152, xb_59153, xe_59154, k_59155, task_59156, s_59157, t_59158, full_output_59159, per_59160, quiet_59161], **kwargs_59162)
    
    # Assigning a type to the variable 'res' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'res', splrep_call_result_59163)
    # Getting the type of 'res' (line 290)
    res_59164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type', res_59164)
    
    # ################# End of 'splrep(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splrep' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_59165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59165)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splrep'
    return stypy_return_type_59165

# Assigning a type to the variable 'splrep' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'splrep', splrep)

@norecursion
def splev(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_59166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 22), 'int')
    int_59167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 29), 'int')
    defaults = [int_59166, int_59167]
    # Create a new context for function 'splev'
    module_type_store = module_type_store.open_function_context('splev', 293, 0, False)
    
    # Passed parameters checking function
    splev.stypy_localization = localization
    splev.stypy_type_of_self = None
    splev.stypy_type_store = module_type_store
    splev.stypy_function_name = 'splev'
    splev.stypy_param_names_list = ['x', 'tck', 'der', 'ext']
    splev.stypy_varargs_param_name = None
    splev.stypy_kwargs_param_name = None
    splev.stypy_call_defaults = defaults
    splev.stypy_call_varargs = varargs
    splev.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splev', ['x', 'tck', 'der', 'ext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splev', localization, ['x', 'tck', 'der', 'ext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splev(...)' code ##################

    str_59168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, (-1)), 'str', '\n    Evaluate a B-spline or its derivatives.\n\n    Given the knots and coefficients of a B-spline representation, evaluate\n    the value of the smoothing polynomial and its derivatives.  This is a\n    wrapper around the FORTRAN routines splev and splder of FITPACK.\n\n    Parameters\n    ----------\n    x : array_like\n        An array of points at which to return the value of the smoothed\n        spline or its derivatives.  If `tck` was returned from `splprep`,\n        then the parameter values, u should be given.\n    tck : 3-tuple or a BSpline object\n        If a tuple, then it should be a sequence of length 3 returned by\n        `splrep` or `splprep` containing the knots, coefficients, and degree\n        of the spline. (Also see Notes.)\n    der : int, optional\n        The order of derivative of the spline to compute (must be less than\n        or equal to k).\n    ext : int, optional\n        Controls the value returned for elements of ``x`` not in the\n        interval defined by the knot sequence.\n\n        * if ext=0, return the extrapolated value.\n        * if ext=1, return 0\n        * if ext=2, raise a ValueError\n        * if ext=3, return the boundary value.\n\n        The default value is 0.\n\n    Returns\n    -------\n    y : ndarray or list of ndarrays\n        An array of values representing the spline function evaluated at\n        the points in `x`.  If `tck` was returned from `splprep`, then this\n        is a list of arrays representing the curve in N-dimensional space.\n\n    Notes\n    -----\n    Manipulating the tck-tuples directly is not recommended. In new code,\n    prefer using `BSpline` objects.\n\n    See Also\n    --------\n    splprep, splrep, sproot, spalde, splint\n    bisplrep, bisplev\n    BSpline\n\n    References\n    ----------\n    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation\n        Theory, 6, p.50-62, 1972.\n    .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths\n        Applics, 10, p.134-149, 1972.\n    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs\n        on Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    
    # Call to isinstance(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'tck' (line 353)
    tck_59170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 18), 'tck', False)
    # Getting the type of 'BSpline' (line 353)
    BSpline_59171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 23), 'BSpline', False)
    # Processing the call keyword arguments (line 353)
    kwargs_59172 = {}
    # Getting the type of 'isinstance' (line 353)
    isinstance_59169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 353)
    isinstance_call_result_59173 = invoke(stypy.reporting.localization.Localization(__file__, 353, 7), isinstance_59169, *[tck_59170, BSpline_59171], **kwargs_59172)
    
    # Testing the type of an if condition (line 353)
    if_condition_59174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 4), isinstance_call_result_59173)
    # Assigning a type to the variable 'if_condition_59174' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'if_condition_59174', if_condition_59174)
    # SSA begins for if statement (line 353)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'tck' (line 354)
    tck_59175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'tck')
    # Obtaining the member 'c' of a type (line 354)
    c_59176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), tck_59175, 'c')
    # Obtaining the member 'ndim' of a type (line 354)
    ndim_59177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), c_59176, 'ndim')
    int_59178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 24), 'int')
    # Applying the binary operator '>' (line 354)
    result_gt_59179 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 11), '>', ndim_59177, int_59178)
    
    # Testing the type of an if condition (line 354)
    if_condition_59180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 8), result_gt_59179)
    # Assigning a type to the variable 'if_condition_59180' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'if_condition_59180', if_condition_59180)
    # SSA begins for if statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 355):
    
    # Assigning a Str to a Name (line 355):
    str_59181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 20), 'str', 'Calling splev() with BSpline objects with c.ndim > 1 is not recommended. Use BSpline.__call__(x) instead.')
    # Assigning a type to the variable 'mesg' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'mesg', str_59181)
    
    # Call to warn(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'mesg' (line 357)
    mesg_59184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 26), 'mesg', False)
    # Getting the type of 'DeprecationWarning' (line 357)
    DeprecationWarning_59185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 32), 'DeprecationWarning', False)
    # Processing the call keyword arguments (line 357)
    kwargs_59186 = {}
    # Getting the type of 'warnings' (line 357)
    warnings_59182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 357)
    warn_59183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 12), warnings_59182, 'warn')
    # Calling warn(args, kwargs) (line 357)
    warn_call_result_59187 = invoke(stypy.reporting.localization.Localization(__file__, 357, 12), warn_59183, *[mesg_59184, DeprecationWarning_59185], **kwargs_59186)
    
    # SSA join for if statement (line 354)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 361):
    
    # Assigning a Subscript to a Name (line 361):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ext' (line 361)
    ext_59188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 38), 'ext')
    
    # Obtaining an instance of the builtin type 'dict' (line 361)
    dict_59189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 26), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 361)
    # Adding element type (key, value) (line 361)
    int_59190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 27), 'int')
    # Getting the type of 'True' (line 361)
    True_59191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 30), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 26), dict_59189, (int_59190, True_59191))
    
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___59192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 26), dict_59189, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 361)
    subscript_call_result_59193 = invoke(stypy.reporting.localization.Localization(__file__, 361, 26), getitem___59192, ext_59188)
    
    # Assigning a type to the variable 'extrapolate' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'extrapolate', subscript_call_result_59193)
    # SSA branch for the except part of a try statement (line 360)
    # SSA branch for the except 'KeyError' branch of a try statement (line 360)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 363)
    # Processing the call arguments (line 363)
    str_59195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 29), 'str', 'Extrapolation mode %s is not supported by BSpline.')
    # Getting the type of 'ext' (line 364)
    ext_59196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 45), 'ext', False)
    # Applying the binary operator '%' (line 363)
    result_mod_59197 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 29), '%', str_59195, ext_59196)
    
    # Processing the call keyword arguments (line 363)
    kwargs_59198 = {}
    # Getting the type of 'ValueError' (line 363)
    ValueError_59194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 363)
    ValueError_call_result_59199 = invoke(stypy.reporting.localization.Localization(__file__, 363, 18), ValueError_59194, *[result_mod_59197], **kwargs_59198)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 363, 12), ValueError_call_result_59199, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 360)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to tck(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'x' (line 366)
    x_59201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'x', False)
    # Getting the type of 'der' (line 366)
    der_59202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'der', False)
    # Processing the call keyword arguments (line 366)
    # Getting the type of 'extrapolate' (line 366)
    extrapolate_59203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 39), 'extrapolate', False)
    keyword_59204 = extrapolate_59203
    kwargs_59205 = {'extrapolate': keyword_59204}
    # Getting the type of 'tck' (line 366)
    tck_59200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'tck', False)
    # Calling tck(args, kwargs) (line 366)
    tck_call_result_59206 = invoke(stypy.reporting.localization.Localization(__file__, 366, 15), tck_59200, *[x_59201, der_59202], **kwargs_59205)
    
    # Assigning a type to the variable 'stypy_return_type' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'stypy_return_type', tck_call_result_59206)
    # SSA branch for the else part of an if statement (line 353)
    module_type_store.open_ssa_branch('else')
    
    # Call to splev(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'x' (line 368)
    x_59209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 27), 'x', False)
    # Getting the type of 'tck' (line 368)
    tck_59210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 30), 'tck', False)
    # Getting the type of 'der' (line 368)
    der_59211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'der', False)
    # Getting the type of 'ext' (line 368)
    ext_59212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 40), 'ext', False)
    # Processing the call keyword arguments (line 368)
    kwargs_59213 = {}
    # Getting the type of '_impl' (line 368)
    _impl_59207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), '_impl', False)
    # Obtaining the member 'splev' of a type (line 368)
    splev_59208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), _impl_59207, 'splev')
    # Calling splev(args, kwargs) (line 368)
    splev_call_result_59214 = invoke(stypy.reporting.localization.Localization(__file__, 368, 15), splev_59208, *[x_59209, tck_59210, der_59211, ext_59212], **kwargs_59213)
    
    # Assigning a type to the variable 'stypy_return_type' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'stypy_return_type', splev_call_result_59214)
    # SSA join for if statement (line 353)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'splev(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splev' in the type store
    # Getting the type of 'stypy_return_type' (line 293)
    stypy_return_type_59215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59215)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splev'
    return stypy_return_type_59215

# Assigning a type to the variable 'splev' (line 293)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 0), 'splev', splev)

@norecursion
def splint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_59216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 34), 'int')
    defaults = [int_59216]
    # Create a new context for function 'splint'
    module_type_store = module_type_store.open_function_context('splint', 371, 0, False)
    
    # Passed parameters checking function
    splint.stypy_localization = localization
    splint.stypy_type_of_self = None
    splint.stypy_type_store = module_type_store
    splint.stypy_function_name = 'splint'
    splint.stypy_param_names_list = ['a', 'b', 'tck', 'full_output']
    splint.stypy_varargs_param_name = None
    splint.stypy_kwargs_param_name = None
    splint.stypy_call_defaults = defaults
    splint.stypy_call_varargs = varargs
    splint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splint', ['a', 'b', 'tck', 'full_output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splint', localization, ['a', 'b', 'tck', 'full_output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splint(...)' code ##################

    str_59217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, (-1)), 'str', '\n    Evaluate the definite integral of a B-spline between two given points.\n\n    Parameters\n    ----------\n    a, b : float\n        The end-points of the integration interval.\n    tck : tuple or a BSpline instance\n        If a tuple, then it should be a sequence of length 3, containing the\n        vector of knots, the B-spline coefficients, and the degree of the\n        spline (see `splev`).\n    full_output : int, optional\n        Non-zero to return optional output.\n\n    Returns\n    -------\n    integral : float\n        The resulting integral.\n    wrk : ndarray\n        An array containing the integrals of the normalized B-splines\n        defined on the set of knots.\n        (Only returned if `full_output` is non-zero)\n\n    Notes\n    -----\n    `splint` silently assumes that the spline function is zero outside the data\n    interval (`a`, `b`).\n\n    Manipulating the tck-tuples directly is not recommended. In new code,\n    prefer using the `BSpline` objects.\n\n    See Also\n    --------\n    splprep, splrep, sproot, spalde, splev\n    bisplrep, bisplev\n    BSpline\n\n    References\n    ----------\n    .. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines",\n        J. Inst. Maths Applics, 17, p.37-41, 1976.\n    .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs\n        on Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    
    # Call to isinstance(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of 'tck' (line 417)
    tck_59219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'tck', False)
    # Getting the type of 'BSpline' (line 417)
    BSpline_59220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 23), 'BSpline', False)
    # Processing the call keyword arguments (line 417)
    kwargs_59221 = {}
    # Getting the type of 'isinstance' (line 417)
    isinstance_59218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 417)
    isinstance_call_result_59222 = invoke(stypy.reporting.localization.Localization(__file__, 417, 7), isinstance_59218, *[tck_59219, BSpline_59220], **kwargs_59221)
    
    # Testing the type of an if condition (line 417)
    if_condition_59223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 4), isinstance_call_result_59222)
    # Assigning a type to the variable 'if_condition_59223' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'if_condition_59223', if_condition_59223)
    # SSA begins for if statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'tck' (line 418)
    tck_59224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), 'tck')
    # Obtaining the member 'c' of a type (line 418)
    c_59225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 11), tck_59224, 'c')
    # Obtaining the member 'ndim' of a type (line 418)
    ndim_59226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 11), c_59225, 'ndim')
    int_59227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 24), 'int')
    # Applying the binary operator '>' (line 418)
    result_gt_59228 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 11), '>', ndim_59226, int_59227)
    
    # Testing the type of an if condition (line 418)
    if_condition_59229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 8), result_gt_59228)
    # Assigning a type to the variable 'if_condition_59229' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'if_condition_59229', if_condition_59229)
    # SSA begins for if statement (line 418)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 419):
    
    # Assigning a Str to a Name (line 419):
    str_59230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 20), 'str', 'Calling splint() with BSpline objects with c.ndim > 1 is not recommended. Use BSpline.integrate() instead.')
    # Assigning a type to the variable 'mesg' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'mesg', str_59230)
    
    # Call to warn(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'mesg' (line 421)
    mesg_59233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 26), 'mesg', False)
    # Getting the type of 'DeprecationWarning' (line 421)
    DeprecationWarning_59234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 32), 'DeprecationWarning', False)
    # Processing the call keyword arguments (line 421)
    kwargs_59235 = {}
    # Getting the type of 'warnings' (line 421)
    warnings_59231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 421)
    warn_59232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 12), warnings_59231, 'warn')
    # Calling warn(args, kwargs) (line 421)
    warn_call_result_59236 = invoke(stypy.reporting.localization.Localization(__file__, 421, 12), warn_59232, *[mesg_59233, DeprecationWarning_59234], **kwargs_59235)
    
    # SSA join for if statement (line 418)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'full_output' (line 423)
    full_output_59237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'full_output')
    int_59238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 26), 'int')
    # Applying the binary operator '!=' (line 423)
    result_ne_59239 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 11), '!=', full_output_59237, int_59238)
    
    # Testing the type of an if condition (line 423)
    if_condition_59240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 8), result_ne_59239)
    # Assigning a type to the variable 'if_condition_59240' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'if_condition_59240', if_condition_59240)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 424):
    
    # Assigning a BinOp to a Name (line 424):
    str_59241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 20), 'str', 'full_output = %s is not supported. Proceeding as if full_output = 0')
    # Getting the type of 'full_output' (line 425)
    full_output_59242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 40), 'full_output')
    # Applying the binary operator '%' (line 424)
    result_mod_59243 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 20), '%', str_59241, full_output_59242)
    
    # Assigning a type to the variable 'mesg' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'mesg', result_mod_59243)
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to integrate(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'a' (line 427)
    a_59246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 29), 'a', False)
    # Getting the type of 'b' (line 427)
    b_59247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 32), 'b', False)
    # Processing the call keyword arguments (line 427)
    # Getting the type of 'False' (line 427)
    False_59248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 47), 'False', False)
    keyword_59249 = False_59248
    kwargs_59250 = {'extrapolate': keyword_59249}
    # Getting the type of 'tck' (line 427)
    tck_59244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'tck', False)
    # Obtaining the member 'integrate' of a type (line 427)
    integrate_59245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 15), tck_59244, 'integrate')
    # Calling integrate(args, kwargs) (line 427)
    integrate_call_result_59251 = invoke(stypy.reporting.localization.Localization(__file__, 427, 15), integrate_59245, *[a_59246, b_59247], **kwargs_59250)
    
    # Assigning a type to the variable 'stypy_return_type' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'stypy_return_type', integrate_call_result_59251)
    # SSA branch for the else part of an if statement (line 417)
    module_type_store.open_ssa_branch('else')
    
    # Call to splint(...): (line 429)
    # Processing the call arguments (line 429)
    # Getting the type of 'a' (line 429)
    a_59254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 28), 'a', False)
    # Getting the type of 'b' (line 429)
    b_59255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 31), 'b', False)
    # Getting the type of 'tck' (line 429)
    tck_59256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 34), 'tck', False)
    # Getting the type of 'full_output' (line 429)
    full_output_59257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 39), 'full_output', False)
    # Processing the call keyword arguments (line 429)
    kwargs_59258 = {}
    # Getting the type of '_impl' (line 429)
    _impl_59252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 15), '_impl', False)
    # Obtaining the member 'splint' of a type (line 429)
    splint_59253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 15), _impl_59252, 'splint')
    # Calling splint(args, kwargs) (line 429)
    splint_call_result_59259 = invoke(stypy.reporting.localization.Localization(__file__, 429, 15), splint_59253, *[a_59254, b_59255, tck_59256, full_output_59257], **kwargs_59258)
    
    # Assigning a type to the variable 'stypy_return_type' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'stypy_return_type', splint_call_result_59259)
    # SSA join for if statement (line 417)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'splint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splint' in the type store
    # Getting the type of 'stypy_return_type' (line 371)
    stypy_return_type_59260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59260)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splint'
    return stypy_return_type_59260

# Assigning a type to the variable 'splint' (line 371)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'splint', splint)

@norecursion
def sproot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_59261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 21), 'int')
    defaults = [int_59261]
    # Create a new context for function 'sproot'
    module_type_store = module_type_store.open_function_context('sproot', 432, 0, False)
    
    # Passed parameters checking function
    sproot.stypy_localization = localization
    sproot.stypy_type_of_self = None
    sproot.stypy_type_store = module_type_store
    sproot.stypy_function_name = 'sproot'
    sproot.stypy_param_names_list = ['tck', 'mest']
    sproot.stypy_varargs_param_name = None
    sproot.stypy_kwargs_param_name = None
    sproot.stypy_call_defaults = defaults
    sproot.stypy_call_varargs = varargs
    sproot.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sproot', ['tck', 'mest'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sproot', localization, ['tck', 'mest'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sproot(...)' code ##################

    str_59262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, (-1)), 'str', '\n    Find the roots of a cubic B-spline.\n\n    Given the knots (>=8) and coefficients of a cubic B-spline return the\n    roots of the spline.\n\n    Parameters\n    ----------\n    tck : tuple or a BSpline object\n        If a tuple, then it should be a sequence of length 3, containing the\n        vector of knots, the B-spline coefficients, and the degree of the\n        spline.\n        The number of knots must be >= 8, and the degree must be 3.\n        The knots must be a montonically increasing sequence.\n    mest : int, optional\n        An estimate of the number of zeros (Default is 10).\n\n    Returns\n    -------\n    zeros : ndarray\n        An array giving the roots of the spline.\n\n    Notes\n    -----\n    Manipulating the tck-tuples directly is not recommended. In new code,\n    prefer using the `BSpline` objects.\n\n    See also\n    --------\n    splprep, splrep, splint, spalde, splev\n    bisplrep, bisplev\n    BSpline\n\n\n    References\n    ----------\n    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation\n        Theory, 6, p.50-62, 1972.\n    .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths\n        Applics, 10, p.134-149, 1972.\n    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs\n        on Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    
    # Call to isinstance(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'tck' (line 477)
    tck_59264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 18), 'tck', False)
    # Getting the type of 'BSpline' (line 477)
    BSpline_59265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 23), 'BSpline', False)
    # Processing the call keyword arguments (line 477)
    kwargs_59266 = {}
    # Getting the type of 'isinstance' (line 477)
    isinstance_59263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 477)
    isinstance_call_result_59267 = invoke(stypy.reporting.localization.Localization(__file__, 477, 7), isinstance_59263, *[tck_59264, BSpline_59265], **kwargs_59266)
    
    # Testing the type of an if condition (line 477)
    if_condition_59268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 4), isinstance_call_result_59267)
    # Assigning a type to the variable 'if_condition_59268' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'if_condition_59268', if_condition_59268)
    # SSA begins for if statement (line 477)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'tck' (line 478)
    tck_59269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'tck')
    # Obtaining the member 'c' of a type (line 478)
    c_59270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 11), tck_59269, 'c')
    # Obtaining the member 'ndim' of a type (line 478)
    ndim_59271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 11), c_59270, 'ndim')
    int_59272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 24), 'int')
    # Applying the binary operator '>' (line 478)
    result_gt_59273 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 11), '>', ndim_59271, int_59272)
    
    # Testing the type of an if condition (line 478)
    if_condition_59274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 8), result_gt_59273)
    # Assigning a type to the variable 'if_condition_59274' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'if_condition_59274', if_condition_59274)
    # SSA begins for if statement (line 478)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 479):
    
    # Assigning a Str to a Name (line 479):
    str_59275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 20), 'str', 'Calling sproot() with BSpline objects with c.ndim > 1 is not recommended.')
    # Assigning a type to the variable 'mesg' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'mesg', str_59275)
    
    # Call to warn(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'mesg' (line 481)
    mesg_59278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 26), 'mesg', False)
    # Getting the type of 'DeprecationWarning' (line 481)
    DeprecationWarning_59279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 32), 'DeprecationWarning', False)
    # Processing the call keyword arguments (line 481)
    kwargs_59280 = {}
    # Getting the type of 'warnings' (line 481)
    warnings_59276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 481)
    warn_59277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), warnings_59276, 'warn')
    # Calling warn(args, kwargs) (line 481)
    warn_call_result_59281 = invoke(stypy.reporting.localization.Localization(__file__, 481, 12), warn_59277, *[mesg_59278, DeprecationWarning_59279], **kwargs_59280)
    
    # SSA join for if statement (line 478)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 483):
    
    # Assigning a Subscript to a Name (line 483):
    
    # Obtaining the type of the subscript
    int_59282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 8), 'int')
    # Getting the type of 'tck' (line 483)
    tck_59283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 18), 'tck')
    # Obtaining the member 'tck' of a type (line 483)
    tck_59284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 18), tck_59283, 'tck')
    # Obtaining the member '__getitem__' of a type (line 483)
    getitem___59285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), tck_59284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 483)
    subscript_call_result_59286 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), getitem___59285, int_59282)
    
    # Assigning a type to the variable 'tuple_var_assignment_59076' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_59076', subscript_call_result_59286)
    
    # Assigning a Subscript to a Name (line 483):
    
    # Obtaining the type of the subscript
    int_59287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 8), 'int')
    # Getting the type of 'tck' (line 483)
    tck_59288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 18), 'tck')
    # Obtaining the member 'tck' of a type (line 483)
    tck_59289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 18), tck_59288, 'tck')
    # Obtaining the member '__getitem__' of a type (line 483)
    getitem___59290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), tck_59289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 483)
    subscript_call_result_59291 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), getitem___59290, int_59287)
    
    # Assigning a type to the variable 'tuple_var_assignment_59077' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_59077', subscript_call_result_59291)
    
    # Assigning a Subscript to a Name (line 483):
    
    # Obtaining the type of the subscript
    int_59292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 8), 'int')
    # Getting the type of 'tck' (line 483)
    tck_59293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 18), 'tck')
    # Obtaining the member 'tck' of a type (line 483)
    tck_59294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 18), tck_59293, 'tck')
    # Obtaining the member '__getitem__' of a type (line 483)
    getitem___59295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), tck_59294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 483)
    subscript_call_result_59296 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), getitem___59295, int_59292)
    
    # Assigning a type to the variable 'tuple_var_assignment_59078' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_59078', subscript_call_result_59296)
    
    # Assigning a Name to a Name (line 483):
    # Getting the type of 'tuple_var_assignment_59076' (line 483)
    tuple_var_assignment_59076_59297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_59076')
    # Assigning a type to the variable 't' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 't', tuple_var_assignment_59076_59297)
    
    # Assigning a Name to a Name (line 483):
    # Getting the type of 'tuple_var_assignment_59077' (line 483)
    tuple_var_assignment_59077_59298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_59077')
    # Assigning a type to the variable 'c' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 11), 'c', tuple_var_assignment_59077_59298)
    
    # Assigning a Name to a Name (line 483):
    # Getting the type of 'tuple_var_assignment_59078' (line 483)
    tuple_var_assignment_59078_59299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_59078')
    # Assigning a type to the variable 'k' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 14), 'k', tuple_var_assignment_59078_59299)
    
    # Assigning a Call to a Name (line 487):
    
    # Assigning a Call to a Name (line 487):
    
    # Call to tuple(...): (line 487)
    # Processing the call arguments (line 487)
    
    # Call to range(...): (line 487)
    # Processing the call arguments (line 487)
    # Getting the type of 'c' (line 487)
    c_59302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 25), 'c', False)
    # Obtaining the member 'ndim' of a type (line 487)
    ndim_59303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 25), c_59302, 'ndim')
    # Processing the call keyword arguments (line 487)
    kwargs_59304 = {}
    # Getting the type of 'range' (line 487)
    range_59301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 19), 'range', False)
    # Calling range(args, kwargs) (line 487)
    range_call_result_59305 = invoke(stypy.reporting.localization.Localization(__file__, 487, 19), range_59301, *[ndim_59303], **kwargs_59304)
    
    # Processing the call keyword arguments (line 487)
    kwargs_59306 = {}
    # Getting the type of 'tuple' (line 487)
    tuple_59300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 13), 'tuple', False)
    # Calling tuple(args, kwargs) (line 487)
    tuple_call_result_59307 = invoke(stypy.reporting.localization.Localization(__file__, 487, 13), tuple_59300, *[range_call_result_59305], **kwargs_59306)
    
    # Assigning a type to the variable 'sh' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'sh', tuple_call_result_59307)
    
    # Assigning a Call to a Name (line 488):
    
    # Assigning a Call to a Name (line 488):
    
    # Call to transpose(...): (line 488)
    # Processing the call arguments (line 488)
    
    # Obtaining the type of the subscript
    int_59310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 27), 'int')
    slice_59311 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 488, 24), int_59310, None, None)
    # Getting the type of 'sh' (line 488)
    sh_59312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 24), 'sh', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___59313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 24), sh_59312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 488)
    subscript_call_result_59314 = invoke(stypy.reporting.localization.Localization(__file__, 488, 24), getitem___59313, slice_59311)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 488)
    tuple_59315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 488)
    # Adding element type (line 488)
    int_59316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 34), tuple_59315, int_59316)
    
    # Applying the binary operator '+' (line 488)
    result_add_59317 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 24), '+', subscript_call_result_59314, tuple_59315)
    
    # Processing the call keyword arguments (line 488)
    kwargs_59318 = {}
    # Getting the type of 'c' (line 488)
    c_59308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'c', False)
    # Obtaining the member 'transpose' of a type (line 488)
    transpose_59309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), c_59308, 'transpose')
    # Calling transpose(args, kwargs) (line 488)
    transpose_call_result_59319 = invoke(stypy.reporting.localization.Localization(__file__, 488, 12), transpose_59309, *[result_add_59317], **kwargs_59318)
    
    # Assigning a type to the variable 'c' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'c', transpose_call_result_59319)
    
    # Call to sproot(...): (line 489)
    # Processing the call arguments (line 489)
    
    # Obtaining an instance of the builtin type 'tuple' (line 489)
    tuple_59322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 489)
    # Adding element type (line 489)
    # Getting the type of 't' (line 489)
    t_59323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 29), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 29), tuple_59322, t_59323)
    # Adding element type (line 489)
    # Getting the type of 'c' (line 489)
    c_59324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 32), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 29), tuple_59322, c_59324)
    # Adding element type (line 489)
    # Getting the type of 'k' (line 489)
    k_59325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 35), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 29), tuple_59322, k_59325)
    
    # Getting the type of 'mest' (line 489)
    mest_59326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 39), 'mest', False)
    # Processing the call keyword arguments (line 489)
    kwargs_59327 = {}
    # Getting the type of '_impl' (line 489)
    _impl_59320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), '_impl', False)
    # Obtaining the member 'sproot' of a type (line 489)
    sproot_59321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 15), _impl_59320, 'sproot')
    # Calling sproot(args, kwargs) (line 489)
    sproot_call_result_59328 = invoke(stypy.reporting.localization.Localization(__file__, 489, 15), sproot_59321, *[tuple_59322, mest_59326], **kwargs_59327)
    
    # Assigning a type to the variable 'stypy_return_type' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'stypy_return_type', sproot_call_result_59328)
    # SSA branch for the else part of an if statement (line 477)
    module_type_store.open_ssa_branch('else')
    
    # Call to sproot(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'tck' (line 491)
    tck_59331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 28), 'tck', False)
    # Getting the type of 'mest' (line 491)
    mest_59332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 33), 'mest', False)
    # Processing the call keyword arguments (line 491)
    kwargs_59333 = {}
    # Getting the type of '_impl' (line 491)
    _impl_59329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 15), '_impl', False)
    # Obtaining the member 'sproot' of a type (line 491)
    sproot_59330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 15), _impl_59329, 'sproot')
    # Calling sproot(args, kwargs) (line 491)
    sproot_call_result_59334 = invoke(stypy.reporting.localization.Localization(__file__, 491, 15), sproot_59330, *[tck_59331, mest_59332], **kwargs_59333)
    
    # Assigning a type to the variable 'stypy_return_type' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'stypy_return_type', sproot_call_result_59334)
    # SSA join for if statement (line 477)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'sproot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sproot' in the type store
    # Getting the type of 'stypy_return_type' (line 432)
    stypy_return_type_59335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59335)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sproot'
    return stypy_return_type_59335

# Assigning a type to the variable 'sproot' (line 432)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), 'sproot', sproot)

@norecursion
def spalde(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'spalde'
    module_type_store = module_type_store.open_function_context('spalde', 494, 0, False)
    
    # Passed parameters checking function
    spalde.stypy_localization = localization
    spalde.stypy_type_of_self = None
    spalde.stypy_type_store = module_type_store
    spalde.stypy_function_name = 'spalde'
    spalde.stypy_param_names_list = ['x', 'tck']
    spalde.stypy_varargs_param_name = None
    spalde.stypy_kwargs_param_name = None
    spalde.stypy_call_defaults = defaults
    spalde.stypy_call_varargs = varargs
    spalde.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spalde', ['x', 'tck'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spalde', localization, ['x', 'tck'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spalde(...)' code ##################

    str_59336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, (-1)), 'str', '\n    Evaluate all derivatives of a B-spline.\n\n    Given the knots and coefficients of a cubic B-spline compute all\n    derivatives up to order k at a point (or set of points).\n\n    Parameters\n    ----------\n    x : array_like\n        A point or a set of points at which to evaluate the derivatives.\n        Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.\n    tck : tuple\n        A tuple ``(t, c, k)``, containing the vector of knots, the B-spline\n        coefficients, and the degree of the spline (see `splev`).\n\n    Returns\n    -------\n    results : {ndarray, list of ndarrays}\n        An array (or a list of arrays) containing all derivatives\n        up to order k inclusive for each point `x`.\n\n    See Also\n    --------\n    splprep, splrep, splint, sproot, splev, bisplrep, bisplev,\n    BSpline\n\n    References\n    ----------\n    .. [1] C. de Boor: On calculating with b-splines, J. Approximation Theory\n       6 (1972) 50-62.\n    .. [2] M. G. Cox : The numerical evaluation of b-splines, J. Inst. Maths\n       applics 10 (1972) 134-149.\n    .. [3] P. Dierckx : Curve and surface fitting with splines, Monographs on\n       Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    
    # Call to isinstance(...): (line 531)
    # Processing the call arguments (line 531)
    # Getting the type of 'tck' (line 531)
    tck_59338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 18), 'tck', False)
    # Getting the type of 'BSpline' (line 531)
    BSpline_59339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 23), 'BSpline', False)
    # Processing the call keyword arguments (line 531)
    kwargs_59340 = {}
    # Getting the type of 'isinstance' (line 531)
    isinstance_59337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 531)
    isinstance_call_result_59341 = invoke(stypy.reporting.localization.Localization(__file__, 531, 7), isinstance_59337, *[tck_59338, BSpline_59339], **kwargs_59340)
    
    # Testing the type of an if condition (line 531)
    if_condition_59342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 4), isinstance_call_result_59341)
    # Assigning a type to the variable 'if_condition_59342' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'if_condition_59342', if_condition_59342)
    # SSA begins for if statement (line 531)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 532)
    # Processing the call arguments (line 532)
    str_59344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 24), 'str', 'spalde does not accept BSpline instances.')
    # Processing the call keyword arguments (line 532)
    kwargs_59345 = {}
    # Getting the type of 'TypeError' (line 532)
    TypeError_59343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 532)
    TypeError_call_result_59346 = invoke(stypy.reporting.localization.Localization(__file__, 532, 14), TypeError_59343, *[str_59344], **kwargs_59345)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 532, 8), TypeError_call_result_59346, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 531)
    module_type_store.open_ssa_branch('else')
    
    # Call to spalde(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 'x' (line 534)
    x_59349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 28), 'x', False)
    # Getting the type of 'tck' (line 534)
    tck_59350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 31), 'tck', False)
    # Processing the call keyword arguments (line 534)
    kwargs_59351 = {}
    # Getting the type of '_impl' (line 534)
    _impl_59347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 15), '_impl', False)
    # Obtaining the member 'spalde' of a type (line 534)
    spalde_59348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 15), _impl_59347, 'spalde')
    # Calling spalde(args, kwargs) (line 534)
    spalde_call_result_59352 = invoke(stypy.reporting.localization.Localization(__file__, 534, 15), spalde_59348, *[x_59349, tck_59350], **kwargs_59351)
    
    # Assigning a type to the variable 'stypy_return_type' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'stypy_return_type', spalde_call_result_59352)
    # SSA join for if statement (line 531)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'spalde(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spalde' in the type store
    # Getting the type of 'stypy_return_type' (line 494)
    stypy_return_type_59353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59353)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spalde'
    return stypy_return_type_59353

# Assigning a type to the variable 'spalde' (line 494)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 0), 'spalde', spalde)

@norecursion
def insert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_59354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 21), 'int')
    int_59355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 28), 'int')
    defaults = [int_59354, int_59355]
    # Create a new context for function 'insert'
    module_type_store = module_type_store.open_function_context('insert', 537, 0, False)
    
    # Passed parameters checking function
    insert.stypy_localization = localization
    insert.stypy_type_of_self = None
    insert.stypy_type_store = module_type_store
    insert.stypy_function_name = 'insert'
    insert.stypy_param_names_list = ['x', 'tck', 'm', 'per']
    insert.stypy_varargs_param_name = None
    insert.stypy_kwargs_param_name = None
    insert.stypy_call_defaults = defaults
    insert.stypy_call_varargs = varargs
    insert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'insert', ['x', 'tck', 'm', 'per'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'insert', localization, ['x', 'tck', 'm', 'per'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'insert(...)' code ##################

    str_59356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, (-1)), 'str', '\n    Insert knots into a B-spline.\n\n    Given the knots and coefficients of a B-spline representation, create a\n    new B-spline with a knot inserted `m` times at point `x`.\n    This is a wrapper around the FORTRAN routine insert of FITPACK.\n\n    Parameters\n    ----------\n    x (u) : array_like\n        A 1-D point at which to insert a new knot(s).  If `tck` was returned\n        from ``splprep``, then the parameter values, u should be given.\n    tck : a `BSpline` instance or a tuple\n        If tuple, then it is expected to be a tuple (t,c,k) containing\n        the vector of knots, the B-spline coefficients, and the degree of\n        the spline.\n    m : int, optional\n        The number of times to insert the given knot (its multiplicity).\n        Default is 1.\n    per : int, optional\n        If non-zero, the input spline is considered periodic.\n\n    Returns\n    -------\n    BSpline instance or a tuple\n        A new B-spline with knots t, coefficients c, and degree k.\n        ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.\n        In case of a periodic spline (``per != 0``) there must be\n        either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``\n        or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.\n        A tuple is returned iff the input argument `tck` is a tuple, otherwise\n        a BSpline object is constructed and returned.\n\n    Notes\n    -----\n    Based on algorithms from [1]_ and [2]_.\n\n    Manipulating the tck-tuples directly is not recommended. In new code,\n    prefer using the `BSpline` objects.\n\n    References\n    ----------\n    .. [1] W. Boehm, "Inserting new knots into b-spline curves.",\n        Computer Aided Design, 12, p.199-201, 1980.\n    .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on\n        Numerical Analysis", Oxford University Press, 1993.\n\n    ')
    
    
    # Call to isinstance(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'tck' (line 586)
    tck_59358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 18), 'tck', False)
    # Getting the type of 'BSpline' (line 586)
    BSpline_59359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 23), 'BSpline', False)
    # Processing the call keyword arguments (line 586)
    kwargs_59360 = {}
    # Getting the type of 'isinstance' (line 586)
    isinstance_59357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 586)
    isinstance_call_result_59361 = invoke(stypy.reporting.localization.Localization(__file__, 586, 7), isinstance_59357, *[tck_59358, BSpline_59359], **kwargs_59360)
    
    # Testing the type of an if condition (line 586)
    if_condition_59362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 586, 4), isinstance_call_result_59361)
    # Assigning a type to the variable 'if_condition_59362' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'if_condition_59362', if_condition_59362)
    # SSA begins for if statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Tuple (line 588):
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_59363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    # Getting the type of 'tck' (line 588)
    tck_59364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 18), 'tck')
    # Obtaining the member 'tck' of a type (line 588)
    tck_59365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 18), tck_59364, 'tck')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___59366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), tck_59365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_59367 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___59366, int_59363)
    
    # Assigning a type to the variable 'tuple_var_assignment_59079' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_59079', subscript_call_result_59367)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_59368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    # Getting the type of 'tck' (line 588)
    tck_59369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 18), 'tck')
    # Obtaining the member 'tck' of a type (line 588)
    tck_59370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 18), tck_59369, 'tck')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___59371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), tck_59370, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_59372 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___59371, int_59368)
    
    # Assigning a type to the variable 'tuple_var_assignment_59080' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_59080', subscript_call_result_59372)
    
    # Assigning a Subscript to a Name (line 588):
    
    # Obtaining the type of the subscript
    int_59373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 8), 'int')
    # Getting the type of 'tck' (line 588)
    tck_59374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 18), 'tck')
    # Obtaining the member 'tck' of a type (line 588)
    tck_59375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 18), tck_59374, 'tck')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___59376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), tck_59375, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_59377 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), getitem___59376, int_59373)
    
    # Assigning a type to the variable 'tuple_var_assignment_59081' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_59081', subscript_call_result_59377)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_59079' (line 588)
    tuple_var_assignment_59079_59378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_59079')
    # Assigning a type to the variable 't' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 't', tuple_var_assignment_59079_59378)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_59080' (line 588)
    tuple_var_assignment_59080_59379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_59080')
    # Assigning a type to the variable 'c' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 11), 'c', tuple_var_assignment_59080_59379)
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'tuple_var_assignment_59081' (line 588)
    tuple_var_assignment_59081_59380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'tuple_var_assignment_59081')
    # Assigning a type to the variable 'k' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 14), 'k', tuple_var_assignment_59081_59380)
    
    # Assigning a Call to a Name (line 592):
    
    # Assigning a Call to a Name (line 592):
    
    # Call to tuple(...): (line 592)
    # Processing the call arguments (line 592)
    
    # Call to range(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'c' (line 592)
    c_59383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 25), 'c', False)
    # Obtaining the member 'ndim' of a type (line 592)
    ndim_59384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 25), c_59383, 'ndim')
    # Processing the call keyword arguments (line 592)
    kwargs_59385 = {}
    # Getting the type of 'range' (line 592)
    range_59382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), 'range', False)
    # Calling range(args, kwargs) (line 592)
    range_call_result_59386 = invoke(stypy.reporting.localization.Localization(__file__, 592, 19), range_59382, *[ndim_59384], **kwargs_59385)
    
    # Processing the call keyword arguments (line 592)
    kwargs_59387 = {}
    # Getting the type of 'tuple' (line 592)
    tuple_59381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 13), 'tuple', False)
    # Calling tuple(args, kwargs) (line 592)
    tuple_call_result_59388 = invoke(stypy.reporting.localization.Localization(__file__, 592, 13), tuple_59381, *[range_call_result_59386], **kwargs_59387)
    
    # Assigning a type to the variable 'sh' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'sh', tuple_call_result_59388)
    
    # Assigning a Call to a Name (line 593):
    
    # Assigning a Call to a Name (line 593):
    
    # Call to transpose(...): (line 593)
    # Processing the call arguments (line 593)
    
    # Obtaining the type of the subscript
    int_59391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 27), 'int')
    slice_59392 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 593, 24), int_59391, None, None)
    # Getting the type of 'sh' (line 593)
    sh_59393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 24), 'sh', False)
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___59394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 24), sh_59393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_59395 = invoke(stypy.reporting.localization.Localization(__file__, 593, 24), getitem___59394, slice_59392)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 593)
    tuple_59396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 593)
    # Adding element type (line 593)
    int_59397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 34), tuple_59396, int_59397)
    
    # Applying the binary operator '+' (line 593)
    result_add_59398 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 24), '+', subscript_call_result_59395, tuple_59396)
    
    # Processing the call keyword arguments (line 593)
    kwargs_59399 = {}
    # Getting the type of 'c' (line 593)
    c_59389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'c', False)
    # Obtaining the member 'transpose' of a type (line 593)
    transpose_59390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 12), c_59389, 'transpose')
    # Calling transpose(args, kwargs) (line 593)
    transpose_call_result_59400 = invoke(stypy.reporting.localization.Localization(__file__, 593, 12), transpose_59390, *[result_add_59398], **kwargs_59399)
    
    # Assigning a type to the variable 'c' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'c', transpose_call_result_59400)
    
    # Assigning a Call to a Tuple (line 594):
    
    # Assigning a Subscript to a Name (line 594):
    
    # Obtaining the type of the subscript
    int_59401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 8), 'int')
    
    # Call to insert(...): (line 594)
    # Processing the call arguments (line 594)
    # Getting the type of 'x' (line 594)
    x_59404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 34), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 594)
    tuple_59405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 594)
    # Adding element type (line 594)
    # Getting the type of 't' (line 594)
    t_59406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 38), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59405, t_59406)
    # Adding element type (line 594)
    # Getting the type of 'c' (line 594)
    c_59407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 41), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59405, c_59407)
    # Adding element type (line 594)
    # Getting the type of 'k' (line 594)
    k_59408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 44), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59405, k_59408)
    
    # Getting the type of 'm' (line 594)
    m_59409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 48), 'm', False)
    # Getting the type of 'per' (line 594)
    per_59410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 51), 'per', False)
    # Processing the call keyword arguments (line 594)
    kwargs_59411 = {}
    # Getting the type of '_impl' (line 594)
    _impl_59402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 21), '_impl', False)
    # Obtaining the member 'insert' of a type (line 594)
    insert_59403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 21), _impl_59402, 'insert')
    # Calling insert(args, kwargs) (line 594)
    insert_call_result_59412 = invoke(stypy.reporting.localization.Localization(__file__, 594, 21), insert_59403, *[x_59404, tuple_59405, m_59409, per_59410], **kwargs_59411)
    
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___59413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 8), insert_call_result_59412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_59414 = invoke(stypy.reporting.localization.Localization(__file__, 594, 8), getitem___59413, int_59401)
    
    # Assigning a type to the variable 'tuple_var_assignment_59082' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'tuple_var_assignment_59082', subscript_call_result_59414)
    
    # Assigning a Subscript to a Name (line 594):
    
    # Obtaining the type of the subscript
    int_59415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 8), 'int')
    
    # Call to insert(...): (line 594)
    # Processing the call arguments (line 594)
    # Getting the type of 'x' (line 594)
    x_59418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 34), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 594)
    tuple_59419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 594)
    # Adding element type (line 594)
    # Getting the type of 't' (line 594)
    t_59420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 38), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59419, t_59420)
    # Adding element type (line 594)
    # Getting the type of 'c' (line 594)
    c_59421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 41), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59419, c_59421)
    # Adding element type (line 594)
    # Getting the type of 'k' (line 594)
    k_59422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 44), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59419, k_59422)
    
    # Getting the type of 'm' (line 594)
    m_59423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 48), 'm', False)
    # Getting the type of 'per' (line 594)
    per_59424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 51), 'per', False)
    # Processing the call keyword arguments (line 594)
    kwargs_59425 = {}
    # Getting the type of '_impl' (line 594)
    _impl_59416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 21), '_impl', False)
    # Obtaining the member 'insert' of a type (line 594)
    insert_59417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 21), _impl_59416, 'insert')
    # Calling insert(args, kwargs) (line 594)
    insert_call_result_59426 = invoke(stypy.reporting.localization.Localization(__file__, 594, 21), insert_59417, *[x_59418, tuple_59419, m_59423, per_59424], **kwargs_59425)
    
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___59427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 8), insert_call_result_59426, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_59428 = invoke(stypy.reporting.localization.Localization(__file__, 594, 8), getitem___59427, int_59415)
    
    # Assigning a type to the variable 'tuple_var_assignment_59083' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'tuple_var_assignment_59083', subscript_call_result_59428)
    
    # Assigning a Subscript to a Name (line 594):
    
    # Obtaining the type of the subscript
    int_59429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 8), 'int')
    
    # Call to insert(...): (line 594)
    # Processing the call arguments (line 594)
    # Getting the type of 'x' (line 594)
    x_59432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 34), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 594)
    tuple_59433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 594)
    # Adding element type (line 594)
    # Getting the type of 't' (line 594)
    t_59434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 38), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59433, t_59434)
    # Adding element type (line 594)
    # Getting the type of 'c' (line 594)
    c_59435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 41), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59433, c_59435)
    # Adding element type (line 594)
    # Getting the type of 'k' (line 594)
    k_59436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 44), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 38), tuple_59433, k_59436)
    
    # Getting the type of 'm' (line 594)
    m_59437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 48), 'm', False)
    # Getting the type of 'per' (line 594)
    per_59438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 51), 'per', False)
    # Processing the call keyword arguments (line 594)
    kwargs_59439 = {}
    # Getting the type of '_impl' (line 594)
    _impl_59430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 21), '_impl', False)
    # Obtaining the member 'insert' of a type (line 594)
    insert_59431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 21), _impl_59430, 'insert')
    # Calling insert(args, kwargs) (line 594)
    insert_call_result_59440 = invoke(stypy.reporting.localization.Localization(__file__, 594, 21), insert_59431, *[x_59432, tuple_59433, m_59437, per_59438], **kwargs_59439)
    
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___59441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 8), insert_call_result_59440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_59442 = invoke(stypy.reporting.localization.Localization(__file__, 594, 8), getitem___59441, int_59429)
    
    # Assigning a type to the variable 'tuple_var_assignment_59084' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'tuple_var_assignment_59084', subscript_call_result_59442)
    
    # Assigning a Name to a Name (line 594):
    # Getting the type of 'tuple_var_assignment_59082' (line 594)
    tuple_var_assignment_59082_59443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'tuple_var_assignment_59082')
    # Assigning a type to the variable 't_' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 't_', tuple_var_assignment_59082_59443)
    
    # Assigning a Name to a Name (line 594):
    # Getting the type of 'tuple_var_assignment_59083' (line 594)
    tuple_var_assignment_59083_59444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'tuple_var_assignment_59083')
    # Assigning a type to the variable 'c_' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'c_', tuple_var_assignment_59083_59444)
    
    # Assigning a Name to a Name (line 594):
    # Getting the type of 'tuple_var_assignment_59084' (line 594)
    tuple_var_assignment_59084_59445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'tuple_var_assignment_59084')
    # Assigning a type to the variable 'k_' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 16), 'k_', tuple_var_assignment_59084_59445)
    
    # Assigning a Call to a Name (line 597):
    
    # Assigning a Call to a Name (line 597):
    
    # Call to asarray(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'c_' (line 597)
    c__59448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 24), 'c_', False)
    # Processing the call keyword arguments (line 597)
    kwargs_59449 = {}
    # Getting the type of 'np' (line 597)
    np_59446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 597)
    asarray_59447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 13), np_59446, 'asarray')
    # Calling asarray(args, kwargs) (line 597)
    asarray_call_result_59450 = invoke(stypy.reporting.localization.Localization(__file__, 597, 13), asarray_59447, *[c__59448], **kwargs_59449)
    
    # Assigning a type to the variable 'c_' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'c_', asarray_call_result_59450)
    
    # Assigning a Call to a Name (line 598):
    
    # Assigning a Call to a Name (line 598):
    
    # Call to transpose(...): (line 598)
    # Processing the call arguments (line 598)
    
    # Obtaining an instance of the builtin type 'tuple' (line 598)
    tuple_59453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 598)
    # Adding element type (line 598)
    
    # Obtaining the type of the subscript
    int_59454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 30), 'int')
    # Getting the type of 'sh' (line 598)
    sh_59455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 27), 'sh', False)
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___59456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 27), sh_59455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 598)
    subscript_call_result_59457 = invoke(stypy.reporting.localization.Localization(__file__, 598, 27), getitem___59456, int_59454)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 27), tuple_59453, subscript_call_result_59457)
    
    
    # Obtaining the type of the subscript
    int_59458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 42), 'int')
    slice_59459 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 598, 38), None, int_59458, None)
    # Getting the type of 'sh' (line 598)
    sh_59460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 38), 'sh', False)
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___59461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 38), sh_59460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 598)
    subscript_call_result_59462 = invoke(stypy.reporting.localization.Localization(__file__, 598, 38), getitem___59461, slice_59459)
    
    # Applying the binary operator '+' (line 598)
    result_add_59463 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 26), '+', tuple_59453, subscript_call_result_59462)
    
    # Processing the call keyword arguments (line 598)
    kwargs_59464 = {}
    # Getting the type of 'c_' (line 598)
    c__59451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 13), 'c_', False)
    # Obtaining the member 'transpose' of a type (line 598)
    transpose_59452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 13), c__59451, 'transpose')
    # Calling transpose(args, kwargs) (line 598)
    transpose_call_result_59465 = invoke(stypy.reporting.localization.Localization(__file__, 598, 13), transpose_59452, *[result_add_59463], **kwargs_59464)
    
    # Assigning a type to the variable 'c_' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'c_', transpose_call_result_59465)
    
    # Call to BSpline(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 't_' (line 599)
    t__59467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 23), 't_', False)
    # Getting the type of 'c_' (line 599)
    c__59468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 27), 'c_', False)
    # Getting the type of 'k_' (line 599)
    k__59469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 31), 'k_', False)
    # Processing the call keyword arguments (line 599)
    kwargs_59470 = {}
    # Getting the type of 'BSpline' (line 599)
    BSpline_59466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 15), 'BSpline', False)
    # Calling BSpline(args, kwargs) (line 599)
    BSpline_call_result_59471 = invoke(stypy.reporting.localization.Localization(__file__, 599, 15), BSpline_59466, *[t__59467, c__59468, k__59469], **kwargs_59470)
    
    # Assigning a type to the variable 'stypy_return_type' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'stypy_return_type', BSpline_call_result_59471)
    # SSA branch for the else part of an if statement (line 586)
    module_type_store.open_ssa_branch('else')
    
    # Call to insert(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'x' (line 601)
    x_59474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 28), 'x', False)
    # Getting the type of 'tck' (line 601)
    tck_59475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 31), 'tck', False)
    # Getting the type of 'm' (line 601)
    m_59476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 36), 'm', False)
    # Getting the type of 'per' (line 601)
    per_59477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 39), 'per', False)
    # Processing the call keyword arguments (line 601)
    kwargs_59478 = {}
    # Getting the type of '_impl' (line 601)
    _impl_59472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 15), '_impl', False)
    # Obtaining the member 'insert' of a type (line 601)
    insert_59473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 15), _impl_59472, 'insert')
    # Calling insert(args, kwargs) (line 601)
    insert_call_result_59479 = invoke(stypy.reporting.localization.Localization(__file__, 601, 15), insert_59473, *[x_59474, tck_59475, m_59476, per_59477], **kwargs_59478)
    
    # Assigning a type to the variable 'stypy_return_type' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'stypy_return_type', insert_call_result_59479)
    # SSA join for if statement (line 586)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'insert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'insert' in the type store
    # Getting the type of 'stypy_return_type' (line 537)
    stypy_return_type_59480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59480)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'insert'
    return stypy_return_type_59480

# Assigning a type to the variable 'insert' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'insert', insert)

@norecursion
def splder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_59481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 18), 'int')
    defaults = [int_59481]
    # Create a new context for function 'splder'
    module_type_store = module_type_store.open_function_context('splder', 604, 0, False)
    
    # Passed parameters checking function
    splder.stypy_localization = localization
    splder.stypy_type_of_self = None
    splder.stypy_type_store = module_type_store
    splder.stypy_function_name = 'splder'
    splder.stypy_param_names_list = ['tck', 'n']
    splder.stypy_varargs_param_name = None
    splder.stypy_kwargs_param_name = None
    splder.stypy_call_defaults = defaults
    splder.stypy_call_varargs = varargs
    splder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splder', ['tck', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splder', localization, ['tck', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splder(...)' code ##################

    str_59482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, (-1)), 'str', "\n    Compute the spline representation of the derivative of a given spline\n\n    Parameters\n    ----------\n    tck : BSpline instance or a tuple of (t, c, k)\n        Spline whose derivative to compute\n    n : int, optional\n        Order of derivative to evaluate. Default: 1\n\n    Returns\n    -------\n    `BSpline` instance or tuple\n        Spline of order k2=k-n representing the derivative\n        of the input spline.\n        A tuple is returned iff the input argument `tck` is a tuple, otherwise\n        a BSpline object is constructed and returned.\n\n    Notes\n    -----\n\n    .. versionadded:: 0.13.0\n\n    See Also\n    --------\n    splantider, splev, spalde\n    BSpline\n\n    Examples\n    --------\n    This can be used for finding maxima of a curve:\n\n    >>> from scipy.interpolate import splrep, splder, sproot\n    >>> x = np.linspace(0, 10, 70)\n    >>> y = np.sin(x)\n    >>> spl = splrep(x, y, k=4)\n\n    Now, differentiate the spline and find the zeros of the\n    derivative. (NB: `sproot` only works for order 3 splines, so we\n    fit an order 4 spline):\n\n    >>> dspl = splder(spl)\n    >>> sproot(dspl) / np.pi\n    array([ 0.50000001,  1.5       ,  2.49999998])\n\n    This agrees well with roots :math:`\\pi/2 + n\\pi` of\n    :math:`\\cos(x) = \\sin'(x)`.\n\n    ")
    
    
    # Call to isinstance(...): (line 654)
    # Processing the call arguments (line 654)
    # Getting the type of 'tck' (line 654)
    tck_59484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 18), 'tck', False)
    # Getting the type of 'BSpline' (line 654)
    BSpline_59485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 23), 'BSpline', False)
    # Processing the call keyword arguments (line 654)
    kwargs_59486 = {}
    # Getting the type of 'isinstance' (line 654)
    isinstance_59483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 654)
    isinstance_call_result_59487 = invoke(stypy.reporting.localization.Localization(__file__, 654, 7), isinstance_59483, *[tck_59484, BSpline_59485], **kwargs_59486)
    
    # Testing the type of an if condition (line 654)
    if_condition_59488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 4), isinstance_call_result_59487)
    # Assigning a type to the variable 'if_condition_59488' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'if_condition_59488', if_condition_59488)
    # SSA begins for if statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to derivative(...): (line 655)
    # Processing the call arguments (line 655)
    # Getting the type of 'n' (line 655)
    n_59491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 30), 'n', False)
    # Processing the call keyword arguments (line 655)
    kwargs_59492 = {}
    # Getting the type of 'tck' (line 655)
    tck_59489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'tck', False)
    # Obtaining the member 'derivative' of a type (line 655)
    derivative_59490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), tck_59489, 'derivative')
    # Calling derivative(args, kwargs) (line 655)
    derivative_call_result_59493 = invoke(stypy.reporting.localization.Localization(__file__, 655, 15), derivative_59490, *[n_59491], **kwargs_59492)
    
    # Assigning a type to the variable 'stypy_return_type' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'stypy_return_type', derivative_call_result_59493)
    # SSA branch for the else part of an if statement (line 654)
    module_type_store.open_ssa_branch('else')
    
    # Call to splder(...): (line 657)
    # Processing the call arguments (line 657)
    # Getting the type of 'tck' (line 657)
    tck_59496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 28), 'tck', False)
    # Getting the type of 'n' (line 657)
    n_59497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 33), 'n', False)
    # Processing the call keyword arguments (line 657)
    kwargs_59498 = {}
    # Getting the type of '_impl' (line 657)
    _impl_59494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 15), '_impl', False)
    # Obtaining the member 'splder' of a type (line 657)
    splder_59495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 15), _impl_59494, 'splder')
    # Calling splder(args, kwargs) (line 657)
    splder_call_result_59499 = invoke(stypy.reporting.localization.Localization(__file__, 657, 15), splder_59495, *[tck_59496, n_59497], **kwargs_59498)
    
    # Assigning a type to the variable 'stypy_return_type' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'stypy_return_type', splder_call_result_59499)
    # SSA join for if statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'splder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splder' in the type store
    # Getting the type of 'stypy_return_type' (line 604)
    stypy_return_type_59500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59500)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splder'
    return stypy_return_type_59500

# Assigning a type to the variable 'splder' (line 604)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 0), 'splder', splder)

@norecursion
def splantider(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_59501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 22), 'int')
    defaults = [int_59501]
    # Create a new context for function 'splantider'
    module_type_store = module_type_store.open_function_context('splantider', 660, 0, False)
    
    # Passed parameters checking function
    splantider.stypy_localization = localization
    splantider.stypy_type_of_self = None
    splantider.stypy_type_store = module_type_store
    splantider.stypy_function_name = 'splantider'
    splantider.stypy_param_names_list = ['tck', 'n']
    splantider.stypy_varargs_param_name = None
    splantider.stypy_kwargs_param_name = None
    splantider.stypy_call_defaults = defaults
    splantider.stypy_call_varargs = varargs
    splantider.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splantider', ['tck', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splantider', localization, ['tck', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splantider(...)' code ##################

    str_59502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, (-1)), 'str', '\n    Compute the spline for the antiderivative (integral) of a given spline.\n\n    Parameters\n    ----------\n    tck : BSpline instance or a tuple of (t, c, k)\n        Spline whose antiderivative to compute\n    n : int, optional\n        Order of antiderivative to evaluate. Default: 1\n\n    Returns\n    -------\n    BSpline instance or a tuple of (t2, c2, k2)\n        Spline of order k2=k+n representing the antiderivative of the input\n        spline.\n        A tuple is returned iff the input argument `tck` is a tuple, otherwise\n        a BSpline object is constructed and returned.\n\n    See Also\n    --------\n    splder, splev, spalde\n    BSpline\n\n    Notes\n    -----\n    The `splder` function is the inverse operation of this function.\n    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo\n    rounding error.\n\n    .. versionadded:: 0.13.0\n\n    Examples\n    --------\n    >>> from scipy.interpolate import splrep, splder, splantider, splev\n    >>> x = np.linspace(0, np.pi/2, 70)\n    >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)\n    >>> spl = splrep(x, y)\n\n    The derivative is the inverse operation of the antiderivative,\n    although some floating point error accumulates:\n\n    >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))\n    (array(2.1565429877197317), array(2.1565429877201865))\n\n    Antiderivative can be used to evaluate definite integrals:\n\n    >>> ispl = splantider(spl)\n    >>> splev(np.pi/2, ispl) - splev(0, ispl)\n    2.2572053588768486\n\n    This is indeed an approximation to the complete elliptic integral\n    :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:\n\n    >>> from scipy.special import ellipk\n    >>> ellipk(0.8)\n    2.2572053268208538\n\n    ')
    
    
    # Call to isinstance(...): (line 719)
    # Processing the call arguments (line 719)
    # Getting the type of 'tck' (line 719)
    tck_59504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 18), 'tck', False)
    # Getting the type of 'BSpline' (line 719)
    BSpline_59505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 23), 'BSpline', False)
    # Processing the call keyword arguments (line 719)
    kwargs_59506 = {}
    # Getting the type of 'isinstance' (line 719)
    isinstance_59503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 719)
    isinstance_call_result_59507 = invoke(stypy.reporting.localization.Localization(__file__, 719, 7), isinstance_59503, *[tck_59504, BSpline_59505], **kwargs_59506)
    
    # Testing the type of an if condition (line 719)
    if_condition_59508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 719, 4), isinstance_call_result_59507)
    # Assigning a type to the variable 'if_condition_59508' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'if_condition_59508', if_condition_59508)
    # SSA begins for if statement (line 719)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to antiderivative(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'n' (line 720)
    n_59511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 34), 'n', False)
    # Processing the call keyword arguments (line 720)
    kwargs_59512 = {}
    # Getting the type of 'tck' (line 720)
    tck_59509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 15), 'tck', False)
    # Obtaining the member 'antiderivative' of a type (line 720)
    antiderivative_59510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 15), tck_59509, 'antiderivative')
    # Calling antiderivative(args, kwargs) (line 720)
    antiderivative_call_result_59513 = invoke(stypy.reporting.localization.Localization(__file__, 720, 15), antiderivative_59510, *[n_59511], **kwargs_59512)
    
    # Assigning a type to the variable 'stypy_return_type' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'stypy_return_type', antiderivative_call_result_59513)
    # SSA branch for the else part of an if statement (line 719)
    module_type_store.open_ssa_branch('else')
    
    # Call to splantider(...): (line 722)
    # Processing the call arguments (line 722)
    # Getting the type of 'tck' (line 722)
    tck_59516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 32), 'tck', False)
    # Getting the type of 'n' (line 722)
    n_59517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 37), 'n', False)
    # Processing the call keyword arguments (line 722)
    kwargs_59518 = {}
    # Getting the type of '_impl' (line 722)
    _impl_59514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 15), '_impl', False)
    # Obtaining the member 'splantider' of a type (line 722)
    splantider_59515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 15), _impl_59514, 'splantider')
    # Calling splantider(args, kwargs) (line 722)
    splantider_call_result_59519 = invoke(stypy.reporting.localization.Localization(__file__, 722, 15), splantider_59515, *[tck_59516, n_59517], **kwargs_59518)
    
    # Assigning a type to the variable 'stypy_return_type' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'stypy_return_type', splantider_call_result_59519)
    # SSA join for if statement (line 719)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'splantider(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splantider' in the type store
    # Getting the type of 'stypy_return_type' (line 660)
    stypy_return_type_59520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59520)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splantider'
    return stypy_return_type_59520

# Assigning a type to the variable 'splantider' (line 660)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 0), 'splantider', splantider)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
