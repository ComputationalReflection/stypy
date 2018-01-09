
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: fitpack (dierckx in netlib) --- A Python-C wrapper to FITPACK (by P. Dierckx).
3:         FITPACK is a collection of FORTRAN programs for curve and surface
4:         fitting with splines and tensor product splines.
5: 
6: See
7:  http://www.cs.kuleuven.ac.be/cwis/research/nalag/research/topics/fitpack.html
8: or
9:  http://www.netlib.org/dierckx/index.html
10: 
11: Copyright 2002 Pearu Peterson all rights reserved,
12: Pearu Peterson <pearu@cens.ioc.ee>
13: Permission to use, modify, and distribute this software is given under the
14: terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
15: this distribution for specifics.
16: 
17: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
18: 
19: TODO: Make interfaces to the following fitpack functions:
20:     For univariate splines: cocosp, concon, fourco, insert
21:     For bivariate splines: profil, regrid, parsur, surev
22: '''
23: from __future__ import division, print_function, absolute_import
24: 
25: 
26: __all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde',
27:            'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']
28: 
29: import warnings
30: import numpy as np
31: from . import _fitpack
32: from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
33:                    empty, iinfo, intc, asarray)
34: 
35: # Try to replace _fitpack interface with
36: #  f2py-generated version
37: from . import dfitpack
38: 
39: 
40: def _intc_overflow(x, msg=None):
41:     '''Cast the value to an intc and raise an OverflowError if the value
42:     cannot fit.
43:     '''
44:     if x > iinfo(intc).max:
45:         if msg is None:
46:             msg = '%r cannot fit into an intc' % x
47:         raise OverflowError(msg)
48:     return intc(x)
49: 
50: 
51: _iermess = {
52:     0: ["The spline has a residual sum of squares fp such that "
53:         "abs(fp-s)/s<=0.001", None],
54:     -1: ["The spline is an interpolating spline (fp=0)", None],
55:     -2: ["The spline is weighted least-squares polynomial of degree k.\n"
56:          "fp gives the upper bound fp0 for the smoothing factor s", None],
57:     1: ["The required storage space exceeds the available storage space.\n"
58:         "Probable causes: data (x,y) size is too small or smoothing parameter"
59:         "\ns is too small (fp>s).", ValueError],
60:     2: ["A theoretically impossible result when finding a smoothing spline\n"
61:         "with fp = s. Probable cause: s too small. (abs(fp-s)/s>0.001)",
62:         ValueError],
63:     3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
64:         "spline with fp=s has been reached. Probable cause: s too small.\n"
65:         "(abs(fp-s)/s>0.001)", ValueError],
66:     10: ["Error on input data", ValueError],
67:     'unknown': ["An error occurred", TypeError]
68: }
69: 
70: _iermess2 = {
71:     0: ["The spline has a residual sum of squares fp such that "
72:         "abs(fp-s)/s<=0.001", None],
73:     -1: ["The spline is an interpolating spline (fp=0)", None],
74:     -2: ["The spline is weighted least-squares polynomial of degree kx and ky."
75:          "\nfp gives the upper bound fp0 for the smoothing factor s", None],
76:     -3: ["Warning. The coefficients of the spline have been computed as the\n"
77:          "minimal norm least-squares solution of a rank deficient system.",
78:          None],
79:     1: ["The required storage space exceeds the available storage space.\n"
80:         "Probable causes: nxest or nyest too small or s is too small. (fp>s)",
81:         ValueError],
82:     2: ["A theoretically impossible result when finding a smoothing spline\n"
83:         "with fp = s. Probable causes: s too small or badly chosen eps.\n"
84:         "(abs(fp-s)/s>0.001)", ValueError],
85:     3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
86:         "spline with fp=s has been reached. Probable cause: s too small.\n"
87:         "(abs(fp-s)/s>0.001)", ValueError],
88:     4: ["No more knots can be added because the number of B-spline\n"
89:         "coefficients already exceeds the number of data points m.\n"
90:         "Probable causes: either s or m too small. (fp>s)", ValueError],
91:     5: ["No more knots can be added because the additional knot would\n"
92:         "coincide with an old one. Probable cause: s too small or too large\n"
93:         "a weight to an inaccurate data point. (fp>s)", ValueError],
94:     10: ["Error on input data", ValueError],
95:     11: ["rwrk2 too small, i.e. there is not enough workspace for computing\n"
96:          "the minimal least-squares solution of a rank deficient system of\n"
97:          "linear equations.", ValueError],
98:     'unknown': ["An error occurred", TypeError]
99: }
100: 
101: _parcur_cache = {'t': array([], float), 'wrk': array([], float),
102:                  'iwrk': array([], intc), 'u': array([], float),
103:                  'ub': 0, 'ue': 1}
104: 
105: 
106: def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
107:             full_output=0, nest=None, per=0, quiet=1):
108:     '''
109:     Find the B-spline representation of an N-dimensional curve.
110: 
111:     Given a list of N rank-1 arrays, `x`, which represent a curve in
112:     N-dimensional space parametrized by `u`, find a smooth approximating
113:     spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.
114: 
115:     Parameters
116:     ----------
117:     x : array_like
118:         A list of sample vector arrays representing the curve.
119:     w : array_like, optional
120:         Strictly positive rank-1 array of weights the same length as `x[0]`.
121:         The weights are used in computing the weighted least-squares spline
122:         fit. If the errors in the `x` values have standard-deviation given by
123:         the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
124:     u : array_like, optional
125:         An array of parameter values. If not given, these values are
126:         calculated automatically as ``M = len(x[0])``, where
127: 
128:             v[0] = 0
129: 
130:             v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)
131: 
132:             u[i] = v[i] / v[M-1]
133: 
134:     ub, ue : int, optional
135:         The end-points of the parameters interval.  Defaults to
136:         u[0] and u[-1].
137:     k : int, optional
138:         Degree of the spline. Cubic splines are recommended.
139:         Even values of `k` should be avoided especially with a small s-value.
140:         ``1 <= k <= 5``, default is 3.
141:     task : int, optional
142:         If task==0 (default), find t and c for a given smoothing factor, s.
143:         If task==1, find t and c for another value of the smoothing factor, s.
144:         There must have been a previous call with task=0 or task=1
145:         for the same set of data.
146:         If task=-1 find the weighted least square spline for a given set of
147:         knots, t.
148:     s : float, optional
149:         A smoothing condition.  The amount of smoothness is determined by
150:         satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
151:         where g(x) is the smoothed interpolation of (x,y).  The user can
152:         use `s` to control the trade-off between closeness and smoothness
153:         of fit.  Larger `s` means more smoothing while smaller values of `s`
154:         indicate less smoothing. Recommended values of `s` depend on the
155:         weights, w.  If the weights represent the inverse of the
156:         standard-deviation of y, then a good `s` value should be found in
157:         the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
158:         data points in x, y, and w.
159:     t : int, optional
160:         The knots needed for task=-1.
161:     full_output : int, optional
162:         If non-zero, then return optional outputs.
163:     nest : int, optional
164:         An over-estimate of the total number of knots of the spline to
165:         help in determining the storage space.  By default nest=m/2.
166:         Always large enough is nest=m+k+1.
167:     per : int, optional
168:        If non-zero, data points are considered periodic with period
169:        ``x[m-1] - x[0]`` and a smooth periodic spline approximation is
170:        returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.
171:     quiet : int, optional
172:          Non-zero to suppress messages.
173:          This parameter is deprecated; use standard Python warning filters
174:          instead.
175: 
176:     Returns
177:     -------
178:     tck : tuple
179:         A tuple (t,c,k) containing the vector of knots, the B-spline
180:         coefficients, and the degree of the spline.
181:     u : array
182:         An array of the values of the parameter.
183:     fp : float
184:         The weighted sum of squared residuals of the spline approximation.
185:     ier : int
186:         An integer flag about splrep success.  Success is indicated
187:         if ier<=0. If ier in [1,2,3] an error occurred but was not raised.
188:         Otherwise an error is raised.
189:     msg : str
190:         A message corresponding to the integer flag, ier.
191: 
192:     See Also
193:     --------
194:     splrep, splev, sproot, spalde, splint,
195:     bisplrep, bisplev
196:     UnivariateSpline, BivariateSpline
197: 
198:     Notes
199:     -----
200:     See `splev` for evaluation of the spline and its derivatives.
201:     The number of dimensions N must be smaller than 11.
202: 
203:     References
204:     ----------
205:     .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
206:         parametric splines, Computer Graphics and Image Processing",
207:         20 (1982) 171-184.
208:     .. [2] P. Dierckx, "Algorithms for smoothing data with periodic and
209:         parametric splines", report tw55, Dept. Computer Science,
210:         K.U.Leuven, 1981.
211:     .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs on
212:         Numerical Analysis, Oxford University Press, 1993.
213: 
214:     '''
215:     if task <= 0:
216:         _parcur_cache = {'t': array([], float), 'wrk': array([], float),
217:                          'iwrk': array([], intc), 'u': array([], float),
218:                          'ub': 0, 'ue': 1}
219:     x = atleast_1d(x)
220:     idim, m = x.shape
221:     if per:
222:         for i in range(idim):
223:             if x[i][0] != x[i][-1]:
224:                 if quiet < 2:
225:                     warnings.warn(RuntimeWarning('Setting x[%d][%d]=x[%d][0]' %
226:                                                  (i, m, i)))
227:                 x[i][-1] = x[i][0]
228:     if not 0 < idim < 11:
229:         raise TypeError('0 < idim < 11 must hold')
230:     if w is None:
231:         w = ones(m, float)
232:     else:
233:         w = atleast_1d(w)
234:     ipar = (u is not None)
235:     if ipar:
236:         _parcur_cache['u'] = u
237:         if ub is None:
238:             _parcur_cache['ub'] = u[0]
239:         else:
240:             _parcur_cache['ub'] = ub
241:         if ue is None:
242:             _parcur_cache['ue'] = u[-1]
243:         else:
244:             _parcur_cache['ue'] = ue
245:     else:
246:         _parcur_cache['u'] = zeros(m, float)
247:     if not (1 <= k <= 5):
248:         raise TypeError('1 <= k= %d <=5 must hold' % k)
249:     if not (-1 <= task <= 1):
250:         raise TypeError('task must be -1, 0 or 1')
251:     if (not len(w) == m) or (ipar == 1 and (not len(u) == m)):
252:         raise TypeError('Mismatch of input dimensions')
253:     if s is None:
254:         s = m - sqrt(2*m)
255:     if t is None and task == -1:
256:         raise TypeError('Knots must be given for task=-1')
257:     if t is not None:
258:         _parcur_cache['t'] = atleast_1d(t)
259:     n = len(_parcur_cache['t'])
260:     if task == -1 and n < 2*k + 2:
261:         raise TypeError('There must be at least 2*k+2 knots for task=-1')
262:     if m <= k:
263:         raise TypeError('m > k must hold')
264:     if nest is None:
265:         nest = m + 2*k
266: 
267:     if (task >= 0 and s == 0) or (nest < 0):
268:         if per:
269:             nest = m + 2*k
270:         else:
271:             nest = m + k + 1
272:     nest = max(nest, 2*k + 3)
273:     u = _parcur_cache['u']
274:     ub = _parcur_cache['ub']
275:     ue = _parcur_cache['ue']
276:     t = _parcur_cache['t']
277:     wrk = _parcur_cache['wrk']
278:     iwrk = _parcur_cache['iwrk']
279:     t, c, o = _fitpack._parcur(ravel(transpose(x)), w, u, ub, ue, k,
280:                                task, ipar, s, t, nest, wrk, iwrk, per)
281:     _parcur_cache['u'] = o['u']
282:     _parcur_cache['ub'] = o['ub']
283:     _parcur_cache['ue'] = o['ue']
284:     _parcur_cache['t'] = t
285:     _parcur_cache['wrk'] = o['wrk']
286:     _parcur_cache['iwrk'] = o['iwrk']
287:     ier = o['ier']
288:     fp = o['fp']
289:     n = len(t)
290:     u = o['u']
291:     c.shape = idim, n - k - 1
292:     tcku = [t, list(c), k], u
293:     if ier <= 0 and not quiet:
294:         warnings.warn(RuntimeWarning(_iermess[ier][0] +
295:                                      "\tk=%d n=%d m=%d fp=%f s=%f" %
296:                                      (k, len(t), m, fp, s)))
297:     if ier > 0 and not full_output:
298:         if ier in [1, 2, 3]:
299:             warnings.warn(RuntimeWarning(_iermess[ier][0]))
300:         else:
301:             try:
302:                 raise _iermess[ier][1](_iermess[ier][0])
303:             except KeyError:
304:                 raise _iermess['unknown'][1](_iermess['unknown'][0])
305:     if full_output:
306:         try:
307:             return tcku, fp, ier, _iermess[ier][0]
308:         except KeyError:
309:             return tcku, fp, ier, _iermess['unknown'][0]
310:     else:
311:         return tcku
312: 
313: _curfit_cache = {'t': array([], float), 'wrk': array([], float),
314:                  'iwrk': array([], intc)}
315: 
316: 
317: def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
318:            full_output=0, per=0, quiet=1):
319:     '''
320:     Find the B-spline representation of 1-D curve.
321: 
322:     Given the set of data points ``(x[i], y[i])`` determine a smooth spline
323:     approximation of degree k on the interval ``xb <= x <= xe``.
324: 
325:     Parameters
326:     ----------
327:     x, y : array_like
328:         The data points defining a curve y = f(x).
329:     w : array_like, optional
330:         Strictly positive rank-1 array of weights the same length as x and y.
331:         The weights are used in computing the weighted least-squares spline
332:         fit. If the errors in the y values have standard-deviation given by the
333:         vector d, then w should be 1/d. Default is ones(len(x)).
334:     xb, xe : float, optional
335:         The interval to fit.  If None, these default to x[0] and x[-1]
336:         respectively.
337:     k : int, optional
338:         The order of the spline fit. It is recommended to use cubic splines.
339:         Even order splines should be avoided especially with small s values.
340:         1 <= k <= 5
341:     task : {1, 0, -1}, optional
342:         If task==0 find t and c for a given smoothing factor, s.
343: 
344:         If task==1 find t and c for another value of the smoothing factor, s.
345:         There must have been a previous call with task=0 or task=1 for the same
346:         set of data (t will be stored an used internally)
347: 
348:         If task=-1 find the weighted least square spline for a given set of
349:         knots, t. These should be interior knots as knots on the ends will be
350:         added automatically.
351:     s : float, optional
352:         A smoothing condition. The amount of smoothness is determined by
353:         satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x)
354:         is the smoothed interpolation of (x,y). The user can use s to control
355:         the tradeoff between closeness and smoothness of fit. Larger s means
356:         more smoothing while smaller values of s indicate less smoothing.
357:         Recommended values of s depend on the weights, w. If the weights
358:         represent the inverse of the standard-deviation of y, then a good s
359:         value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is
360:         the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if
361:         weights are supplied. s = 0.0 (interpolating) if no weights are
362:         supplied.
363:     t : array_like, optional
364:         The knots needed for task=-1. If given then task is automatically set
365:         to -1.
366:     full_output : bool, optional
367:         If non-zero, then return optional outputs.
368:     per : bool, optional
369:         If non-zero, data points are considered periodic with period x[m-1] -
370:         x[0] and a smooth periodic spline approximation is returned. Values of
371:         y[m-1] and w[m-1] are not used.
372:     quiet : bool, optional
373:         Non-zero to suppress messages.
374:         This parameter is deprecated; use standard Python warning filters
375:         instead.
376: 
377:     Returns
378:     -------
379:     tck : tuple
380:         (t,c,k) a tuple containing the vector of knots, the B-spline
381:         coefficients, and the degree of the spline.
382:     fp : array, optional
383:         The weighted sum of squared residuals of the spline approximation.
384:     ier : int, optional
385:         An integer flag about splrep success. Success is indicated if ier<=0.
386:         If ier in [1,2,3] an error occurred but was not raised. Otherwise an
387:         error is raised.
388:     msg : str, optional
389:         A message corresponding to the integer flag, ier.
390: 
391:     Notes
392:     -----
393:     See splev for evaluation of the spline and its derivatives.
394: 
395:     The user is responsible for assuring that the values of *x* are unique.
396:     Otherwise, *splrep* will not return sensible results.
397: 
398:     See Also
399:     --------
400:     UnivariateSpline, BivariateSpline
401:     splprep, splev, sproot, spalde, splint
402:     bisplrep, bisplev
403: 
404:     Notes
405:     -----
406:     See splev for evaluation of the spline and its derivatives. Uses the
407:     FORTRAN routine curfit from FITPACK.
408: 
409:     If provided, knots `t` must satisfy the Schoenberg-Whitney conditions,
410:     i.e., there must be a subset of data points ``x[j]`` such that
411:     ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.
412: 
413:     References
414:     ----------
415:     Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:
416: 
417:     .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and
418:        integration of experimental data using spline functions",
419:        J.Comp.Appl.Maths 1 (1975) 165-184.
420:     .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular
421:        grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)
422:        1286-1304.
423:     .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline
424:        functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.
425:     .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on
426:        Numerical Analysis, Oxford University Press, 1993.
427: 
428:     Examples
429:     --------
430: 
431:     >>> import matplotlib.pyplot as plt
432:     >>> from scipy.interpolate import splev, splrep
433:     >>> x = np.linspace(0, 10, 10)
434:     >>> y = np.sin(x)
435:     >>> tck = splrep(x, y)
436:     >>> x2 = np.linspace(0, 10, 200)
437:     >>> y2 = splev(x2, tck)
438:     >>> plt.plot(x, y, 'o', x2, y2)
439:     >>> plt.show()
440: 
441:     '''
442:     if task <= 0:
443:         _curfit_cache = {}
444:     x, y = map(atleast_1d, [x, y])
445:     m = len(x)
446:     if w is None:
447:         w = ones(m, float)
448:         if s is None:
449:             s = 0.0
450:     else:
451:         w = atleast_1d(w)
452:         if s is None:
453:             s = m - sqrt(2*m)
454:     if not len(w) == m:
455:         raise TypeError('len(w)=%d is not equal to m=%d' % (len(w), m))
456:     if (m != len(y)) or (m != len(w)):
457:         raise TypeError('Lengths of the first three arguments (x,y,w) must '
458:                         'be equal')
459:     if not (1 <= k <= 5):
460:         raise TypeError('Given degree of the spline (k=%d) is not supported. '
461:                         '(1<=k<=5)' % k)
462:     if m <= k:
463:         raise TypeError('m > k must hold')
464:     if xb is None:
465:         xb = x[0]
466:     if xe is None:
467:         xe = x[-1]
468:     if not (-1 <= task <= 1):
469:         raise TypeError('task must be -1, 0 or 1')
470:     if t is not None:
471:         task = -1
472:     if task == -1:
473:         if t is None:
474:             raise TypeError('Knots must be given for task=-1')
475:         numknots = len(t)
476:         _curfit_cache['t'] = empty((numknots + 2*k + 2,), float)
477:         _curfit_cache['t'][k+1:-k-1] = t
478:         nest = len(_curfit_cache['t'])
479:     elif task == 0:
480:         if per:
481:             nest = max(m + 2*k, 2*k + 3)
482:         else:
483:             nest = max(m + k + 1, 2*k + 3)
484:         t = empty((nest,), float)
485:         _curfit_cache['t'] = t
486:     if task <= 0:
487:         if per:
488:             _curfit_cache['wrk'] = empty((m*(k + 1) + nest*(8 + 5*k),), float)
489:         else:
490:             _curfit_cache['wrk'] = empty((m*(k + 1) + nest*(7 + 3*k),), float)
491:         _curfit_cache['iwrk'] = empty((nest,), intc)
492:     try:
493:         t = _curfit_cache['t']
494:         wrk = _curfit_cache['wrk']
495:         iwrk = _curfit_cache['iwrk']
496:     except KeyError:
497:         raise TypeError("must call with task=1 only after"
498:                         " call with task=0,-1")
499:     if not per:
500:         n, c, fp, ier = dfitpack.curfit(task, x, y, w, t, wrk, iwrk,
501:                                         xb, xe, k, s)
502:     else:
503:         n, c, fp, ier = dfitpack.percur(task, x, y, w, t, wrk, iwrk, k, s)
504:     tck = (t[:n], c[:n], k)
505:     if ier <= 0 and not quiet:
506:         _mess = (_iermess[ier][0] + "\tk=%d n=%d m=%d fp=%f s=%f" %
507:                  (k, len(t), m, fp, s))
508:         warnings.warn(RuntimeWarning(_mess))
509:     if ier > 0 and not full_output:
510:         if ier in [1, 2, 3]:
511:             warnings.warn(RuntimeWarning(_iermess[ier][0]))
512:         else:
513:             try:
514:                 raise _iermess[ier][1](_iermess[ier][0])
515:             except KeyError:
516:                 raise _iermess['unknown'][1](_iermess['unknown'][0])
517:     if full_output:
518:         try:
519:             return tck, fp, ier, _iermess[ier][0]
520:         except KeyError:
521:             return tck, fp, ier, _iermess['unknown'][0]
522:     else:
523:         return tck
524: 
525: 
526: def splev(x, tck, der=0, ext=0):
527:     '''
528:     Evaluate a B-spline or its derivatives.
529: 
530:     Given the knots and coefficients of a B-spline representation, evaluate
531:     the value of the smoothing polynomial and its derivatives.  This is a
532:     wrapper around the FORTRAN routines splev and splder of FITPACK.
533: 
534:     Parameters
535:     ----------
536:     x : array_like
537:         An array of points at which to return the value of the smoothed
538:         spline or its derivatives.  If `tck` was returned from `splprep`,
539:         then the parameter values, u should be given.
540:     tck : tuple
541:         A sequence of length 3 returned by `splrep` or `splprep` containing
542:         the knots, coefficients, and degree of the spline.
543:     der : int, optional
544:         The order of derivative of the spline to compute (must be less than
545:         or equal to k).
546:     ext : int, optional
547:         Controls the value returned for elements of ``x`` not in the
548:         interval defined by the knot sequence.
549: 
550:         * if ext=0, return the extrapolated value.
551:         * if ext=1, return 0
552:         * if ext=2, raise a ValueError
553:         * if ext=3, return the boundary value.
554: 
555:         The default value is 0.
556: 
557:     Returns
558:     -------
559:     y : ndarray or list of ndarrays
560:         An array of values representing the spline function evaluated at
561:         the points in ``x``.  If `tck` was returned from `splprep`, then this
562:         is a list of arrays representing the curve in N-dimensional space.
563: 
564:     See Also
565:     --------
566:     splprep, splrep, sproot, spalde, splint
567:     bisplrep, bisplev
568: 
569:     References
570:     ----------
571:     .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
572:         Theory, 6, p.50-62, 1972.
573:     .. [2] M.G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
574:         Applics, 10, p.134-149, 1972.
575:     .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
576:         on Numerical Analysis, Oxford University Press, 1993.
577: 
578:     '''
579:     t, c, k = tck
580:     try:
581:         c[0][0]
582:         parametric = True
583:     except:
584:         parametric = False
585:     if parametric:
586:         return list(map(lambda c, x=x, t=t, k=k, der=der:
587:                         splev(x, [t, c, k], der, ext), c))
588:     else:
589:         if not (0 <= der <= k):
590:             raise ValueError("0<=der=%d<=k=%d must hold" % (der, k))
591:         if ext not in (0, 1, 2, 3):
592:             raise ValueError("ext = %s not in (0, 1, 2, 3) " % ext)
593: 
594:         x = asarray(x)
595:         shape = x.shape
596:         x = atleast_1d(x).ravel()
597:         y, ier = _fitpack._spl_(x, der, t, c, k, ext)
598: 
599:         if ier == 10:
600:             raise ValueError("Invalid input data")
601:         if ier == 1:
602:             raise ValueError("Found x value not in the domain")
603:         if ier:
604:             raise TypeError("An error occurred")
605: 
606:         return y.reshape(shape)
607: 
608: 
609: def splint(a, b, tck, full_output=0):
610:     '''
611:     Evaluate the definite integral of a B-spline.
612: 
613:     Given the knots and coefficients of a B-spline, evaluate the definite
614:     integral of the smoothing polynomial between two given points.
615: 
616:     Parameters
617:     ----------
618:     a, b : float
619:         The end-points of the integration interval.
620:     tck : tuple
621:         A tuple (t,c,k) containing the vector of knots, the B-spline
622:         coefficients, and the degree of the spline (see `splev`).
623:     full_output : int, optional
624:         Non-zero to return optional output.
625: 
626:     Returns
627:     -------
628:     integral : float
629:         The resulting integral.
630:     wrk : ndarray
631:         An array containing the integrals of the normalized B-splines
632:         defined on the set of knots.
633: 
634:     Notes
635:     -----
636:     splint silently assumes that the spline function is zero outside the data
637:     interval (a, b).
638: 
639:     See Also
640:     --------
641:     splprep, splrep, sproot, spalde, splev
642:     bisplrep, bisplev
643:     UnivariateSpline, BivariateSpline
644: 
645:     References
646:     ----------
647:     .. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines",
648:         J. Inst. Maths Applics, 17, p.37-41, 1976.
649:     .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs
650:         on Numerical Analysis, Oxford University Press, 1993.
651: 
652:     '''
653:     t, c, k = tck
654:     try:
655:         c[0][0]
656:         parametric = True
657:     except:
658:         parametric = False
659:     if parametric:
660:         return list(map(lambda c, a=a, b=b, t=t, k=k:
661:                         splint(a, b, [t, c, k]), c))
662:     else:
663:         aint, wrk = _fitpack._splint(t, c, k, a, b)
664:         if full_output:
665:             return aint, wrk
666:         else:
667:             return aint
668: 
669: 
670: def sproot(tck, mest=10):
671:     '''
672:     Find the roots of a cubic B-spline.
673: 
674:     Given the knots (>=8) and coefficients of a cubic B-spline return the
675:     roots of the spline.
676: 
677:     Parameters
678:     ----------
679:     tck : tuple
680:         A tuple (t,c,k) containing the vector of knots,
681:         the B-spline coefficients, and the degree of the spline.
682:         The number of knots must be >= 8, and the degree must be 3.
683:         The knots must be a montonically increasing sequence.
684:     mest : int, optional
685:         An estimate of the number of zeros (Default is 10).
686: 
687:     Returns
688:     -------
689:     zeros : ndarray
690:         An array giving the roots of the spline.
691: 
692:     See also
693:     --------
694:     splprep, splrep, splint, spalde, splev
695:     bisplrep, bisplev
696:     UnivariateSpline, BivariateSpline
697: 
698: 
699:     References
700:     ----------
701:     .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
702:         Theory, 6, p.50-62, 1972.
703:     .. [2] M.G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
704:         Applics, 10, p.134-149, 1972.
705:     .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
706:         on Numerical Analysis, Oxford University Press, 1993.
707: 
708:     '''
709:     t, c, k = tck
710:     if k != 3:
711:         raise ValueError("sproot works only for cubic (k=3) splines")
712:     try:
713:         c[0][0]
714:         parametric = True
715:     except:
716:         parametric = False
717:     if parametric:
718:         return list(map(lambda c, t=t, k=k, mest=mest:
719:                         sproot([t, c, k], mest), c))
720:     else:
721:         if len(t) < 8:
722:             raise TypeError("The number of knots %d>=8" % len(t))
723:         z, ier = _fitpack._sproot(t, c, k, mest)
724:         if ier == 10:
725:             raise TypeError("Invalid input data. "
726:                             "t1<=..<=t4<t5<..<tn-3<=..<=tn must hold.")
727:         if ier == 0:
728:             return z
729:         if ier == 1:
730:             warnings.warn(RuntimeWarning("The number of zeros exceeds mest"))
731:             return z
732:         raise TypeError("Unknown error")
733: 
734: 
735: def spalde(x, tck):
736:     '''
737:     Evaluate all derivatives of a B-spline.
738: 
739:     Given the knots and coefficients of a cubic B-spline compute all
740:     derivatives up to order k at a point (or set of points).
741: 
742:     Parameters
743:     ----------
744:     x : array_like
745:         A point or a set of points at which to evaluate the derivatives.
746:         Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.
747:     tck : tuple
748:         A tuple (t,c,k) containing the vector of knots,
749:         the B-spline coefficients, and the degree of the spline.
750: 
751:     Returns
752:     -------
753:     results : {ndarray, list of ndarrays}
754:         An array (or a list of arrays) containing all derivatives
755:         up to order k inclusive for each point `x`.
756: 
757:     See Also
758:     --------
759:     splprep, splrep, splint, sproot, splev, bisplrep, bisplev,
760:     UnivariateSpline, BivariateSpline
761: 
762:     References
763:     ----------
764:     .. [1] de Boor C : On calculating with b-splines, J. Approximation Theory
765:        6 (1972) 50-62.
766:     .. [2] Cox M.G. : The numerical evaluation of b-splines, J. Inst. Maths
767:        applics 10 (1972) 134-149.
768:     .. [3] Dierckx P. : Curve and surface fitting with splines, Monographs on
769:        Numerical Analysis, Oxford University Press, 1993.
770: 
771:     '''
772:     t, c, k = tck
773:     try:
774:         c[0][0]
775:         parametric = True
776:     except:
777:         parametric = False
778:     if parametric:
779:         return list(map(lambda c, x=x, t=t, k=k:
780:                         spalde(x, [t, c, k]), c))
781:     else:
782:         x = atleast_1d(x)
783:         if len(x) > 1:
784:             return list(map(lambda x, tck=tck: spalde(x, tck), x))
785:         d, ier = _fitpack._spalde(t, c, k, x[0])
786:         if ier == 0:
787:             return d
788:         if ier == 10:
789:             raise TypeError("Invalid input data. t(k)<=x<=t(n-k+1) must hold.")
790:         raise TypeError("Unknown error")
791: 
792: # def _curfit(x,y,w=None,xb=None,xe=None,k=3,task=0,s=None,t=None,
793: #           full_output=0,nest=None,per=0,quiet=1):
794: 
795: _surfit_cache = {'tx': array([], float), 'ty': array([], float),
796:                  'wrk': array([], float), 'iwrk': array([], intc)}
797: 
798: 
799: def bisplrep(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None,
800:              kx=3, ky=3, task=0, s=None, eps=1e-16, tx=None, ty=None,
801:              full_output=0, nxest=None, nyest=None, quiet=1):
802:     '''
803:     Find a bivariate B-spline representation of a surface.
804: 
805:     Given a set of data points (x[i], y[i], z[i]) representing a surface
806:     z=f(x,y), compute a B-spline representation of the surface. Based on
807:     the routine SURFIT from FITPACK.
808: 
809:     Parameters
810:     ----------
811:     x, y, z : ndarray
812:         Rank-1 arrays of data points.
813:     w : ndarray, optional
814:         Rank-1 array of weights. By default ``w=np.ones(len(x))``.
815:     xb, xe : float, optional
816:         End points of approximation interval in `x`.
817:         By default ``xb = x.min(), xe=x.max()``.
818:     yb, ye : float, optional
819:         End points of approximation interval in `y`.
820:         By default ``yb=y.min(), ye = y.max()``.
821:     kx, ky : int, optional
822:         The degrees of the spline (1 <= kx, ky <= 5).
823:         Third order (kx=ky=3) is recommended.
824:     task : int, optional
825:         If task=0, find knots in x and y and coefficients for a given
826:         smoothing factor, s.
827:         If task=1, find knots and coefficients for another value of the
828:         smoothing factor, s.  bisplrep must have been previously called
829:         with task=0 or task=1.
830:         If task=-1, find coefficients for a given set of knots tx, ty.
831:     s : float, optional
832:         A non-negative smoothing factor.  If weights correspond
833:         to the inverse of the standard-deviation of the errors in z,
834:         then a good s-value should be found in the range
835:         ``(m-sqrt(2*m),m+sqrt(2*m))`` where m=len(x).
836:     eps : float, optional
837:         A threshold for determining the effective rank of an
838:         over-determined linear system of equations (0 < eps < 1).
839:         `eps` is not likely to need changing.
840:     tx, ty : ndarray, optional
841:         Rank-1 arrays of the knots of the spline for task=-1
842:     full_output : int, optional
843:         Non-zero to return optional outputs.
844:     nxest, nyest : int, optional
845:         Over-estimates of the total number of knots. If None then
846:         ``nxest = max(kx+sqrt(m/2),2*kx+3)``,
847:         ``nyest = max(ky+sqrt(m/2),2*ky+3)``.
848:     quiet : int, optional
849:         Non-zero to suppress printing of messages.
850:         This parameter is deprecated; use standard Python warning filters
851:         instead.
852: 
853:     Returns
854:     -------
855:     tck : array_like
856:         A list [tx, ty, c, kx, ky] containing the knots (tx, ty) and
857:         coefficients (c) of the bivariate B-spline representation of the
858:         surface along with the degree of the spline.
859:     fp : ndarray
860:         The weighted sum of squared residuals of the spline approximation.
861:     ier : int
862:         An integer flag about splrep success.  Success is indicated if
863:         ier<=0. If ier in [1,2,3] an error occurred but was not raised.
864:         Otherwise an error is raised.
865:     msg : str
866:         A message corresponding to the integer flag, ier.
867: 
868:     See Also
869:     --------
870:     splprep, splrep, splint, sproot, splev
871:     UnivariateSpline, BivariateSpline
872: 
873:     Notes
874:     -----
875:     See `bisplev` to evaluate the value of the B-spline given its tck
876:     representation.
877: 
878:     References
879:     ----------
880:     .. [1] Dierckx P.:An algorithm for surface fitting with spline functions
881:        Ima J. Numer. Anal. 1 (1981) 267-283.
882:     .. [2] Dierckx P.:An algorithm for surface fitting with spline functions
883:        report tw50, Dept. Computer Science,K.U.Leuven, 1980.
884:     .. [3] Dierckx P.:Curve and surface fitting with splines, Monographs on
885:        Numerical Analysis, Oxford University Press, 1993.
886: 
887:     '''
888:     x, y, z = map(ravel, [x, y, z])  # ensure 1-d arrays.
889:     m = len(x)
890:     if not (m == len(y) == len(z)):
891:         raise TypeError('len(x)==len(y)==len(z) must hold.')
892:     if w is None:
893:         w = ones(m, float)
894:     else:
895:         w = atleast_1d(w)
896:     if not len(w) == m:
897:         raise TypeError('len(w)=%d is not equal to m=%d' % (len(w), m))
898:     if xb is None:
899:         xb = x.min()
900:     if xe is None:
901:         xe = x.max()
902:     if yb is None:
903:         yb = y.min()
904:     if ye is None:
905:         ye = y.max()
906:     if not (-1 <= task <= 1):
907:         raise TypeError('task must be -1, 0 or 1')
908:     if s is None:
909:         s = m - sqrt(2*m)
910:     if tx is None and task == -1:
911:         raise TypeError('Knots_x must be given for task=-1')
912:     if tx is not None:
913:         _surfit_cache['tx'] = atleast_1d(tx)
914:     nx = len(_surfit_cache['tx'])
915:     if ty is None and task == -1:
916:         raise TypeError('Knots_y must be given for task=-1')
917:     if ty is not None:
918:         _surfit_cache['ty'] = atleast_1d(ty)
919:     ny = len(_surfit_cache['ty'])
920:     if task == -1 and nx < 2*kx+2:
921:         raise TypeError('There must be at least 2*kx+2 knots_x for task=-1')
922:     if task == -1 and ny < 2*ky+2:
923:         raise TypeError('There must be at least 2*ky+2 knots_x for task=-1')
924:     if not ((1 <= kx <= 5) and (1 <= ky <= 5)):
925:         raise TypeError('Given degree of the spline (kx,ky=%d,%d) is not '
926:                         'supported. (1<=k<=5)' % (kx, ky))
927:     if m < (kx + 1)*(ky + 1):
928:         raise TypeError('m >= (kx+1)(ky+1) must hold')
929:     if nxest is None:
930:         nxest = int(kx + sqrt(m/2))
931:     if nyest is None:
932:         nyest = int(ky + sqrt(m/2))
933:     nxest, nyest = max(nxest, 2*kx + 3), max(nyest, 2*ky + 3)
934:     if task >= 0 and s == 0:
935:         nxest = int(kx + sqrt(3*m))
936:         nyest = int(ky + sqrt(3*m))
937:     if task == -1:
938:         _surfit_cache['tx'] = atleast_1d(tx)
939:         _surfit_cache['ty'] = atleast_1d(ty)
940:     tx, ty = _surfit_cache['tx'], _surfit_cache['ty']
941:     wrk = _surfit_cache['wrk']
942:     u = nxest - kx - 1
943:     v = nyest - ky - 1
944:     km = max(kx, ky) + 1
945:     ne = max(nxest, nyest)
946:     bx, by = kx*v + ky + 1, ky*u + kx + 1
947:     b1, b2 = bx, bx + v - ky
948:     if bx > by:
949:         b1, b2 = by, by + u - kx
950:     msg = "Too many data points to interpolate"
951:     lwrk1 = _intc_overflow(u*v*(2 + b1 + b2) +
952:                            2*(u + v + km*(m + ne) + ne - kx - ky) + b2 + 1,
953:                            msg=msg)
954:     lwrk2 = _intc_overflow(u*v*(b2 + 1) + b2, msg=msg)
955:     tx, ty, c, o = _fitpack._surfit(x, y, z, w, xb, xe, yb, ye, kx, ky,
956:                                     task, s, eps, tx, ty, nxest, nyest,
957:                                     wrk, lwrk1, lwrk2)
958:     _curfit_cache['tx'] = tx
959:     _curfit_cache['ty'] = ty
960:     _curfit_cache['wrk'] = o['wrk']
961:     ier, fp = o['ier'], o['fp']
962:     tck = [tx, ty, c, kx, ky]
963: 
964:     ierm = min(11, max(-3, ier))
965:     if ierm <= 0 and not quiet:
966:         _mess = (_iermess2[ierm][0] +
967:                  "\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f" %
968:                  (kx, ky, len(tx), len(ty), m, fp, s))
969:         warnings.warn(RuntimeWarning(_mess))
970:     if ierm > 0 and not full_output:
971:         if ier in [1, 2, 3, 4, 5]:
972:             _mess = ("\n\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f" %
973:                      (kx, ky, len(tx), len(ty), m, fp, s))
974:             warnings.warn(RuntimeWarning(_iermess2[ierm][0] + _mess))
975:         else:
976:             try:
977:                 raise _iermess2[ierm][1](_iermess2[ierm][0])
978:             except KeyError:
979:                 raise _iermess2['unknown'][1](_iermess2['unknown'][0])
980:     if full_output:
981:         try:
982:             return tck, fp, ier, _iermess2[ierm][0]
983:         except KeyError:
984:             return tck, fp, ier, _iermess2['unknown'][0]
985:     else:
986:         return tck
987: 
988: 
989: def bisplev(x, y, tck, dx=0, dy=0):
990:     '''
991:     Evaluate a bivariate B-spline and its derivatives.
992: 
993:     Return a rank-2 array of spline function values (or spline derivative
994:     values) at points given by the cross-product of the rank-1 arrays `x` and
995:     `y`.  In special cases, return an array or just a float if either `x` or
996:     `y` or both are floats.  Based on BISPEV from FITPACK.
997: 
998:     Parameters
999:     ----------
1000:     x, y : ndarray
1001:         Rank-1 arrays specifying the domain over which to evaluate the
1002:         spline or its derivative.
1003:     tck : tuple
1004:         A sequence of length 5 returned by `bisplrep` containing the knot
1005:         locations, the coefficients, and the degree of the spline:
1006:         [tx, ty, c, kx, ky].
1007:     dx, dy : int, optional
1008:         The orders of the partial derivatives in `x` and `y` respectively.
1009: 
1010:     Returns
1011:     -------
1012:     vals : ndarray
1013:         The B-spline or its derivative evaluated over the set formed by
1014:         the cross-product of `x` and `y`.
1015: 
1016:     See Also
1017:     --------
1018:     splprep, splrep, splint, sproot, splev
1019:     UnivariateSpline, BivariateSpline
1020: 
1021:     Notes
1022:     -----
1023:         See `bisplrep` to generate the `tck` representation.
1024: 
1025:     References
1026:     ----------
1027:     .. [1] Dierckx P. : An algorithm for surface fitting
1028:        with spline functions
1029:        Ima J. Numer. Anal. 1 (1981) 267-283.
1030:     .. [2] Dierckx P. : An algorithm for surface fitting
1031:        with spline functions
1032:        report tw50, Dept. Computer Science,K.U.Leuven, 1980.
1033:     .. [3] Dierckx P. : Curve and surface fitting with splines,
1034:        Monographs on Numerical Analysis, Oxford University Press, 1993.
1035: 
1036:     '''
1037:     tx, ty, c, kx, ky = tck
1038:     if not (0 <= dx < kx):
1039:         raise ValueError("0 <= dx = %d < kx = %d must hold" % (dx, kx))
1040:     if not (0 <= dy < ky):
1041:         raise ValueError("0 <= dy = %d < ky = %d must hold" % (dy, ky))
1042:     x, y = map(atleast_1d, [x, y])
1043:     if (len(x.shape) != 1) or (len(y.shape) != 1):
1044:         raise ValueError("First two entries should be rank-1 arrays.")
1045:     z, ier = _fitpack._bispev(tx, ty, c, kx, ky, x, y, dx, dy)
1046:     if ier == 10:
1047:         raise ValueError("Invalid input data")
1048:     if ier:
1049:         raise TypeError("An error occurred")
1050:     z.shape = len(x), len(y)
1051:     if len(z) > 1:
1052:         return z
1053:     if len(z[0]) > 1:
1054:         return z[0]
1055:     return z[0][0]
1056: 
1057: 
1058: def dblint(xa, xb, ya, yb, tck):
1059:     '''Evaluate the integral of a spline over area [xa,xb] x [ya,yb].
1060: 
1061:     Parameters
1062:     ----------
1063:     xa, xb : float
1064:         The end-points of the x integration interval.
1065:     ya, yb : float
1066:         The end-points of the y integration interval.
1067:     tck : list [tx, ty, c, kx, ky]
1068:         A sequence of length 5 returned by bisplrep containing the knot
1069:         locations tx, ty, the coefficients c, and the degrees kx, ky
1070:         of the spline.
1071: 
1072:     Returns
1073:     -------
1074:     integ : float
1075:         The value of the resulting integral.
1076:     '''
1077:     tx, ty, c, kx, ky = tck
1078:     return dfitpack.dblint(tx, ty, c, kx, ky, xa, xb, ya, yb)
1079: 
1080: 
1081: def insert(x, tck, m=1, per=0):
1082:     '''
1083:     Insert knots into a B-spline.
1084: 
1085:     Given the knots and coefficients of a B-spline representation, create a
1086:     new B-spline with a knot inserted `m` times at point `x`.
1087:     This is a wrapper around the FORTRAN routine insert of FITPACK.
1088: 
1089:     Parameters
1090:     ----------
1091:     x (u) : array_like
1092:         A 1-D point at which to insert a new knot(s).  If `tck` was returned
1093:         from ``splprep``, then the parameter values, u should be given.
1094:     tck : tuple
1095:         A tuple (t,c,k) returned by ``splrep`` or ``splprep`` containing
1096:         the vector of knots, the B-spline coefficients,
1097:         and the degree of the spline.
1098:     m : int, optional
1099:         The number of times to insert the given knot (its multiplicity).
1100:         Default is 1.
1101:     per : int, optional
1102:         If non-zero, the input spline is considered periodic.
1103: 
1104:     Returns
1105:     -------
1106:     tck : tuple
1107:         A tuple (t,c,k) containing the vector of knots, the B-spline
1108:         coefficients, and the degree of the new spline.
1109:         ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.
1110:         In case of a periodic spline (``per != 0``) there must be
1111:         either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``
1112:         or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.
1113: 
1114:     Notes
1115:     -----
1116:     Based on algorithms from [1]_ and [2]_.
1117: 
1118:     References
1119:     ----------
1120:     .. [1] W. Boehm, "Inserting new knots into b-spline curves.",
1121:         Computer Aided Design, 12, p.199-201, 1980.
1122:     .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on
1123:         Numerical Analysis", Oxford University Press, 1993.
1124: 
1125:     '''
1126:     t, c, k = tck
1127:     try:
1128:         c[0][0]
1129:         parametric = True
1130:     except:
1131:         parametric = False
1132:     if parametric:
1133:         cc = []
1134:         for c_vals in c:
1135:             tt, cc_val, kk = insert(x, [t, c_vals, k], m)
1136:             cc.append(cc_val)
1137:         return (tt, cc, kk)
1138:     else:
1139:         tt, cc, ier = _fitpack._insert(per, t, c, k, x, m)
1140:         if ier == 10:
1141:             raise ValueError("Invalid input data")
1142:         if ier:
1143:             raise TypeError("An error occurred")
1144:         return (tt, cc, k)
1145: 
1146: 
1147: def splder(tck, n=1):
1148:     '''
1149:     Compute the spline representation of the derivative of a given spline
1150: 
1151:     Parameters
1152:     ----------
1153:     tck : tuple of (t, c, k)
1154:         Spline whose derivative to compute
1155:     n : int, optional
1156:         Order of derivative to evaluate. Default: 1
1157: 
1158:     Returns
1159:     -------
1160:     tck_der : tuple of (t2, c2, k2)
1161:         Spline of order k2=k-n representing the derivative
1162:         of the input spline.
1163: 
1164:     Notes
1165:     -----
1166: 
1167:     .. versionadded:: 0.13.0
1168: 
1169:     See Also
1170:     --------
1171:     splantider, splev, spalde
1172: 
1173:     Examples
1174:     --------
1175:     This can be used for finding maxima of a curve:
1176: 
1177:     >>> from scipy.interpolate import splrep, splder, sproot
1178:     >>> x = np.linspace(0, 10, 70)
1179:     >>> y = np.sin(x)
1180:     >>> spl = splrep(x, y, k=4)
1181: 
1182:     Now, differentiate the spline and find the zeros of the
1183:     derivative. (NB: `sproot` only works for order 3 splines, so we
1184:     fit an order 4 spline):
1185: 
1186:     >>> dspl = splder(spl)
1187:     >>> sproot(dspl) / np.pi
1188:     array([ 0.50000001,  1.5       ,  2.49999998])
1189: 
1190:     This agrees well with roots :math:`\\pi/2 + n\\pi` of
1191:     :math:`\\cos(x) = \\sin'(x)`.
1192: 
1193:     '''
1194:     if n < 0:
1195:         return splantider(tck, -n)
1196: 
1197:     t, c, k = tck
1198: 
1199:     if n > k:
1200:         raise ValueError(("Order of derivative (n = %r) must be <= "
1201:                           "order of spline (k = %r)") % (n, tck[2]))
1202: 
1203:     # Extra axes for the trailing dims of the `c` array:
1204:     sh = (slice(None),) + ((None,)*len(c.shape[1:]))
1205: 
1206:     with np.errstate(invalid='raise', divide='raise'):
1207:         try:
1208:             for j in range(n):
1209:                 # See e.g. Schumaker, Spline Functions: Basic Theory, Chapter 5
1210: 
1211:                 # Compute the denominator in the differentiation formula.
1212:                 # (and append traling dims, if necessary)
1213:                 dt = t[k+1:-1] - t[1:-k-1]
1214:                 dt = dt[sh]
1215:                 # Compute the new coefficients
1216:                 c = (c[1:-1-k] - c[:-2-k]) * k / dt
1217:                 # Pad coefficient array to same size as knots (FITPACK
1218:                 # convention)
1219:                 c = np.r_[c, np.zeros((k,) + c.shape[1:])]
1220:                 # Adjust knots
1221:                 t = t[1:-1]
1222:                 k -= 1
1223:         except FloatingPointError:
1224:             raise ValueError(("The spline has internal repeated knots "
1225:                               "and is not differentiable %d times") % n)
1226: 
1227:     return t, c, k
1228: 
1229: 
1230: def splantider(tck, n=1):
1231:     '''
1232:     Compute the spline for the antiderivative (integral) of a given spline.
1233: 
1234:     Parameters
1235:     ----------
1236:     tck : tuple of (t, c, k)
1237:         Spline whose antiderivative to compute
1238:     n : int, optional
1239:         Order of antiderivative to evaluate. Default: 1
1240: 
1241:     Returns
1242:     -------
1243:     tck_ader : tuple of (t2, c2, k2)
1244:         Spline of order k2=k+n representing the antiderivative of the input
1245:         spline.
1246: 
1247:     See Also
1248:     --------
1249:     splder, splev, spalde
1250: 
1251:     Notes
1252:     -----
1253:     The `splder` function is the inverse operation of this function.
1254:     Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
1255:     rounding error.
1256: 
1257:     .. versionadded:: 0.13.0
1258: 
1259:     Examples
1260:     --------
1261:     >>> from scipy.interpolate import splrep, splder, splantider, splev
1262:     >>> x = np.linspace(0, np.pi/2, 70)
1263:     >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
1264:     >>> spl = splrep(x, y)
1265: 
1266:     The derivative is the inverse operation of the antiderivative,
1267:     although some floating point error accumulates:
1268: 
1269:     >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
1270:     (array(2.1565429877197317), array(2.1565429877201865))
1271: 
1272:     Antiderivative can be used to evaluate definite integrals:
1273: 
1274:     >>> ispl = splantider(spl)
1275:     >>> splev(np.pi/2, ispl) - splev(0, ispl)
1276:     2.2572053588768486
1277: 
1278:     This is indeed an approximation to the complete elliptic integral
1279:     :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:
1280: 
1281:     >>> from scipy.special import ellipk
1282:     >>> ellipk(0.8)
1283:     2.2572053268208538
1284: 
1285:     '''
1286:     if n < 0:
1287:         return splder(tck, -n)
1288: 
1289:     t, c, k = tck
1290: 
1291:     # Extra axes for the trailing dims of the `c` array:
1292:     sh = (slice(None),) + (None,)*len(c.shape[1:])
1293: 
1294:     for j in range(n):
1295:         # This is the inverse set of operations to splder.
1296: 
1297:         # Compute the multiplier in the antiderivative formula.
1298:         dt = t[k+1:] - t[:-k-1]
1299:         dt = dt[sh]
1300:         # Compute the new coefficients
1301:         c = np.cumsum(c[:-k-1] * dt, axis=0) / (k + 1)
1302:         c = np.r_[np.zeros((1,) + c.shape[1:]),
1303:                   c,
1304:                   [c[-1]] * (k+2)]
1305:         # New knots
1306:         t = np.r_[t[0], t, t[-1]]
1307:         k += 1
1308: 
1309:     return t, c, k
1310: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_78360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\nfitpack (dierckx in netlib) --- A Python-C wrapper to FITPACK (by P. Dierckx).\n        FITPACK is a collection of FORTRAN programs for curve and surface\n        fitting with splines and tensor product splines.\n\nSee\n http://www.cs.kuleuven.ac.be/cwis/research/nalag/research/topics/fitpack.html\nor\n http://www.netlib.org/dierckx/index.html\n\nCopyright 2002 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@cens.ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the SciPy (BSD style) license.  See LICENSE.txt that came with\nthis distribution for specifics.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n\nTODO: Make interfaces to the following fitpack functions:\n    For univariate splines: cocosp, concon, fourco, insert\n    For bivariate splines: profil, regrid, parsur, surev\n')

# Assigning a List to a Name (line 26):

# Assigning a List to a Name (line 26):
__all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde', 'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']
module_type_store.set_exportable_members(['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde', 'bisplrep', 'bisplev', 'insert', 'splder', 'splantider'])

# Obtaining an instance of the builtin type 'list' (line 26)
list_78361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_78362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', 'splrep')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78362)
# Adding element type (line 26)
str_78363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'str', 'splprep')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78363)
# Adding element type (line 26)
str_78364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'str', 'splev')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78364)
# Adding element type (line 26)
str_78365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 41), 'str', 'splint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78365)
# Adding element type (line 26)
str_78366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 51), 'str', 'sproot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78366)
# Adding element type (line 26)
str_78367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 61), 'str', 'spalde')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78367)
# Adding element type (line 26)
str_78368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'str', 'bisplrep')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78368)
# Adding element type (line 26)
str_78369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'str', 'bisplev')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78369)
# Adding element type (line 26)
str_78370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'str', 'insert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78370)
# Adding element type (line 26)
str_78371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 44), 'str', 'splder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78371)
# Adding element type (line 26)
str_78372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 54), 'str', 'splantider')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_78361, str_78372)

# Assigning a type to the variable '__all__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__all__', list_78361)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'import warnings' statement (line 29)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import numpy' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_78373 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy')

if (type(import_78373) is not StypyTypeError):

    if (import_78373 != 'pyd_module'):
        __import__(import_78373)
        sys_modules_78374 = sys.modules[import_78373]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'np', sys_modules_78374.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy', import_78373)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from scipy.interpolate import _fitpack' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_78375 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy.interpolate')

if (type(import_78375) is not StypyTypeError):

    if (import_78375 != 'pyd_module'):
        __import__(import_78375)
        sys_modules_78376 = sys.modules[import_78375]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy.interpolate', sys_modules_78376.module_type_store, module_type_store, ['_fitpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_78376, sys_modules_78376.module_type_store, module_type_store)
    else:
        from scipy.interpolate import _fitpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy.interpolate', None, module_type_store, ['_fitpack'], [_fitpack])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy.interpolate', import_78375)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from numpy import atleast_1d, array, ones, zeros, sqrt, ravel, transpose, empty, iinfo, intc, asarray' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_78377 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy')

if (type(import_78377) is not StypyTypeError):

    if (import_78377 != 'pyd_module'):
        __import__(import_78377)
        sys_modules_78378 = sys.modules[import_78377]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy', sys_modules_78378.module_type_store, module_type_store, ['atleast_1d', 'array', 'ones', 'zeros', 'sqrt', 'ravel', 'transpose', 'empty', 'iinfo', 'intc', 'asarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_78378, sys_modules_78378.module_type_store, module_type_store)
    else:
        from numpy import atleast_1d, array, ones, zeros, sqrt, ravel, transpose, empty, iinfo, intc, asarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy', None, module_type_store, ['atleast_1d', 'array', 'ones', 'zeros', 'sqrt', 'ravel', 'transpose', 'empty', 'iinfo', 'intc', 'asarray'], [atleast_1d, array, ones, zeros, sqrt, ravel, transpose, empty, iinfo, intc, asarray])

else:
    # Assigning a type to the variable 'numpy' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy', import_78377)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from scipy.interpolate import dfitpack' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_78379 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.interpolate')

if (type(import_78379) is not StypyTypeError):

    if (import_78379 != 'pyd_module'):
        __import__(import_78379)
        sys_modules_78380 = sys.modules[import_78379]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.interpolate', sys_modules_78380.module_type_store, module_type_store, ['dfitpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_78380, sys_modules_78380.module_type_store, module_type_store)
    else:
        from scipy.interpolate import dfitpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.interpolate', None, module_type_store, ['dfitpack'], [dfitpack])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.interpolate', import_78379)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


@norecursion
def _intc_overflow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 40)
    None_78381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'None')
    defaults = [None_78381]
    # Create a new context for function '_intc_overflow'
    module_type_store = module_type_store.open_function_context('_intc_overflow', 40, 0, False)
    
    # Passed parameters checking function
    _intc_overflow.stypy_localization = localization
    _intc_overflow.stypy_type_of_self = None
    _intc_overflow.stypy_type_store = module_type_store
    _intc_overflow.stypy_function_name = '_intc_overflow'
    _intc_overflow.stypy_param_names_list = ['x', 'msg']
    _intc_overflow.stypy_varargs_param_name = None
    _intc_overflow.stypy_kwargs_param_name = None
    _intc_overflow.stypy_call_defaults = defaults
    _intc_overflow.stypy_call_varargs = varargs
    _intc_overflow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_intc_overflow', ['x', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_intc_overflow', localization, ['x', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_intc_overflow(...)' code ##################

    str_78382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', 'Cast the value to an intc and raise an OverflowError if the value\n    cannot fit.\n    ')
    
    
    # Getting the type of 'x' (line 44)
    x_78383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'x')
    
    # Call to iinfo(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'intc' (line 44)
    intc_78385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'intc', False)
    # Processing the call keyword arguments (line 44)
    kwargs_78386 = {}
    # Getting the type of 'iinfo' (line 44)
    iinfo_78384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'iinfo', False)
    # Calling iinfo(args, kwargs) (line 44)
    iinfo_call_result_78387 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), iinfo_78384, *[intc_78385], **kwargs_78386)
    
    # Obtaining the member 'max' of a type (line 44)
    max_78388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), iinfo_call_result_78387, 'max')
    # Applying the binary operator '>' (line 44)
    result_gt_78389 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 7), '>', x_78383, max_78388)
    
    # Testing the type of an if condition (line 44)
    if_condition_78390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), result_gt_78389)
    # Assigning a type to the variable 'if_condition_78390' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_78390', if_condition_78390)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 45)
    # Getting the type of 'msg' (line 45)
    msg_78391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'msg')
    # Getting the type of 'None' (line 45)
    None_78392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'None')
    
    (may_be_78393, more_types_in_union_78394) = may_be_none(msg_78391, None_78392)

    if may_be_78393:

        if more_types_in_union_78394:
            # Runtime conditional SSA (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 46):
        
        # Assigning a BinOp to a Name (line 46):
        str_78395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'str', '%r cannot fit into an intc')
        # Getting the type of 'x' (line 46)
        x_78396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 49), 'x')
        # Applying the binary operator '%' (line 46)
        result_mod_78397 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 18), '%', str_78395, x_78396)
        
        # Assigning a type to the variable 'msg' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'msg', result_mod_78397)

        if more_types_in_union_78394:
            # SSA join for if statement (line 45)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to OverflowError(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'msg' (line 47)
    msg_78399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 28), 'msg', False)
    # Processing the call keyword arguments (line 47)
    kwargs_78400 = {}
    # Getting the type of 'OverflowError' (line 47)
    OverflowError_78398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'OverflowError', False)
    # Calling OverflowError(args, kwargs) (line 47)
    OverflowError_call_result_78401 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), OverflowError_78398, *[msg_78399], **kwargs_78400)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 47, 8), OverflowError_call_result_78401, 'raise parameter', BaseException)
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to intc(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'x' (line 48)
    x_78403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'x', False)
    # Processing the call keyword arguments (line 48)
    kwargs_78404 = {}
    # Getting the type of 'intc' (line 48)
    intc_78402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'intc', False)
    # Calling intc(args, kwargs) (line 48)
    intc_call_result_78405 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), intc_78402, *[x_78403], **kwargs_78404)
    
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', intc_call_result_78405)
    
    # ################# End of '_intc_overflow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_intc_overflow' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_78406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_78406)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_intc_overflow'
    return stypy_return_type_78406

# Assigning a type to the variable '_intc_overflow' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '_intc_overflow', _intc_overflow)

# Assigning a Dict to a Name (line 51):

# Assigning a Dict to a Name (line 51):

# Obtaining an instance of the builtin type 'dict' (line 51)
dict_78407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 51)
# Adding element type (key, value) (line 51)
int_78408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 52)
list_78409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 52)
# Adding element type (line 52)
str_78410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'str', 'The spline has a residual sum of squares fp such that abs(fp-s)/s<=0.001')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 7), list_78409, str_78410)
# Adding element type (line 52)
# Getting the type of 'None' (line 53)
None_78411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 7), list_78409, None_78411)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), dict_78407, (int_78408, list_78409))
# Adding element type (key, value) (line 51)
int_78412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 54)
list_78413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 54)
# Adding element type (line 54)
str_78414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'str', 'The spline is an interpolating spline (fp=0)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_78413, str_78414)
# Adding element type (line 54)
# Getting the type of 'None' (line 54)
None_78415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 57), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_78413, None_78415)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), dict_78407, (int_78412, list_78413))
# Adding element type (key, value) (line 51)
int_78416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 55)
list_78417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 55)
# Adding element type (line 55)
str_78418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'str', 'The spline is weighted least-squares polynomial of degree k.\nfp gives the upper bound fp0 for the smoothing factor s')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_78417, str_78418)
# Adding element type (line 55)
# Getting the type of 'None' (line 56)
None_78419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 68), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_78417, None_78419)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), dict_78407, (int_78416, list_78417))
# Adding element type (key, value) (line 51)
int_78420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 57)
list_78421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
str_78422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'str', 'The required storage space exceeds the available storage space.\nProbable causes: data (x,y) size is too small or smoothing parameter\ns is too small (fp>s).')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 7), list_78421, str_78422)
# Adding element type (line 57)
# Getting the type of 'ValueError' (line 59)
ValueError_78423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 7), list_78421, ValueError_78423)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), dict_78407, (int_78420, list_78421))
# Adding element type (key, value) (line 51)
int_78424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 60)
list_78425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)
str_78426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'str', 'A theoretically impossible result when finding a smoothing spline\nwith fp = s. Probable cause: s too small. (abs(fp-s)/s>0.001)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 7), list_78425, str_78426)
# Adding element type (line 60)
# Getting the type of 'ValueError' (line 62)
ValueError_78427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 7), list_78425, ValueError_78427)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), dict_78407, (int_78424, list_78425))
# Adding element type (key, value) (line 51)
int_78428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 63)
list_78429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 63)
# Adding element type (line 63)
str_78430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'str', 'The maximal number of iterations (20) allowed for finding smoothing\nspline with fp=s has been reached. Probable cause: s too small.\n(abs(fp-s)/s>0.001)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 7), list_78429, str_78430)
# Adding element type (line 63)
# Getting the type of 'ValueError' (line 65)
ValueError_78431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 7), list_78429, ValueError_78431)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), dict_78407, (int_78428, list_78429))
# Adding element type (key, value) (line 51)
int_78432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 66)
list_78433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 66)
# Adding element type (line 66)
str_78434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'str', 'Error on input data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 8), list_78433, str_78434)
# Adding element type (line 66)
# Getting the type of 'ValueError' (line 66)
ValueError_78435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 8), list_78433, ValueError_78435)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), dict_78407, (int_78432, list_78433))
# Adding element type (key, value) (line 51)
str_78436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', 'unknown')

# Obtaining an instance of the builtin type 'list' (line 67)
list_78437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 67)
# Adding element type (line 67)
str_78438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'str', 'An error occurred')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 15), list_78437, str_78438)
# Adding element type (line 67)
# Getting the type of 'TypeError' (line 67)
TypeError_78439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'TypeError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 15), list_78437, TypeError_78439)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), dict_78407, (str_78436, list_78437))

# Assigning a type to the variable '_iermess' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '_iermess', dict_78407)

# Assigning a Dict to a Name (line 70):

# Assigning a Dict to a Name (line 70):

# Obtaining an instance of the builtin type 'dict' (line 70)
dict_78440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 70)
# Adding element type (key, value) (line 70)
int_78441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 71)
list_78442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 71)
# Adding element type (line 71)
str_78443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'str', 'The spline has a residual sum of squares fp such that abs(fp-s)/s<=0.001')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 7), list_78442, str_78443)
# Adding element type (line 71)
# Getting the type of 'None' (line 72)
None_78444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 7), list_78442, None_78444)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78441, list_78442))
# Adding element type (key, value) (line 70)
int_78445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 73)
list_78446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 73)
# Adding element type (line 73)
str_78447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 9), 'str', 'The spline is an interpolating spline (fp=0)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 8), list_78446, str_78447)
# Adding element type (line 73)
# Getting the type of 'None' (line 73)
None_78448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 57), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 8), list_78446, None_78448)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78445, list_78446))
# Adding element type (key, value) (line 70)
int_78449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 74)
list_78450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 74)
# Adding element type (line 74)
str_78451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 9), 'str', 'The spline is weighted least-squares polynomial of degree kx and ky.\nfp gives the upper bound fp0 for the smoothing factor s')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 8), list_78450, str_78451)
# Adding element type (line 74)
# Getting the type of 'None' (line 75)
None_78452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 70), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 8), list_78450, None_78452)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78449, list_78450))
# Adding element type (key, value) (line 70)
int_78453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 76)
list_78454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 76)
# Adding element type (line 76)
str_78455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'str', 'Warning. The coefficients of the spline have been computed as the\nminimal norm least-squares solution of a rank deficient system.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 8), list_78454, str_78455)
# Adding element type (line 76)
# Getting the type of 'None' (line 78)
None_78456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 9), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 8), list_78454, None_78456)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78453, list_78454))
# Adding element type (key, value) (line 70)
int_78457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 79)
list_78458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 79)
# Adding element type (line 79)
str_78459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'str', 'The required storage space exceeds the available storage space.\nProbable causes: nxest or nyest too small or s is too small. (fp>s)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 7), list_78458, str_78459)
# Adding element type (line 79)
# Getting the type of 'ValueError' (line 81)
ValueError_78460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 7), list_78458, ValueError_78460)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78457, list_78458))
# Adding element type (key, value) (line 70)
int_78461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 82)
list_78462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 82)
# Adding element type (line 82)
str_78463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'str', 'A theoretically impossible result when finding a smoothing spline\nwith fp = s. Probable causes: s too small or badly chosen eps.\n(abs(fp-s)/s>0.001)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 7), list_78462, str_78463)
# Adding element type (line 82)
# Getting the type of 'ValueError' (line 84)
ValueError_78464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 7), list_78462, ValueError_78464)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78461, list_78462))
# Adding element type (key, value) (line 70)
int_78465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 85)
list_78466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 85)
# Adding element type (line 85)
str_78467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'str', 'The maximal number of iterations (20) allowed for finding smoothing\nspline with fp=s has been reached. Probable cause: s too small.\n(abs(fp-s)/s>0.001)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 7), list_78466, str_78467)
# Adding element type (line 85)
# Getting the type of 'ValueError' (line 87)
ValueError_78468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 31), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 7), list_78466, ValueError_78468)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78465, list_78466))
# Adding element type (key, value) (line 70)
int_78469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 88)
list_78470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 88)
# Adding element type (line 88)
str_78471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'str', 'No more knots can be added because the number of B-spline\ncoefficients already exceeds the number of data points m.\nProbable causes: either s or m too small. (fp>s)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 7), list_78470, str_78471)
# Adding element type (line 88)
# Getting the type of 'ValueError' (line 90)
ValueError_78472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 60), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 7), list_78470, ValueError_78472)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78469, list_78470))
# Adding element type (key, value) (line 70)
int_78473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 91)
list_78474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 91)
# Adding element type (line 91)
str_78475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'str', 'No more knots can be added because the additional knot would\ncoincide with an old one. Probable cause: s too small or too large\na weight to an inaccurate data point. (fp>s)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 7), list_78474, str_78475)
# Adding element type (line 91)
# Getting the type of 'ValueError' (line 93)
ValueError_78476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 56), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 7), list_78474, ValueError_78476)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78473, list_78474))
# Adding element type (key, value) (line 70)
int_78477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 94)
list_78478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 94)
# Adding element type (line 94)
str_78479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 9), 'str', 'Error on input data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 8), list_78478, str_78479)
# Adding element type (line 94)
# Getting the type of 'ValueError' (line 94)
ValueError_78480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 8), list_78478, ValueError_78480)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78477, list_78478))
# Adding element type (key, value) (line 70)
int_78481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 4), 'int')

# Obtaining an instance of the builtin type 'list' (line 95)
list_78482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 95)
# Adding element type (line 95)
str_78483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 9), 'str', 'rwrk2 too small, i.e. there is not enough workspace for computing\nthe minimal least-squares solution of a rank deficient system of\nlinear equations.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), list_78482, str_78483)
# Adding element type (line 95)
# Getting the type of 'ValueError' (line 97)
ValueError_78484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 'ValueError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), list_78482, ValueError_78484)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (int_78481, list_78482))
# Adding element type (key, value) (line 70)
str_78485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'str', 'unknown')

# Obtaining an instance of the builtin type 'list' (line 98)
list_78486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 98)
# Adding element type (line 98)
str_78487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'str', 'An error occurred')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 15), list_78486, str_78487)
# Adding element type (line 98)
# Getting the type of 'TypeError' (line 98)
TypeError_78488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'TypeError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 15), list_78486, TypeError_78488)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_78440, (str_78485, list_78486))

# Assigning a type to the variable '_iermess2' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), '_iermess2', dict_78440)

# Assigning a Dict to a Name (line 101):

# Assigning a Dict to a Name (line 101):

# Obtaining an instance of the builtin type 'dict' (line 101)
dict_78489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 101)
# Adding element type (key, value) (line 101)
str_78490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'str', 't')

# Call to array(...): (line 101)
# Processing the call arguments (line 101)

# Obtaining an instance of the builtin type 'list' (line 101)
list_78492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 101)

# Getting the type of 'float' (line 101)
float_78493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'float', False)
# Processing the call keyword arguments (line 101)
kwargs_78494 = {}
# Getting the type of 'array' (line 101)
array_78491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'array', False)
# Calling array(args, kwargs) (line 101)
array_call_result_78495 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), array_78491, *[list_78492, float_78493], **kwargs_78494)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 16), dict_78489, (str_78490, array_call_result_78495))
# Adding element type (key, value) (line 101)
str_78496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 40), 'str', 'wrk')

# Call to array(...): (line 101)
# Processing the call arguments (line 101)

# Obtaining an instance of the builtin type 'list' (line 101)
list_78498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 53), 'list')
# Adding type elements to the builtin type 'list' instance (line 101)

# Getting the type of 'float' (line 101)
float_78499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 57), 'float', False)
# Processing the call keyword arguments (line 101)
kwargs_78500 = {}
# Getting the type of 'array' (line 101)
array_78497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 47), 'array', False)
# Calling array(args, kwargs) (line 101)
array_call_result_78501 = invoke(stypy.reporting.localization.Localization(__file__, 101, 47), array_78497, *[list_78498, float_78499], **kwargs_78500)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 16), dict_78489, (str_78496, array_call_result_78501))
# Adding element type (key, value) (line 101)
str_78502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 17), 'str', 'iwrk')

# Call to array(...): (line 102)
# Processing the call arguments (line 102)

# Obtaining an instance of the builtin type 'list' (line 102)
list_78504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 102)

# Getting the type of 'intc' (line 102)
intc_78505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'intc', False)
# Processing the call keyword arguments (line 102)
kwargs_78506 = {}
# Getting the type of 'array' (line 102)
array_78503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'array', False)
# Calling array(args, kwargs) (line 102)
array_call_result_78507 = invoke(stypy.reporting.localization.Localization(__file__, 102, 25), array_78503, *[list_78504, intc_78505], **kwargs_78506)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 16), dict_78489, (str_78502, array_call_result_78507))
# Adding element type (key, value) (line 101)
str_78508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 42), 'str', 'u')

# Call to array(...): (line 102)
# Processing the call arguments (line 102)

# Obtaining an instance of the builtin type 'list' (line 102)
list_78510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 53), 'list')
# Adding type elements to the builtin type 'list' instance (line 102)

# Getting the type of 'float' (line 102)
float_78511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 57), 'float', False)
# Processing the call keyword arguments (line 102)
kwargs_78512 = {}
# Getting the type of 'array' (line 102)
array_78509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 47), 'array', False)
# Calling array(args, kwargs) (line 102)
array_call_result_78513 = invoke(stypy.reporting.localization.Localization(__file__, 102, 47), array_78509, *[list_78510, float_78511], **kwargs_78512)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 16), dict_78489, (str_78508, array_call_result_78513))
# Adding element type (key, value) (line 101)
str_78514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 17), 'str', 'ub')
int_78515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 16), dict_78489, (str_78514, int_78515))
# Adding element type (key, value) (line 101)
str_78516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'str', 'ue')
int_78517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 32), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 16), dict_78489, (str_78516, int_78517))

# Assigning a type to the variable '_parcur_cache' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), '_parcur_cache', dict_78489)

@norecursion
def splprep(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 106)
    None_78518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'None')
    # Getting the type of 'None' (line 106)
    None_78519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'None')
    # Getting the type of 'None' (line 106)
    None_78520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'None')
    # Getting the type of 'None' (line 106)
    None_78521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 43), 'None')
    int_78522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 51), 'int')
    int_78523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 59), 'int')
    # Getting the type of 'None' (line 106)
    None_78524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 64), 'None')
    # Getting the type of 'None' (line 106)
    None_78525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 72), 'None')
    int_78526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'int')
    # Getting the type of 'None' (line 107)
    None_78527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'None')
    int_78528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 42), 'int')
    int_78529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 51), 'int')
    defaults = [None_78518, None_78519, None_78520, None_78521, int_78522, int_78523, None_78524, None_78525, int_78526, None_78527, int_78528, int_78529]
    # Create a new context for function 'splprep'
    module_type_store = module_type_store.open_function_context('splprep', 106, 0, False)
    
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

    str_78530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '\n    Find the B-spline representation of an N-dimensional curve.\n\n    Given a list of N rank-1 arrays, `x`, which represent a curve in\n    N-dimensional space parametrized by `u`, find a smooth approximating\n    spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.\n\n    Parameters\n    ----------\n    x : array_like\n        A list of sample vector arrays representing the curve.\n    w : array_like, optional\n        Strictly positive rank-1 array of weights the same length as `x[0]`.\n        The weights are used in computing the weighted least-squares spline\n        fit. If the errors in the `x` values have standard-deviation given by\n        the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.\n    u : array_like, optional\n        An array of parameter values. If not given, these values are\n        calculated automatically as ``M = len(x[0])``, where\n\n            v[0] = 0\n\n            v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)\n\n            u[i] = v[i] / v[M-1]\n\n    ub, ue : int, optional\n        The end-points of the parameters interval.  Defaults to\n        u[0] and u[-1].\n    k : int, optional\n        Degree of the spline. Cubic splines are recommended.\n        Even values of `k` should be avoided especially with a small s-value.\n        ``1 <= k <= 5``, default is 3.\n    task : int, optional\n        If task==0 (default), find t and c for a given smoothing factor, s.\n        If task==1, find t and c for another value of the smoothing factor, s.\n        There must have been a previous call with task=0 or task=1\n        for the same set of data.\n        If task=-1 find the weighted least square spline for a given set of\n        knots, t.\n    s : float, optional\n        A smoothing condition.  The amount of smoothness is determined by\n        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,\n        where g(x) is the smoothed interpolation of (x,y).  The user can\n        use `s` to control the trade-off between closeness and smoothness\n        of fit.  Larger `s` means more smoothing while smaller values of `s`\n        indicate less smoothing. Recommended values of `s` depend on the\n        weights, w.  If the weights represent the inverse of the\n        standard-deviation of y, then a good `s` value should be found in\n        the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of\n        data points in x, y, and w.\n    t : int, optional\n        The knots needed for task=-1.\n    full_output : int, optional\n        If non-zero, then return optional outputs.\n    nest : int, optional\n        An over-estimate of the total number of knots of the spline to\n        help in determining the storage space.  By default nest=m/2.\n        Always large enough is nest=m+k+1.\n    per : int, optional\n       If non-zero, data points are considered periodic with period\n       ``x[m-1] - x[0]`` and a smooth periodic spline approximation is\n       returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.\n    quiet : int, optional\n         Non-zero to suppress messages.\n         This parameter is deprecated; use standard Python warning filters\n         instead.\n\n    Returns\n    -------\n    tck : tuple\n        A tuple (t,c,k) containing the vector of knots, the B-spline\n        coefficients, and the degree of the spline.\n    u : array\n        An array of the values of the parameter.\n    fp : float\n        The weighted sum of squared residuals of the spline approximation.\n    ier : int\n        An integer flag about splrep success.  Success is indicated\n        if ier<=0. If ier in [1,2,3] an error occurred but was not raised.\n        Otherwise an error is raised.\n    msg : str\n        A message corresponding to the integer flag, ier.\n\n    See Also\n    --------\n    splrep, splev, sproot, spalde, splint,\n    bisplrep, bisplev\n    UnivariateSpline, BivariateSpline\n\n    Notes\n    -----\n    See `splev` for evaluation of the spline and its derivatives.\n    The number of dimensions N must be smaller than 11.\n\n    References\n    ----------\n    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and\n        parametric splines, Computer Graphics and Image Processing",\n        20 (1982) 171-184.\n    .. [2] P. Dierckx, "Algorithms for smoothing data with periodic and\n        parametric splines", report tw55, Dept. Computer Science,\n        K.U.Leuven, 1981.\n    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs on\n        Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    
    # Getting the type of 'task' (line 215)
    task_78531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 7), 'task')
    int_78532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 15), 'int')
    # Applying the binary operator '<=' (line 215)
    result_le_78533 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 7), '<=', task_78531, int_78532)
    
    # Testing the type of an if condition (line 215)
    if_condition_78534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), result_le_78533)
    # Assigning a type to the variable 'if_condition_78534' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_78534', if_condition_78534)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Dict to a Name (line 216):
    
    # Assigning a Dict to a Name (line 216):
    
    # Obtaining an instance of the builtin type 'dict' (line 216)
    dict_78535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 24), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 216)
    # Adding element type (key, value) (line 216)
    str_78536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 25), 'str', 't')
    
    # Call to array(...): (line 216)
    # Processing the call arguments (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_78538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    
    # Getting the type of 'float' (line 216)
    float_78539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 40), 'float', False)
    # Processing the call keyword arguments (line 216)
    kwargs_78540 = {}
    # Getting the type of 'array' (line 216)
    array_78537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 30), 'array', False)
    # Calling array(args, kwargs) (line 216)
    array_call_result_78541 = invoke(stypy.reporting.localization.Localization(__file__, 216, 30), array_78537, *[list_78538, float_78539], **kwargs_78540)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 24), dict_78535, (str_78536, array_call_result_78541))
    # Adding element type (key, value) (line 216)
    str_78542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 48), 'str', 'wrk')
    
    # Call to array(...): (line 216)
    # Processing the call arguments (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_78544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    
    # Getting the type of 'float' (line 216)
    float_78545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 65), 'float', False)
    # Processing the call keyword arguments (line 216)
    kwargs_78546 = {}
    # Getting the type of 'array' (line 216)
    array_78543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 55), 'array', False)
    # Calling array(args, kwargs) (line 216)
    array_call_result_78547 = invoke(stypy.reporting.localization.Localization(__file__, 216, 55), array_78543, *[list_78544, float_78545], **kwargs_78546)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 24), dict_78535, (str_78542, array_call_result_78547))
    # Adding element type (key, value) (line 216)
    str_78548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 25), 'str', 'iwrk')
    
    # Call to array(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Obtaining an instance of the builtin type 'list' (line 217)
    list_78550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 217)
    
    # Getting the type of 'intc' (line 217)
    intc_78551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 43), 'intc', False)
    # Processing the call keyword arguments (line 217)
    kwargs_78552 = {}
    # Getting the type of 'array' (line 217)
    array_78549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 33), 'array', False)
    # Calling array(args, kwargs) (line 217)
    array_call_result_78553 = invoke(stypy.reporting.localization.Localization(__file__, 217, 33), array_78549, *[list_78550, intc_78551], **kwargs_78552)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 24), dict_78535, (str_78548, array_call_result_78553))
    # Adding element type (key, value) (line 216)
    str_78554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 50), 'str', 'u')
    
    # Call to array(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Obtaining an instance of the builtin type 'list' (line 217)
    list_78556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 217)
    
    # Getting the type of 'float' (line 217)
    float_78557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 65), 'float', False)
    # Processing the call keyword arguments (line 217)
    kwargs_78558 = {}
    # Getting the type of 'array' (line 217)
    array_78555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 55), 'array', False)
    # Calling array(args, kwargs) (line 217)
    array_call_result_78559 = invoke(stypy.reporting.localization.Localization(__file__, 217, 55), array_78555, *[list_78556, float_78557], **kwargs_78558)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 24), dict_78535, (str_78554, array_call_result_78559))
    # Adding element type (key, value) (line 216)
    str_78560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'str', 'ub')
    int_78561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 31), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 24), dict_78535, (str_78560, int_78561))
    # Adding element type (key, value) (line 216)
    str_78562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 34), 'str', 'ue')
    int_78563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 40), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 24), dict_78535, (str_78562, int_78563))
    
    # Assigning a type to the variable '_parcur_cache' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), '_parcur_cache', dict_78535)
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 219):
    
    # Assigning a Call to a Name (line 219):
    
    # Call to atleast_1d(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'x' (line 219)
    x_78565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), 'x', False)
    # Processing the call keyword arguments (line 219)
    kwargs_78566 = {}
    # Getting the type of 'atleast_1d' (line 219)
    atleast_1d_78564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 219)
    atleast_1d_call_result_78567 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), atleast_1d_78564, *[x_78565], **kwargs_78566)
    
    # Assigning a type to the variable 'x' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'x', atleast_1d_call_result_78567)
    
    # Assigning a Attribute to a Tuple (line 220):
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    int_78568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 4), 'int')
    # Getting the type of 'x' (line 220)
    x_78569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 14), 'x')
    # Obtaining the member 'shape' of a type (line 220)
    shape_78570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 14), x_78569, 'shape')
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___78571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 4), shape_78570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_78572 = invoke(stypy.reporting.localization.Localization(__file__, 220, 4), getitem___78571, int_78568)
    
    # Assigning a type to the variable 'tuple_var_assignment_78277' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'tuple_var_assignment_78277', subscript_call_result_78572)
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    int_78573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 4), 'int')
    # Getting the type of 'x' (line 220)
    x_78574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 14), 'x')
    # Obtaining the member 'shape' of a type (line 220)
    shape_78575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 14), x_78574, 'shape')
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___78576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 4), shape_78575, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_78577 = invoke(stypy.reporting.localization.Localization(__file__, 220, 4), getitem___78576, int_78573)
    
    # Assigning a type to the variable 'tuple_var_assignment_78278' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'tuple_var_assignment_78278', subscript_call_result_78577)
    
    # Assigning a Name to a Name (line 220):
    # Getting the type of 'tuple_var_assignment_78277' (line 220)
    tuple_var_assignment_78277_78578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'tuple_var_assignment_78277')
    # Assigning a type to the variable 'idim' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'idim', tuple_var_assignment_78277_78578)
    
    # Assigning a Name to a Name (line 220):
    # Getting the type of 'tuple_var_assignment_78278' (line 220)
    tuple_var_assignment_78278_78579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'tuple_var_assignment_78278')
    # Assigning a type to the variable 'm' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 10), 'm', tuple_var_assignment_78278_78579)
    
    # Getting the type of 'per' (line 221)
    per_78580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 7), 'per')
    # Testing the type of an if condition (line 221)
    if_condition_78581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 4), per_78580)
    # Assigning a type to the variable 'if_condition_78581' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'if_condition_78581', if_condition_78581)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'idim' (line 222)
    idim_78583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'idim', False)
    # Processing the call keyword arguments (line 222)
    kwargs_78584 = {}
    # Getting the type of 'range' (line 222)
    range_78582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 17), 'range', False)
    # Calling range(args, kwargs) (line 222)
    range_call_result_78585 = invoke(stypy.reporting.localization.Localization(__file__, 222, 17), range_78582, *[idim_78583], **kwargs_78584)
    
    # Testing the type of a for loop iterable (line 222)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 8), range_call_result_78585)
    # Getting the type of the for loop variable (line 222)
    for_loop_var_78586 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 8), range_call_result_78585)
    # Assigning a type to the variable 'i' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'i', for_loop_var_78586)
    # SSA begins for a for statement (line 222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    int_78587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 20), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 223)
    i_78588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'i')
    # Getting the type of 'x' (line 223)
    x_78589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'x')
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___78590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), x_78589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_78591 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), getitem___78590, i_78588)
    
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___78592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), subscript_call_result_78591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_78593 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), getitem___78592, int_78587)
    
    
    # Obtaining the type of the subscript
    int_78594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 31), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 223)
    i_78595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 28), 'i')
    # Getting the type of 'x' (line 223)
    x_78596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'x')
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___78597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 26), x_78596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_78598 = invoke(stypy.reporting.localization.Localization(__file__, 223, 26), getitem___78597, i_78595)
    
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___78599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 26), subscript_call_result_78598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_78600 = invoke(stypy.reporting.localization.Localization(__file__, 223, 26), getitem___78599, int_78594)
    
    # Applying the binary operator '!=' (line 223)
    result_ne_78601 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 15), '!=', subscript_call_result_78593, subscript_call_result_78600)
    
    # Testing the type of an if condition (line 223)
    if_condition_78602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 12), result_ne_78601)
    # Assigning a type to the variable 'if_condition_78602' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'if_condition_78602', if_condition_78602)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'quiet' (line 224)
    quiet_78603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'quiet')
    int_78604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 27), 'int')
    # Applying the binary operator '<' (line 224)
    result_lt_78605 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 19), '<', quiet_78603, int_78604)
    
    # Testing the type of an if condition (line 224)
    if_condition_78606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 16), result_lt_78605)
    # Assigning a type to the variable 'if_condition_78606' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'if_condition_78606', if_condition_78606)
    # SSA begins for if statement (line 224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 225)
    # Processing the call arguments (line 225)
    
    # Call to RuntimeWarning(...): (line 225)
    # Processing the call arguments (line 225)
    str_78610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 49), 'str', 'Setting x[%d][%d]=x[%d][0]')
    
    # Obtaining an instance of the builtin type 'tuple' (line 226)
    tuple_78611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 226)
    # Adding element type (line 226)
    # Getting the type of 'i' (line 226)
    i_78612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 50), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 50), tuple_78611, i_78612)
    # Adding element type (line 226)
    # Getting the type of 'm' (line 226)
    m_78613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 53), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 50), tuple_78611, m_78613)
    # Adding element type (line 226)
    # Getting the type of 'i' (line 226)
    i_78614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 56), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 50), tuple_78611, i_78614)
    
    # Applying the binary operator '%' (line 225)
    result_mod_78615 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 49), '%', str_78610, tuple_78611)
    
    # Processing the call keyword arguments (line 225)
    kwargs_78616 = {}
    # Getting the type of 'RuntimeWarning' (line 225)
    RuntimeWarning_78609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 34), 'RuntimeWarning', False)
    # Calling RuntimeWarning(args, kwargs) (line 225)
    RuntimeWarning_call_result_78617 = invoke(stypy.reporting.localization.Localization(__file__, 225, 34), RuntimeWarning_78609, *[result_mod_78615], **kwargs_78616)
    
    # Processing the call keyword arguments (line 225)
    kwargs_78618 = {}
    # Getting the type of 'warnings' (line 225)
    warnings_78607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 225)
    warn_78608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 20), warnings_78607, 'warn')
    # Calling warn(args, kwargs) (line 225)
    warn_call_result_78619 = invoke(stypy.reporting.localization.Localization(__file__, 225, 20), warn_78608, *[RuntimeWarning_call_result_78617], **kwargs_78618)
    
    # SSA join for if statement (line 224)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 227):
    
    # Assigning a Subscript to a Subscript (line 227):
    
    # Obtaining the type of the subscript
    int_78620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 32), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 227)
    i_78621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'i')
    # Getting the type of 'x' (line 227)
    x_78622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'x')
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___78623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 27), x_78622, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_78624 = invoke(stypy.reporting.localization.Localization(__file__, 227, 27), getitem___78623, i_78621)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___78625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 27), subscript_call_result_78624, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_78626 = invoke(stypy.reporting.localization.Localization(__file__, 227, 27), getitem___78625, int_78620)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 227)
    i_78627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 18), 'i')
    # Getting the type of 'x' (line 227)
    x_78628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'x')
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___78629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), x_78628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_78630 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), getitem___78629, i_78627)
    
    int_78631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 21), 'int')
    # Storing an element on a container (line 227)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 16), subscript_call_result_78630, (int_78631, subscript_call_result_78626))
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    int_78632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 11), 'int')
    # Getting the type of 'idim' (line 228)
    idim_78633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'idim')
    # Applying the binary operator '<' (line 228)
    result_lt_78634 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 11), '<', int_78632, idim_78633)
    int_78635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'int')
    # Applying the binary operator '<' (line 228)
    result_lt_78636 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 11), '<', idim_78633, int_78635)
    # Applying the binary operator '&' (line 228)
    result_and__78637 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 11), '&', result_lt_78634, result_lt_78636)
    
    # Applying the 'not' unary operator (line 228)
    result_not__78638 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 7), 'not', result_and__78637)
    
    # Testing the type of an if condition (line 228)
    if_condition_78639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 4), result_not__78638)
    # Assigning a type to the variable 'if_condition_78639' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'if_condition_78639', if_condition_78639)
    # SSA begins for if statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 229)
    # Processing the call arguments (line 229)
    str_78641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'str', '0 < idim < 11 must hold')
    # Processing the call keyword arguments (line 229)
    kwargs_78642 = {}
    # Getting the type of 'TypeError' (line 229)
    TypeError_78640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 229)
    TypeError_call_result_78643 = invoke(stypy.reporting.localization.Localization(__file__, 229, 14), TypeError_78640, *[str_78641], **kwargs_78642)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 229, 8), TypeError_call_result_78643, 'raise parameter', BaseException)
    # SSA join for if statement (line 228)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 230)
    # Getting the type of 'w' (line 230)
    w_78644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 7), 'w')
    # Getting the type of 'None' (line 230)
    None_78645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'None')
    
    (may_be_78646, more_types_in_union_78647) = may_be_none(w_78644, None_78645)

    if may_be_78646:

        if more_types_in_union_78647:
            # Runtime conditional SSA (line 230)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to ones(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'm' (line 231)
        m_78649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'm', False)
        # Getting the type of 'float' (line 231)
        float_78650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'float', False)
        # Processing the call keyword arguments (line 231)
        kwargs_78651 = {}
        # Getting the type of 'ones' (line 231)
        ones_78648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 231)
        ones_call_result_78652 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), ones_78648, *[m_78649, float_78650], **kwargs_78651)
        
        # Assigning a type to the variable 'w' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'w', ones_call_result_78652)

        if more_types_in_union_78647:
            # Runtime conditional SSA for else branch (line 230)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_78646) or more_types_in_union_78647):
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to atleast_1d(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'w' (line 233)
        w_78654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 23), 'w', False)
        # Processing the call keyword arguments (line 233)
        kwargs_78655 = {}
        # Getting the type of 'atleast_1d' (line 233)
        atleast_1d_78653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 233)
        atleast_1d_call_result_78656 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), atleast_1d_78653, *[w_78654], **kwargs_78655)
        
        # Assigning a type to the variable 'w' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'w', atleast_1d_call_result_78656)

        if (may_be_78646 and more_types_in_union_78647):
            # SSA join for if statement (line 230)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Compare to a Name (line 234):
    
    # Assigning a Compare to a Name (line 234):
    
    # Getting the type of 'u' (line 234)
    u_78657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'u')
    # Getting the type of 'None' (line 234)
    None_78658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'None')
    # Applying the binary operator 'isnot' (line 234)
    result_is_not_78659 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 12), 'isnot', u_78657, None_78658)
    
    # Assigning a type to the variable 'ipar' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'ipar', result_is_not_78659)
    
    # Getting the type of 'ipar' (line 235)
    ipar_78660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'ipar')
    # Testing the type of an if condition (line 235)
    if_condition_78661 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 4), ipar_78660)
    # Assigning a type to the variable 'if_condition_78661' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'if_condition_78661', if_condition_78661)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 236):
    
    # Assigning a Name to a Subscript (line 236):
    # Getting the type of 'u' (line 236)
    u_78662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 29), 'u')
    # Getting the type of '_parcur_cache' (line 236)
    _parcur_cache_78663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), '_parcur_cache')
    str_78664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 22), 'str', 'u')
    # Storing an element on a container (line 236)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 8), _parcur_cache_78663, (str_78664, u_78662))
    
    # Type idiom detected: calculating its left and rigth part (line 237)
    # Getting the type of 'ub' (line 237)
    ub_78665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'ub')
    # Getting the type of 'None' (line 237)
    None_78666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'None')
    
    (may_be_78667, more_types_in_union_78668) = may_be_none(ub_78665, None_78666)

    if may_be_78667:

        if more_types_in_union_78668:
            # Runtime conditional SSA (line 237)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Subscript (line 238):
        
        # Assigning a Subscript to a Subscript (line 238):
        
        # Obtaining the type of the subscript
        int_78669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 36), 'int')
        # Getting the type of 'u' (line 238)
        u_78670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 34), 'u')
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___78671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 34), u_78670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 238)
        subscript_call_result_78672 = invoke(stypy.reporting.localization.Localization(__file__, 238, 34), getitem___78671, int_78669)
        
        # Getting the type of '_parcur_cache' (line 238)
        _parcur_cache_78673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), '_parcur_cache')
        str_78674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 26), 'str', 'ub')
        # Storing an element on a container (line 238)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 12), _parcur_cache_78673, (str_78674, subscript_call_result_78672))

        if more_types_in_union_78668:
            # Runtime conditional SSA for else branch (line 237)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_78667) or more_types_in_union_78668):
        
        # Assigning a Name to a Subscript (line 240):
        
        # Assigning a Name to a Subscript (line 240):
        # Getting the type of 'ub' (line 240)
        ub_78675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 34), 'ub')
        # Getting the type of '_parcur_cache' (line 240)
        _parcur_cache_78676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), '_parcur_cache')
        str_78677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 26), 'str', 'ub')
        # Storing an element on a container (line 240)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 12), _parcur_cache_78676, (str_78677, ub_78675))

        if (may_be_78667 and more_types_in_union_78668):
            # SSA join for if statement (line 237)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 241)
    # Getting the type of 'ue' (line 241)
    ue_78678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'ue')
    # Getting the type of 'None' (line 241)
    None_78679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'None')
    
    (may_be_78680, more_types_in_union_78681) = may_be_none(ue_78678, None_78679)

    if may_be_78680:

        if more_types_in_union_78681:
            # Runtime conditional SSA (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Subscript (line 242):
        
        # Assigning a Subscript to a Subscript (line 242):
        
        # Obtaining the type of the subscript
        int_78682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 36), 'int')
        # Getting the type of 'u' (line 242)
        u_78683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'u')
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___78684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 34), u_78683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_78685 = invoke(stypy.reporting.localization.Localization(__file__, 242, 34), getitem___78684, int_78682)
        
        # Getting the type of '_parcur_cache' (line 242)
        _parcur_cache_78686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), '_parcur_cache')
        str_78687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 26), 'str', 'ue')
        # Storing an element on a container (line 242)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 12), _parcur_cache_78686, (str_78687, subscript_call_result_78685))

        if more_types_in_union_78681:
            # Runtime conditional SSA for else branch (line 241)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_78680) or more_types_in_union_78681):
        
        # Assigning a Name to a Subscript (line 244):
        
        # Assigning a Name to a Subscript (line 244):
        # Getting the type of 'ue' (line 244)
        ue_78688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 34), 'ue')
        # Getting the type of '_parcur_cache' (line 244)
        _parcur_cache_78689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), '_parcur_cache')
        str_78690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 26), 'str', 'ue')
        # Storing an element on a container (line 244)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 12), _parcur_cache_78689, (str_78690, ue_78688))

        if (may_be_78680 and more_types_in_union_78681):
            # SSA join for if statement (line 241)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 235)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Subscript (line 246):
    
    # Assigning a Call to a Subscript (line 246):
    
    # Call to zeros(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'm' (line 246)
    m_78692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 35), 'm', False)
    # Getting the type of 'float' (line 246)
    float_78693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'float', False)
    # Processing the call keyword arguments (line 246)
    kwargs_78694 = {}
    # Getting the type of 'zeros' (line 246)
    zeros_78691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'zeros', False)
    # Calling zeros(args, kwargs) (line 246)
    zeros_call_result_78695 = invoke(stypy.reporting.localization.Localization(__file__, 246, 29), zeros_78691, *[m_78692, float_78693], **kwargs_78694)
    
    # Getting the type of '_parcur_cache' (line 246)
    _parcur_cache_78696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), '_parcur_cache')
    str_78697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 22), 'str', 'u')
    # Storing an element on a container (line 246)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 8), _parcur_cache_78696, (str_78697, zeros_call_result_78695))
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    int_78698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 12), 'int')
    # Getting the type of 'k' (line 247)
    k_78699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'k')
    # Applying the binary operator '<=' (line 247)
    result_le_78700 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 12), '<=', int_78698, k_78699)
    int_78701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 22), 'int')
    # Applying the binary operator '<=' (line 247)
    result_le_78702 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 12), '<=', k_78699, int_78701)
    # Applying the binary operator '&' (line 247)
    result_and__78703 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 12), '&', result_le_78700, result_le_78702)
    
    # Applying the 'not' unary operator (line 247)
    result_not__78704 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 7), 'not', result_and__78703)
    
    # Testing the type of an if condition (line 247)
    if_condition_78705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 4), result_not__78704)
    # Assigning a type to the variable 'if_condition_78705' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'if_condition_78705', if_condition_78705)
    # SSA begins for if statement (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 248)
    # Processing the call arguments (line 248)
    str_78707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 24), 'str', '1 <= k= %d <=5 must hold')
    # Getting the type of 'k' (line 248)
    k_78708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 53), 'k', False)
    # Applying the binary operator '%' (line 248)
    result_mod_78709 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 24), '%', str_78707, k_78708)
    
    # Processing the call keyword arguments (line 248)
    kwargs_78710 = {}
    # Getting the type of 'TypeError' (line 248)
    TypeError_78706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 248)
    TypeError_call_result_78711 = invoke(stypy.reporting.localization.Localization(__file__, 248, 14), TypeError_78706, *[result_mod_78709], **kwargs_78710)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 248, 8), TypeError_call_result_78711, 'raise parameter', BaseException)
    # SSA join for if statement (line 247)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    int_78712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'int')
    # Getting the type of 'task' (line 249)
    task_78713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'task')
    # Applying the binary operator '<=' (line 249)
    result_le_78714 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 12), '<=', int_78712, task_78713)
    int_78715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 26), 'int')
    # Applying the binary operator '<=' (line 249)
    result_le_78716 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 12), '<=', task_78713, int_78715)
    # Applying the binary operator '&' (line 249)
    result_and__78717 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 12), '&', result_le_78714, result_le_78716)
    
    # Applying the 'not' unary operator (line 249)
    result_not__78718 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 7), 'not', result_and__78717)
    
    # Testing the type of an if condition (line 249)
    if_condition_78719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 4), result_not__78718)
    # Assigning a type to the variable 'if_condition_78719' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'if_condition_78719', if_condition_78719)
    # SSA begins for if statement (line 249)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 250)
    # Processing the call arguments (line 250)
    str_78721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 24), 'str', 'task must be -1, 0 or 1')
    # Processing the call keyword arguments (line 250)
    kwargs_78722 = {}
    # Getting the type of 'TypeError' (line 250)
    TypeError_78720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 250)
    TypeError_call_result_78723 = invoke(stypy.reporting.localization.Localization(__file__, 250, 14), TypeError_78720, *[str_78721], **kwargs_78722)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 250, 8), TypeError_call_result_78723, 'raise parameter', BaseException)
    # SSA join for if statement (line 249)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    
    # Call to len(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'w' (line 251)
    w_78725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'w', False)
    # Processing the call keyword arguments (line 251)
    kwargs_78726 = {}
    # Getting the type of 'len' (line 251)
    len_78724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'len', False)
    # Calling len(args, kwargs) (line 251)
    len_call_result_78727 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), len_78724, *[w_78725], **kwargs_78726)
    
    # Getting the type of 'm' (line 251)
    m_78728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 22), 'm')
    # Applying the binary operator '==' (line 251)
    result_eq_78729 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 12), '==', len_call_result_78727, m_78728)
    
    # Applying the 'not' unary operator (line 251)
    result_not__78730 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 8), 'not', result_eq_78729)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ipar' (line 251)
    ipar_78731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 29), 'ipar')
    int_78732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 37), 'int')
    # Applying the binary operator '==' (line 251)
    result_eq_78733 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 29), '==', ipar_78731, int_78732)
    
    
    
    
    # Call to len(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'u' (line 251)
    u_78735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 52), 'u', False)
    # Processing the call keyword arguments (line 251)
    kwargs_78736 = {}
    # Getting the type of 'len' (line 251)
    len_78734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 48), 'len', False)
    # Calling len(args, kwargs) (line 251)
    len_call_result_78737 = invoke(stypy.reporting.localization.Localization(__file__, 251, 48), len_78734, *[u_78735], **kwargs_78736)
    
    # Getting the type of 'm' (line 251)
    m_78738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 58), 'm')
    # Applying the binary operator '==' (line 251)
    result_eq_78739 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 48), '==', len_call_result_78737, m_78738)
    
    # Applying the 'not' unary operator (line 251)
    result_not__78740 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 44), 'not', result_eq_78739)
    
    # Applying the binary operator 'and' (line 251)
    result_and_keyword_78741 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 29), 'and', result_eq_78733, result_not__78740)
    
    # Applying the binary operator 'or' (line 251)
    result_or_keyword_78742 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 7), 'or', result_not__78730, result_and_keyword_78741)
    
    # Testing the type of an if condition (line 251)
    if_condition_78743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 4), result_or_keyword_78742)
    # Assigning a type to the variable 'if_condition_78743' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'if_condition_78743', if_condition_78743)
    # SSA begins for if statement (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 252)
    # Processing the call arguments (line 252)
    str_78745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 24), 'str', 'Mismatch of input dimensions')
    # Processing the call keyword arguments (line 252)
    kwargs_78746 = {}
    # Getting the type of 'TypeError' (line 252)
    TypeError_78744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 252)
    TypeError_call_result_78747 = invoke(stypy.reporting.localization.Localization(__file__, 252, 14), TypeError_78744, *[str_78745], **kwargs_78746)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 252, 8), TypeError_call_result_78747, 'raise parameter', BaseException)
    # SSA join for if statement (line 251)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 253)
    # Getting the type of 's' (line 253)
    s_78748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 7), 's')
    # Getting the type of 'None' (line 253)
    None_78749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'None')
    
    (may_be_78750, more_types_in_union_78751) = may_be_none(s_78748, None_78749)

    if may_be_78750:

        if more_types_in_union_78751:
            # Runtime conditional SSA (line 253)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 254):
        
        # Assigning a BinOp to a Name (line 254):
        # Getting the type of 'm' (line 254)
        m_78752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'm')
        
        # Call to sqrt(...): (line 254)
        # Processing the call arguments (line 254)
        int_78754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 21), 'int')
        # Getting the type of 'm' (line 254)
        m_78755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'm', False)
        # Applying the binary operator '*' (line 254)
        result_mul_78756 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 21), '*', int_78754, m_78755)
        
        # Processing the call keyword arguments (line 254)
        kwargs_78757 = {}
        # Getting the type of 'sqrt' (line 254)
        sqrt_78753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 254)
        sqrt_call_result_78758 = invoke(stypy.reporting.localization.Localization(__file__, 254, 16), sqrt_78753, *[result_mul_78756], **kwargs_78757)
        
        # Applying the binary operator '-' (line 254)
        result_sub_78759 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 12), '-', m_78752, sqrt_call_result_78758)
        
        # Assigning a type to the variable 's' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 's', result_sub_78759)

        if more_types_in_union_78751:
            # SSA join for if statement (line 253)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 't' (line 255)
    t_78760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 7), 't')
    # Getting the type of 'None' (line 255)
    None_78761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'None')
    # Applying the binary operator 'is' (line 255)
    result_is__78762 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 7), 'is', t_78760, None_78761)
    
    
    # Getting the type of 'task' (line 255)
    task_78763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'task')
    int_78764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 29), 'int')
    # Applying the binary operator '==' (line 255)
    result_eq_78765 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 21), '==', task_78763, int_78764)
    
    # Applying the binary operator 'and' (line 255)
    result_and_keyword_78766 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 7), 'and', result_is__78762, result_eq_78765)
    
    # Testing the type of an if condition (line 255)
    if_condition_78767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 4), result_and_keyword_78766)
    # Assigning a type to the variable 'if_condition_78767' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'if_condition_78767', if_condition_78767)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 256)
    # Processing the call arguments (line 256)
    str_78769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 24), 'str', 'Knots must be given for task=-1')
    # Processing the call keyword arguments (line 256)
    kwargs_78770 = {}
    # Getting the type of 'TypeError' (line 256)
    TypeError_78768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 256)
    TypeError_call_result_78771 = invoke(stypy.reporting.localization.Localization(__file__, 256, 14), TypeError_78768, *[str_78769], **kwargs_78770)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 8), TypeError_call_result_78771, 'raise parameter', BaseException)
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 257)
    # Getting the type of 't' (line 257)
    t_78772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 't')
    # Getting the type of 'None' (line 257)
    None_78773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'None')
    
    (may_be_78774, more_types_in_union_78775) = may_not_be_none(t_78772, None_78773)

    if may_be_78774:

        if more_types_in_union_78775:
            # Runtime conditional SSA (line 257)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 258):
        
        # Assigning a Call to a Subscript (line 258):
        
        # Call to atleast_1d(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 't' (line 258)
        t_78777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 40), 't', False)
        # Processing the call keyword arguments (line 258)
        kwargs_78778 = {}
        # Getting the type of 'atleast_1d' (line 258)
        atleast_1d_78776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 29), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 258)
        atleast_1d_call_result_78779 = invoke(stypy.reporting.localization.Localization(__file__, 258, 29), atleast_1d_78776, *[t_78777], **kwargs_78778)
        
        # Getting the type of '_parcur_cache' (line 258)
        _parcur_cache_78780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), '_parcur_cache')
        str_78781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 22), 'str', 't')
        # Storing an element on a container (line 258)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 8), _parcur_cache_78780, (str_78781, atleast_1d_call_result_78779))

        if more_types_in_union_78775:
            # SSA join for if statement (line 257)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 259):
    
    # Call to len(...): (line 259)
    # Processing the call arguments (line 259)
    
    # Obtaining the type of the subscript
    str_78783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 26), 'str', 't')
    # Getting the type of '_parcur_cache' (line 259)
    _parcur_cache_78784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), '_parcur_cache', False)
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___78785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), _parcur_cache_78784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_78786 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), getitem___78785, str_78783)
    
    # Processing the call keyword arguments (line 259)
    kwargs_78787 = {}
    # Getting the type of 'len' (line 259)
    len_78782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'len', False)
    # Calling len(args, kwargs) (line 259)
    len_call_result_78788 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), len_78782, *[subscript_call_result_78786], **kwargs_78787)
    
    # Assigning a type to the variable 'n' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'n', len_call_result_78788)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'task' (line 260)
    task_78789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 7), 'task')
    int_78790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 15), 'int')
    # Applying the binary operator '==' (line 260)
    result_eq_78791 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 7), '==', task_78789, int_78790)
    
    
    # Getting the type of 'n' (line 260)
    n_78792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'n')
    int_78793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 26), 'int')
    # Getting the type of 'k' (line 260)
    k_78794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 28), 'k')
    # Applying the binary operator '*' (line 260)
    result_mul_78795 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 26), '*', int_78793, k_78794)
    
    int_78796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'int')
    # Applying the binary operator '+' (line 260)
    result_add_78797 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 26), '+', result_mul_78795, int_78796)
    
    # Applying the binary operator '<' (line 260)
    result_lt_78798 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 22), '<', n_78792, result_add_78797)
    
    # Applying the binary operator 'and' (line 260)
    result_and_keyword_78799 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 7), 'and', result_eq_78791, result_lt_78798)
    
    # Testing the type of an if condition (line 260)
    if_condition_78800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 4), result_and_keyword_78799)
    # Assigning a type to the variable 'if_condition_78800' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'if_condition_78800', if_condition_78800)
    # SSA begins for if statement (line 260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 261)
    # Processing the call arguments (line 261)
    str_78802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 24), 'str', 'There must be at least 2*k+2 knots for task=-1')
    # Processing the call keyword arguments (line 261)
    kwargs_78803 = {}
    # Getting the type of 'TypeError' (line 261)
    TypeError_78801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 261)
    TypeError_call_result_78804 = invoke(stypy.reporting.localization.Localization(__file__, 261, 14), TypeError_78801, *[str_78802], **kwargs_78803)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 261, 8), TypeError_call_result_78804, 'raise parameter', BaseException)
    # SSA join for if statement (line 260)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 262)
    m_78805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 7), 'm')
    # Getting the type of 'k' (line 262)
    k_78806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'k')
    # Applying the binary operator '<=' (line 262)
    result_le_78807 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 7), '<=', m_78805, k_78806)
    
    # Testing the type of an if condition (line 262)
    if_condition_78808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 4), result_le_78807)
    # Assigning a type to the variable 'if_condition_78808' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'if_condition_78808', if_condition_78808)
    # SSA begins for if statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 263)
    # Processing the call arguments (line 263)
    str_78810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 24), 'str', 'm > k must hold')
    # Processing the call keyword arguments (line 263)
    kwargs_78811 = {}
    # Getting the type of 'TypeError' (line 263)
    TypeError_78809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 263)
    TypeError_call_result_78812 = invoke(stypy.reporting.localization.Localization(__file__, 263, 14), TypeError_78809, *[str_78810], **kwargs_78811)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 263, 8), TypeError_call_result_78812, 'raise parameter', BaseException)
    # SSA join for if statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 264)
    # Getting the type of 'nest' (line 264)
    nest_78813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 7), 'nest')
    # Getting the type of 'None' (line 264)
    None_78814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'None')
    
    (may_be_78815, more_types_in_union_78816) = may_be_none(nest_78813, None_78814)

    if may_be_78815:

        if more_types_in_union_78816:
            # Runtime conditional SSA (line 264)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 265):
        
        # Assigning a BinOp to a Name (line 265):
        # Getting the type of 'm' (line 265)
        m_78817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'm')
        int_78818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 19), 'int')
        # Getting the type of 'k' (line 265)
        k_78819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'k')
        # Applying the binary operator '*' (line 265)
        result_mul_78820 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 19), '*', int_78818, k_78819)
        
        # Applying the binary operator '+' (line 265)
        result_add_78821 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), '+', m_78817, result_mul_78820)
        
        # Assigning a type to the variable 'nest' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'nest', result_add_78821)

        if more_types_in_union_78816:
            # SSA join for if statement (line 264)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'task' (line 267)
    task_78822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'task')
    int_78823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 16), 'int')
    # Applying the binary operator '>=' (line 267)
    result_ge_78824 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 8), '>=', task_78822, int_78823)
    
    
    # Getting the type of 's' (line 267)
    s_78825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 22), 's')
    int_78826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 27), 'int')
    # Applying the binary operator '==' (line 267)
    result_eq_78827 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 22), '==', s_78825, int_78826)
    
    # Applying the binary operator 'and' (line 267)
    result_and_keyword_78828 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 8), 'and', result_ge_78824, result_eq_78827)
    
    
    # Getting the type of 'nest' (line 267)
    nest_78829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 34), 'nest')
    int_78830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 41), 'int')
    # Applying the binary operator '<' (line 267)
    result_lt_78831 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 34), '<', nest_78829, int_78830)
    
    # Applying the binary operator 'or' (line 267)
    result_or_keyword_78832 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 7), 'or', result_and_keyword_78828, result_lt_78831)
    
    # Testing the type of an if condition (line 267)
    if_condition_78833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 4), result_or_keyword_78832)
    # Assigning a type to the variable 'if_condition_78833' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'if_condition_78833', if_condition_78833)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'per' (line 268)
    per_78834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'per')
    # Testing the type of an if condition (line 268)
    if_condition_78835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), per_78834)
    # Assigning a type to the variable 'if_condition_78835' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_78835', if_condition_78835)
    # SSA begins for if statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 269):
    
    # Assigning a BinOp to a Name (line 269):
    # Getting the type of 'm' (line 269)
    m_78836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'm')
    int_78837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 23), 'int')
    # Getting the type of 'k' (line 269)
    k_78838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'k')
    # Applying the binary operator '*' (line 269)
    result_mul_78839 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 23), '*', int_78837, k_78838)
    
    # Applying the binary operator '+' (line 269)
    result_add_78840 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 19), '+', m_78836, result_mul_78839)
    
    # Assigning a type to the variable 'nest' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'nest', result_add_78840)
    # SSA branch for the else part of an if statement (line 268)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 271):
    
    # Assigning a BinOp to a Name (line 271):
    # Getting the type of 'm' (line 271)
    m_78841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'm')
    # Getting the type of 'k' (line 271)
    k_78842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'k')
    # Applying the binary operator '+' (line 271)
    result_add_78843 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 19), '+', m_78841, k_78842)
    
    int_78844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 27), 'int')
    # Applying the binary operator '+' (line 271)
    result_add_78845 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 25), '+', result_add_78843, int_78844)
    
    # Assigning a type to the variable 'nest' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'nest', result_add_78845)
    # SSA join for if statement (line 268)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 272):
    
    # Assigning a Call to a Name (line 272):
    
    # Call to max(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'nest' (line 272)
    nest_78847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'nest', False)
    int_78848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 21), 'int')
    # Getting the type of 'k' (line 272)
    k_78849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'k', False)
    # Applying the binary operator '*' (line 272)
    result_mul_78850 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 21), '*', int_78848, k_78849)
    
    int_78851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 27), 'int')
    # Applying the binary operator '+' (line 272)
    result_add_78852 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 21), '+', result_mul_78850, int_78851)
    
    # Processing the call keyword arguments (line 272)
    kwargs_78853 = {}
    # Getting the type of 'max' (line 272)
    max_78846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'max', False)
    # Calling max(args, kwargs) (line 272)
    max_call_result_78854 = invoke(stypy.reporting.localization.Localization(__file__, 272, 11), max_78846, *[nest_78847, result_add_78852], **kwargs_78853)
    
    # Assigning a type to the variable 'nest' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'nest', max_call_result_78854)
    
    # Assigning a Subscript to a Name (line 273):
    
    # Assigning a Subscript to a Name (line 273):
    
    # Obtaining the type of the subscript
    str_78855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 22), 'str', 'u')
    # Getting the type of '_parcur_cache' (line 273)
    _parcur_cache_78856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), '_parcur_cache')
    # Obtaining the member '__getitem__' of a type (line 273)
    getitem___78857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), _parcur_cache_78856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 273)
    subscript_call_result_78858 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), getitem___78857, str_78855)
    
    # Assigning a type to the variable 'u' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'u', subscript_call_result_78858)
    
    # Assigning a Subscript to a Name (line 274):
    
    # Assigning a Subscript to a Name (line 274):
    
    # Obtaining the type of the subscript
    str_78859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 23), 'str', 'ub')
    # Getting the type of '_parcur_cache' (line 274)
    _parcur_cache_78860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 9), '_parcur_cache')
    # Obtaining the member '__getitem__' of a type (line 274)
    getitem___78861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 9), _parcur_cache_78860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 274)
    subscript_call_result_78862 = invoke(stypy.reporting.localization.Localization(__file__, 274, 9), getitem___78861, str_78859)
    
    # Assigning a type to the variable 'ub' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'ub', subscript_call_result_78862)
    
    # Assigning a Subscript to a Name (line 275):
    
    # Assigning a Subscript to a Name (line 275):
    
    # Obtaining the type of the subscript
    str_78863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 23), 'str', 'ue')
    # Getting the type of '_parcur_cache' (line 275)
    _parcur_cache_78864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), '_parcur_cache')
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___78865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 9), _parcur_cache_78864, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_78866 = invoke(stypy.reporting.localization.Localization(__file__, 275, 9), getitem___78865, str_78863)
    
    # Assigning a type to the variable 'ue' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'ue', subscript_call_result_78866)
    
    # Assigning a Subscript to a Name (line 276):
    
    # Assigning a Subscript to a Name (line 276):
    
    # Obtaining the type of the subscript
    str_78867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 22), 'str', 't')
    # Getting the type of '_parcur_cache' (line 276)
    _parcur_cache_78868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), '_parcur_cache')
    # Obtaining the member '__getitem__' of a type (line 276)
    getitem___78869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), _parcur_cache_78868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 276)
    subscript_call_result_78870 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), getitem___78869, str_78867)
    
    # Assigning a type to the variable 't' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 't', subscript_call_result_78870)
    
    # Assigning a Subscript to a Name (line 277):
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    str_78871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'str', 'wrk')
    # Getting the type of '_parcur_cache' (line 277)
    _parcur_cache_78872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 10), '_parcur_cache')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___78873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 10), _parcur_cache_78872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_78874 = invoke(stypy.reporting.localization.Localization(__file__, 277, 10), getitem___78873, str_78871)
    
    # Assigning a type to the variable 'wrk' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'wrk', subscript_call_result_78874)
    
    # Assigning a Subscript to a Name (line 278):
    
    # Assigning a Subscript to a Name (line 278):
    
    # Obtaining the type of the subscript
    str_78875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 25), 'str', 'iwrk')
    # Getting the type of '_parcur_cache' (line 278)
    _parcur_cache_78876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), '_parcur_cache')
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___78877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 11), _parcur_cache_78876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_78878 = invoke(stypy.reporting.localization.Localization(__file__, 278, 11), getitem___78877, str_78875)
    
    # Assigning a type to the variable 'iwrk' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'iwrk', subscript_call_result_78878)
    
    # Assigning a Call to a Tuple (line 279):
    
    # Assigning a Subscript to a Name (line 279):
    
    # Obtaining the type of the subscript
    int_78879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 4), 'int')
    
    # Call to _parcur(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Call to ravel(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Call to transpose(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'x' (line 279)
    x_78884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 47), 'x', False)
    # Processing the call keyword arguments (line 279)
    kwargs_78885 = {}
    # Getting the type of 'transpose' (line 279)
    transpose_78883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 37), 'transpose', False)
    # Calling transpose(args, kwargs) (line 279)
    transpose_call_result_78886 = invoke(stypy.reporting.localization.Localization(__file__, 279, 37), transpose_78883, *[x_78884], **kwargs_78885)
    
    # Processing the call keyword arguments (line 279)
    kwargs_78887 = {}
    # Getting the type of 'ravel' (line 279)
    ravel_78882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'ravel', False)
    # Calling ravel(args, kwargs) (line 279)
    ravel_call_result_78888 = invoke(stypy.reporting.localization.Localization(__file__, 279, 31), ravel_78882, *[transpose_call_result_78886], **kwargs_78887)
    
    # Getting the type of 'w' (line 279)
    w_78889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 52), 'w', False)
    # Getting the type of 'u' (line 279)
    u_78890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 55), 'u', False)
    # Getting the type of 'ub' (line 279)
    ub_78891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 58), 'ub', False)
    # Getting the type of 'ue' (line 279)
    ue_78892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 62), 'ue', False)
    # Getting the type of 'k' (line 279)
    k_78893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 66), 'k', False)
    # Getting the type of 'task' (line 280)
    task_78894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 31), 'task', False)
    # Getting the type of 'ipar' (line 280)
    ipar_78895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), 'ipar', False)
    # Getting the type of 's' (line 280)
    s_78896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 43), 's', False)
    # Getting the type of 't' (line 280)
    t_78897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 46), 't', False)
    # Getting the type of 'nest' (line 280)
    nest_78898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 49), 'nest', False)
    # Getting the type of 'wrk' (line 280)
    wrk_78899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 55), 'wrk', False)
    # Getting the type of 'iwrk' (line 280)
    iwrk_78900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 60), 'iwrk', False)
    # Getting the type of 'per' (line 280)
    per_78901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 66), 'per', False)
    # Processing the call keyword arguments (line 279)
    kwargs_78902 = {}
    # Getting the type of '_fitpack' (line 279)
    _fitpack_78880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 14), '_fitpack', False)
    # Obtaining the member '_parcur' of a type (line 279)
    _parcur_78881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 14), _fitpack_78880, '_parcur')
    # Calling _parcur(args, kwargs) (line 279)
    _parcur_call_result_78903 = invoke(stypy.reporting.localization.Localization(__file__, 279, 14), _parcur_78881, *[ravel_call_result_78888, w_78889, u_78890, ub_78891, ue_78892, k_78893, task_78894, ipar_78895, s_78896, t_78897, nest_78898, wrk_78899, iwrk_78900, per_78901], **kwargs_78902)
    
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___78904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 4), _parcur_call_result_78903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_78905 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), getitem___78904, int_78879)
    
    # Assigning a type to the variable 'tuple_var_assignment_78279' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'tuple_var_assignment_78279', subscript_call_result_78905)
    
    # Assigning a Subscript to a Name (line 279):
    
    # Obtaining the type of the subscript
    int_78906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 4), 'int')
    
    # Call to _parcur(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Call to ravel(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Call to transpose(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'x' (line 279)
    x_78911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 47), 'x', False)
    # Processing the call keyword arguments (line 279)
    kwargs_78912 = {}
    # Getting the type of 'transpose' (line 279)
    transpose_78910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 37), 'transpose', False)
    # Calling transpose(args, kwargs) (line 279)
    transpose_call_result_78913 = invoke(stypy.reporting.localization.Localization(__file__, 279, 37), transpose_78910, *[x_78911], **kwargs_78912)
    
    # Processing the call keyword arguments (line 279)
    kwargs_78914 = {}
    # Getting the type of 'ravel' (line 279)
    ravel_78909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'ravel', False)
    # Calling ravel(args, kwargs) (line 279)
    ravel_call_result_78915 = invoke(stypy.reporting.localization.Localization(__file__, 279, 31), ravel_78909, *[transpose_call_result_78913], **kwargs_78914)
    
    # Getting the type of 'w' (line 279)
    w_78916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 52), 'w', False)
    # Getting the type of 'u' (line 279)
    u_78917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 55), 'u', False)
    # Getting the type of 'ub' (line 279)
    ub_78918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 58), 'ub', False)
    # Getting the type of 'ue' (line 279)
    ue_78919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 62), 'ue', False)
    # Getting the type of 'k' (line 279)
    k_78920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 66), 'k', False)
    # Getting the type of 'task' (line 280)
    task_78921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 31), 'task', False)
    # Getting the type of 'ipar' (line 280)
    ipar_78922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), 'ipar', False)
    # Getting the type of 's' (line 280)
    s_78923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 43), 's', False)
    # Getting the type of 't' (line 280)
    t_78924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 46), 't', False)
    # Getting the type of 'nest' (line 280)
    nest_78925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 49), 'nest', False)
    # Getting the type of 'wrk' (line 280)
    wrk_78926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 55), 'wrk', False)
    # Getting the type of 'iwrk' (line 280)
    iwrk_78927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 60), 'iwrk', False)
    # Getting the type of 'per' (line 280)
    per_78928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 66), 'per', False)
    # Processing the call keyword arguments (line 279)
    kwargs_78929 = {}
    # Getting the type of '_fitpack' (line 279)
    _fitpack_78907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 14), '_fitpack', False)
    # Obtaining the member '_parcur' of a type (line 279)
    _parcur_78908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 14), _fitpack_78907, '_parcur')
    # Calling _parcur(args, kwargs) (line 279)
    _parcur_call_result_78930 = invoke(stypy.reporting.localization.Localization(__file__, 279, 14), _parcur_78908, *[ravel_call_result_78915, w_78916, u_78917, ub_78918, ue_78919, k_78920, task_78921, ipar_78922, s_78923, t_78924, nest_78925, wrk_78926, iwrk_78927, per_78928], **kwargs_78929)
    
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___78931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 4), _parcur_call_result_78930, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_78932 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), getitem___78931, int_78906)
    
    # Assigning a type to the variable 'tuple_var_assignment_78280' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'tuple_var_assignment_78280', subscript_call_result_78932)
    
    # Assigning a Subscript to a Name (line 279):
    
    # Obtaining the type of the subscript
    int_78933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 4), 'int')
    
    # Call to _parcur(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Call to ravel(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Call to transpose(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'x' (line 279)
    x_78938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 47), 'x', False)
    # Processing the call keyword arguments (line 279)
    kwargs_78939 = {}
    # Getting the type of 'transpose' (line 279)
    transpose_78937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 37), 'transpose', False)
    # Calling transpose(args, kwargs) (line 279)
    transpose_call_result_78940 = invoke(stypy.reporting.localization.Localization(__file__, 279, 37), transpose_78937, *[x_78938], **kwargs_78939)
    
    # Processing the call keyword arguments (line 279)
    kwargs_78941 = {}
    # Getting the type of 'ravel' (line 279)
    ravel_78936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'ravel', False)
    # Calling ravel(args, kwargs) (line 279)
    ravel_call_result_78942 = invoke(stypy.reporting.localization.Localization(__file__, 279, 31), ravel_78936, *[transpose_call_result_78940], **kwargs_78941)
    
    # Getting the type of 'w' (line 279)
    w_78943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 52), 'w', False)
    # Getting the type of 'u' (line 279)
    u_78944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 55), 'u', False)
    # Getting the type of 'ub' (line 279)
    ub_78945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 58), 'ub', False)
    # Getting the type of 'ue' (line 279)
    ue_78946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 62), 'ue', False)
    # Getting the type of 'k' (line 279)
    k_78947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 66), 'k', False)
    # Getting the type of 'task' (line 280)
    task_78948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 31), 'task', False)
    # Getting the type of 'ipar' (line 280)
    ipar_78949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), 'ipar', False)
    # Getting the type of 's' (line 280)
    s_78950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 43), 's', False)
    # Getting the type of 't' (line 280)
    t_78951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 46), 't', False)
    # Getting the type of 'nest' (line 280)
    nest_78952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 49), 'nest', False)
    # Getting the type of 'wrk' (line 280)
    wrk_78953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 55), 'wrk', False)
    # Getting the type of 'iwrk' (line 280)
    iwrk_78954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 60), 'iwrk', False)
    # Getting the type of 'per' (line 280)
    per_78955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 66), 'per', False)
    # Processing the call keyword arguments (line 279)
    kwargs_78956 = {}
    # Getting the type of '_fitpack' (line 279)
    _fitpack_78934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 14), '_fitpack', False)
    # Obtaining the member '_parcur' of a type (line 279)
    _parcur_78935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 14), _fitpack_78934, '_parcur')
    # Calling _parcur(args, kwargs) (line 279)
    _parcur_call_result_78957 = invoke(stypy.reporting.localization.Localization(__file__, 279, 14), _parcur_78935, *[ravel_call_result_78942, w_78943, u_78944, ub_78945, ue_78946, k_78947, task_78948, ipar_78949, s_78950, t_78951, nest_78952, wrk_78953, iwrk_78954, per_78955], **kwargs_78956)
    
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___78958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 4), _parcur_call_result_78957, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_78959 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), getitem___78958, int_78933)
    
    # Assigning a type to the variable 'tuple_var_assignment_78281' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'tuple_var_assignment_78281', subscript_call_result_78959)
    
    # Assigning a Name to a Name (line 279):
    # Getting the type of 'tuple_var_assignment_78279' (line 279)
    tuple_var_assignment_78279_78960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'tuple_var_assignment_78279')
    # Assigning a type to the variable 't' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 't', tuple_var_assignment_78279_78960)
    
    # Assigning a Name to a Name (line 279):
    # Getting the type of 'tuple_var_assignment_78280' (line 279)
    tuple_var_assignment_78280_78961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'tuple_var_assignment_78280')
    # Assigning a type to the variable 'c' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 7), 'c', tuple_var_assignment_78280_78961)
    
    # Assigning a Name to a Name (line 279):
    # Getting the type of 'tuple_var_assignment_78281' (line 279)
    tuple_var_assignment_78281_78962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'tuple_var_assignment_78281')
    # Assigning a type to the variable 'o' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 10), 'o', tuple_var_assignment_78281_78962)
    
    # Assigning a Subscript to a Subscript (line 281):
    
    # Assigning a Subscript to a Subscript (line 281):
    
    # Obtaining the type of the subscript
    str_78963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 27), 'str', 'u')
    # Getting the type of 'o' (line 281)
    o_78964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 25), 'o')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___78965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 25), o_78964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_78966 = invoke(stypy.reporting.localization.Localization(__file__, 281, 25), getitem___78965, str_78963)
    
    # Getting the type of '_parcur_cache' (line 281)
    _parcur_cache_78967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), '_parcur_cache')
    str_78968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 18), 'str', 'u')
    # Storing an element on a container (line 281)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 4), _parcur_cache_78967, (str_78968, subscript_call_result_78966))
    
    # Assigning a Subscript to a Subscript (line 282):
    
    # Assigning a Subscript to a Subscript (line 282):
    
    # Obtaining the type of the subscript
    str_78969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 28), 'str', 'ub')
    # Getting the type of 'o' (line 282)
    o_78970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'o')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___78971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 26), o_78970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_78972 = invoke(stypy.reporting.localization.Localization(__file__, 282, 26), getitem___78971, str_78969)
    
    # Getting the type of '_parcur_cache' (line 282)
    _parcur_cache_78973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), '_parcur_cache')
    str_78974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'str', 'ub')
    # Storing an element on a container (line 282)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 4), _parcur_cache_78973, (str_78974, subscript_call_result_78972))
    
    # Assigning a Subscript to a Subscript (line 283):
    
    # Assigning a Subscript to a Subscript (line 283):
    
    # Obtaining the type of the subscript
    str_78975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'str', 'ue')
    # Getting the type of 'o' (line 283)
    o_78976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'o')
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___78977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 26), o_78976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_78978 = invoke(stypy.reporting.localization.Localization(__file__, 283, 26), getitem___78977, str_78975)
    
    # Getting the type of '_parcur_cache' (line 283)
    _parcur_cache_78979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), '_parcur_cache')
    str_78980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 18), 'str', 'ue')
    # Storing an element on a container (line 283)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 4), _parcur_cache_78979, (str_78980, subscript_call_result_78978))
    
    # Assigning a Name to a Subscript (line 284):
    
    # Assigning a Name to a Subscript (line 284):
    # Getting the type of 't' (line 284)
    t_78981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 't')
    # Getting the type of '_parcur_cache' (line 284)
    _parcur_cache_78982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), '_parcur_cache')
    str_78983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 18), 'str', 't')
    # Storing an element on a container (line 284)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 4), _parcur_cache_78982, (str_78983, t_78981))
    
    # Assigning a Subscript to a Subscript (line 285):
    
    # Assigning a Subscript to a Subscript (line 285):
    
    # Obtaining the type of the subscript
    str_78984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 29), 'str', 'wrk')
    # Getting the type of 'o' (line 285)
    o_78985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 27), 'o')
    # Obtaining the member '__getitem__' of a type (line 285)
    getitem___78986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 27), o_78985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 285)
    subscript_call_result_78987 = invoke(stypy.reporting.localization.Localization(__file__, 285, 27), getitem___78986, str_78984)
    
    # Getting the type of '_parcur_cache' (line 285)
    _parcur_cache_78988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), '_parcur_cache')
    str_78989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 18), 'str', 'wrk')
    # Storing an element on a container (line 285)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 4), _parcur_cache_78988, (str_78989, subscript_call_result_78987))
    
    # Assigning a Subscript to a Subscript (line 286):
    
    # Assigning a Subscript to a Subscript (line 286):
    
    # Obtaining the type of the subscript
    str_78990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 30), 'str', 'iwrk')
    # Getting the type of 'o' (line 286)
    o_78991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 28), 'o')
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___78992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 28), o_78991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_78993 = invoke(stypy.reporting.localization.Localization(__file__, 286, 28), getitem___78992, str_78990)
    
    # Getting the type of '_parcur_cache' (line 286)
    _parcur_cache_78994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), '_parcur_cache')
    str_78995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 18), 'str', 'iwrk')
    # Storing an element on a container (line 286)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 4), _parcur_cache_78994, (str_78995, subscript_call_result_78993))
    
    # Assigning a Subscript to a Name (line 287):
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    str_78996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 12), 'str', 'ier')
    # Getting the type of 'o' (line 287)
    o_78997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 10), 'o')
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___78998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 10), o_78997, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_78999 = invoke(stypy.reporting.localization.Localization(__file__, 287, 10), getitem___78998, str_78996)
    
    # Assigning a type to the variable 'ier' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'ier', subscript_call_result_78999)
    
    # Assigning a Subscript to a Name (line 288):
    
    # Assigning a Subscript to a Name (line 288):
    
    # Obtaining the type of the subscript
    str_79000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 11), 'str', 'fp')
    # Getting the type of 'o' (line 288)
    o_79001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 9), 'o')
    # Obtaining the member '__getitem__' of a type (line 288)
    getitem___79002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 9), o_79001, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 288)
    subscript_call_result_79003 = invoke(stypy.reporting.localization.Localization(__file__, 288, 9), getitem___79002, str_79000)
    
    # Assigning a type to the variable 'fp' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'fp', subscript_call_result_79003)
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to len(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 't' (line 289)
    t_79005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 't', False)
    # Processing the call keyword arguments (line 289)
    kwargs_79006 = {}
    # Getting the type of 'len' (line 289)
    len_79004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'len', False)
    # Calling len(args, kwargs) (line 289)
    len_call_result_79007 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), len_79004, *[t_79005], **kwargs_79006)
    
    # Assigning a type to the variable 'n' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'n', len_call_result_79007)
    
    # Assigning a Subscript to a Name (line 290):
    
    # Assigning a Subscript to a Name (line 290):
    
    # Obtaining the type of the subscript
    str_79008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 10), 'str', 'u')
    # Getting the type of 'o' (line 290)
    o_79009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'o')
    # Obtaining the member '__getitem__' of a type (line 290)
    getitem___79010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), o_79009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 290)
    subscript_call_result_79011 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___79010, str_79008)
    
    # Assigning a type to the variable 'u' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'u', subscript_call_result_79011)
    
    # Assigning a Tuple to a Attribute (line 291):
    
    # Assigning a Tuple to a Attribute (line 291):
    
    # Obtaining an instance of the builtin type 'tuple' (line 291)
    tuple_79012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 291)
    # Adding element type (line 291)
    # Getting the type of 'idim' (line 291)
    idim_79013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 14), 'idim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 14), tuple_79012, idim_79013)
    # Adding element type (line 291)
    # Getting the type of 'n' (line 291)
    n_79014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'n')
    # Getting the type of 'k' (line 291)
    k_79015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'k')
    # Applying the binary operator '-' (line 291)
    result_sub_79016 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 20), '-', n_79014, k_79015)
    
    int_79017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 28), 'int')
    # Applying the binary operator '-' (line 291)
    result_sub_79018 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 26), '-', result_sub_79016, int_79017)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 14), tuple_79012, result_sub_79018)
    
    # Getting the type of 'c' (line 291)
    c_79019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'c')
    # Setting the type of the member 'shape' of a type (line 291)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 4), c_79019, 'shape', tuple_79012)
    
    # Assigning a Tuple to a Name (line 292):
    
    # Assigning a Tuple to a Name (line 292):
    
    # Obtaining an instance of the builtin type 'tuple' (line 292)
    tuple_79020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 292)
    # Adding element type (line 292)
    
    # Obtaining an instance of the builtin type 'list' (line 292)
    list_79021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 292)
    # Adding element type (line 292)
    # Getting the type of 't' (line 292)
    t_79022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 11), list_79021, t_79022)
    # Adding element type (line 292)
    
    # Call to list(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'c' (line 292)
    c_79024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 20), 'c', False)
    # Processing the call keyword arguments (line 292)
    kwargs_79025 = {}
    # Getting the type of 'list' (line 292)
    list_79023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'list', False)
    # Calling list(args, kwargs) (line 292)
    list_call_result_79026 = invoke(stypy.reporting.localization.Localization(__file__, 292, 15), list_79023, *[c_79024], **kwargs_79025)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 11), list_79021, list_call_result_79026)
    # Adding element type (line 292)
    # Getting the type of 'k' (line 292)
    k_79027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 11), list_79021, k_79027)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 11), tuple_79020, list_79021)
    # Adding element type (line 292)
    # Getting the type of 'u' (line 292)
    u_79028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 28), 'u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 11), tuple_79020, u_79028)
    
    # Assigning a type to the variable 'tcku' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'tcku', tuple_79020)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ier' (line 293)
    ier_79029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 7), 'ier')
    int_79030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 14), 'int')
    # Applying the binary operator '<=' (line 293)
    result_le_79031 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 7), '<=', ier_79029, int_79030)
    
    
    # Getting the type of 'quiet' (line 293)
    quiet_79032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'quiet')
    # Applying the 'not' unary operator (line 293)
    result_not__79033 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 20), 'not', quiet_79032)
    
    # Applying the binary operator 'and' (line 293)
    result_and_keyword_79034 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 7), 'and', result_le_79031, result_not__79033)
    
    # Testing the type of an if condition (line 293)
    if_condition_79035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 4), result_and_keyword_79034)
    # Assigning a type to the variable 'if_condition_79035' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'if_condition_79035', if_condition_79035)
    # SSA begins for if statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Call to RuntimeWarning(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Obtaining the type of the subscript
    int_79039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 51), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 294)
    ier_79040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 46), 'ier', False)
    # Getting the type of '_iermess' (line 294)
    _iermess_79041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 37), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 294)
    getitem___79042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 37), _iermess_79041, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 294)
    subscript_call_result_79043 = invoke(stypy.reporting.localization.Localization(__file__, 294, 37), getitem___79042, ier_79040)
    
    # Obtaining the member '__getitem__' of a type (line 294)
    getitem___79044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 37), subscript_call_result_79043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 294)
    subscript_call_result_79045 = invoke(stypy.reporting.localization.Localization(__file__, 294, 37), getitem___79044, int_79039)
    
    str_79046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 37), 'str', '\tk=%d n=%d m=%d fp=%f s=%f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 296)
    tuple_79047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 296)
    # Adding element type (line 296)
    # Getting the type of 'k' (line 296)
    k_79048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 38), tuple_79047, k_79048)
    # Adding element type (line 296)
    
    # Call to len(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 't' (line 296)
    t_79050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 45), 't', False)
    # Processing the call keyword arguments (line 296)
    kwargs_79051 = {}
    # Getting the type of 'len' (line 296)
    len_79049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 41), 'len', False)
    # Calling len(args, kwargs) (line 296)
    len_call_result_79052 = invoke(stypy.reporting.localization.Localization(__file__, 296, 41), len_79049, *[t_79050], **kwargs_79051)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 38), tuple_79047, len_call_result_79052)
    # Adding element type (line 296)
    # Getting the type of 'm' (line 296)
    m_79053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 49), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 38), tuple_79047, m_79053)
    # Adding element type (line 296)
    # Getting the type of 'fp' (line 296)
    fp_79054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 52), 'fp', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 38), tuple_79047, fp_79054)
    # Adding element type (line 296)
    # Getting the type of 's' (line 296)
    s_79055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 56), 's', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 38), tuple_79047, s_79055)
    
    # Applying the binary operator '%' (line 295)
    result_mod_79056 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 37), '%', str_79046, tuple_79047)
    
    # Applying the binary operator '+' (line 294)
    result_add_79057 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 37), '+', subscript_call_result_79045, result_mod_79056)
    
    # Processing the call keyword arguments (line 294)
    kwargs_79058 = {}
    # Getting the type of 'RuntimeWarning' (line 294)
    RuntimeWarning_79038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'RuntimeWarning', False)
    # Calling RuntimeWarning(args, kwargs) (line 294)
    RuntimeWarning_call_result_79059 = invoke(stypy.reporting.localization.Localization(__file__, 294, 22), RuntimeWarning_79038, *[result_add_79057], **kwargs_79058)
    
    # Processing the call keyword arguments (line 294)
    kwargs_79060 = {}
    # Getting the type of 'warnings' (line 294)
    warnings_79036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 294)
    warn_79037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), warnings_79036, 'warn')
    # Calling warn(args, kwargs) (line 294)
    warn_call_result_79061 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), warn_79037, *[RuntimeWarning_call_result_79059], **kwargs_79060)
    
    # SSA join for if statement (line 293)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ier' (line 297)
    ier_79062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 7), 'ier')
    int_79063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 13), 'int')
    # Applying the binary operator '>' (line 297)
    result_gt_79064 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 7), '>', ier_79062, int_79063)
    
    
    # Getting the type of 'full_output' (line 297)
    full_output_79065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 23), 'full_output')
    # Applying the 'not' unary operator (line 297)
    result_not__79066 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 19), 'not', full_output_79065)
    
    # Applying the binary operator 'and' (line 297)
    result_and_keyword_79067 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 7), 'and', result_gt_79064, result_not__79066)
    
    # Testing the type of an if condition (line 297)
    if_condition_79068 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 4), result_and_keyword_79067)
    # Assigning a type to the variable 'if_condition_79068' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'if_condition_79068', if_condition_79068)
    # SSA begins for if statement (line 297)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'ier' (line 298)
    ier_79069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'ier')
    
    # Obtaining an instance of the builtin type 'list' (line 298)
    list_79070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 298)
    # Adding element type (line 298)
    int_79071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 18), list_79070, int_79071)
    # Adding element type (line 298)
    int_79072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 18), list_79070, int_79072)
    # Adding element type (line 298)
    int_79073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 18), list_79070, int_79073)
    
    # Applying the binary operator 'in' (line 298)
    result_contains_79074 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 11), 'in', ier_79069, list_79070)
    
    # Testing the type of an if condition (line 298)
    if_condition_79075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), result_contains_79074)
    # Assigning a type to the variable 'if_condition_79075' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_79075', if_condition_79075)
    # SSA begins for if statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Call to RuntimeWarning(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Obtaining the type of the subscript
    int_79079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 55), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 299)
    ier_79080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 50), 'ier', False)
    # Getting the type of '_iermess' (line 299)
    _iermess_79081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 41), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___79082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 41), _iermess_79081, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_79083 = invoke(stypy.reporting.localization.Localization(__file__, 299, 41), getitem___79082, ier_79080)
    
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___79084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 41), subscript_call_result_79083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_79085 = invoke(stypy.reporting.localization.Localization(__file__, 299, 41), getitem___79084, int_79079)
    
    # Processing the call keyword arguments (line 299)
    kwargs_79086 = {}
    # Getting the type of 'RuntimeWarning' (line 299)
    RuntimeWarning_79078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'RuntimeWarning', False)
    # Calling RuntimeWarning(args, kwargs) (line 299)
    RuntimeWarning_call_result_79087 = invoke(stypy.reporting.localization.Localization(__file__, 299, 26), RuntimeWarning_79078, *[subscript_call_result_79085], **kwargs_79086)
    
    # Processing the call keyword arguments (line 299)
    kwargs_79088 = {}
    # Getting the type of 'warnings' (line 299)
    warnings_79076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 299)
    warn_79077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), warnings_79076, 'warn')
    # Calling warn(args, kwargs) (line 299)
    warn_call_result_79089 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), warn_79077, *[RuntimeWarning_call_result_79087], **kwargs_79088)
    
    # SSA branch for the else part of an if statement (line 298)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 301)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to (...): (line 302)
    # Processing the call arguments (line 302)
    
    # Obtaining the type of the subscript
    int_79097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 53), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 302)
    ier_79098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 48), 'ier', False)
    # Getting the type of '_iermess' (line 302)
    _iermess_79099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 39), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___79100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 39), _iermess_79099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_79101 = invoke(stypy.reporting.localization.Localization(__file__, 302, 39), getitem___79100, ier_79098)
    
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___79102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 39), subscript_call_result_79101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_79103 = invoke(stypy.reporting.localization.Localization(__file__, 302, 39), getitem___79102, int_79097)
    
    # Processing the call keyword arguments (line 302)
    kwargs_79104 = {}
    
    # Obtaining the type of the subscript
    int_79090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 36), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 302)
    ier_79091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 31), 'ier', False)
    # Getting the type of '_iermess' (line 302)
    _iermess_79092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 22), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___79093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 22), _iermess_79092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_79094 = invoke(stypy.reporting.localization.Localization(__file__, 302, 22), getitem___79093, ier_79091)
    
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___79095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 22), subscript_call_result_79094, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_79096 = invoke(stypy.reporting.localization.Localization(__file__, 302, 22), getitem___79095, int_79090)
    
    # Calling (args, kwargs) (line 302)
    _call_result_79105 = invoke(stypy.reporting.localization.Localization(__file__, 302, 22), subscript_call_result_79096, *[subscript_call_result_79103], **kwargs_79104)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 302, 16), _call_result_79105, 'raise parameter', BaseException)
    # SSA branch for the except part of a try statement (line 301)
    # SSA branch for the except 'KeyError' branch of a try statement (line 301)
    module_type_store.open_ssa_branch('except')
    
    # Call to (...): (line 304)
    # Processing the call arguments (line 304)
    
    # Obtaining the type of the subscript
    int_79113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 65), 'int')
    
    # Obtaining the type of the subscript
    str_79114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 54), 'str', 'unknown')
    # Getting the type of '_iermess' (line 304)
    _iermess_79115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 45), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___79116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 45), _iermess_79115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_79117 = invoke(stypy.reporting.localization.Localization(__file__, 304, 45), getitem___79116, str_79114)
    
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___79118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 45), subscript_call_result_79117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_79119 = invoke(stypy.reporting.localization.Localization(__file__, 304, 45), getitem___79118, int_79113)
    
    # Processing the call keyword arguments (line 304)
    kwargs_79120 = {}
    
    # Obtaining the type of the subscript
    int_79106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 42), 'int')
    
    # Obtaining the type of the subscript
    str_79107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 31), 'str', 'unknown')
    # Getting the type of '_iermess' (line 304)
    _iermess_79108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___79109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 22), _iermess_79108, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_79110 = invoke(stypy.reporting.localization.Localization(__file__, 304, 22), getitem___79109, str_79107)
    
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___79111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 22), subscript_call_result_79110, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_79112 = invoke(stypy.reporting.localization.Localization(__file__, 304, 22), getitem___79111, int_79106)
    
    # Calling (args, kwargs) (line 304)
    _call_result_79121 = invoke(stypy.reporting.localization.Localization(__file__, 304, 22), subscript_call_result_79112, *[subscript_call_result_79119], **kwargs_79120)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 304, 16), _call_result_79121, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 301)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 298)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 297)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full_output' (line 305)
    full_output_79122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 7), 'full_output')
    # Testing the type of an if condition (line 305)
    if_condition_79123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 4), full_output_79122)
    # Assigning a type to the variable 'if_condition_79123' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'if_condition_79123', if_condition_79123)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 307)
    tuple_79124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 307)
    # Adding element type (line 307)
    # Getting the type of 'tcku' (line 307)
    tcku_79125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'tcku')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 19), tuple_79124, tcku_79125)
    # Adding element type (line 307)
    # Getting the type of 'fp' (line 307)
    fp_79126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 25), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 19), tuple_79124, fp_79126)
    # Adding element type (line 307)
    # Getting the type of 'ier' (line 307)
    ier_79127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 29), 'ier')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 19), tuple_79124, ier_79127)
    # Adding element type (line 307)
    
    # Obtaining the type of the subscript
    int_79128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 48), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 307)
    ier_79129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 43), 'ier')
    # Getting the type of '_iermess' (line 307)
    _iermess_79130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 34), '_iermess')
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___79131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 34), _iermess_79130, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_79132 = invoke(stypy.reporting.localization.Localization(__file__, 307, 34), getitem___79131, ier_79129)
    
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___79133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 34), subscript_call_result_79132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_79134 = invoke(stypy.reporting.localization.Localization(__file__, 307, 34), getitem___79133, int_79128)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 19), tuple_79124, subscript_call_result_79134)
    
    # Assigning a type to the variable 'stypy_return_type' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'stypy_return_type', tuple_79124)
    # SSA branch for the except part of a try statement (line 306)
    # SSA branch for the except 'KeyError' branch of a try statement (line 306)
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 309)
    tuple_79135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 309)
    # Adding element type (line 309)
    # Getting the type of 'tcku' (line 309)
    tcku_79136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'tcku')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 19), tuple_79135, tcku_79136)
    # Adding element type (line 309)
    # Getting the type of 'fp' (line 309)
    fp_79137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 25), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 19), tuple_79135, fp_79137)
    # Adding element type (line 309)
    # Getting the type of 'ier' (line 309)
    ier_79138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 29), 'ier')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 19), tuple_79135, ier_79138)
    # Adding element type (line 309)
    
    # Obtaining the type of the subscript
    int_79139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 54), 'int')
    
    # Obtaining the type of the subscript
    str_79140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 43), 'str', 'unknown')
    # Getting the type of '_iermess' (line 309)
    _iermess_79141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 34), '_iermess')
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___79142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 34), _iermess_79141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_79143 = invoke(stypy.reporting.localization.Localization(__file__, 309, 34), getitem___79142, str_79140)
    
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___79144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 34), subscript_call_result_79143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_79145 = invoke(stypy.reporting.localization.Localization(__file__, 309, 34), getitem___79144, int_79139)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 19), tuple_79135, subscript_call_result_79145)
    
    # Assigning a type to the variable 'stypy_return_type' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'stypy_return_type', tuple_79135)
    # SSA join for try-except statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 305)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'tcku' (line 311)
    tcku_79146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), 'tcku')
    # Assigning a type to the variable 'stypy_return_type' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'stypy_return_type', tcku_79146)
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'splprep(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splprep' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_79147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_79147)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splprep'
    return stypy_return_type_79147

# Assigning a type to the variable 'splprep' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'splprep', splprep)

# Assigning a Dict to a Name (line 313):

# Assigning a Dict to a Name (line 313):

# Obtaining an instance of the builtin type 'dict' (line 313)
dict_79148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 313)
# Adding element type (key, value) (line 313)
str_79149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 17), 'str', 't')

# Call to array(...): (line 313)
# Processing the call arguments (line 313)

# Obtaining an instance of the builtin type 'list' (line 313)
list_79151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 313)

# Getting the type of 'float' (line 313)
float_79152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 32), 'float', False)
# Processing the call keyword arguments (line 313)
kwargs_79153 = {}
# Getting the type of 'array' (line 313)
array_79150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 22), 'array', False)
# Calling array(args, kwargs) (line 313)
array_call_result_79154 = invoke(stypy.reporting.localization.Localization(__file__, 313, 22), array_79150, *[list_79151, float_79152], **kwargs_79153)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 16), dict_79148, (str_79149, array_call_result_79154))
# Adding element type (key, value) (line 313)
str_79155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 40), 'str', 'wrk')

# Call to array(...): (line 313)
# Processing the call arguments (line 313)

# Obtaining an instance of the builtin type 'list' (line 313)
list_79157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 53), 'list')
# Adding type elements to the builtin type 'list' instance (line 313)

# Getting the type of 'float' (line 313)
float_79158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 57), 'float', False)
# Processing the call keyword arguments (line 313)
kwargs_79159 = {}
# Getting the type of 'array' (line 313)
array_79156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 47), 'array', False)
# Calling array(args, kwargs) (line 313)
array_call_result_79160 = invoke(stypy.reporting.localization.Localization(__file__, 313, 47), array_79156, *[list_79157, float_79158], **kwargs_79159)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 16), dict_79148, (str_79155, array_call_result_79160))
# Adding element type (key, value) (line 313)
str_79161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 17), 'str', 'iwrk')

# Call to array(...): (line 314)
# Processing the call arguments (line 314)

# Obtaining an instance of the builtin type 'list' (line 314)
list_79163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 314)

# Getting the type of 'intc' (line 314)
intc_79164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 35), 'intc', False)
# Processing the call keyword arguments (line 314)
kwargs_79165 = {}
# Getting the type of 'array' (line 314)
array_79162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 25), 'array', False)
# Calling array(args, kwargs) (line 314)
array_call_result_79166 = invoke(stypy.reporting.localization.Localization(__file__, 314, 25), array_79162, *[list_79163, intc_79164], **kwargs_79165)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 16), dict_79148, (str_79161, array_call_result_79166))

# Assigning a type to the variable '_curfit_cache' (line 313)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 0), '_curfit_cache', dict_79148)

@norecursion
def splrep(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 317)
    None_79167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'None')
    # Getting the type of 'None' (line 317)
    None_79168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 28), 'None')
    # Getting the type of 'None' (line 317)
    None_79169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 37), 'None')
    int_79170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 45), 'int')
    int_79171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 53), 'int')
    # Getting the type of 'None' (line 317)
    None_79172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 58), 'None')
    # Getting the type of 'None' (line 317)
    None_79173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 66), 'None')
    int_79174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 23), 'int')
    int_79175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 30), 'int')
    int_79176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 39), 'int')
    defaults = [None_79167, None_79168, None_79169, int_79170, int_79171, None_79172, None_79173, int_79174, int_79175, int_79176]
    # Create a new context for function 'splrep'
    module_type_store = module_type_store.open_function_context('splrep', 317, 0, False)
    
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

    str_79177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, (-1)), 'str', '\n    Find the B-spline representation of 1-D curve.\n\n    Given the set of data points ``(x[i], y[i])`` determine a smooth spline\n    approximation of degree k on the interval ``xb <= x <= xe``.\n\n    Parameters\n    ----------\n    x, y : array_like\n        The data points defining a curve y = f(x).\n    w : array_like, optional\n        Strictly positive rank-1 array of weights the same length as x and y.\n        The weights are used in computing the weighted least-squares spline\n        fit. If the errors in the y values have standard-deviation given by the\n        vector d, then w should be 1/d. Default is ones(len(x)).\n    xb, xe : float, optional\n        The interval to fit.  If None, these default to x[0] and x[-1]\n        respectively.\n    k : int, optional\n        The order of the spline fit. It is recommended to use cubic splines.\n        Even order splines should be avoided especially with small s values.\n        1 <= k <= 5\n    task : {1, 0, -1}, optional\n        If task==0 find t and c for a given smoothing factor, s.\n\n        If task==1 find t and c for another value of the smoothing factor, s.\n        There must have been a previous call with task=0 or task=1 for the same\n        set of data (t will be stored an used internally)\n\n        If task=-1 find the weighted least square spline for a given set of\n        knots, t. These should be interior knots as knots on the ends will be\n        added automatically.\n    s : float, optional\n        A smoothing condition. The amount of smoothness is determined by\n        satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x)\n        is the smoothed interpolation of (x,y). The user can use s to control\n        the tradeoff between closeness and smoothness of fit. Larger s means\n        more smoothing while smaller values of s indicate less smoothing.\n        Recommended values of s depend on the weights, w. If the weights\n        represent the inverse of the standard-deviation of y, then a good s\n        value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is\n        the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if\n        weights are supplied. s = 0.0 (interpolating) if no weights are\n        supplied.\n    t : array_like, optional\n        The knots needed for task=-1. If given then task is automatically set\n        to -1.\n    full_output : bool, optional\n        If non-zero, then return optional outputs.\n    per : bool, optional\n        If non-zero, data points are considered periodic with period x[m-1] -\n        x[0] and a smooth periodic spline approximation is returned. Values of\n        y[m-1] and w[m-1] are not used.\n    quiet : bool, optional\n        Non-zero to suppress messages.\n        This parameter is deprecated; use standard Python warning filters\n        instead.\n\n    Returns\n    -------\n    tck : tuple\n        (t,c,k) a tuple containing the vector of knots, the B-spline\n        coefficients, and the degree of the spline.\n    fp : array, optional\n        The weighted sum of squared residuals of the spline approximation.\n    ier : int, optional\n        An integer flag about splrep success. Success is indicated if ier<=0.\n        If ier in [1,2,3] an error occurred but was not raised. Otherwise an\n        error is raised.\n    msg : str, optional\n        A message corresponding to the integer flag, ier.\n\n    Notes\n    -----\n    See splev for evaluation of the spline and its derivatives.\n\n    The user is responsible for assuring that the values of *x* are unique.\n    Otherwise, *splrep* will not return sensible results.\n\n    See Also\n    --------\n    UnivariateSpline, BivariateSpline\n    splprep, splev, sproot, spalde, splint\n    bisplrep, bisplev\n\n    Notes\n    -----\n    See splev for evaluation of the spline and its derivatives. Uses the\n    FORTRAN routine curfit from FITPACK.\n\n    If provided, knots `t` must satisfy the Schoenberg-Whitney conditions,\n    i.e., there must be a subset of data points ``x[j]`` such that\n    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.\n\n    References\n    ----------\n    Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:\n\n    .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and\n       integration of experimental data using spline functions",\n       J.Comp.Appl.Maths 1 (1975) 165-184.\n    .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular\n       grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)\n       1286-1304.\n    .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline\n       functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.\n    .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on\n       Numerical Analysis, Oxford University Press, 1993.\n\n    Examples\n    --------\n\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.interpolate import splev, splrep\n    >>> x = np.linspace(0, 10, 10)\n    >>> y = np.sin(x)\n    >>> tck = splrep(x, y)\n    >>> x2 = np.linspace(0, 10, 200)\n    >>> y2 = splev(x2, tck)\n    >>> plt.plot(x, y, \'o\', x2, y2)\n    >>> plt.show()\n\n    ')
    
    
    # Getting the type of 'task' (line 442)
    task_79178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 7), 'task')
    int_79179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 15), 'int')
    # Applying the binary operator '<=' (line 442)
    result_le_79180 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 7), '<=', task_79178, int_79179)
    
    # Testing the type of an if condition (line 442)
    if_condition_79181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 4), result_le_79180)
    # Assigning a type to the variable 'if_condition_79181' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'if_condition_79181', if_condition_79181)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Dict to a Name (line 443):
    
    # Assigning a Dict to a Name (line 443):
    
    # Obtaining an instance of the builtin type 'dict' (line 443)
    dict_79182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 24), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 443)
    
    # Assigning a type to the variable '_curfit_cache' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), '_curfit_cache', dict_79182)
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 444):
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_79183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to map(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'atleast_1d' (line 444)
    atleast_1d_79185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 15), 'atleast_1d', False)
    
    # Obtaining an instance of the builtin type 'list' (line 444)
    list_79186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 444)
    # Adding element type (line 444)
    # Getting the type of 'x' (line 444)
    x_79187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 27), list_79186, x_79187)
    # Adding element type (line 444)
    # Getting the type of 'y' (line 444)
    y_79188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 27), list_79186, y_79188)
    
    # Processing the call keyword arguments (line 444)
    kwargs_79189 = {}
    # Getting the type of 'map' (line 444)
    map_79184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'map', False)
    # Calling map(args, kwargs) (line 444)
    map_call_result_79190 = invoke(stypy.reporting.localization.Localization(__file__, 444, 11), map_79184, *[atleast_1d_79185, list_79186], **kwargs_79189)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___79191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), map_call_result_79190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_79192 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___79191, int_79183)
    
    # Assigning a type to the variable 'tuple_var_assignment_78282' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_78282', subscript_call_result_79192)
    
    # Assigning a Subscript to a Name (line 444):
    
    # Obtaining the type of the subscript
    int_79193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'int')
    
    # Call to map(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'atleast_1d' (line 444)
    atleast_1d_79195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 15), 'atleast_1d', False)
    
    # Obtaining an instance of the builtin type 'list' (line 444)
    list_79196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 444)
    # Adding element type (line 444)
    # Getting the type of 'x' (line 444)
    x_79197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 27), list_79196, x_79197)
    # Adding element type (line 444)
    # Getting the type of 'y' (line 444)
    y_79198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 27), list_79196, y_79198)
    
    # Processing the call keyword arguments (line 444)
    kwargs_79199 = {}
    # Getting the type of 'map' (line 444)
    map_79194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'map', False)
    # Calling map(args, kwargs) (line 444)
    map_call_result_79200 = invoke(stypy.reporting.localization.Localization(__file__, 444, 11), map_79194, *[atleast_1d_79195, list_79196], **kwargs_79199)
    
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___79201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), map_call_result_79200, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_79202 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), getitem___79201, int_79193)
    
    # Assigning a type to the variable 'tuple_var_assignment_78283' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_78283', subscript_call_result_79202)
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_78282' (line 444)
    tuple_var_assignment_78282_79203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_78282')
    # Assigning a type to the variable 'x' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'x', tuple_var_assignment_78282_79203)
    
    # Assigning a Name to a Name (line 444):
    # Getting the type of 'tuple_var_assignment_78283' (line 444)
    tuple_var_assignment_78283_79204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'tuple_var_assignment_78283')
    # Assigning a type to the variable 'y' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 7), 'y', tuple_var_assignment_78283_79204)
    
    # Assigning a Call to a Name (line 445):
    
    # Assigning a Call to a Name (line 445):
    
    # Call to len(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'x' (line 445)
    x_79206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'x', False)
    # Processing the call keyword arguments (line 445)
    kwargs_79207 = {}
    # Getting the type of 'len' (line 445)
    len_79205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'len', False)
    # Calling len(args, kwargs) (line 445)
    len_call_result_79208 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), len_79205, *[x_79206], **kwargs_79207)
    
    # Assigning a type to the variable 'm' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'm', len_call_result_79208)
    
    # Type idiom detected: calculating its left and rigth part (line 446)
    # Getting the type of 'w' (line 446)
    w_79209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 7), 'w')
    # Getting the type of 'None' (line 446)
    None_79210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'None')
    
    (may_be_79211, more_types_in_union_79212) = may_be_none(w_79209, None_79210)

    if may_be_79211:

        if more_types_in_union_79212:
            # Runtime conditional SSA (line 446)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 447):
        
        # Assigning a Call to a Name (line 447):
        
        # Call to ones(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'm' (line 447)
        m_79214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 17), 'm', False)
        # Getting the type of 'float' (line 447)
        float_79215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 20), 'float', False)
        # Processing the call keyword arguments (line 447)
        kwargs_79216 = {}
        # Getting the type of 'ones' (line 447)
        ones_79213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 447)
        ones_call_result_79217 = invoke(stypy.reporting.localization.Localization(__file__, 447, 12), ones_79213, *[m_79214, float_79215], **kwargs_79216)
        
        # Assigning a type to the variable 'w' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'w', ones_call_result_79217)
        
        # Type idiom detected: calculating its left and rigth part (line 448)
        # Getting the type of 's' (line 448)
        s_79218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 's')
        # Getting the type of 'None' (line 448)
        None_79219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'None')
        
        (may_be_79220, more_types_in_union_79221) = may_be_none(s_79218, None_79219)

        if may_be_79220:

            if more_types_in_union_79221:
                # Runtime conditional SSA (line 448)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 449):
            
            # Assigning a Num to a Name (line 449):
            float_79222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 16), 'float')
            # Assigning a type to the variable 's' (line 449)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 's', float_79222)

            if more_types_in_union_79221:
                # SSA join for if statement (line 448)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_79212:
            # Runtime conditional SSA for else branch (line 446)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_79211) or more_types_in_union_79212):
        
        # Assigning a Call to a Name (line 451):
        
        # Assigning a Call to a Name (line 451):
        
        # Call to atleast_1d(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'w' (line 451)
        w_79224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'w', False)
        # Processing the call keyword arguments (line 451)
        kwargs_79225 = {}
        # Getting the type of 'atleast_1d' (line 451)
        atleast_1d_79223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 451)
        atleast_1d_call_result_79226 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), atleast_1d_79223, *[w_79224], **kwargs_79225)
        
        # Assigning a type to the variable 'w' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'w', atleast_1d_call_result_79226)
        
        # Type idiom detected: calculating its left and rigth part (line 452)
        # Getting the type of 's' (line 452)
        s_79227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 11), 's')
        # Getting the type of 'None' (line 452)
        None_79228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'None')
        
        (may_be_79229, more_types_in_union_79230) = may_be_none(s_79227, None_79228)

        if may_be_79229:

            if more_types_in_union_79230:
                # Runtime conditional SSA (line 452)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 453):
            
            # Assigning a BinOp to a Name (line 453):
            # Getting the type of 'm' (line 453)
            m_79231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'm')
            
            # Call to sqrt(...): (line 453)
            # Processing the call arguments (line 453)
            int_79233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 25), 'int')
            # Getting the type of 'm' (line 453)
            m_79234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 27), 'm', False)
            # Applying the binary operator '*' (line 453)
            result_mul_79235 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 25), '*', int_79233, m_79234)
            
            # Processing the call keyword arguments (line 453)
            kwargs_79236 = {}
            # Getting the type of 'sqrt' (line 453)
            sqrt_79232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 20), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 453)
            sqrt_call_result_79237 = invoke(stypy.reporting.localization.Localization(__file__, 453, 20), sqrt_79232, *[result_mul_79235], **kwargs_79236)
            
            # Applying the binary operator '-' (line 453)
            result_sub_79238 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 16), '-', m_79231, sqrt_call_result_79237)
            
            # Assigning a type to the variable 's' (line 453)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 's', result_sub_79238)

            if more_types_in_union_79230:
                # SSA join for if statement (line 452)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_79211 and more_types_in_union_79212):
            # SSA join for if statement (line 446)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    
    # Call to len(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'w' (line 454)
    w_79240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'w', False)
    # Processing the call keyword arguments (line 454)
    kwargs_79241 = {}
    # Getting the type of 'len' (line 454)
    len_79239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'len', False)
    # Calling len(args, kwargs) (line 454)
    len_call_result_79242 = invoke(stypy.reporting.localization.Localization(__file__, 454, 11), len_79239, *[w_79240], **kwargs_79241)
    
    # Getting the type of 'm' (line 454)
    m_79243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 21), 'm')
    # Applying the binary operator '==' (line 454)
    result_eq_79244 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 11), '==', len_call_result_79242, m_79243)
    
    # Applying the 'not' unary operator (line 454)
    result_not__79245 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 7), 'not', result_eq_79244)
    
    # Testing the type of an if condition (line 454)
    if_condition_79246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 4), result_not__79245)
    # Assigning a type to the variable 'if_condition_79246' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'if_condition_79246', if_condition_79246)
    # SSA begins for if statement (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 455)
    # Processing the call arguments (line 455)
    str_79248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 24), 'str', 'len(w)=%d is not equal to m=%d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 455)
    tuple_79249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 455)
    # Adding element type (line 455)
    
    # Call to len(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'w' (line 455)
    w_79251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 64), 'w', False)
    # Processing the call keyword arguments (line 455)
    kwargs_79252 = {}
    # Getting the type of 'len' (line 455)
    len_79250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 60), 'len', False)
    # Calling len(args, kwargs) (line 455)
    len_call_result_79253 = invoke(stypy.reporting.localization.Localization(__file__, 455, 60), len_79250, *[w_79251], **kwargs_79252)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 60), tuple_79249, len_call_result_79253)
    # Adding element type (line 455)
    # Getting the type of 'm' (line 455)
    m_79254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 68), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 60), tuple_79249, m_79254)
    
    # Applying the binary operator '%' (line 455)
    result_mod_79255 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 24), '%', str_79248, tuple_79249)
    
    # Processing the call keyword arguments (line 455)
    kwargs_79256 = {}
    # Getting the type of 'TypeError' (line 455)
    TypeError_79247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 455)
    TypeError_call_result_79257 = invoke(stypy.reporting.localization.Localization(__file__, 455, 14), TypeError_79247, *[result_mod_79255], **kwargs_79256)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 455, 8), TypeError_call_result_79257, 'raise parameter', BaseException)
    # SSA join for if statement (line 454)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'm' (line 456)
    m_79258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'm')
    
    # Call to len(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'y' (line 456)
    y_79260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'y', False)
    # Processing the call keyword arguments (line 456)
    kwargs_79261 = {}
    # Getting the type of 'len' (line 456)
    len_79259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 13), 'len', False)
    # Calling len(args, kwargs) (line 456)
    len_call_result_79262 = invoke(stypy.reporting.localization.Localization(__file__, 456, 13), len_79259, *[y_79260], **kwargs_79261)
    
    # Applying the binary operator '!=' (line 456)
    result_ne_79263 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 8), '!=', m_79258, len_call_result_79262)
    
    
    # Getting the type of 'm' (line 456)
    m_79264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 25), 'm')
    
    # Call to len(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'w' (line 456)
    w_79266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 34), 'w', False)
    # Processing the call keyword arguments (line 456)
    kwargs_79267 = {}
    # Getting the type of 'len' (line 456)
    len_79265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'len', False)
    # Calling len(args, kwargs) (line 456)
    len_call_result_79268 = invoke(stypy.reporting.localization.Localization(__file__, 456, 30), len_79265, *[w_79266], **kwargs_79267)
    
    # Applying the binary operator '!=' (line 456)
    result_ne_79269 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 25), '!=', m_79264, len_call_result_79268)
    
    # Applying the binary operator 'or' (line 456)
    result_or_keyword_79270 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 7), 'or', result_ne_79263, result_ne_79269)
    
    # Testing the type of an if condition (line 456)
    if_condition_79271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 4), result_or_keyword_79270)
    # Assigning a type to the variable 'if_condition_79271' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'if_condition_79271', if_condition_79271)
    # SSA begins for if statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 457)
    # Processing the call arguments (line 457)
    str_79273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 24), 'str', 'Lengths of the first three arguments (x,y,w) must be equal')
    # Processing the call keyword arguments (line 457)
    kwargs_79274 = {}
    # Getting the type of 'TypeError' (line 457)
    TypeError_79272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 457)
    TypeError_call_result_79275 = invoke(stypy.reporting.localization.Localization(__file__, 457, 14), TypeError_79272, *[str_79273], **kwargs_79274)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 457, 8), TypeError_call_result_79275, 'raise parameter', BaseException)
    # SSA join for if statement (line 456)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    int_79276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 12), 'int')
    # Getting the type of 'k' (line 459)
    k_79277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'k')
    # Applying the binary operator '<=' (line 459)
    result_le_79278 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 12), '<=', int_79276, k_79277)
    int_79279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 22), 'int')
    # Applying the binary operator '<=' (line 459)
    result_le_79280 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 12), '<=', k_79277, int_79279)
    # Applying the binary operator '&' (line 459)
    result_and__79281 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 12), '&', result_le_79278, result_le_79280)
    
    # Applying the 'not' unary operator (line 459)
    result_not__79282 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 7), 'not', result_and__79281)
    
    # Testing the type of an if condition (line 459)
    if_condition_79283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 4), result_not__79282)
    # Assigning a type to the variable 'if_condition_79283' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'if_condition_79283', if_condition_79283)
    # SSA begins for if statement (line 459)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 460)
    # Processing the call arguments (line 460)
    str_79285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 24), 'str', 'Given degree of the spline (k=%d) is not supported. (1<=k<=5)')
    # Getting the type of 'k' (line 461)
    k_79286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 38), 'k', False)
    # Applying the binary operator '%' (line 460)
    result_mod_79287 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 24), '%', str_79285, k_79286)
    
    # Processing the call keyword arguments (line 460)
    kwargs_79288 = {}
    # Getting the type of 'TypeError' (line 460)
    TypeError_79284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 460)
    TypeError_call_result_79289 = invoke(stypy.reporting.localization.Localization(__file__, 460, 14), TypeError_79284, *[result_mod_79287], **kwargs_79288)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 460, 8), TypeError_call_result_79289, 'raise parameter', BaseException)
    # SSA join for if statement (line 459)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 462)
    m_79290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 7), 'm')
    # Getting the type of 'k' (line 462)
    k_79291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'k')
    # Applying the binary operator '<=' (line 462)
    result_le_79292 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 7), '<=', m_79290, k_79291)
    
    # Testing the type of an if condition (line 462)
    if_condition_79293 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 4), result_le_79292)
    # Assigning a type to the variable 'if_condition_79293' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'if_condition_79293', if_condition_79293)
    # SSA begins for if statement (line 462)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 463)
    # Processing the call arguments (line 463)
    str_79295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 24), 'str', 'm > k must hold')
    # Processing the call keyword arguments (line 463)
    kwargs_79296 = {}
    # Getting the type of 'TypeError' (line 463)
    TypeError_79294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 463)
    TypeError_call_result_79297 = invoke(stypy.reporting.localization.Localization(__file__, 463, 14), TypeError_79294, *[str_79295], **kwargs_79296)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 463, 8), TypeError_call_result_79297, 'raise parameter', BaseException)
    # SSA join for if statement (line 462)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 464)
    # Getting the type of 'xb' (line 464)
    xb_79298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 7), 'xb')
    # Getting the type of 'None' (line 464)
    None_79299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 13), 'None')
    
    (may_be_79300, more_types_in_union_79301) = may_be_none(xb_79298, None_79299)

    if may_be_79300:

        if more_types_in_union_79301:
            # Runtime conditional SSA (line 464)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 465):
        
        # Assigning a Subscript to a Name (line 465):
        
        # Obtaining the type of the subscript
        int_79302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 15), 'int')
        # Getting the type of 'x' (line 465)
        x_79303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 13), 'x')
        # Obtaining the member '__getitem__' of a type (line 465)
        getitem___79304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 13), x_79303, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 465)
        subscript_call_result_79305 = invoke(stypy.reporting.localization.Localization(__file__, 465, 13), getitem___79304, int_79302)
        
        # Assigning a type to the variable 'xb' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'xb', subscript_call_result_79305)

        if more_types_in_union_79301:
            # SSA join for if statement (line 464)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 466)
    # Getting the type of 'xe' (line 466)
    xe_79306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 7), 'xe')
    # Getting the type of 'None' (line 466)
    None_79307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 13), 'None')
    
    (may_be_79308, more_types_in_union_79309) = may_be_none(xe_79306, None_79307)

    if may_be_79308:

        if more_types_in_union_79309:
            # Runtime conditional SSA (line 466)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 467):
        
        # Assigning a Subscript to a Name (line 467):
        
        # Obtaining the type of the subscript
        int_79310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 15), 'int')
        # Getting the type of 'x' (line 467)
        x_79311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 13), 'x')
        # Obtaining the member '__getitem__' of a type (line 467)
        getitem___79312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 13), x_79311, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 467)
        subscript_call_result_79313 = invoke(stypy.reporting.localization.Localization(__file__, 467, 13), getitem___79312, int_79310)
        
        # Assigning a type to the variable 'xe' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'xe', subscript_call_result_79313)

        if more_types_in_union_79309:
            # SSA join for if statement (line 466)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    int_79314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 12), 'int')
    # Getting the type of 'task' (line 468)
    task_79315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 18), 'task')
    # Applying the binary operator '<=' (line 468)
    result_le_79316 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 12), '<=', int_79314, task_79315)
    int_79317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 26), 'int')
    # Applying the binary operator '<=' (line 468)
    result_le_79318 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 12), '<=', task_79315, int_79317)
    # Applying the binary operator '&' (line 468)
    result_and__79319 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 12), '&', result_le_79316, result_le_79318)
    
    # Applying the 'not' unary operator (line 468)
    result_not__79320 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 7), 'not', result_and__79319)
    
    # Testing the type of an if condition (line 468)
    if_condition_79321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 4), result_not__79320)
    # Assigning a type to the variable 'if_condition_79321' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'if_condition_79321', if_condition_79321)
    # SSA begins for if statement (line 468)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 469)
    # Processing the call arguments (line 469)
    str_79323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 24), 'str', 'task must be -1, 0 or 1')
    # Processing the call keyword arguments (line 469)
    kwargs_79324 = {}
    # Getting the type of 'TypeError' (line 469)
    TypeError_79322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 469)
    TypeError_call_result_79325 = invoke(stypy.reporting.localization.Localization(__file__, 469, 14), TypeError_79322, *[str_79323], **kwargs_79324)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 469, 8), TypeError_call_result_79325, 'raise parameter', BaseException)
    # SSA join for if statement (line 468)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 470)
    # Getting the type of 't' (line 470)
    t_79326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 't')
    # Getting the type of 'None' (line 470)
    None_79327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 16), 'None')
    
    (may_be_79328, more_types_in_union_79329) = may_not_be_none(t_79326, None_79327)

    if may_be_79328:

        if more_types_in_union_79329:
            # Runtime conditional SSA (line 470)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 471):
        
        # Assigning a Num to a Name (line 471):
        int_79330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 15), 'int')
        # Assigning a type to the variable 'task' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'task', int_79330)

        if more_types_in_union_79329:
            # SSA join for if statement (line 470)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'task' (line 472)
    task_79331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 7), 'task')
    int_79332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 15), 'int')
    # Applying the binary operator '==' (line 472)
    result_eq_79333 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 7), '==', task_79331, int_79332)
    
    # Testing the type of an if condition (line 472)
    if_condition_79334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 4), result_eq_79333)
    # Assigning a type to the variable 'if_condition_79334' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'if_condition_79334', if_condition_79334)
    # SSA begins for if statement (line 472)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 473)
    # Getting the type of 't' (line 473)
    t_79335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 11), 't')
    # Getting the type of 'None' (line 473)
    None_79336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 16), 'None')
    
    (may_be_79337, more_types_in_union_79338) = may_be_none(t_79335, None_79336)

    if may_be_79337:

        if more_types_in_union_79338:
            # Runtime conditional SSA (line 473)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to TypeError(...): (line 474)
        # Processing the call arguments (line 474)
        str_79340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 28), 'str', 'Knots must be given for task=-1')
        # Processing the call keyword arguments (line 474)
        kwargs_79341 = {}
        # Getting the type of 'TypeError' (line 474)
        TypeError_79339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 474)
        TypeError_call_result_79342 = invoke(stypy.reporting.localization.Localization(__file__, 474, 18), TypeError_79339, *[str_79340], **kwargs_79341)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 474, 12), TypeError_call_result_79342, 'raise parameter', BaseException)

        if more_types_in_union_79338:
            # SSA join for if statement (line 473)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to len(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 't' (line 475)
    t_79344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 23), 't', False)
    # Processing the call keyword arguments (line 475)
    kwargs_79345 = {}
    # Getting the type of 'len' (line 475)
    len_79343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'len', False)
    # Calling len(args, kwargs) (line 475)
    len_call_result_79346 = invoke(stypy.reporting.localization.Localization(__file__, 475, 19), len_79343, *[t_79344], **kwargs_79345)
    
    # Assigning a type to the variable 'numknots' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'numknots', len_call_result_79346)
    
    # Assigning a Call to a Subscript (line 476):
    
    # Assigning a Call to a Subscript (line 476):
    
    # Call to empty(...): (line 476)
    # Processing the call arguments (line 476)
    
    # Obtaining an instance of the builtin type 'tuple' (line 476)
    tuple_79348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 476)
    # Adding element type (line 476)
    # Getting the type of 'numknots' (line 476)
    numknots_79349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 36), 'numknots', False)
    int_79350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 47), 'int')
    # Getting the type of 'k' (line 476)
    k_79351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 49), 'k', False)
    # Applying the binary operator '*' (line 476)
    result_mul_79352 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 47), '*', int_79350, k_79351)
    
    # Applying the binary operator '+' (line 476)
    result_add_79353 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 36), '+', numknots_79349, result_mul_79352)
    
    int_79354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 53), 'int')
    # Applying the binary operator '+' (line 476)
    result_add_79355 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 51), '+', result_add_79353, int_79354)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 36), tuple_79348, result_add_79355)
    
    # Getting the type of 'float' (line 476)
    float_79356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 58), 'float', False)
    # Processing the call keyword arguments (line 476)
    kwargs_79357 = {}
    # Getting the type of 'empty' (line 476)
    empty_79347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 29), 'empty', False)
    # Calling empty(args, kwargs) (line 476)
    empty_call_result_79358 = invoke(stypy.reporting.localization.Localization(__file__, 476, 29), empty_79347, *[tuple_79348, float_79356], **kwargs_79357)
    
    # Getting the type of '_curfit_cache' (line 476)
    _curfit_cache_79359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), '_curfit_cache')
    str_79360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 22), 'str', 't')
    # Storing an element on a container (line 476)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 8), _curfit_cache_79359, (str_79360, empty_call_result_79358))
    
    # Assigning a Name to a Subscript (line 477):
    
    # Assigning a Name to a Subscript (line 477):
    # Getting the type of 't' (line 477)
    t_79361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 39), 't')
    
    # Obtaining the type of the subscript
    str_79362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 22), 'str', 't')
    # Getting the type of '_curfit_cache' (line 477)
    _curfit_cache_79363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), '_curfit_cache')
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___79364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), _curfit_cache_79363, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_79365 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___79364, str_79362)
    
    # Getting the type of 'k' (line 477)
    k_79366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 27), 'k')
    int_79367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 29), 'int')
    # Applying the binary operator '+' (line 477)
    result_add_79368 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 27), '+', k_79366, int_79367)
    
    
    # Getting the type of 'k' (line 477)
    k_79369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 32), 'k')
    # Applying the 'usub' unary operator (line 477)
    result___neg___79370 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 31), 'usub', k_79369)
    
    int_79371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 34), 'int')
    # Applying the binary operator '-' (line 477)
    result_sub_79372 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 31), '-', result___neg___79370, int_79371)
    
    slice_79373 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 477, 8), result_add_79368, result_sub_79372, None)
    # Storing an element on a container (line 477)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 8), subscript_call_result_79365, (slice_79373, t_79361))
    
    # Assigning a Call to a Name (line 478):
    
    # Assigning a Call to a Name (line 478):
    
    # Call to len(...): (line 478)
    # Processing the call arguments (line 478)
    
    # Obtaining the type of the subscript
    str_79375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 33), 'str', 't')
    # Getting the type of '_curfit_cache' (line 478)
    _curfit_cache_79376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 19), '_curfit_cache', False)
    # Obtaining the member '__getitem__' of a type (line 478)
    getitem___79377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 19), _curfit_cache_79376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 478)
    subscript_call_result_79378 = invoke(stypy.reporting.localization.Localization(__file__, 478, 19), getitem___79377, str_79375)
    
    # Processing the call keyword arguments (line 478)
    kwargs_79379 = {}
    # Getting the type of 'len' (line 478)
    len_79374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 15), 'len', False)
    # Calling len(args, kwargs) (line 478)
    len_call_result_79380 = invoke(stypy.reporting.localization.Localization(__file__, 478, 15), len_79374, *[subscript_call_result_79378], **kwargs_79379)
    
    # Assigning a type to the variable 'nest' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'nest', len_call_result_79380)
    # SSA branch for the else part of an if statement (line 472)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'task' (line 479)
    task_79381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 9), 'task')
    int_79382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 17), 'int')
    # Applying the binary operator '==' (line 479)
    result_eq_79383 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 9), '==', task_79381, int_79382)
    
    # Testing the type of an if condition (line 479)
    if_condition_79384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 9), result_eq_79383)
    # Assigning a type to the variable 'if_condition_79384' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 9), 'if_condition_79384', if_condition_79384)
    # SSA begins for if statement (line 479)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'per' (line 480)
    per_79385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 11), 'per')
    # Testing the type of an if condition (line 480)
    if_condition_79386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 8), per_79385)
    # Assigning a type to the variable 'if_condition_79386' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'if_condition_79386', if_condition_79386)
    # SSA begins for if statement (line 480)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 481):
    
    # Assigning a Call to a Name (line 481):
    
    # Call to max(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'm' (line 481)
    m_79388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 23), 'm', False)
    int_79389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 27), 'int')
    # Getting the type of 'k' (line 481)
    k_79390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 29), 'k', False)
    # Applying the binary operator '*' (line 481)
    result_mul_79391 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 27), '*', int_79389, k_79390)
    
    # Applying the binary operator '+' (line 481)
    result_add_79392 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 23), '+', m_79388, result_mul_79391)
    
    int_79393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 32), 'int')
    # Getting the type of 'k' (line 481)
    k_79394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 34), 'k', False)
    # Applying the binary operator '*' (line 481)
    result_mul_79395 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 32), '*', int_79393, k_79394)
    
    int_79396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 38), 'int')
    # Applying the binary operator '+' (line 481)
    result_add_79397 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 32), '+', result_mul_79395, int_79396)
    
    # Processing the call keyword arguments (line 481)
    kwargs_79398 = {}
    # Getting the type of 'max' (line 481)
    max_79387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 19), 'max', False)
    # Calling max(args, kwargs) (line 481)
    max_call_result_79399 = invoke(stypy.reporting.localization.Localization(__file__, 481, 19), max_79387, *[result_add_79392, result_add_79397], **kwargs_79398)
    
    # Assigning a type to the variable 'nest' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'nest', max_call_result_79399)
    # SSA branch for the else part of an if statement (line 480)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 483):
    
    # Assigning a Call to a Name (line 483):
    
    # Call to max(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'm' (line 483)
    m_79401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 23), 'm', False)
    # Getting the type of 'k' (line 483)
    k_79402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 27), 'k', False)
    # Applying the binary operator '+' (line 483)
    result_add_79403 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 23), '+', m_79401, k_79402)
    
    int_79404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 31), 'int')
    # Applying the binary operator '+' (line 483)
    result_add_79405 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 29), '+', result_add_79403, int_79404)
    
    int_79406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 34), 'int')
    # Getting the type of 'k' (line 483)
    k_79407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 36), 'k', False)
    # Applying the binary operator '*' (line 483)
    result_mul_79408 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 34), '*', int_79406, k_79407)
    
    int_79409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 40), 'int')
    # Applying the binary operator '+' (line 483)
    result_add_79410 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 34), '+', result_mul_79408, int_79409)
    
    # Processing the call keyword arguments (line 483)
    kwargs_79411 = {}
    # Getting the type of 'max' (line 483)
    max_79400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 19), 'max', False)
    # Calling max(args, kwargs) (line 483)
    max_call_result_79412 = invoke(stypy.reporting.localization.Localization(__file__, 483, 19), max_79400, *[result_add_79405, result_add_79410], **kwargs_79411)
    
    # Assigning a type to the variable 'nest' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'nest', max_call_result_79412)
    # SSA join for if statement (line 480)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 484):
    
    # Assigning a Call to a Name (line 484):
    
    # Call to empty(...): (line 484)
    # Processing the call arguments (line 484)
    
    # Obtaining an instance of the builtin type 'tuple' (line 484)
    tuple_79414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 484)
    # Adding element type (line 484)
    # Getting the type of 'nest' (line 484)
    nest_79415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 19), 'nest', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 19), tuple_79414, nest_79415)
    
    # Getting the type of 'float' (line 484)
    float_79416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 27), 'float', False)
    # Processing the call keyword arguments (line 484)
    kwargs_79417 = {}
    # Getting the type of 'empty' (line 484)
    empty_79413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'empty', False)
    # Calling empty(args, kwargs) (line 484)
    empty_call_result_79418 = invoke(stypy.reporting.localization.Localization(__file__, 484, 12), empty_79413, *[tuple_79414, float_79416], **kwargs_79417)
    
    # Assigning a type to the variable 't' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 't', empty_call_result_79418)
    
    # Assigning a Name to a Subscript (line 485):
    
    # Assigning a Name to a Subscript (line 485):
    # Getting the type of 't' (line 485)
    t_79419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 29), 't')
    # Getting the type of '_curfit_cache' (line 485)
    _curfit_cache_79420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), '_curfit_cache')
    str_79421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 22), 'str', 't')
    # Storing an element on a container (line 485)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 8), _curfit_cache_79420, (str_79421, t_79419))
    # SSA join for if statement (line 479)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 472)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'task' (line 486)
    task_79422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 7), 'task')
    int_79423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 15), 'int')
    # Applying the binary operator '<=' (line 486)
    result_le_79424 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 7), '<=', task_79422, int_79423)
    
    # Testing the type of an if condition (line 486)
    if_condition_79425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 4), result_le_79424)
    # Assigning a type to the variable 'if_condition_79425' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'if_condition_79425', if_condition_79425)
    # SSA begins for if statement (line 486)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'per' (line 487)
    per_79426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 11), 'per')
    # Testing the type of an if condition (line 487)
    if_condition_79427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 487, 8), per_79426)
    # Assigning a type to the variable 'if_condition_79427' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'if_condition_79427', if_condition_79427)
    # SSA begins for if statement (line 487)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 488):
    
    # Assigning a Call to a Subscript (line 488):
    
    # Call to empty(...): (line 488)
    # Processing the call arguments (line 488)
    
    # Obtaining an instance of the builtin type 'tuple' (line 488)
    tuple_79429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 488)
    # Adding element type (line 488)
    # Getting the type of 'm' (line 488)
    m_79430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 42), 'm', False)
    # Getting the type of 'k' (line 488)
    k_79431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 45), 'k', False)
    int_79432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 49), 'int')
    # Applying the binary operator '+' (line 488)
    result_add_79433 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 45), '+', k_79431, int_79432)
    
    # Applying the binary operator '*' (line 488)
    result_mul_79434 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 42), '*', m_79430, result_add_79433)
    
    # Getting the type of 'nest' (line 488)
    nest_79435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 54), 'nest', False)
    int_79436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 60), 'int')
    int_79437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 64), 'int')
    # Getting the type of 'k' (line 488)
    k_79438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 66), 'k', False)
    # Applying the binary operator '*' (line 488)
    result_mul_79439 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 64), '*', int_79437, k_79438)
    
    # Applying the binary operator '+' (line 488)
    result_add_79440 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 60), '+', int_79436, result_mul_79439)
    
    # Applying the binary operator '*' (line 488)
    result_mul_79441 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 54), '*', nest_79435, result_add_79440)
    
    # Applying the binary operator '+' (line 488)
    result_add_79442 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 42), '+', result_mul_79434, result_mul_79441)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 42), tuple_79429, result_add_79442)
    
    # Getting the type of 'float' (line 488)
    float_79443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 72), 'float', False)
    # Processing the call keyword arguments (line 488)
    kwargs_79444 = {}
    # Getting the type of 'empty' (line 488)
    empty_79428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 35), 'empty', False)
    # Calling empty(args, kwargs) (line 488)
    empty_call_result_79445 = invoke(stypy.reporting.localization.Localization(__file__, 488, 35), empty_79428, *[tuple_79429, float_79443], **kwargs_79444)
    
    # Getting the type of '_curfit_cache' (line 488)
    _curfit_cache_79446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), '_curfit_cache')
    str_79447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 26), 'str', 'wrk')
    # Storing an element on a container (line 488)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 12), _curfit_cache_79446, (str_79447, empty_call_result_79445))
    # SSA branch for the else part of an if statement (line 487)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Subscript (line 490):
    
    # Assigning a Call to a Subscript (line 490):
    
    # Call to empty(...): (line 490)
    # Processing the call arguments (line 490)
    
    # Obtaining an instance of the builtin type 'tuple' (line 490)
    tuple_79449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 490)
    # Adding element type (line 490)
    # Getting the type of 'm' (line 490)
    m_79450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 42), 'm', False)
    # Getting the type of 'k' (line 490)
    k_79451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 45), 'k', False)
    int_79452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 49), 'int')
    # Applying the binary operator '+' (line 490)
    result_add_79453 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 45), '+', k_79451, int_79452)
    
    # Applying the binary operator '*' (line 490)
    result_mul_79454 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 42), '*', m_79450, result_add_79453)
    
    # Getting the type of 'nest' (line 490)
    nest_79455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 54), 'nest', False)
    int_79456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 60), 'int')
    int_79457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 64), 'int')
    # Getting the type of 'k' (line 490)
    k_79458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 66), 'k', False)
    # Applying the binary operator '*' (line 490)
    result_mul_79459 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 64), '*', int_79457, k_79458)
    
    # Applying the binary operator '+' (line 490)
    result_add_79460 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 60), '+', int_79456, result_mul_79459)
    
    # Applying the binary operator '*' (line 490)
    result_mul_79461 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 54), '*', nest_79455, result_add_79460)
    
    # Applying the binary operator '+' (line 490)
    result_add_79462 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 42), '+', result_mul_79454, result_mul_79461)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 42), tuple_79449, result_add_79462)
    
    # Getting the type of 'float' (line 490)
    float_79463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 72), 'float', False)
    # Processing the call keyword arguments (line 490)
    kwargs_79464 = {}
    # Getting the type of 'empty' (line 490)
    empty_79448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 35), 'empty', False)
    # Calling empty(args, kwargs) (line 490)
    empty_call_result_79465 = invoke(stypy.reporting.localization.Localization(__file__, 490, 35), empty_79448, *[tuple_79449, float_79463], **kwargs_79464)
    
    # Getting the type of '_curfit_cache' (line 490)
    _curfit_cache_79466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), '_curfit_cache')
    str_79467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 26), 'str', 'wrk')
    # Storing an element on a container (line 490)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 12), _curfit_cache_79466, (str_79467, empty_call_result_79465))
    # SSA join for if statement (line 487)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 491):
    
    # Assigning a Call to a Subscript (line 491):
    
    # Call to empty(...): (line 491)
    # Processing the call arguments (line 491)
    
    # Obtaining an instance of the builtin type 'tuple' (line 491)
    tuple_79469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 491)
    # Adding element type (line 491)
    # Getting the type of 'nest' (line 491)
    nest_79470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 39), 'nest', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 39), tuple_79469, nest_79470)
    
    # Getting the type of 'intc' (line 491)
    intc_79471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 47), 'intc', False)
    # Processing the call keyword arguments (line 491)
    kwargs_79472 = {}
    # Getting the type of 'empty' (line 491)
    empty_79468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 32), 'empty', False)
    # Calling empty(args, kwargs) (line 491)
    empty_call_result_79473 = invoke(stypy.reporting.localization.Localization(__file__, 491, 32), empty_79468, *[tuple_79469, intc_79471], **kwargs_79472)
    
    # Getting the type of '_curfit_cache' (line 491)
    _curfit_cache_79474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), '_curfit_cache')
    str_79475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 22), 'str', 'iwrk')
    # Storing an element on a container (line 491)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 8), _curfit_cache_79474, (str_79475, empty_call_result_79473))
    # SSA join for if statement (line 486)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 492)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 493):
    
    # Assigning a Subscript to a Name (line 493):
    
    # Obtaining the type of the subscript
    str_79476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 26), 'str', 't')
    # Getting the type of '_curfit_cache' (line 493)
    _curfit_cache_79477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), '_curfit_cache')
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___79478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 12), _curfit_cache_79477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_79479 = invoke(stypy.reporting.localization.Localization(__file__, 493, 12), getitem___79478, str_79476)
    
    # Assigning a type to the variable 't' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 't', subscript_call_result_79479)
    
    # Assigning a Subscript to a Name (line 494):
    
    # Assigning a Subscript to a Name (line 494):
    
    # Obtaining the type of the subscript
    str_79480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 28), 'str', 'wrk')
    # Getting the type of '_curfit_cache' (line 494)
    _curfit_cache_79481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 14), '_curfit_cache')
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___79482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 14), _curfit_cache_79481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 494)
    subscript_call_result_79483 = invoke(stypy.reporting.localization.Localization(__file__, 494, 14), getitem___79482, str_79480)
    
    # Assigning a type to the variable 'wrk' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'wrk', subscript_call_result_79483)
    
    # Assigning a Subscript to a Name (line 495):
    
    # Assigning a Subscript to a Name (line 495):
    
    # Obtaining the type of the subscript
    str_79484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 29), 'str', 'iwrk')
    # Getting the type of '_curfit_cache' (line 495)
    _curfit_cache_79485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 15), '_curfit_cache')
    # Obtaining the member '__getitem__' of a type (line 495)
    getitem___79486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 15), _curfit_cache_79485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 495)
    subscript_call_result_79487 = invoke(stypy.reporting.localization.Localization(__file__, 495, 15), getitem___79486, str_79484)
    
    # Assigning a type to the variable 'iwrk' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'iwrk', subscript_call_result_79487)
    # SSA branch for the except part of a try statement (line 492)
    # SSA branch for the except 'KeyError' branch of a try statement (line 492)
    module_type_store.open_ssa_branch('except')
    
    # Call to TypeError(...): (line 497)
    # Processing the call arguments (line 497)
    str_79489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 24), 'str', 'must call with task=1 only after call with task=0,-1')
    # Processing the call keyword arguments (line 497)
    kwargs_79490 = {}
    # Getting the type of 'TypeError' (line 497)
    TypeError_79488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 497)
    TypeError_call_result_79491 = invoke(stypy.reporting.localization.Localization(__file__, 497, 14), TypeError_79488, *[str_79489], **kwargs_79490)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 497, 8), TypeError_call_result_79491, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 492)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'per' (line 499)
    per_79492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'per')
    # Applying the 'not' unary operator (line 499)
    result_not__79493 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 7), 'not', per_79492)
    
    # Testing the type of an if condition (line 499)
    if_condition_79494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 4), result_not__79493)
    # Assigning a type to the variable 'if_condition_79494' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'if_condition_79494', if_condition_79494)
    # SSA begins for if statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 500):
    
    # Assigning a Subscript to a Name (line 500):
    
    # Obtaining the type of the subscript
    int_79495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 8), 'int')
    
    # Call to curfit(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'task' (line 500)
    task_79498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 40), 'task', False)
    # Getting the type of 'x' (line 500)
    x_79499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 46), 'x', False)
    # Getting the type of 'y' (line 500)
    y_79500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 49), 'y', False)
    # Getting the type of 'w' (line 500)
    w_79501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 52), 'w', False)
    # Getting the type of 't' (line 500)
    t_79502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 55), 't', False)
    # Getting the type of 'wrk' (line 500)
    wrk_79503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 58), 'wrk', False)
    # Getting the type of 'iwrk' (line 500)
    iwrk_79504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 63), 'iwrk', False)
    # Getting the type of 'xb' (line 501)
    xb_79505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'xb', False)
    # Getting the type of 'xe' (line 501)
    xe_79506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 44), 'xe', False)
    # Getting the type of 'k' (line 501)
    k_79507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 48), 'k', False)
    # Getting the type of 's' (line 501)
    s_79508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 51), 's', False)
    # Processing the call keyword arguments (line 500)
    kwargs_79509 = {}
    # Getting the type of 'dfitpack' (line 500)
    dfitpack_79496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 24), 'dfitpack', False)
    # Obtaining the member 'curfit' of a type (line 500)
    curfit_79497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 24), dfitpack_79496, 'curfit')
    # Calling curfit(args, kwargs) (line 500)
    curfit_call_result_79510 = invoke(stypy.reporting.localization.Localization(__file__, 500, 24), curfit_79497, *[task_79498, x_79499, y_79500, w_79501, t_79502, wrk_79503, iwrk_79504, xb_79505, xe_79506, k_79507, s_79508], **kwargs_79509)
    
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___79511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), curfit_call_result_79510, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_79512 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), getitem___79511, int_79495)
    
    # Assigning a type to the variable 'tuple_var_assignment_78284' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple_var_assignment_78284', subscript_call_result_79512)
    
    # Assigning a Subscript to a Name (line 500):
    
    # Obtaining the type of the subscript
    int_79513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 8), 'int')
    
    # Call to curfit(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'task' (line 500)
    task_79516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 40), 'task', False)
    # Getting the type of 'x' (line 500)
    x_79517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 46), 'x', False)
    # Getting the type of 'y' (line 500)
    y_79518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 49), 'y', False)
    # Getting the type of 'w' (line 500)
    w_79519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 52), 'w', False)
    # Getting the type of 't' (line 500)
    t_79520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 55), 't', False)
    # Getting the type of 'wrk' (line 500)
    wrk_79521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 58), 'wrk', False)
    # Getting the type of 'iwrk' (line 500)
    iwrk_79522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 63), 'iwrk', False)
    # Getting the type of 'xb' (line 501)
    xb_79523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'xb', False)
    # Getting the type of 'xe' (line 501)
    xe_79524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 44), 'xe', False)
    # Getting the type of 'k' (line 501)
    k_79525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 48), 'k', False)
    # Getting the type of 's' (line 501)
    s_79526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 51), 's', False)
    # Processing the call keyword arguments (line 500)
    kwargs_79527 = {}
    # Getting the type of 'dfitpack' (line 500)
    dfitpack_79514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 24), 'dfitpack', False)
    # Obtaining the member 'curfit' of a type (line 500)
    curfit_79515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 24), dfitpack_79514, 'curfit')
    # Calling curfit(args, kwargs) (line 500)
    curfit_call_result_79528 = invoke(stypy.reporting.localization.Localization(__file__, 500, 24), curfit_79515, *[task_79516, x_79517, y_79518, w_79519, t_79520, wrk_79521, iwrk_79522, xb_79523, xe_79524, k_79525, s_79526], **kwargs_79527)
    
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___79529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), curfit_call_result_79528, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_79530 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), getitem___79529, int_79513)
    
    # Assigning a type to the variable 'tuple_var_assignment_78285' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple_var_assignment_78285', subscript_call_result_79530)
    
    # Assigning a Subscript to a Name (line 500):
    
    # Obtaining the type of the subscript
    int_79531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 8), 'int')
    
    # Call to curfit(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'task' (line 500)
    task_79534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 40), 'task', False)
    # Getting the type of 'x' (line 500)
    x_79535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 46), 'x', False)
    # Getting the type of 'y' (line 500)
    y_79536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 49), 'y', False)
    # Getting the type of 'w' (line 500)
    w_79537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 52), 'w', False)
    # Getting the type of 't' (line 500)
    t_79538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 55), 't', False)
    # Getting the type of 'wrk' (line 500)
    wrk_79539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 58), 'wrk', False)
    # Getting the type of 'iwrk' (line 500)
    iwrk_79540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 63), 'iwrk', False)
    # Getting the type of 'xb' (line 501)
    xb_79541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'xb', False)
    # Getting the type of 'xe' (line 501)
    xe_79542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 44), 'xe', False)
    # Getting the type of 'k' (line 501)
    k_79543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 48), 'k', False)
    # Getting the type of 's' (line 501)
    s_79544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 51), 's', False)
    # Processing the call keyword arguments (line 500)
    kwargs_79545 = {}
    # Getting the type of 'dfitpack' (line 500)
    dfitpack_79532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 24), 'dfitpack', False)
    # Obtaining the member 'curfit' of a type (line 500)
    curfit_79533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 24), dfitpack_79532, 'curfit')
    # Calling curfit(args, kwargs) (line 500)
    curfit_call_result_79546 = invoke(stypy.reporting.localization.Localization(__file__, 500, 24), curfit_79533, *[task_79534, x_79535, y_79536, w_79537, t_79538, wrk_79539, iwrk_79540, xb_79541, xe_79542, k_79543, s_79544], **kwargs_79545)
    
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___79547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), curfit_call_result_79546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_79548 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), getitem___79547, int_79531)
    
    # Assigning a type to the variable 'tuple_var_assignment_78286' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple_var_assignment_78286', subscript_call_result_79548)
    
    # Assigning a Subscript to a Name (line 500):
    
    # Obtaining the type of the subscript
    int_79549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 8), 'int')
    
    # Call to curfit(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'task' (line 500)
    task_79552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 40), 'task', False)
    # Getting the type of 'x' (line 500)
    x_79553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 46), 'x', False)
    # Getting the type of 'y' (line 500)
    y_79554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 49), 'y', False)
    # Getting the type of 'w' (line 500)
    w_79555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 52), 'w', False)
    # Getting the type of 't' (line 500)
    t_79556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 55), 't', False)
    # Getting the type of 'wrk' (line 500)
    wrk_79557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 58), 'wrk', False)
    # Getting the type of 'iwrk' (line 500)
    iwrk_79558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 63), 'iwrk', False)
    # Getting the type of 'xb' (line 501)
    xb_79559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'xb', False)
    # Getting the type of 'xe' (line 501)
    xe_79560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 44), 'xe', False)
    # Getting the type of 'k' (line 501)
    k_79561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 48), 'k', False)
    # Getting the type of 's' (line 501)
    s_79562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 51), 's', False)
    # Processing the call keyword arguments (line 500)
    kwargs_79563 = {}
    # Getting the type of 'dfitpack' (line 500)
    dfitpack_79550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 24), 'dfitpack', False)
    # Obtaining the member 'curfit' of a type (line 500)
    curfit_79551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 24), dfitpack_79550, 'curfit')
    # Calling curfit(args, kwargs) (line 500)
    curfit_call_result_79564 = invoke(stypy.reporting.localization.Localization(__file__, 500, 24), curfit_79551, *[task_79552, x_79553, y_79554, w_79555, t_79556, wrk_79557, iwrk_79558, xb_79559, xe_79560, k_79561, s_79562], **kwargs_79563)
    
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___79565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), curfit_call_result_79564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_79566 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), getitem___79565, int_79549)
    
    # Assigning a type to the variable 'tuple_var_assignment_78287' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple_var_assignment_78287', subscript_call_result_79566)
    
    # Assigning a Name to a Name (line 500):
    # Getting the type of 'tuple_var_assignment_78284' (line 500)
    tuple_var_assignment_78284_79567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple_var_assignment_78284')
    # Assigning a type to the variable 'n' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'n', tuple_var_assignment_78284_79567)
    
    # Assigning a Name to a Name (line 500):
    # Getting the type of 'tuple_var_assignment_78285' (line 500)
    tuple_var_assignment_78285_79568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple_var_assignment_78285')
    # Assigning a type to the variable 'c' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 11), 'c', tuple_var_assignment_78285_79568)
    
    # Assigning a Name to a Name (line 500):
    # Getting the type of 'tuple_var_assignment_78286' (line 500)
    tuple_var_assignment_78286_79569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple_var_assignment_78286')
    # Assigning a type to the variable 'fp' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 14), 'fp', tuple_var_assignment_78286_79569)
    
    # Assigning a Name to a Name (line 500):
    # Getting the type of 'tuple_var_assignment_78287' (line 500)
    tuple_var_assignment_78287_79570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple_var_assignment_78287')
    # Assigning a type to the variable 'ier' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 18), 'ier', tuple_var_assignment_78287_79570)
    # SSA branch for the else part of an if statement (line 499)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 503):
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_79571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    
    # Call to percur(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'task' (line 503)
    task_79574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'task', False)
    # Getting the type of 'x' (line 503)
    x_79575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 46), 'x', False)
    # Getting the type of 'y' (line 503)
    y_79576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 49), 'y', False)
    # Getting the type of 'w' (line 503)
    w_79577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 52), 'w', False)
    # Getting the type of 't' (line 503)
    t_79578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 55), 't', False)
    # Getting the type of 'wrk' (line 503)
    wrk_79579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 58), 'wrk', False)
    # Getting the type of 'iwrk' (line 503)
    iwrk_79580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 63), 'iwrk', False)
    # Getting the type of 'k' (line 503)
    k_79581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 69), 'k', False)
    # Getting the type of 's' (line 503)
    s_79582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 72), 's', False)
    # Processing the call keyword arguments (line 503)
    kwargs_79583 = {}
    # Getting the type of 'dfitpack' (line 503)
    dfitpack_79572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'dfitpack', False)
    # Obtaining the member 'percur' of a type (line 503)
    percur_79573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 24), dfitpack_79572, 'percur')
    # Calling percur(args, kwargs) (line 503)
    percur_call_result_79584 = invoke(stypy.reporting.localization.Localization(__file__, 503, 24), percur_79573, *[task_79574, x_79575, y_79576, w_79577, t_79578, wrk_79579, iwrk_79580, k_79581, s_79582], **kwargs_79583)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___79585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), percur_call_result_79584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_79586 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), getitem___79585, int_79571)
    
    # Assigning a type to the variable 'tuple_var_assignment_78288' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_78288', subscript_call_result_79586)
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_79587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    
    # Call to percur(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'task' (line 503)
    task_79590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'task', False)
    # Getting the type of 'x' (line 503)
    x_79591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 46), 'x', False)
    # Getting the type of 'y' (line 503)
    y_79592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 49), 'y', False)
    # Getting the type of 'w' (line 503)
    w_79593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 52), 'w', False)
    # Getting the type of 't' (line 503)
    t_79594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 55), 't', False)
    # Getting the type of 'wrk' (line 503)
    wrk_79595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 58), 'wrk', False)
    # Getting the type of 'iwrk' (line 503)
    iwrk_79596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 63), 'iwrk', False)
    # Getting the type of 'k' (line 503)
    k_79597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 69), 'k', False)
    # Getting the type of 's' (line 503)
    s_79598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 72), 's', False)
    # Processing the call keyword arguments (line 503)
    kwargs_79599 = {}
    # Getting the type of 'dfitpack' (line 503)
    dfitpack_79588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'dfitpack', False)
    # Obtaining the member 'percur' of a type (line 503)
    percur_79589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 24), dfitpack_79588, 'percur')
    # Calling percur(args, kwargs) (line 503)
    percur_call_result_79600 = invoke(stypy.reporting.localization.Localization(__file__, 503, 24), percur_79589, *[task_79590, x_79591, y_79592, w_79593, t_79594, wrk_79595, iwrk_79596, k_79597, s_79598], **kwargs_79599)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___79601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), percur_call_result_79600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_79602 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), getitem___79601, int_79587)
    
    # Assigning a type to the variable 'tuple_var_assignment_78289' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_78289', subscript_call_result_79602)
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_79603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    
    # Call to percur(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'task' (line 503)
    task_79606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'task', False)
    # Getting the type of 'x' (line 503)
    x_79607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 46), 'x', False)
    # Getting the type of 'y' (line 503)
    y_79608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 49), 'y', False)
    # Getting the type of 'w' (line 503)
    w_79609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 52), 'w', False)
    # Getting the type of 't' (line 503)
    t_79610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 55), 't', False)
    # Getting the type of 'wrk' (line 503)
    wrk_79611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 58), 'wrk', False)
    # Getting the type of 'iwrk' (line 503)
    iwrk_79612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 63), 'iwrk', False)
    # Getting the type of 'k' (line 503)
    k_79613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 69), 'k', False)
    # Getting the type of 's' (line 503)
    s_79614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 72), 's', False)
    # Processing the call keyword arguments (line 503)
    kwargs_79615 = {}
    # Getting the type of 'dfitpack' (line 503)
    dfitpack_79604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'dfitpack', False)
    # Obtaining the member 'percur' of a type (line 503)
    percur_79605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 24), dfitpack_79604, 'percur')
    # Calling percur(args, kwargs) (line 503)
    percur_call_result_79616 = invoke(stypy.reporting.localization.Localization(__file__, 503, 24), percur_79605, *[task_79606, x_79607, y_79608, w_79609, t_79610, wrk_79611, iwrk_79612, k_79613, s_79614], **kwargs_79615)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___79617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), percur_call_result_79616, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_79618 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), getitem___79617, int_79603)
    
    # Assigning a type to the variable 'tuple_var_assignment_78290' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_78290', subscript_call_result_79618)
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_79619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    
    # Call to percur(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'task' (line 503)
    task_79622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'task', False)
    # Getting the type of 'x' (line 503)
    x_79623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 46), 'x', False)
    # Getting the type of 'y' (line 503)
    y_79624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 49), 'y', False)
    # Getting the type of 'w' (line 503)
    w_79625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 52), 'w', False)
    # Getting the type of 't' (line 503)
    t_79626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 55), 't', False)
    # Getting the type of 'wrk' (line 503)
    wrk_79627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 58), 'wrk', False)
    # Getting the type of 'iwrk' (line 503)
    iwrk_79628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 63), 'iwrk', False)
    # Getting the type of 'k' (line 503)
    k_79629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 69), 'k', False)
    # Getting the type of 's' (line 503)
    s_79630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 72), 's', False)
    # Processing the call keyword arguments (line 503)
    kwargs_79631 = {}
    # Getting the type of 'dfitpack' (line 503)
    dfitpack_79620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'dfitpack', False)
    # Obtaining the member 'percur' of a type (line 503)
    percur_79621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 24), dfitpack_79620, 'percur')
    # Calling percur(args, kwargs) (line 503)
    percur_call_result_79632 = invoke(stypy.reporting.localization.Localization(__file__, 503, 24), percur_79621, *[task_79622, x_79623, y_79624, w_79625, t_79626, wrk_79627, iwrk_79628, k_79629, s_79630], **kwargs_79631)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___79633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), percur_call_result_79632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_79634 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), getitem___79633, int_79619)
    
    # Assigning a type to the variable 'tuple_var_assignment_78291' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_78291', subscript_call_result_79634)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_78288' (line 503)
    tuple_var_assignment_78288_79635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_78288')
    # Assigning a type to the variable 'n' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'n', tuple_var_assignment_78288_79635)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_78289' (line 503)
    tuple_var_assignment_78289_79636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_78289')
    # Assigning a type to the variable 'c' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 11), 'c', tuple_var_assignment_78289_79636)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_78290' (line 503)
    tuple_var_assignment_78290_79637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_78290')
    # Assigning a type to the variable 'fp' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 14), 'fp', tuple_var_assignment_78290_79637)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_78291' (line 503)
    tuple_var_assignment_78291_79638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_78291')
    # Assigning a type to the variable 'ier' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 18), 'ier', tuple_var_assignment_78291_79638)
    # SSA join for if statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 504):
    
    # Assigning a Tuple to a Name (line 504):
    
    # Obtaining an instance of the builtin type 'tuple' (line 504)
    tuple_79639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 504)
    # Adding element type (line 504)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 504)
    n_79640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 14), 'n')
    slice_79641 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 504, 11), None, n_79640, None)
    # Getting the type of 't' (line 504)
    t_79642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 11), 't')
    # Obtaining the member '__getitem__' of a type (line 504)
    getitem___79643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 11), t_79642, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 504)
    subscript_call_result_79644 = invoke(stypy.reporting.localization.Localization(__file__, 504, 11), getitem___79643, slice_79641)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 11), tuple_79639, subscript_call_result_79644)
    # Adding element type (line 504)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 504)
    n_79645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 21), 'n')
    slice_79646 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 504, 18), None, n_79645, None)
    # Getting the type of 'c' (line 504)
    c_79647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 18), 'c')
    # Obtaining the member '__getitem__' of a type (line 504)
    getitem___79648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 18), c_79647, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 504)
    subscript_call_result_79649 = invoke(stypy.reporting.localization.Localization(__file__, 504, 18), getitem___79648, slice_79646)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 11), tuple_79639, subscript_call_result_79649)
    # Adding element type (line 504)
    # Getting the type of 'k' (line 504)
    k_79650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 25), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 11), tuple_79639, k_79650)
    
    # Assigning a type to the variable 'tck' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'tck', tuple_79639)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ier' (line 505)
    ier_79651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 7), 'ier')
    int_79652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 14), 'int')
    # Applying the binary operator '<=' (line 505)
    result_le_79653 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), '<=', ier_79651, int_79652)
    
    
    # Getting the type of 'quiet' (line 505)
    quiet_79654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 24), 'quiet')
    # Applying the 'not' unary operator (line 505)
    result_not__79655 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 20), 'not', quiet_79654)
    
    # Applying the binary operator 'and' (line 505)
    result_and_keyword_79656 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), 'and', result_le_79653, result_not__79655)
    
    # Testing the type of an if condition (line 505)
    if_condition_79657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 4), result_and_keyword_79656)
    # Assigning a type to the variable 'if_condition_79657' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'if_condition_79657', if_condition_79657)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 506):
    
    # Assigning a BinOp to a Name (line 506):
    
    # Obtaining the type of the subscript
    int_79658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 31), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 506)
    ier_79659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 26), 'ier')
    # Getting the type of '_iermess' (line 506)
    _iermess_79660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), '_iermess')
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___79661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 17), _iermess_79660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_79662 = invoke(stypy.reporting.localization.Localization(__file__, 506, 17), getitem___79661, ier_79659)
    
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___79663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 17), subscript_call_result_79662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_79664 = invoke(stypy.reporting.localization.Localization(__file__, 506, 17), getitem___79663, int_79658)
    
    str_79665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 36), 'str', '\tk=%d n=%d m=%d fp=%f s=%f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 507)
    tuple_79666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 507)
    # Adding element type (line 507)
    # Getting the type of 'k' (line 507)
    k_79667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 18), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 18), tuple_79666, k_79667)
    # Adding element type (line 507)
    
    # Call to len(...): (line 507)
    # Processing the call arguments (line 507)
    # Getting the type of 't' (line 507)
    t_79669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 25), 't', False)
    # Processing the call keyword arguments (line 507)
    kwargs_79670 = {}
    # Getting the type of 'len' (line 507)
    len_79668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 21), 'len', False)
    # Calling len(args, kwargs) (line 507)
    len_call_result_79671 = invoke(stypy.reporting.localization.Localization(__file__, 507, 21), len_79668, *[t_79669], **kwargs_79670)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 18), tuple_79666, len_call_result_79671)
    # Adding element type (line 507)
    # Getting the type of 'm' (line 507)
    m_79672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 29), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 18), tuple_79666, m_79672)
    # Adding element type (line 507)
    # Getting the type of 'fp' (line 507)
    fp_79673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 32), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 18), tuple_79666, fp_79673)
    # Adding element type (line 507)
    # Getting the type of 's' (line 507)
    s_79674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 36), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 18), tuple_79666, s_79674)
    
    # Applying the binary operator '%' (line 506)
    result_mod_79675 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 36), '%', str_79665, tuple_79666)
    
    # Applying the binary operator '+' (line 506)
    result_add_79676 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 17), '+', subscript_call_result_79664, result_mod_79675)
    
    # Assigning a type to the variable '_mess' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), '_mess', result_add_79676)
    
    # Call to warn(...): (line 508)
    # Processing the call arguments (line 508)
    
    # Call to RuntimeWarning(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of '_mess' (line 508)
    _mess_79680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 37), '_mess', False)
    # Processing the call keyword arguments (line 508)
    kwargs_79681 = {}
    # Getting the type of 'RuntimeWarning' (line 508)
    RuntimeWarning_79679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 22), 'RuntimeWarning', False)
    # Calling RuntimeWarning(args, kwargs) (line 508)
    RuntimeWarning_call_result_79682 = invoke(stypy.reporting.localization.Localization(__file__, 508, 22), RuntimeWarning_79679, *[_mess_79680], **kwargs_79681)
    
    # Processing the call keyword arguments (line 508)
    kwargs_79683 = {}
    # Getting the type of 'warnings' (line 508)
    warnings_79677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 508)
    warn_79678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), warnings_79677, 'warn')
    # Calling warn(args, kwargs) (line 508)
    warn_call_result_79684 = invoke(stypy.reporting.localization.Localization(__file__, 508, 8), warn_79678, *[RuntimeWarning_call_result_79682], **kwargs_79683)
    
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ier' (line 509)
    ier_79685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 7), 'ier')
    int_79686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 13), 'int')
    # Applying the binary operator '>' (line 509)
    result_gt_79687 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 7), '>', ier_79685, int_79686)
    
    
    # Getting the type of 'full_output' (line 509)
    full_output_79688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 23), 'full_output')
    # Applying the 'not' unary operator (line 509)
    result_not__79689 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 19), 'not', full_output_79688)
    
    # Applying the binary operator 'and' (line 509)
    result_and_keyword_79690 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 7), 'and', result_gt_79687, result_not__79689)
    
    # Testing the type of an if condition (line 509)
    if_condition_79691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 4), result_and_keyword_79690)
    # Assigning a type to the variable 'if_condition_79691' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'if_condition_79691', if_condition_79691)
    # SSA begins for if statement (line 509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'ier' (line 510)
    ier_79692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 11), 'ier')
    
    # Obtaining an instance of the builtin type 'list' (line 510)
    list_79693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 510)
    # Adding element type (line 510)
    int_79694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 18), list_79693, int_79694)
    # Adding element type (line 510)
    int_79695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 18), list_79693, int_79695)
    # Adding element type (line 510)
    int_79696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 18), list_79693, int_79696)
    
    # Applying the binary operator 'in' (line 510)
    result_contains_79697 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 11), 'in', ier_79692, list_79693)
    
    # Testing the type of an if condition (line 510)
    if_condition_79698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 8), result_contains_79697)
    # Assigning a type to the variable 'if_condition_79698' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'if_condition_79698', if_condition_79698)
    # SSA begins for if statement (line 510)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 511)
    # Processing the call arguments (line 511)
    
    # Call to RuntimeWarning(...): (line 511)
    # Processing the call arguments (line 511)
    
    # Obtaining the type of the subscript
    int_79702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 55), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 511)
    ier_79703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 50), 'ier', False)
    # Getting the type of '_iermess' (line 511)
    _iermess_79704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 41), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 511)
    getitem___79705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 41), _iermess_79704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 511)
    subscript_call_result_79706 = invoke(stypy.reporting.localization.Localization(__file__, 511, 41), getitem___79705, ier_79703)
    
    # Obtaining the member '__getitem__' of a type (line 511)
    getitem___79707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 41), subscript_call_result_79706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 511)
    subscript_call_result_79708 = invoke(stypy.reporting.localization.Localization(__file__, 511, 41), getitem___79707, int_79702)
    
    # Processing the call keyword arguments (line 511)
    kwargs_79709 = {}
    # Getting the type of 'RuntimeWarning' (line 511)
    RuntimeWarning_79701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 26), 'RuntimeWarning', False)
    # Calling RuntimeWarning(args, kwargs) (line 511)
    RuntimeWarning_call_result_79710 = invoke(stypy.reporting.localization.Localization(__file__, 511, 26), RuntimeWarning_79701, *[subscript_call_result_79708], **kwargs_79709)
    
    # Processing the call keyword arguments (line 511)
    kwargs_79711 = {}
    # Getting the type of 'warnings' (line 511)
    warnings_79699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 511)
    warn_79700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 12), warnings_79699, 'warn')
    # Calling warn(args, kwargs) (line 511)
    warn_call_result_79712 = invoke(stypy.reporting.localization.Localization(__file__, 511, 12), warn_79700, *[RuntimeWarning_call_result_79710], **kwargs_79711)
    
    # SSA branch for the else part of an if statement (line 510)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 513)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to (...): (line 514)
    # Processing the call arguments (line 514)
    
    # Obtaining the type of the subscript
    int_79720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 53), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 514)
    ier_79721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 48), 'ier', False)
    # Getting the type of '_iermess' (line 514)
    _iermess_79722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 39), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 514)
    getitem___79723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 39), _iermess_79722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 514)
    subscript_call_result_79724 = invoke(stypy.reporting.localization.Localization(__file__, 514, 39), getitem___79723, ier_79721)
    
    # Obtaining the member '__getitem__' of a type (line 514)
    getitem___79725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 39), subscript_call_result_79724, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 514)
    subscript_call_result_79726 = invoke(stypy.reporting.localization.Localization(__file__, 514, 39), getitem___79725, int_79720)
    
    # Processing the call keyword arguments (line 514)
    kwargs_79727 = {}
    
    # Obtaining the type of the subscript
    int_79713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 36), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 514)
    ier_79714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 31), 'ier', False)
    # Getting the type of '_iermess' (line 514)
    _iermess_79715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 22), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 514)
    getitem___79716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 22), _iermess_79715, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 514)
    subscript_call_result_79717 = invoke(stypy.reporting.localization.Localization(__file__, 514, 22), getitem___79716, ier_79714)
    
    # Obtaining the member '__getitem__' of a type (line 514)
    getitem___79718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 22), subscript_call_result_79717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 514)
    subscript_call_result_79719 = invoke(stypy.reporting.localization.Localization(__file__, 514, 22), getitem___79718, int_79713)
    
    # Calling (args, kwargs) (line 514)
    _call_result_79728 = invoke(stypy.reporting.localization.Localization(__file__, 514, 22), subscript_call_result_79719, *[subscript_call_result_79726], **kwargs_79727)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 514, 16), _call_result_79728, 'raise parameter', BaseException)
    # SSA branch for the except part of a try statement (line 513)
    # SSA branch for the except 'KeyError' branch of a try statement (line 513)
    module_type_store.open_ssa_branch('except')
    
    # Call to (...): (line 516)
    # Processing the call arguments (line 516)
    
    # Obtaining the type of the subscript
    int_79736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 65), 'int')
    
    # Obtaining the type of the subscript
    str_79737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 54), 'str', 'unknown')
    # Getting the type of '_iermess' (line 516)
    _iermess_79738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 45), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___79739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 45), _iermess_79738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_79740 = invoke(stypy.reporting.localization.Localization(__file__, 516, 45), getitem___79739, str_79737)
    
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___79741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 45), subscript_call_result_79740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_79742 = invoke(stypy.reporting.localization.Localization(__file__, 516, 45), getitem___79741, int_79736)
    
    # Processing the call keyword arguments (line 516)
    kwargs_79743 = {}
    
    # Obtaining the type of the subscript
    int_79729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 42), 'int')
    
    # Obtaining the type of the subscript
    str_79730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 31), 'str', 'unknown')
    # Getting the type of '_iermess' (line 516)
    _iermess_79731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 22), '_iermess', False)
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___79732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 22), _iermess_79731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_79733 = invoke(stypy.reporting.localization.Localization(__file__, 516, 22), getitem___79732, str_79730)
    
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___79734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 22), subscript_call_result_79733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_79735 = invoke(stypy.reporting.localization.Localization(__file__, 516, 22), getitem___79734, int_79729)
    
    # Calling (args, kwargs) (line 516)
    _call_result_79744 = invoke(stypy.reporting.localization.Localization(__file__, 516, 22), subscript_call_result_79735, *[subscript_call_result_79742], **kwargs_79743)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 516, 16), _call_result_79744, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 513)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 510)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 509)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full_output' (line 517)
    full_output_79745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 7), 'full_output')
    # Testing the type of an if condition (line 517)
    if_condition_79746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 4), full_output_79745)
    # Assigning a type to the variable 'if_condition_79746' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'if_condition_79746', if_condition_79746)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 518)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 519)
    tuple_79747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 519)
    # Adding element type (line 519)
    # Getting the type of 'tck' (line 519)
    tck_79748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 19), 'tck')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), tuple_79747, tck_79748)
    # Adding element type (line 519)
    # Getting the type of 'fp' (line 519)
    fp_79749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 24), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), tuple_79747, fp_79749)
    # Adding element type (line 519)
    # Getting the type of 'ier' (line 519)
    ier_79750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 28), 'ier')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), tuple_79747, ier_79750)
    # Adding element type (line 519)
    
    # Obtaining the type of the subscript
    int_79751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 47), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ier' (line 519)
    ier_79752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 42), 'ier')
    # Getting the type of '_iermess' (line 519)
    _iermess_79753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 33), '_iermess')
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___79754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 33), _iermess_79753, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_79755 = invoke(stypy.reporting.localization.Localization(__file__, 519, 33), getitem___79754, ier_79752)
    
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___79756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 33), subscript_call_result_79755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_79757 = invoke(stypy.reporting.localization.Localization(__file__, 519, 33), getitem___79756, int_79751)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), tuple_79747, subscript_call_result_79757)
    
    # Assigning a type to the variable 'stypy_return_type' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'stypy_return_type', tuple_79747)
    # SSA branch for the except part of a try statement (line 518)
    # SSA branch for the except 'KeyError' branch of a try statement (line 518)
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 521)
    tuple_79758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 521)
    # Adding element type (line 521)
    # Getting the type of 'tck' (line 521)
    tck_79759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 19), 'tck')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 19), tuple_79758, tck_79759)
    # Adding element type (line 521)
    # Getting the type of 'fp' (line 521)
    fp_79760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 24), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 19), tuple_79758, fp_79760)
    # Adding element type (line 521)
    # Getting the type of 'ier' (line 521)
    ier_79761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 28), 'ier')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 19), tuple_79758, ier_79761)
    # Adding element type (line 521)
    
    # Obtaining the type of the subscript
    int_79762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 53), 'int')
    
    # Obtaining the type of the subscript
    str_79763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 42), 'str', 'unknown')
    # Getting the type of '_iermess' (line 521)
    _iermess_79764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 33), '_iermess')
    # Obtaining the member '__getitem__' of a type (line 521)
    getitem___79765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 33), _iermess_79764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 521)
    subscript_call_result_79766 = invoke(stypy.reporting.localization.Localization(__file__, 521, 33), getitem___79765, str_79763)
    
    # Obtaining the member '__getitem__' of a type (line 521)
    getitem___79767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 33), subscript_call_result_79766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 521)
    subscript_call_result_79768 = invoke(stypy.reporting.localization.Localization(__file__, 521, 33), getitem___79767, int_79762)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 19), tuple_79758, subscript_call_result_79768)
    
    # Assigning a type to the variable 'stypy_return_type' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'stypy_return_type', tuple_79758)
    # SSA join for try-except statement (line 518)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 517)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'tck' (line 523)
    tck_79769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 15), 'tck')
    # Assigning a type to the variable 'stypy_return_type' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'stypy_return_type', tck_79769)
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'splrep(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splrep' in the type store
    # Getting the type of 'stypy_return_type' (line 317)
    stypy_return_type_79770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_79770)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splrep'
    return stypy_return_type_79770

# Assigning a type to the variable 'splrep' (line 317)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'splrep', splrep)

@norecursion
def splev(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_79771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 22), 'int')
    int_79772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 29), 'int')
    defaults = [int_79771, int_79772]
    # Create a new context for function 'splev'
    module_type_store = module_type_store.open_function_context('splev', 526, 0, False)
    
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

    str_79773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, (-1)), 'str', '\n    Evaluate a B-spline or its derivatives.\n\n    Given the knots and coefficients of a B-spline representation, evaluate\n    the value of the smoothing polynomial and its derivatives.  This is a\n    wrapper around the FORTRAN routines splev and splder of FITPACK.\n\n    Parameters\n    ----------\n    x : array_like\n        An array of points at which to return the value of the smoothed\n        spline or its derivatives.  If `tck` was returned from `splprep`,\n        then the parameter values, u should be given.\n    tck : tuple\n        A sequence of length 3 returned by `splrep` or `splprep` containing\n        the knots, coefficients, and degree of the spline.\n    der : int, optional\n        The order of derivative of the spline to compute (must be less than\n        or equal to k).\n    ext : int, optional\n        Controls the value returned for elements of ``x`` not in the\n        interval defined by the knot sequence.\n\n        * if ext=0, return the extrapolated value.\n        * if ext=1, return 0\n        * if ext=2, raise a ValueError\n        * if ext=3, return the boundary value.\n\n        The default value is 0.\n\n    Returns\n    -------\n    y : ndarray or list of ndarrays\n        An array of values representing the spline function evaluated at\n        the points in ``x``.  If `tck` was returned from `splprep`, then this\n        is a list of arrays representing the curve in N-dimensional space.\n\n    See Also\n    --------\n    splprep, splrep, sproot, spalde, splint\n    bisplrep, bisplev\n\n    References\n    ----------\n    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation\n        Theory, 6, p.50-62, 1972.\n    .. [2] M.G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths\n        Applics, 10, p.134-149, 1972.\n    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs\n        on Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    # Assigning a Name to a Tuple (line 579):
    
    # Assigning a Subscript to a Name (line 579):
    
    # Obtaining the type of the subscript
    int_79774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 4), 'int')
    # Getting the type of 'tck' (line 579)
    tck_79775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___79776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 4), tck_79775, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_79777 = invoke(stypy.reporting.localization.Localization(__file__, 579, 4), getitem___79776, int_79774)
    
    # Assigning a type to the variable 'tuple_var_assignment_78292' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'tuple_var_assignment_78292', subscript_call_result_79777)
    
    # Assigning a Subscript to a Name (line 579):
    
    # Obtaining the type of the subscript
    int_79778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 4), 'int')
    # Getting the type of 'tck' (line 579)
    tck_79779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___79780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 4), tck_79779, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_79781 = invoke(stypy.reporting.localization.Localization(__file__, 579, 4), getitem___79780, int_79778)
    
    # Assigning a type to the variable 'tuple_var_assignment_78293' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'tuple_var_assignment_78293', subscript_call_result_79781)
    
    # Assigning a Subscript to a Name (line 579):
    
    # Obtaining the type of the subscript
    int_79782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 4), 'int')
    # Getting the type of 'tck' (line 579)
    tck_79783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___79784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 4), tck_79783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_79785 = invoke(stypy.reporting.localization.Localization(__file__, 579, 4), getitem___79784, int_79782)
    
    # Assigning a type to the variable 'tuple_var_assignment_78294' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'tuple_var_assignment_78294', subscript_call_result_79785)
    
    # Assigning a Name to a Name (line 579):
    # Getting the type of 'tuple_var_assignment_78292' (line 579)
    tuple_var_assignment_78292_79786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'tuple_var_assignment_78292')
    # Assigning a type to the variable 't' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 't', tuple_var_assignment_78292_79786)
    
    # Assigning a Name to a Name (line 579):
    # Getting the type of 'tuple_var_assignment_78293' (line 579)
    tuple_var_assignment_78293_79787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'tuple_var_assignment_78293')
    # Assigning a type to the variable 'c' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 7), 'c', tuple_var_assignment_78293_79787)
    
    # Assigning a Name to a Name (line 579):
    # Getting the type of 'tuple_var_assignment_78294' (line 579)
    tuple_var_assignment_78294_79788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'tuple_var_assignment_78294')
    # Assigning a type to the variable 'k' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 10), 'k', tuple_var_assignment_78294_79788)
    
    
    # SSA begins for try-except statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    int_79789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 13), 'int')
    
    # Obtaining the type of the subscript
    int_79790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 10), 'int')
    # Getting the type of 'c' (line 581)
    c_79791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'c')
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___79792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), c_79791, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_79793 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___79792, int_79790)
    
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___79794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), subscript_call_result_79793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_79795 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___79794, int_79789)
    
    
    # Assigning a Name to a Name (line 582):
    
    # Assigning a Name to a Name (line 582):
    # Getting the type of 'True' (line 582)
    True_79796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 21), 'True')
    # Assigning a type to the variable 'parametric' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'parametric', True_79796)
    # SSA branch for the except part of a try statement (line 580)
    # SSA branch for the except '<any exception>' branch of a try statement (line 580)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 584):
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'False' (line 584)
    False_79797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 21), 'False')
    # Assigning a type to the variable 'parametric' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'parametric', False_79797)
    # SSA join for try-except statement (line 580)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'parametric' (line 585)
    parametric_79798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 7), 'parametric')
    # Testing the type of an if condition (line 585)
    if_condition_79799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 4), parametric_79798)
    # Assigning a type to the variable 'if_condition_79799' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'if_condition_79799', if_condition_79799)
    # SSA begins for if statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to list(...): (line 586)
    # Processing the call arguments (line 586)
    
    # Call to map(...): (line 586)
    # Processing the call arguments (line 586)

    @norecursion
    def _stypy_temp_lambda_49(localization, *varargs, **kwargs):
        global module_type_store
        # Getting the type of 'x' (line 586)
        x_79802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 36), 'x', False)
        # Getting the type of 't' (line 586)
        t_79803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 41), 't', False)
        # Getting the type of 'k' (line 586)
        k_79804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 46), 'k', False)
        # Getting the type of 'der' (line 586)
        der_79805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 53), 'der', False)
        # Assign values to the parameters with defaults
        defaults = [x_79802, t_79803, k_79804, der_79805]
        # Create a new context for function '_stypy_temp_lambda_49'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_49', 586, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_49.stypy_localization = localization
        _stypy_temp_lambda_49.stypy_type_of_self = None
        _stypy_temp_lambda_49.stypy_type_store = module_type_store
        _stypy_temp_lambda_49.stypy_function_name = '_stypy_temp_lambda_49'
        _stypy_temp_lambda_49.stypy_param_names_list = ['c', 'x', 't', 'k', 'der']
        _stypy_temp_lambda_49.stypy_varargs_param_name = None
        _stypy_temp_lambda_49.stypy_kwargs_param_name = None
        _stypy_temp_lambda_49.stypy_call_defaults = defaults
        _stypy_temp_lambda_49.stypy_call_varargs = varargs
        _stypy_temp_lambda_49.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_49', ['c', 'x', 't', 'k', 'der'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_49', ['c', 'x', 't', 'k', 'der'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to splev(...): (line 587)
        # Processing the call arguments (line 587)
        # Getting the type of 'x' (line 587)
        x_79807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 587)
        list_79808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 587)
        # Adding element type (line 587)
        # Getting the type of 't' (line 587)
        t_79809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 34), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 33), list_79808, t_79809)
        # Adding element type (line 587)
        # Getting the type of 'c' (line 587)
        c_79810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 37), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 33), list_79808, c_79810)
        # Adding element type (line 587)
        # Getting the type of 'k' (line 587)
        k_79811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 40), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 33), list_79808, k_79811)
        
        # Getting the type of 'der' (line 587)
        der_79812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 44), 'der', False)
        # Getting the type of 'ext' (line 587)
        ext_79813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 49), 'ext', False)
        # Processing the call keyword arguments (line 587)
        kwargs_79814 = {}
        # Getting the type of 'splev' (line 587)
        splev_79806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 24), 'splev', False)
        # Calling splev(args, kwargs) (line 587)
        splev_call_result_79815 = invoke(stypy.reporting.localization.Localization(__file__, 587, 24), splev_79806, *[x_79807, list_79808, der_79812, ext_79813], **kwargs_79814)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 24), 'stypy_return_type', splev_call_result_79815)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_49' in the type store
        # Getting the type of 'stypy_return_type' (line 586)
        stypy_return_type_79816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_79816)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_49'
        return stypy_return_type_79816

    # Assigning a type to the variable '_stypy_temp_lambda_49' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 24), '_stypy_temp_lambda_49', _stypy_temp_lambda_49)
    # Getting the type of '_stypy_temp_lambda_49' (line 586)
    _stypy_temp_lambda_49_79817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 24), '_stypy_temp_lambda_49')
    # Getting the type of 'c' (line 587)
    c_79818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 55), 'c', False)
    # Processing the call keyword arguments (line 586)
    kwargs_79819 = {}
    # Getting the type of 'map' (line 586)
    map_79801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'map', False)
    # Calling map(args, kwargs) (line 586)
    map_call_result_79820 = invoke(stypy.reporting.localization.Localization(__file__, 586, 20), map_79801, *[_stypy_temp_lambda_49_79817, c_79818], **kwargs_79819)
    
    # Processing the call keyword arguments (line 586)
    kwargs_79821 = {}
    # Getting the type of 'list' (line 586)
    list_79800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 15), 'list', False)
    # Calling list(args, kwargs) (line 586)
    list_call_result_79822 = invoke(stypy.reporting.localization.Localization(__file__, 586, 15), list_79800, *[map_call_result_79820], **kwargs_79821)
    
    # Assigning a type to the variable 'stypy_return_type' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'stypy_return_type', list_call_result_79822)
    # SSA branch for the else part of an if statement (line 585)
    module_type_store.open_ssa_branch('else')
    
    
    
    int_79823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 16), 'int')
    # Getting the type of 'der' (line 589)
    der_79824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'der')
    # Applying the binary operator '<=' (line 589)
    result_le_79825 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 16), '<=', int_79823, der_79824)
    # Getting the type of 'k' (line 589)
    k_79826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 28), 'k')
    # Applying the binary operator '<=' (line 589)
    result_le_79827 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 16), '<=', der_79824, k_79826)
    # Applying the binary operator '&' (line 589)
    result_and__79828 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 16), '&', result_le_79825, result_le_79827)
    
    # Applying the 'not' unary operator (line 589)
    result_not__79829 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 11), 'not', result_and__79828)
    
    # Testing the type of an if condition (line 589)
    if_condition_79830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 589, 8), result_not__79829)
    # Assigning a type to the variable 'if_condition_79830' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'if_condition_79830', if_condition_79830)
    # SSA begins for if statement (line 589)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 590)
    # Processing the call arguments (line 590)
    str_79832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 29), 'str', '0<=der=%d<=k=%d must hold')
    
    # Obtaining an instance of the builtin type 'tuple' (line 590)
    tuple_79833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 590)
    # Adding element type (line 590)
    # Getting the type of 'der' (line 590)
    der_79834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 60), 'der', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 60), tuple_79833, der_79834)
    # Adding element type (line 590)
    # Getting the type of 'k' (line 590)
    k_79835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 65), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 60), tuple_79833, k_79835)
    
    # Applying the binary operator '%' (line 590)
    result_mod_79836 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 29), '%', str_79832, tuple_79833)
    
    # Processing the call keyword arguments (line 590)
    kwargs_79837 = {}
    # Getting the type of 'ValueError' (line 590)
    ValueError_79831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 590)
    ValueError_call_result_79838 = invoke(stypy.reporting.localization.Localization(__file__, 590, 18), ValueError_79831, *[result_mod_79836], **kwargs_79837)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 590, 12), ValueError_call_result_79838, 'raise parameter', BaseException)
    # SSA join for if statement (line 589)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ext' (line 591)
    ext_79839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 11), 'ext')
    
    # Obtaining an instance of the builtin type 'tuple' (line 591)
    tuple_79840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 591)
    # Adding element type (line 591)
    int_79841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 23), tuple_79840, int_79841)
    # Adding element type (line 591)
    int_79842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 23), tuple_79840, int_79842)
    # Adding element type (line 591)
    int_79843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 23), tuple_79840, int_79843)
    # Adding element type (line 591)
    int_79844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 23), tuple_79840, int_79844)
    
    # Applying the binary operator 'notin' (line 591)
    result_contains_79845 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 11), 'notin', ext_79839, tuple_79840)
    
    # Testing the type of an if condition (line 591)
    if_condition_79846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 591, 8), result_contains_79845)
    # Assigning a type to the variable 'if_condition_79846' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'if_condition_79846', if_condition_79846)
    # SSA begins for if statement (line 591)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 592)
    # Processing the call arguments (line 592)
    str_79848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 29), 'str', 'ext = %s not in (0, 1, 2, 3) ')
    # Getting the type of 'ext' (line 592)
    ext_79849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 63), 'ext', False)
    # Applying the binary operator '%' (line 592)
    result_mod_79850 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 29), '%', str_79848, ext_79849)
    
    # Processing the call keyword arguments (line 592)
    kwargs_79851 = {}
    # Getting the type of 'ValueError' (line 592)
    ValueError_79847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 592)
    ValueError_call_result_79852 = invoke(stypy.reporting.localization.Localization(__file__, 592, 18), ValueError_79847, *[result_mod_79850], **kwargs_79851)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 592, 12), ValueError_call_result_79852, 'raise parameter', BaseException)
    # SSA join for if statement (line 591)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 594):
    
    # Assigning a Call to a Name (line 594):
    
    # Call to asarray(...): (line 594)
    # Processing the call arguments (line 594)
    # Getting the type of 'x' (line 594)
    x_79854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 20), 'x', False)
    # Processing the call keyword arguments (line 594)
    kwargs_79855 = {}
    # Getting the type of 'asarray' (line 594)
    asarray_79853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'asarray', False)
    # Calling asarray(args, kwargs) (line 594)
    asarray_call_result_79856 = invoke(stypy.reporting.localization.Localization(__file__, 594, 12), asarray_79853, *[x_79854], **kwargs_79855)
    
    # Assigning a type to the variable 'x' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'x', asarray_call_result_79856)
    
    # Assigning a Attribute to a Name (line 595):
    
    # Assigning a Attribute to a Name (line 595):
    # Getting the type of 'x' (line 595)
    x_79857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 16), 'x')
    # Obtaining the member 'shape' of a type (line 595)
    shape_79858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 16), x_79857, 'shape')
    # Assigning a type to the variable 'shape' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'shape', shape_79858)
    
    # Assigning a Call to a Name (line 596):
    
    # Assigning a Call to a Name (line 596):
    
    # Call to ravel(...): (line 596)
    # Processing the call keyword arguments (line 596)
    kwargs_79864 = {}
    
    # Call to atleast_1d(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'x' (line 596)
    x_79860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 23), 'x', False)
    # Processing the call keyword arguments (line 596)
    kwargs_79861 = {}
    # Getting the type of 'atleast_1d' (line 596)
    atleast_1d_79859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 12), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 596)
    atleast_1d_call_result_79862 = invoke(stypy.reporting.localization.Localization(__file__, 596, 12), atleast_1d_79859, *[x_79860], **kwargs_79861)
    
    # Obtaining the member 'ravel' of a type (line 596)
    ravel_79863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 12), atleast_1d_call_result_79862, 'ravel')
    # Calling ravel(args, kwargs) (line 596)
    ravel_call_result_79865 = invoke(stypy.reporting.localization.Localization(__file__, 596, 12), ravel_79863, *[], **kwargs_79864)
    
    # Assigning a type to the variable 'x' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'x', ravel_call_result_79865)
    
    # Assigning a Call to a Tuple (line 597):
    
    # Assigning a Subscript to a Name (line 597):
    
    # Obtaining the type of the subscript
    int_79866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 8), 'int')
    
    # Call to _spl_(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'x' (line 597)
    x_79869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 32), 'x', False)
    # Getting the type of 'der' (line 597)
    der_79870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 35), 'der', False)
    # Getting the type of 't' (line 597)
    t_79871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 40), 't', False)
    # Getting the type of 'c' (line 597)
    c_79872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 43), 'c', False)
    # Getting the type of 'k' (line 597)
    k_79873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 46), 'k', False)
    # Getting the type of 'ext' (line 597)
    ext_79874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 49), 'ext', False)
    # Processing the call keyword arguments (line 597)
    kwargs_79875 = {}
    # Getting the type of '_fitpack' (line 597)
    _fitpack_79867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 17), '_fitpack', False)
    # Obtaining the member '_spl_' of a type (line 597)
    _spl__79868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 17), _fitpack_79867, '_spl_')
    # Calling _spl_(args, kwargs) (line 597)
    _spl__call_result_79876 = invoke(stypy.reporting.localization.Localization(__file__, 597, 17), _spl__79868, *[x_79869, der_79870, t_79871, c_79872, k_79873, ext_79874], **kwargs_79875)
    
    # Obtaining the member '__getitem__' of a type (line 597)
    getitem___79877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 8), _spl__call_result_79876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 597)
    subscript_call_result_79878 = invoke(stypy.reporting.localization.Localization(__file__, 597, 8), getitem___79877, int_79866)
    
    # Assigning a type to the variable 'tuple_var_assignment_78295' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'tuple_var_assignment_78295', subscript_call_result_79878)
    
    # Assigning a Subscript to a Name (line 597):
    
    # Obtaining the type of the subscript
    int_79879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 8), 'int')
    
    # Call to _spl_(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'x' (line 597)
    x_79882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 32), 'x', False)
    # Getting the type of 'der' (line 597)
    der_79883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 35), 'der', False)
    # Getting the type of 't' (line 597)
    t_79884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 40), 't', False)
    # Getting the type of 'c' (line 597)
    c_79885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 43), 'c', False)
    # Getting the type of 'k' (line 597)
    k_79886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 46), 'k', False)
    # Getting the type of 'ext' (line 597)
    ext_79887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 49), 'ext', False)
    # Processing the call keyword arguments (line 597)
    kwargs_79888 = {}
    # Getting the type of '_fitpack' (line 597)
    _fitpack_79880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 17), '_fitpack', False)
    # Obtaining the member '_spl_' of a type (line 597)
    _spl__79881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 17), _fitpack_79880, '_spl_')
    # Calling _spl_(args, kwargs) (line 597)
    _spl__call_result_79889 = invoke(stypy.reporting.localization.Localization(__file__, 597, 17), _spl__79881, *[x_79882, der_79883, t_79884, c_79885, k_79886, ext_79887], **kwargs_79888)
    
    # Obtaining the member '__getitem__' of a type (line 597)
    getitem___79890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 8), _spl__call_result_79889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 597)
    subscript_call_result_79891 = invoke(stypy.reporting.localization.Localization(__file__, 597, 8), getitem___79890, int_79879)
    
    # Assigning a type to the variable 'tuple_var_assignment_78296' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'tuple_var_assignment_78296', subscript_call_result_79891)
    
    # Assigning a Name to a Name (line 597):
    # Getting the type of 'tuple_var_assignment_78295' (line 597)
    tuple_var_assignment_78295_79892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'tuple_var_assignment_78295')
    # Assigning a type to the variable 'y' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'y', tuple_var_assignment_78295_79892)
    
    # Assigning a Name to a Name (line 597):
    # Getting the type of 'tuple_var_assignment_78296' (line 597)
    tuple_var_assignment_78296_79893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'tuple_var_assignment_78296')
    # Assigning a type to the variable 'ier' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 11), 'ier', tuple_var_assignment_78296_79893)
    
    
    # Getting the type of 'ier' (line 599)
    ier_79894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 11), 'ier')
    int_79895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 18), 'int')
    # Applying the binary operator '==' (line 599)
    result_eq_79896 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 11), '==', ier_79894, int_79895)
    
    # Testing the type of an if condition (line 599)
    if_condition_79897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 8), result_eq_79896)
    # Assigning a type to the variable 'if_condition_79897' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'if_condition_79897', if_condition_79897)
    # SSA begins for if statement (line 599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 600)
    # Processing the call arguments (line 600)
    str_79899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 29), 'str', 'Invalid input data')
    # Processing the call keyword arguments (line 600)
    kwargs_79900 = {}
    # Getting the type of 'ValueError' (line 600)
    ValueError_79898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 600)
    ValueError_call_result_79901 = invoke(stypy.reporting.localization.Localization(__file__, 600, 18), ValueError_79898, *[str_79899], **kwargs_79900)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 600, 12), ValueError_call_result_79901, 'raise parameter', BaseException)
    # SSA join for if statement (line 599)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ier' (line 601)
    ier_79902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 11), 'ier')
    int_79903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 18), 'int')
    # Applying the binary operator '==' (line 601)
    result_eq_79904 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 11), '==', ier_79902, int_79903)
    
    # Testing the type of an if condition (line 601)
    if_condition_79905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 601, 8), result_eq_79904)
    # Assigning a type to the variable 'if_condition_79905' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'if_condition_79905', if_condition_79905)
    # SSA begins for if statement (line 601)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 602)
    # Processing the call arguments (line 602)
    str_79907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 29), 'str', 'Found x value not in the domain')
    # Processing the call keyword arguments (line 602)
    kwargs_79908 = {}
    # Getting the type of 'ValueError' (line 602)
    ValueError_79906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 602)
    ValueError_call_result_79909 = invoke(stypy.reporting.localization.Localization(__file__, 602, 18), ValueError_79906, *[str_79907], **kwargs_79908)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 602, 12), ValueError_call_result_79909, 'raise parameter', BaseException)
    # SSA join for if statement (line 601)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ier' (line 603)
    ier_79910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 11), 'ier')
    # Testing the type of an if condition (line 603)
    if_condition_79911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 8), ier_79910)
    # Assigning a type to the variable 'if_condition_79911' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'if_condition_79911', if_condition_79911)
    # SSA begins for if statement (line 603)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 604)
    # Processing the call arguments (line 604)
    str_79913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 28), 'str', 'An error occurred')
    # Processing the call keyword arguments (line 604)
    kwargs_79914 = {}
    # Getting the type of 'TypeError' (line 604)
    TypeError_79912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 604)
    TypeError_call_result_79915 = invoke(stypy.reporting.localization.Localization(__file__, 604, 18), TypeError_79912, *[str_79913], **kwargs_79914)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 604, 12), TypeError_call_result_79915, 'raise parameter', BaseException)
    # SSA join for if statement (line 603)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reshape(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'shape' (line 606)
    shape_79918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 25), 'shape', False)
    # Processing the call keyword arguments (line 606)
    kwargs_79919 = {}
    # Getting the type of 'y' (line 606)
    y_79916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 15), 'y', False)
    # Obtaining the member 'reshape' of a type (line 606)
    reshape_79917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 15), y_79916, 'reshape')
    # Calling reshape(args, kwargs) (line 606)
    reshape_call_result_79920 = invoke(stypy.reporting.localization.Localization(__file__, 606, 15), reshape_79917, *[shape_79918], **kwargs_79919)
    
    # Assigning a type to the variable 'stypy_return_type' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'stypy_return_type', reshape_call_result_79920)
    # SSA join for if statement (line 585)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'splev(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splev' in the type store
    # Getting the type of 'stypy_return_type' (line 526)
    stypy_return_type_79921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_79921)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splev'
    return stypy_return_type_79921

# Assigning a type to the variable 'splev' (line 526)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 0), 'splev', splev)

@norecursion
def splint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_79922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 34), 'int')
    defaults = [int_79922]
    # Create a new context for function 'splint'
    module_type_store = module_type_store.open_function_context('splint', 609, 0, False)
    
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

    str_79923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, (-1)), 'str', '\n    Evaluate the definite integral of a B-spline.\n\n    Given the knots and coefficients of a B-spline, evaluate the definite\n    integral of the smoothing polynomial between two given points.\n\n    Parameters\n    ----------\n    a, b : float\n        The end-points of the integration interval.\n    tck : tuple\n        A tuple (t,c,k) containing the vector of knots, the B-spline\n        coefficients, and the degree of the spline (see `splev`).\n    full_output : int, optional\n        Non-zero to return optional output.\n\n    Returns\n    -------\n    integral : float\n        The resulting integral.\n    wrk : ndarray\n        An array containing the integrals of the normalized B-splines\n        defined on the set of knots.\n\n    Notes\n    -----\n    splint silently assumes that the spline function is zero outside the data\n    interval (a, b).\n\n    See Also\n    --------\n    splprep, splrep, sproot, spalde, splev\n    bisplrep, bisplev\n    UnivariateSpline, BivariateSpline\n\n    References\n    ----------\n    .. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines",\n        J. Inst. Maths Applics, 17, p.37-41, 1976.\n    .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs\n        on Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    # Assigning a Name to a Tuple (line 653):
    
    # Assigning a Subscript to a Name (line 653):
    
    # Obtaining the type of the subscript
    int_79924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 4), 'int')
    # Getting the type of 'tck' (line 653)
    tck_79925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___79926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 4), tck_79925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_79927 = invoke(stypy.reporting.localization.Localization(__file__, 653, 4), getitem___79926, int_79924)
    
    # Assigning a type to the variable 'tuple_var_assignment_78297' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'tuple_var_assignment_78297', subscript_call_result_79927)
    
    # Assigning a Subscript to a Name (line 653):
    
    # Obtaining the type of the subscript
    int_79928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 4), 'int')
    # Getting the type of 'tck' (line 653)
    tck_79929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___79930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 4), tck_79929, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_79931 = invoke(stypy.reporting.localization.Localization(__file__, 653, 4), getitem___79930, int_79928)
    
    # Assigning a type to the variable 'tuple_var_assignment_78298' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'tuple_var_assignment_78298', subscript_call_result_79931)
    
    # Assigning a Subscript to a Name (line 653):
    
    # Obtaining the type of the subscript
    int_79932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 4), 'int')
    # Getting the type of 'tck' (line 653)
    tck_79933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___79934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 4), tck_79933, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_79935 = invoke(stypy.reporting.localization.Localization(__file__, 653, 4), getitem___79934, int_79932)
    
    # Assigning a type to the variable 'tuple_var_assignment_78299' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'tuple_var_assignment_78299', subscript_call_result_79935)
    
    # Assigning a Name to a Name (line 653):
    # Getting the type of 'tuple_var_assignment_78297' (line 653)
    tuple_var_assignment_78297_79936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'tuple_var_assignment_78297')
    # Assigning a type to the variable 't' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 't', tuple_var_assignment_78297_79936)
    
    # Assigning a Name to a Name (line 653):
    # Getting the type of 'tuple_var_assignment_78298' (line 653)
    tuple_var_assignment_78298_79937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'tuple_var_assignment_78298')
    # Assigning a type to the variable 'c' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 7), 'c', tuple_var_assignment_78298_79937)
    
    # Assigning a Name to a Name (line 653):
    # Getting the type of 'tuple_var_assignment_78299' (line 653)
    tuple_var_assignment_78299_79938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'tuple_var_assignment_78299')
    # Assigning a type to the variable 'k' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 10), 'k', tuple_var_assignment_78299_79938)
    
    
    # SSA begins for try-except statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    int_79939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 13), 'int')
    
    # Obtaining the type of the subscript
    int_79940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 10), 'int')
    # Getting the type of 'c' (line 655)
    c_79941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'c')
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___79942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 8), c_79941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_79943 = invoke(stypy.reporting.localization.Localization(__file__, 655, 8), getitem___79942, int_79940)
    
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___79944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 8), subscript_call_result_79943, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 655)
    subscript_call_result_79945 = invoke(stypy.reporting.localization.Localization(__file__, 655, 8), getitem___79944, int_79939)
    
    
    # Assigning a Name to a Name (line 656):
    
    # Assigning a Name to a Name (line 656):
    # Getting the type of 'True' (line 656)
    True_79946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 21), 'True')
    # Assigning a type to the variable 'parametric' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'parametric', True_79946)
    # SSA branch for the except part of a try statement (line 654)
    # SSA branch for the except '<any exception>' branch of a try statement (line 654)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 658):
    
    # Assigning a Name to a Name (line 658):
    # Getting the type of 'False' (line 658)
    False_79947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 21), 'False')
    # Assigning a type to the variable 'parametric' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'parametric', False_79947)
    # SSA join for try-except statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'parametric' (line 659)
    parametric_79948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 7), 'parametric')
    # Testing the type of an if condition (line 659)
    if_condition_79949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 4), parametric_79948)
    # Assigning a type to the variable 'if_condition_79949' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'if_condition_79949', if_condition_79949)
    # SSA begins for if statement (line 659)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to list(...): (line 660)
    # Processing the call arguments (line 660)
    
    # Call to map(...): (line 660)
    # Processing the call arguments (line 660)

    @norecursion
    def _stypy_temp_lambda_50(localization, *varargs, **kwargs):
        global module_type_store
        # Getting the type of 'a' (line 660)
        a_79952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 36), 'a', False)
        # Getting the type of 'b' (line 660)
        b_79953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 41), 'b', False)
        # Getting the type of 't' (line 660)
        t_79954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 46), 't', False)
        # Getting the type of 'k' (line 660)
        k_79955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 51), 'k', False)
        # Assign values to the parameters with defaults
        defaults = [a_79952, b_79953, t_79954, k_79955]
        # Create a new context for function '_stypy_temp_lambda_50'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_50', 660, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_50.stypy_localization = localization
        _stypy_temp_lambda_50.stypy_type_of_self = None
        _stypy_temp_lambda_50.stypy_type_store = module_type_store
        _stypy_temp_lambda_50.stypy_function_name = '_stypy_temp_lambda_50'
        _stypy_temp_lambda_50.stypy_param_names_list = ['c', 'a', 'b', 't', 'k']
        _stypy_temp_lambda_50.stypy_varargs_param_name = None
        _stypy_temp_lambda_50.stypy_kwargs_param_name = None
        _stypy_temp_lambda_50.stypy_call_defaults = defaults
        _stypy_temp_lambda_50.stypy_call_varargs = varargs
        _stypy_temp_lambda_50.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_50', ['c', 'a', 'b', 't', 'k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_50', ['c', 'a', 'b', 't', 'k'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to splint(...): (line 661)
        # Processing the call arguments (line 661)
        # Getting the type of 'a' (line 661)
        a_79957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 31), 'a', False)
        # Getting the type of 'b' (line 661)
        b_79958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 34), 'b', False)
        
        # Obtaining an instance of the builtin type 'list' (line 661)
        list_79959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 661)
        # Adding element type (line 661)
        # Getting the type of 't' (line 661)
        t_79960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 38), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 661, 37), list_79959, t_79960)
        # Adding element type (line 661)
        # Getting the type of 'c' (line 661)
        c_79961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 41), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 661, 37), list_79959, c_79961)
        # Adding element type (line 661)
        # Getting the type of 'k' (line 661)
        k_79962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 44), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 661, 37), list_79959, k_79962)
        
        # Processing the call keyword arguments (line 661)
        kwargs_79963 = {}
        # Getting the type of 'splint' (line 661)
        splint_79956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 24), 'splint', False)
        # Calling splint(args, kwargs) (line 661)
        splint_call_result_79964 = invoke(stypy.reporting.localization.Localization(__file__, 661, 24), splint_79956, *[a_79957, b_79958, list_79959], **kwargs_79963)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 24), 'stypy_return_type', splint_call_result_79964)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_50' in the type store
        # Getting the type of 'stypy_return_type' (line 660)
        stypy_return_type_79965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_79965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_50'
        return stypy_return_type_79965

    # Assigning a type to the variable '_stypy_temp_lambda_50' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 24), '_stypy_temp_lambda_50', _stypy_temp_lambda_50)
    # Getting the type of '_stypy_temp_lambda_50' (line 660)
    _stypy_temp_lambda_50_79966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 24), '_stypy_temp_lambda_50')
    # Getting the type of 'c' (line 661)
    c_79967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 49), 'c', False)
    # Processing the call keyword arguments (line 660)
    kwargs_79968 = {}
    # Getting the type of 'map' (line 660)
    map_79951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 20), 'map', False)
    # Calling map(args, kwargs) (line 660)
    map_call_result_79969 = invoke(stypy.reporting.localization.Localization(__file__, 660, 20), map_79951, *[_stypy_temp_lambda_50_79966, c_79967], **kwargs_79968)
    
    # Processing the call keyword arguments (line 660)
    kwargs_79970 = {}
    # Getting the type of 'list' (line 660)
    list_79950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 15), 'list', False)
    # Calling list(args, kwargs) (line 660)
    list_call_result_79971 = invoke(stypy.reporting.localization.Localization(__file__, 660, 15), list_79950, *[map_call_result_79969], **kwargs_79970)
    
    # Assigning a type to the variable 'stypy_return_type' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'stypy_return_type', list_call_result_79971)
    # SSA branch for the else part of an if statement (line 659)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 663):
    
    # Assigning a Subscript to a Name (line 663):
    
    # Obtaining the type of the subscript
    int_79972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 8), 'int')
    
    # Call to _splint(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 't' (line 663)
    t_79975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 37), 't', False)
    # Getting the type of 'c' (line 663)
    c_79976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 40), 'c', False)
    # Getting the type of 'k' (line 663)
    k_79977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 43), 'k', False)
    # Getting the type of 'a' (line 663)
    a_79978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 46), 'a', False)
    # Getting the type of 'b' (line 663)
    b_79979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 49), 'b', False)
    # Processing the call keyword arguments (line 663)
    kwargs_79980 = {}
    # Getting the type of '_fitpack' (line 663)
    _fitpack_79973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 20), '_fitpack', False)
    # Obtaining the member '_splint' of a type (line 663)
    _splint_79974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 20), _fitpack_79973, '_splint')
    # Calling _splint(args, kwargs) (line 663)
    _splint_call_result_79981 = invoke(stypy.reporting.localization.Localization(__file__, 663, 20), _splint_79974, *[t_79975, c_79976, k_79977, a_79978, b_79979], **kwargs_79980)
    
    # Obtaining the member '__getitem__' of a type (line 663)
    getitem___79982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 8), _splint_call_result_79981, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 663)
    subscript_call_result_79983 = invoke(stypy.reporting.localization.Localization(__file__, 663, 8), getitem___79982, int_79972)
    
    # Assigning a type to the variable 'tuple_var_assignment_78300' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'tuple_var_assignment_78300', subscript_call_result_79983)
    
    # Assigning a Subscript to a Name (line 663):
    
    # Obtaining the type of the subscript
    int_79984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 8), 'int')
    
    # Call to _splint(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 't' (line 663)
    t_79987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 37), 't', False)
    # Getting the type of 'c' (line 663)
    c_79988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 40), 'c', False)
    # Getting the type of 'k' (line 663)
    k_79989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 43), 'k', False)
    # Getting the type of 'a' (line 663)
    a_79990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 46), 'a', False)
    # Getting the type of 'b' (line 663)
    b_79991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 49), 'b', False)
    # Processing the call keyword arguments (line 663)
    kwargs_79992 = {}
    # Getting the type of '_fitpack' (line 663)
    _fitpack_79985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 20), '_fitpack', False)
    # Obtaining the member '_splint' of a type (line 663)
    _splint_79986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 20), _fitpack_79985, '_splint')
    # Calling _splint(args, kwargs) (line 663)
    _splint_call_result_79993 = invoke(stypy.reporting.localization.Localization(__file__, 663, 20), _splint_79986, *[t_79987, c_79988, k_79989, a_79990, b_79991], **kwargs_79992)
    
    # Obtaining the member '__getitem__' of a type (line 663)
    getitem___79994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 8), _splint_call_result_79993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 663)
    subscript_call_result_79995 = invoke(stypy.reporting.localization.Localization(__file__, 663, 8), getitem___79994, int_79984)
    
    # Assigning a type to the variable 'tuple_var_assignment_78301' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'tuple_var_assignment_78301', subscript_call_result_79995)
    
    # Assigning a Name to a Name (line 663):
    # Getting the type of 'tuple_var_assignment_78300' (line 663)
    tuple_var_assignment_78300_79996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'tuple_var_assignment_78300')
    # Assigning a type to the variable 'aint' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'aint', tuple_var_assignment_78300_79996)
    
    # Assigning a Name to a Name (line 663):
    # Getting the type of 'tuple_var_assignment_78301' (line 663)
    tuple_var_assignment_78301_79997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'tuple_var_assignment_78301')
    # Assigning a type to the variable 'wrk' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 14), 'wrk', tuple_var_assignment_78301_79997)
    
    # Getting the type of 'full_output' (line 664)
    full_output_79998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 11), 'full_output')
    # Testing the type of an if condition (line 664)
    if_condition_79999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 664, 8), full_output_79998)
    # Assigning a type to the variable 'if_condition_79999' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'if_condition_79999', if_condition_79999)
    # SSA begins for if statement (line 664)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 665)
    tuple_80000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 665)
    # Adding element type (line 665)
    # Getting the type of 'aint' (line 665)
    aint_80001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 19), 'aint')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 19), tuple_80000, aint_80001)
    # Adding element type (line 665)
    # Getting the type of 'wrk' (line 665)
    wrk_80002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 25), 'wrk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 19), tuple_80000, wrk_80002)
    
    # Assigning a type to the variable 'stypy_return_type' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'stypy_return_type', tuple_80000)
    # SSA branch for the else part of an if statement (line 664)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'aint' (line 667)
    aint_80003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 19), 'aint')
    # Assigning a type to the variable 'stypy_return_type' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'stypy_return_type', aint_80003)
    # SSA join for if statement (line 664)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 659)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'splint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splint' in the type store
    # Getting the type of 'stypy_return_type' (line 609)
    stypy_return_type_80004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_80004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splint'
    return stypy_return_type_80004

# Assigning a type to the variable 'splint' (line 609)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 0), 'splint', splint)

@norecursion
def sproot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_80005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 21), 'int')
    defaults = [int_80005]
    # Create a new context for function 'sproot'
    module_type_store = module_type_store.open_function_context('sproot', 670, 0, False)
    
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

    str_80006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, (-1)), 'str', '\n    Find the roots of a cubic B-spline.\n\n    Given the knots (>=8) and coefficients of a cubic B-spline return the\n    roots of the spline.\n\n    Parameters\n    ----------\n    tck : tuple\n        A tuple (t,c,k) containing the vector of knots,\n        the B-spline coefficients, and the degree of the spline.\n        The number of knots must be >= 8, and the degree must be 3.\n        The knots must be a montonically increasing sequence.\n    mest : int, optional\n        An estimate of the number of zeros (Default is 10).\n\n    Returns\n    -------\n    zeros : ndarray\n        An array giving the roots of the spline.\n\n    See also\n    --------\n    splprep, splrep, splint, spalde, splev\n    bisplrep, bisplev\n    UnivariateSpline, BivariateSpline\n\n\n    References\n    ----------\n    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation\n        Theory, 6, p.50-62, 1972.\n    .. [2] M.G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths\n        Applics, 10, p.134-149, 1972.\n    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs\n        on Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    # Assigning a Name to a Tuple (line 709):
    
    # Assigning a Subscript to a Name (line 709):
    
    # Obtaining the type of the subscript
    int_80007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 4), 'int')
    # Getting the type of 'tck' (line 709)
    tck_80008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 709)
    getitem___80009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 4), tck_80008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 709)
    subscript_call_result_80010 = invoke(stypy.reporting.localization.Localization(__file__, 709, 4), getitem___80009, int_80007)
    
    # Assigning a type to the variable 'tuple_var_assignment_78302' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'tuple_var_assignment_78302', subscript_call_result_80010)
    
    # Assigning a Subscript to a Name (line 709):
    
    # Obtaining the type of the subscript
    int_80011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 4), 'int')
    # Getting the type of 'tck' (line 709)
    tck_80012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 709)
    getitem___80013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 4), tck_80012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 709)
    subscript_call_result_80014 = invoke(stypy.reporting.localization.Localization(__file__, 709, 4), getitem___80013, int_80011)
    
    # Assigning a type to the variable 'tuple_var_assignment_78303' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'tuple_var_assignment_78303', subscript_call_result_80014)
    
    # Assigning a Subscript to a Name (line 709):
    
    # Obtaining the type of the subscript
    int_80015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 4), 'int')
    # Getting the type of 'tck' (line 709)
    tck_80016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 709)
    getitem___80017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 4), tck_80016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 709)
    subscript_call_result_80018 = invoke(stypy.reporting.localization.Localization(__file__, 709, 4), getitem___80017, int_80015)
    
    # Assigning a type to the variable 'tuple_var_assignment_78304' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'tuple_var_assignment_78304', subscript_call_result_80018)
    
    # Assigning a Name to a Name (line 709):
    # Getting the type of 'tuple_var_assignment_78302' (line 709)
    tuple_var_assignment_78302_80019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'tuple_var_assignment_78302')
    # Assigning a type to the variable 't' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 't', tuple_var_assignment_78302_80019)
    
    # Assigning a Name to a Name (line 709):
    # Getting the type of 'tuple_var_assignment_78303' (line 709)
    tuple_var_assignment_78303_80020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'tuple_var_assignment_78303')
    # Assigning a type to the variable 'c' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 7), 'c', tuple_var_assignment_78303_80020)
    
    # Assigning a Name to a Name (line 709):
    # Getting the type of 'tuple_var_assignment_78304' (line 709)
    tuple_var_assignment_78304_80021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'tuple_var_assignment_78304')
    # Assigning a type to the variable 'k' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 10), 'k', tuple_var_assignment_78304_80021)
    
    
    # Getting the type of 'k' (line 710)
    k_80022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 7), 'k')
    int_80023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 12), 'int')
    # Applying the binary operator '!=' (line 710)
    result_ne_80024 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 7), '!=', k_80022, int_80023)
    
    # Testing the type of an if condition (line 710)
    if_condition_80025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 710, 4), result_ne_80024)
    # Assigning a type to the variable 'if_condition_80025' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'if_condition_80025', if_condition_80025)
    # SSA begins for if statement (line 710)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 711)
    # Processing the call arguments (line 711)
    str_80027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 25), 'str', 'sproot works only for cubic (k=3) splines')
    # Processing the call keyword arguments (line 711)
    kwargs_80028 = {}
    # Getting the type of 'ValueError' (line 711)
    ValueError_80026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 711)
    ValueError_call_result_80029 = invoke(stypy.reporting.localization.Localization(__file__, 711, 14), ValueError_80026, *[str_80027], **kwargs_80028)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 711, 8), ValueError_call_result_80029, 'raise parameter', BaseException)
    # SSA join for if statement (line 710)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 712)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    int_80030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 13), 'int')
    
    # Obtaining the type of the subscript
    int_80031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 10), 'int')
    # Getting the type of 'c' (line 713)
    c_80032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'c')
    # Obtaining the member '__getitem__' of a type (line 713)
    getitem___80033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), c_80032, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 713)
    subscript_call_result_80034 = invoke(stypy.reporting.localization.Localization(__file__, 713, 8), getitem___80033, int_80031)
    
    # Obtaining the member '__getitem__' of a type (line 713)
    getitem___80035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), subscript_call_result_80034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 713)
    subscript_call_result_80036 = invoke(stypy.reporting.localization.Localization(__file__, 713, 8), getitem___80035, int_80030)
    
    
    # Assigning a Name to a Name (line 714):
    
    # Assigning a Name to a Name (line 714):
    # Getting the type of 'True' (line 714)
    True_80037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 21), 'True')
    # Assigning a type to the variable 'parametric' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'parametric', True_80037)
    # SSA branch for the except part of a try statement (line 712)
    # SSA branch for the except '<any exception>' branch of a try statement (line 712)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 716):
    
    # Assigning a Name to a Name (line 716):
    # Getting the type of 'False' (line 716)
    False_80038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 21), 'False')
    # Assigning a type to the variable 'parametric' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'parametric', False_80038)
    # SSA join for try-except statement (line 712)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'parametric' (line 717)
    parametric_80039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 7), 'parametric')
    # Testing the type of an if condition (line 717)
    if_condition_80040 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 717, 4), parametric_80039)
    # Assigning a type to the variable 'if_condition_80040' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'if_condition_80040', if_condition_80040)
    # SSA begins for if statement (line 717)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to list(...): (line 718)
    # Processing the call arguments (line 718)
    
    # Call to map(...): (line 718)
    # Processing the call arguments (line 718)

    @norecursion
    def _stypy_temp_lambda_51(localization, *varargs, **kwargs):
        global module_type_store
        # Getting the type of 't' (line 718)
        t_80043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 36), 't', False)
        # Getting the type of 'k' (line 718)
        k_80044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 41), 'k', False)
        # Getting the type of 'mest' (line 718)
        mest_80045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 49), 'mest', False)
        # Assign values to the parameters with defaults
        defaults = [t_80043, k_80044, mest_80045]
        # Create a new context for function '_stypy_temp_lambda_51'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_51', 718, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_51.stypy_localization = localization
        _stypy_temp_lambda_51.stypy_type_of_self = None
        _stypy_temp_lambda_51.stypy_type_store = module_type_store
        _stypy_temp_lambda_51.stypy_function_name = '_stypy_temp_lambda_51'
        _stypy_temp_lambda_51.stypy_param_names_list = ['c', 't', 'k', 'mest']
        _stypy_temp_lambda_51.stypy_varargs_param_name = None
        _stypy_temp_lambda_51.stypy_kwargs_param_name = None
        _stypy_temp_lambda_51.stypy_call_defaults = defaults
        _stypy_temp_lambda_51.stypy_call_varargs = varargs
        _stypy_temp_lambda_51.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_51', ['c', 't', 'k', 'mest'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_51', ['c', 't', 'k', 'mest'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to sproot(...): (line 719)
        # Processing the call arguments (line 719)
        
        # Obtaining an instance of the builtin type 'list' (line 719)
        list_80047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 719)
        # Adding element type (line 719)
        # Getting the type of 't' (line 719)
        t_80048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 32), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 31), list_80047, t_80048)
        # Adding element type (line 719)
        # Getting the type of 'c' (line 719)
        c_80049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 35), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 31), list_80047, c_80049)
        # Adding element type (line 719)
        # Getting the type of 'k' (line 719)
        k_80050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 38), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 31), list_80047, k_80050)
        
        # Getting the type of 'mest' (line 719)
        mest_80051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 42), 'mest', False)
        # Processing the call keyword arguments (line 719)
        kwargs_80052 = {}
        # Getting the type of 'sproot' (line 719)
        sproot_80046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 24), 'sproot', False)
        # Calling sproot(args, kwargs) (line 719)
        sproot_call_result_80053 = invoke(stypy.reporting.localization.Localization(__file__, 719, 24), sproot_80046, *[list_80047, mest_80051], **kwargs_80052)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 24), 'stypy_return_type', sproot_call_result_80053)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_51' in the type store
        # Getting the type of 'stypy_return_type' (line 718)
        stypy_return_type_80054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_80054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_51'
        return stypy_return_type_80054

    # Assigning a type to the variable '_stypy_temp_lambda_51' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 24), '_stypy_temp_lambda_51', _stypy_temp_lambda_51)
    # Getting the type of '_stypy_temp_lambda_51' (line 718)
    _stypy_temp_lambda_51_80055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 24), '_stypy_temp_lambda_51')
    # Getting the type of 'c' (line 719)
    c_80056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 49), 'c', False)
    # Processing the call keyword arguments (line 718)
    kwargs_80057 = {}
    # Getting the type of 'map' (line 718)
    map_80042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 20), 'map', False)
    # Calling map(args, kwargs) (line 718)
    map_call_result_80058 = invoke(stypy.reporting.localization.Localization(__file__, 718, 20), map_80042, *[_stypy_temp_lambda_51_80055, c_80056], **kwargs_80057)
    
    # Processing the call keyword arguments (line 718)
    kwargs_80059 = {}
    # Getting the type of 'list' (line 718)
    list_80041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 15), 'list', False)
    # Calling list(args, kwargs) (line 718)
    list_call_result_80060 = invoke(stypy.reporting.localization.Localization(__file__, 718, 15), list_80041, *[map_call_result_80058], **kwargs_80059)
    
    # Assigning a type to the variable 'stypy_return_type' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'stypy_return_type', list_call_result_80060)
    # SSA branch for the else part of an if statement (line 717)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 721)
    # Processing the call arguments (line 721)
    # Getting the type of 't' (line 721)
    t_80062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 15), 't', False)
    # Processing the call keyword arguments (line 721)
    kwargs_80063 = {}
    # Getting the type of 'len' (line 721)
    len_80061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 11), 'len', False)
    # Calling len(args, kwargs) (line 721)
    len_call_result_80064 = invoke(stypy.reporting.localization.Localization(__file__, 721, 11), len_80061, *[t_80062], **kwargs_80063)
    
    int_80065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 20), 'int')
    # Applying the binary operator '<' (line 721)
    result_lt_80066 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 11), '<', len_call_result_80064, int_80065)
    
    # Testing the type of an if condition (line 721)
    if_condition_80067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 721, 8), result_lt_80066)
    # Assigning a type to the variable 'if_condition_80067' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'if_condition_80067', if_condition_80067)
    # SSA begins for if statement (line 721)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 722)
    # Processing the call arguments (line 722)
    str_80069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 28), 'str', 'The number of knots %d>=8')
    
    # Call to len(...): (line 722)
    # Processing the call arguments (line 722)
    # Getting the type of 't' (line 722)
    t_80071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 62), 't', False)
    # Processing the call keyword arguments (line 722)
    kwargs_80072 = {}
    # Getting the type of 'len' (line 722)
    len_80070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 58), 'len', False)
    # Calling len(args, kwargs) (line 722)
    len_call_result_80073 = invoke(stypy.reporting.localization.Localization(__file__, 722, 58), len_80070, *[t_80071], **kwargs_80072)
    
    # Applying the binary operator '%' (line 722)
    result_mod_80074 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 28), '%', str_80069, len_call_result_80073)
    
    # Processing the call keyword arguments (line 722)
    kwargs_80075 = {}
    # Getting the type of 'TypeError' (line 722)
    TypeError_80068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 722)
    TypeError_call_result_80076 = invoke(stypy.reporting.localization.Localization(__file__, 722, 18), TypeError_80068, *[result_mod_80074], **kwargs_80075)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 722, 12), TypeError_call_result_80076, 'raise parameter', BaseException)
    # SSA join for if statement (line 721)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 723):
    
    # Assigning a Subscript to a Name (line 723):
    
    # Obtaining the type of the subscript
    int_80077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 8), 'int')
    
    # Call to _sproot(...): (line 723)
    # Processing the call arguments (line 723)
    # Getting the type of 't' (line 723)
    t_80080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 34), 't', False)
    # Getting the type of 'c' (line 723)
    c_80081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 37), 'c', False)
    # Getting the type of 'k' (line 723)
    k_80082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 40), 'k', False)
    # Getting the type of 'mest' (line 723)
    mest_80083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 43), 'mest', False)
    # Processing the call keyword arguments (line 723)
    kwargs_80084 = {}
    # Getting the type of '_fitpack' (line 723)
    _fitpack_80078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 17), '_fitpack', False)
    # Obtaining the member '_sproot' of a type (line 723)
    _sproot_80079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 17), _fitpack_80078, '_sproot')
    # Calling _sproot(args, kwargs) (line 723)
    _sproot_call_result_80085 = invoke(stypy.reporting.localization.Localization(__file__, 723, 17), _sproot_80079, *[t_80080, c_80081, k_80082, mest_80083], **kwargs_80084)
    
    # Obtaining the member '__getitem__' of a type (line 723)
    getitem___80086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 8), _sproot_call_result_80085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 723)
    subscript_call_result_80087 = invoke(stypy.reporting.localization.Localization(__file__, 723, 8), getitem___80086, int_80077)
    
    # Assigning a type to the variable 'tuple_var_assignment_78305' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'tuple_var_assignment_78305', subscript_call_result_80087)
    
    # Assigning a Subscript to a Name (line 723):
    
    # Obtaining the type of the subscript
    int_80088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 8), 'int')
    
    # Call to _sproot(...): (line 723)
    # Processing the call arguments (line 723)
    # Getting the type of 't' (line 723)
    t_80091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 34), 't', False)
    # Getting the type of 'c' (line 723)
    c_80092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 37), 'c', False)
    # Getting the type of 'k' (line 723)
    k_80093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 40), 'k', False)
    # Getting the type of 'mest' (line 723)
    mest_80094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 43), 'mest', False)
    # Processing the call keyword arguments (line 723)
    kwargs_80095 = {}
    # Getting the type of '_fitpack' (line 723)
    _fitpack_80089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 17), '_fitpack', False)
    # Obtaining the member '_sproot' of a type (line 723)
    _sproot_80090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 17), _fitpack_80089, '_sproot')
    # Calling _sproot(args, kwargs) (line 723)
    _sproot_call_result_80096 = invoke(stypy.reporting.localization.Localization(__file__, 723, 17), _sproot_80090, *[t_80091, c_80092, k_80093, mest_80094], **kwargs_80095)
    
    # Obtaining the member '__getitem__' of a type (line 723)
    getitem___80097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 8), _sproot_call_result_80096, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 723)
    subscript_call_result_80098 = invoke(stypy.reporting.localization.Localization(__file__, 723, 8), getitem___80097, int_80088)
    
    # Assigning a type to the variable 'tuple_var_assignment_78306' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'tuple_var_assignment_78306', subscript_call_result_80098)
    
    # Assigning a Name to a Name (line 723):
    # Getting the type of 'tuple_var_assignment_78305' (line 723)
    tuple_var_assignment_78305_80099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'tuple_var_assignment_78305')
    # Assigning a type to the variable 'z' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'z', tuple_var_assignment_78305_80099)
    
    # Assigning a Name to a Name (line 723):
    # Getting the type of 'tuple_var_assignment_78306' (line 723)
    tuple_var_assignment_78306_80100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'tuple_var_assignment_78306')
    # Assigning a type to the variable 'ier' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 11), 'ier', tuple_var_assignment_78306_80100)
    
    
    # Getting the type of 'ier' (line 724)
    ier_80101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 11), 'ier')
    int_80102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 18), 'int')
    # Applying the binary operator '==' (line 724)
    result_eq_80103 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 11), '==', ier_80101, int_80102)
    
    # Testing the type of an if condition (line 724)
    if_condition_80104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 724, 8), result_eq_80103)
    # Assigning a type to the variable 'if_condition_80104' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'if_condition_80104', if_condition_80104)
    # SSA begins for if statement (line 724)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 725)
    # Processing the call arguments (line 725)
    str_80106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 28), 'str', 'Invalid input data. t1<=..<=t4<t5<..<tn-3<=..<=tn must hold.')
    # Processing the call keyword arguments (line 725)
    kwargs_80107 = {}
    # Getting the type of 'TypeError' (line 725)
    TypeError_80105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 725)
    TypeError_call_result_80108 = invoke(stypy.reporting.localization.Localization(__file__, 725, 18), TypeError_80105, *[str_80106], **kwargs_80107)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 725, 12), TypeError_call_result_80108, 'raise parameter', BaseException)
    # SSA join for if statement (line 724)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ier' (line 727)
    ier_80109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 11), 'ier')
    int_80110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 18), 'int')
    # Applying the binary operator '==' (line 727)
    result_eq_80111 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 11), '==', ier_80109, int_80110)
    
    # Testing the type of an if condition (line 727)
    if_condition_80112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 727, 8), result_eq_80111)
    # Assigning a type to the variable 'if_condition_80112' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'if_condition_80112', if_condition_80112)
    # SSA begins for if statement (line 727)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'z' (line 728)
    z_80113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 19), 'z')
    # Assigning a type to the variable 'stypy_return_type' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 12), 'stypy_return_type', z_80113)
    # SSA join for if statement (line 727)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ier' (line 729)
    ier_80114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 11), 'ier')
    int_80115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 18), 'int')
    # Applying the binary operator '==' (line 729)
    result_eq_80116 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 11), '==', ier_80114, int_80115)
    
    # Testing the type of an if condition (line 729)
    if_condition_80117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 8), result_eq_80116)
    # Assigning a type to the variable 'if_condition_80117' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'if_condition_80117', if_condition_80117)
    # SSA begins for if statement (line 729)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 730)
    # Processing the call arguments (line 730)
    
    # Call to RuntimeWarning(...): (line 730)
    # Processing the call arguments (line 730)
    str_80121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 41), 'str', 'The number of zeros exceeds mest')
    # Processing the call keyword arguments (line 730)
    kwargs_80122 = {}
    # Getting the type of 'RuntimeWarning' (line 730)
    RuntimeWarning_80120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 26), 'RuntimeWarning', False)
    # Calling RuntimeWarning(args, kwargs) (line 730)
    RuntimeWarning_call_result_80123 = invoke(stypy.reporting.localization.Localization(__file__, 730, 26), RuntimeWarning_80120, *[str_80121], **kwargs_80122)
    
    # Processing the call keyword arguments (line 730)
    kwargs_80124 = {}
    # Getting the type of 'warnings' (line 730)
    warnings_80118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 730)
    warn_80119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 12), warnings_80118, 'warn')
    # Calling warn(args, kwargs) (line 730)
    warn_call_result_80125 = invoke(stypy.reporting.localization.Localization(__file__, 730, 12), warn_80119, *[RuntimeWarning_call_result_80123], **kwargs_80124)
    
    # Getting the type of 'z' (line 731)
    z_80126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 19), 'z')
    # Assigning a type to the variable 'stypy_return_type' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 12), 'stypy_return_type', z_80126)
    # SSA join for if statement (line 729)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to TypeError(...): (line 732)
    # Processing the call arguments (line 732)
    str_80128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 24), 'str', 'Unknown error')
    # Processing the call keyword arguments (line 732)
    kwargs_80129 = {}
    # Getting the type of 'TypeError' (line 732)
    TypeError_80127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 732)
    TypeError_call_result_80130 = invoke(stypy.reporting.localization.Localization(__file__, 732, 14), TypeError_80127, *[str_80128], **kwargs_80129)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 732, 8), TypeError_call_result_80130, 'raise parameter', BaseException)
    # SSA join for if statement (line 717)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'sproot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sproot' in the type store
    # Getting the type of 'stypy_return_type' (line 670)
    stypy_return_type_80131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_80131)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sproot'
    return stypy_return_type_80131

# Assigning a type to the variable 'sproot' (line 670)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 0), 'sproot', sproot)

@norecursion
def spalde(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'spalde'
    module_type_store = module_type_store.open_function_context('spalde', 735, 0, False)
    
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

    str_80132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, (-1)), 'str', '\n    Evaluate all derivatives of a B-spline.\n\n    Given the knots and coefficients of a cubic B-spline compute all\n    derivatives up to order k at a point (or set of points).\n\n    Parameters\n    ----------\n    x : array_like\n        A point or a set of points at which to evaluate the derivatives.\n        Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.\n    tck : tuple\n        A tuple (t,c,k) containing the vector of knots,\n        the B-spline coefficients, and the degree of the spline.\n\n    Returns\n    -------\n    results : {ndarray, list of ndarrays}\n        An array (or a list of arrays) containing all derivatives\n        up to order k inclusive for each point `x`.\n\n    See Also\n    --------\n    splprep, splrep, splint, sproot, splev, bisplrep, bisplev,\n    UnivariateSpline, BivariateSpline\n\n    References\n    ----------\n    .. [1] de Boor C : On calculating with b-splines, J. Approximation Theory\n       6 (1972) 50-62.\n    .. [2] Cox M.G. : The numerical evaluation of b-splines, J. Inst. Maths\n       applics 10 (1972) 134-149.\n    .. [3] Dierckx P. : Curve and surface fitting with splines, Monographs on\n       Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    # Assigning a Name to a Tuple (line 772):
    
    # Assigning a Subscript to a Name (line 772):
    
    # Obtaining the type of the subscript
    int_80133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 4), 'int')
    # Getting the type of 'tck' (line 772)
    tck_80134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 772)
    getitem___80135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 4), tck_80134, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 772)
    subscript_call_result_80136 = invoke(stypy.reporting.localization.Localization(__file__, 772, 4), getitem___80135, int_80133)
    
    # Assigning a type to the variable 'tuple_var_assignment_78307' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'tuple_var_assignment_78307', subscript_call_result_80136)
    
    # Assigning a Subscript to a Name (line 772):
    
    # Obtaining the type of the subscript
    int_80137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 4), 'int')
    # Getting the type of 'tck' (line 772)
    tck_80138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 772)
    getitem___80139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 4), tck_80138, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 772)
    subscript_call_result_80140 = invoke(stypy.reporting.localization.Localization(__file__, 772, 4), getitem___80139, int_80137)
    
    # Assigning a type to the variable 'tuple_var_assignment_78308' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'tuple_var_assignment_78308', subscript_call_result_80140)
    
    # Assigning a Subscript to a Name (line 772):
    
    # Obtaining the type of the subscript
    int_80141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 4), 'int')
    # Getting the type of 'tck' (line 772)
    tck_80142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 772)
    getitem___80143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 4), tck_80142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 772)
    subscript_call_result_80144 = invoke(stypy.reporting.localization.Localization(__file__, 772, 4), getitem___80143, int_80141)
    
    # Assigning a type to the variable 'tuple_var_assignment_78309' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'tuple_var_assignment_78309', subscript_call_result_80144)
    
    # Assigning a Name to a Name (line 772):
    # Getting the type of 'tuple_var_assignment_78307' (line 772)
    tuple_var_assignment_78307_80145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'tuple_var_assignment_78307')
    # Assigning a type to the variable 't' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 't', tuple_var_assignment_78307_80145)
    
    # Assigning a Name to a Name (line 772):
    # Getting the type of 'tuple_var_assignment_78308' (line 772)
    tuple_var_assignment_78308_80146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'tuple_var_assignment_78308')
    # Assigning a type to the variable 'c' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 7), 'c', tuple_var_assignment_78308_80146)
    
    # Assigning a Name to a Name (line 772):
    # Getting the type of 'tuple_var_assignment_78309' (line 772)
    tuple_var_assignment_78309_80147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'tuple_var_assignment_78309')
    # Assigning a type to the variable 'k' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 10), 'k', tuple_var_assignment_78309_80147)
    
    
    # SSA begins for try-except statement (line 773)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    int_80148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 13), 'int')
    
    # Obtaining the type of the subscript
    int_80149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 10), 'int')
    # Getting the type of 'c' (line 774)
    c_80150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'c')
    # Obtaining the member '__getitem__' of a type (line 774)
    getitem___80151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 8), c_80150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 774)
    subscript_call_result_80152 = invoke(stypy.reporting.localization.Localization(__file__, 774, 8), getitem___80151, int_80149)
    
    # Obtaining the member '__getitem__' of a type (line 774)
    getitem___80153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 8), subscript_call_result_80152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 774)
    subscript_call_result_80154 = invoke(stypy.reporting.localization.Localization(__file__, 774, 8), getitem___80153, int_80148)
    
    
    # Assigning a Name to a Name (line 775):
    
    # Assigning a Name to a Name (line 775):
    # Getting the type of 'True' (line 775)
    True_80155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 21), 'True')
    # Assigning a type to the variable 'parametric' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'parametric', True_80155)
    # SSA branch for the except part of a try statement (line 773)
    # SSA branch for the except '<any exception>' branch of a try statement (line 773)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 777):
    
    # Assigning a Name to a Name (line 777):
    # Getting the type of 'False' (line 777)
    False_80156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 21), 'False')
    # Assigning a type to the variable 'parametric' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'parametric', False_80156)
    # SSA join for try-except statement (line 773)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'parametric' (line 778)
    parametric_80157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 7), 'parametric')
    # Testing the type of an if condition (line 778)
    if_condition_80158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 778, 4), parametric_80157)
    # Assigning a type to the variable 'if_condition_80158' (line 778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'if_condition_80158', if_condition_80158)
    # SSA begins for if statement (line 778)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to list(...): (line 779)
    # Processing the call arguments (line 779)
    
    # Call to map(...): (line 779)
    # Processing the call arguments (line 779)

    @norecursion
    def _stypy_temp_lambda_52(localization, *varargs, **kwargs):
        global module_type_store
        # Getting the type of 'x' (line 779)
        x_80161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 36), 'x', False)
        # Getting the type of 't' (line 779)
        t_80162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 41), 't', False)
        # Getting the type of 'k' (line 779)
        k_80163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 46), 'k', False)
        # Assign values to the parameters with defaults
        defaults = [x_80161, t_80162, k_80163]
        # Create a new context for function '_stypy_temp_lambda_52'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_52', 779, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_52.stypy_localization = localization
        _stypy_temp_lambda_52.stypy_type_of_self = None
        _stypy_temp_lambda_52.stypy_type_store = module_type_store
        _stypy_temp_lambda_52.stypy_function_name = '_stypy_temp_lambda_52'
        _stypy_temp_lambda_52.stypy_param_names_list = ['c', 'x', 't', 'k']
        _stypy_temp_lambda_52.stypy_varargs_param_name = None
        _stypy_temp_lambda_52.stypy_kwargs_param_name = None
        _stypy_temp_lambda_52.stypy_call_defaults = defaults
        _stypy_temp_lambda_52.stypy_call_varargs = varargs
        _stypy_temp_lambda_52.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_52', ['c', 'x', 't', 'k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_52', ['c', 'x', 't', 'k'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to spalde(...): (line 780)
        # Processing the call arguments (line 780)
        # Getting the type of 'x' (line 780)
        x_80165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 31), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 780)
        list_80166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 780)
        # Adding element type (line 780)
        # Getting the type of 't' (line 780)
        t_80167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 35), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 34), list_80166, t_80167)
        # Adding element type (line 780)
        # Getting the type of 'c' (line 780)
        c_80168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 38), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 34), list_80166, c_80168)
        # Adding element type (line 780)
        # Getting the type of 'k' (line 780)
        k_80169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 41), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 34), list_80166, k_80169)
        
        # Processing the call keyword arguments (line 780)
        kwargs_80170 = {}
        # Getting the type of 'spalde' (line 780)
        spalde_80164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 24), 'spalde', False)
        # Calling spalde(args, kwargs) (line 780)
        spalde_call_result_80171 = invoke(stypy.reporting.localization.Localization(__file__, 780, 24), spalde_80164, *[x_80165, list_80166], **kwargs_80170)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 24), 'stypy_return_type', spalde_call_result_80171)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_52' in the type store
        # Getting the type of 'stypy_return_type' (line 779)
        stypy_return_type_80172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_80172)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_52'
        return stypy_return_type_80172

    # Assigning a type to the variable '_stypy_temp_lambda_52' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 24), '_stypy_temp_lambda_52', _stypy_temp_lambda_52)
    # Getting the type of '_stypy_temp_lambda_52' (line 779)
    _stypy_temp_lambda_52_80173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 24), '_stypy_temp_lambda_52')
    # Getting the type of 'c' (line 780)
    c_80174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 46), 'c', False)
    # Processing the call keyword arguments (line 779)
    kwargs_80175 = {}
    # Getting the type of 'map' (line 779)
    map_80160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 20), 'map', False)
    # Calling map(args, kwargs) (line 779)
    map_call_result_80176 = invoke(stypy.reporting.localization.Localization(__file__, 779, 20), map_80160, *[_stypy_temp_lambda_52_80173, c_80174], **kwargs_80175)
    
    # Processing the call keyword arguments (line 779)
    kwargs_80177 = {}
    # Getting the type of 'list' (line 779)
    list_80159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 15), 'list', False)
    # Calling list(args, kwargs) (line 779)
    list_call_result_80178 = invoke(stypy.reporting.localization.Localization(__file__, 779, 15), list_80159, *[map_call_result_80176], **kwargs_80177)
    
    # Assigning a type to the variable 'stypy_return_type' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'stypy_return_type', list_call_result_80178)
    # SSA branch for the else part of an if statement (line 778)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 782):
    
    # Assigning a Call to a Name (line 782):
    
    # Call to atleast_1d(...): (line 782)
    # Processing the call arguments (line 782)
    # Getting the type of 'x' (line 782)
    x_80180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 23), 'x', False)
    # Processing the call keyword arguments (line 782)
    kwargs_80181 = {}
    # Getting the type of 'atleast_1d' (line 782)
    atleast_1d_80179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 12), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 782)
    atleast_1d_call_result_80182 = invoke(stypy.reporting.localization.Localization(__file__, 782, 12), atleast_1d_80179, *[x_80180], **kwargs_80181)
    
    # Assigning a type to the variable 'x' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'x', atleast_1d_call_result_80182)
    
    
    
    # Call to len(...): (line 783)
    # Processing the call arguments (line 783)
    # Getting the type of 'x' (line 783)
    x_80184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 15), 'x', False)
    # Processing the call keyword arguments (line 783)
    kwargs_80185 = {}
    # Getting the type of 'len' (line 783)
    len_80183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 11), 'len', False)
    # Calling len(args, kwargs) (line 783)
    len_call_result_80186 = invoke(stypy.reporting.localization.Localization(__file__, 783, 11), len_80183, *[x_80184], **kwargs_80185)
    
    int_80187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 20), 'int')
    # Applying the binary operator '>' (line 783)
    result_gt_80188 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 11), '>', len_call_result_80186, int_80187)
    
    # Testing the type of an if condition (line 783)
    if_condition_80189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 783, 8), result_gt_80188)
    # Assigning a type to the variable 'if_condition_80189' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'if_condition_80189', if_condition_80189)
    # SSA begins for if statement (line 783)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to list(...): (line 784)
    # Processing the call arguments (line 784)
    
    # Call to map(...): (line 784)
    # Processing the call arguments (line 784)

    @norecursion
    def _stypy_temp_lambda_53(localization, *varargs, **kwargs):
        global module_type_store
        # Getting the type of 'tck' (line 784)
        tck_80192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 42), 'tck', False)
        # Assign values to the parameters with defaults
        defaults = [tck_80192]
        # Create a new context for function '_stypy_temp_lambda_53'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_53', 784, 28, True)
        # Passed parameters checking function
        _stypy_temp_lambda_53.stypy_localization = localization
        _stypy_temp_lambda_53.stypy_type_of_self = None
        _stypy_temp_lambda_53.stypy_type_store = module_type_store
        _stypy_temp_lambda_53.stypy_function_name = '_stypy_temp_lambda_53'
        _stypy_temp_lambda_53.stypy_param_names_list = ['x', 'tck']
        _stypy_temp_lambda_53.stypy_varargs_param_name = None
        _stypy_temp_lambda_53.stypy_kwargs_param_name = None
        _stypy_temp_lambda_53.stypy_call_defaults = defaults
        _stypy_temp_lambda_53.stypy_call_varargs = varargs
        _stypy_temp_lambda_53.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_53', ['x', 'tck'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_53', ['x', 'tck'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to spalde(...): (line 784)
        # Processing the call arguments (line 784)
        # Getting the type of 'x' (line 784)
        x_80194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 54), 'x', False)
        # Getting the type of 'tck' (line 784)
        tck_80195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 57), 'tck', False)
        # Processing the call keyword arguments (line 784)
        kwargs_80196 = {}
        # Getting the type of 'spalde' (line 784)
        spalde_80193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 47), 'spalde', False)
        # Calling spalde(args, kwargs) (line 784)
        spalde_call_result_80197 = invoke(stypy.reporting.localization.Localization(__file__, 784, 47), spalde_80193, *[x_80194, tck_80195], **kwargs_80196)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 784)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 28), 'stypy_return_type', spalde_call_result_80197)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_53' in the type store
        # Getting the type of 'stypy_return_type' (line 784)
        stypy_return_type_80198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 28), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_80198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_53'
        return stypy_return_type_80198

    # Assigning a type to the variable '_stypy_temp_lambda_53' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 28), '_stypy_temp_lambda_53', _stypy_temp_lambda_53)
    # Getting the type of '_stypy_temp_lambda_53' (line 784)
    _stypy_temp_lambda_53_80199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 28), '_stypy_temp_lambda_53')
    # Getting the type of 'x' (line 784)
    x_80200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 63), 'x', False)
    # Processing the call keyword arguments (line 784)
    kwargs_80201 = {}
    # Getting the type of 'map' (line 784)
    map_80191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 24), 'map', False)
    # Calling map(args, kwargs) (line 784)
    map_call_result_80202 = invoke(stypy.reporting.localization.Localization(__file__, 784, 24), map_80191, *[_stypy_temp_lambda_53_80199, x_80200], **kwargs_80201)
    
    # Processing the call keyword arguments (line 784)
    kwargs_80203 = {}
    # Getting the type of 'list' (line 784)
    list_80190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 19), 'list', False)
    # Calling list(args, kwargs) (line 784)
    list_call_result_80204 = invoke(stypy.reporting.localization.Localization(__file__, 784, 19), list_80190, *[map_call_result_80202], **kwargs_80203)
    
    # Assigning a type to the variable 'stypy_return_type' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 12), 'stypy_return_type', list_call_result_80204)
    # SSA join for if statement (line 783)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 785):
    
    # Assigning a Subscript to a Name (line 785):
    
    # Obtaining the type of the subscript
    int_80205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 8), 'int')
    
    # Call to _spalde(...): (line 785)
    # Processing the call arguments (line 785)
    # Getting the type of 't' (line 785)
    t_80208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 34), 't', False)
    # Getting the type of 'c' (line 785)
    c_80209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 37), 'c', False)
    # Getting the type of 'k' (line 785)
    k_80210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'k', False)
    
    # Obtaining the type of the subscript
    int_80211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 45), 'int')
    # Getting the type of 'x' (line 785)
    x_80212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 43), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___80213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 43), x_80212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_80214 = invoke(stypy.reporting.localization.Localization(__file__, 785, 43), getitem___80213, int_80211)
    
    # Processing the call keyword arguments (line 785)
    kwargs_80215 = {}
    # Getting the type of '_fitpack' (line 785)
    _fitpack_80206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 17), '_fitpack', False)
    # Obtaining the member '_spalde' of a type (line 785)
    _spalde_80207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 17), _fitpack_80206, '_spalde')
    # Calling _spalde(args, kwargs) (line 785)
    _spalde_call_result_80216 = invoke(stypy.reporting.localization.Localization(__file__, 785, 17), _spalde_80207, *[t_80208, c_80209, k_80210, subscript_call_result_80214], **kwargs_80215)
    
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___80217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 8), _spalde_call_result_80216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_80218 = invoke(stypy.reporting.localization.Localization(__file__, 785, 8), getitem___80217, int_80205)
    
    # Assigning a type to the variable 'tuple_var_assignment_78310' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'tuple_var_assignment_78310', subscript_call_result_80218)
    
    # Assigning a Subscript to a Name (line 785):
    
    # Obtaining the type of the subscript
    int_80219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 8), 'int')
    
    # Call to _spalde(...): (line 785)
    # Processing the call arguments (line 785)
    # Getting the type of 't' (line 785)
    t_80222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 34), 't', False)
    # Getting the type of 'c' (line 785)
    c_80223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 37), 'c', False)
    # Getting the type of 'k' (line 785)
    k_80224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'k', False)
    
    # Obtaining the type of the subscript
    int_80225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 45), 'int')
    # Getting the type of 'x' (line 785)
    x_80226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 43), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___80227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 43), x_80226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_80228 = invoke(stypy.reporting.localization.Localization(__file__, 785, 43), getitem___80227, int_80225)
    
    # Processing the call keyword arguments (line 785)
    kwargs_80229 = {}
    # Getting the type of '_fitpack' (line 785)
    _fitpack_80220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 17), '_fitpack', False)
    # Obtaining the member '_spalde' of a type (line 785)
    _spalde_80221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 17), _fitpack_80220, '_spalde')
    # Calling _spalde(args, kwargs) (line 785)
    _spalde_call_result_80230 = invoke(stypy.reporting.localization.Localization(__file__, 785, 17), _spalde_80221, *[t_80222, c_80223, k_80224, subscript_call_result_80228], **kwargs_80229)
    
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___80231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 8), _spalde_call_result_80230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_80232 = invoke(stypy.reporting.localization.Localization(__file__, 785, 8), getitem___80231, int_80219)
    
    # Assigning a type to the variable 'tuple_var_assignment_78311' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'tuple_var_assignment_78311', subscript_call_result_80232)
    
    # Assigning a Name to a Name (line 785):
    # Getting the type of 'tuple_var_assignment_78310' (line 785)
    tuple_var_assignment_78310_80233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'tuple_var_assignment_78310')
    # Assigning a type to the variable 'd' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'd', tuple_var_assignment_78310_80233)
    
    # Assigning a Name to a Name (line 785):
    # Getting the type of 'tuple_var_assignment_78311' (line 785)
    tuple_var_assignment_78311_80234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'tuple_var_assignment_78311')
    # Assigning a type to the variable 'ier' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 11), 'ier', tuple_var_assignment_78311_80234)
    
    
    # Getting the type of 'ier' (line 786)
    ier_80235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 11), 'ier')
    int_80236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 18), 'int')
    # Applying the binary operator '==' (line 786)
    result_eq_80237 = python_operator(stypy.reporting.localization.Localization(__file__, 786, 11), '==', ier_80235, int_80236)
    
    # Testing the type of an if condition (line 786)
    if_condition_80238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 786, 8), result_eq_80237)
    # Assigning a type to the variable 'if_condition_80238' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'if_condition_80238', if_condition_80238)
    # SSA begins for if statement (line 786)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'd' (line 787)
    d_80239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 19), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 12), 'stypy_return_type', d_80239)
    # SSA join for if statement (line 786)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ier' (line 788)
    ier_80240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 11), 'ier')
    int_80241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 18), 'int')
    # Applying the binary operator '==' (line 788)
    result_eq_80242 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 11), '==', ier_80240, int_80241)
    
    # Testing the type of an if condition (line 788)
    if_condition_80243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 788, 8), result_eq_80242)
    # Assigning a type to the variable 'if_condition_80243' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'if_condition_80243', if_condition_80243)
    # SSA begins for if statement (line 788)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 789)
    # Processing the call arguments (line 789)
    str_80245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 28), 'str', 'Invalid input data. t(k)<=x<=t(n-k+1) must hold.')
    # Processing the call keyword arguments (line 789)
    kwargs_80246 = {}
    # Getting the type of 'TypeError' (line 789)
    TypeError_80244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 789)
    TypeError_call_result_80247 = invoke(stypy.reporting.localization.Localization(__file__, 789, 18), TypeError_80244, *[str_80245], **kwargs_80246)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 789, 12), TypeError_call_result_80247, 'raise parameter', BaseException)
    # SSA join for if statement (line 788)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to TypeError(...): (line 790)
    # Processing the call arguments (line 790)
    str_80249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 24), 'str', 'Unknown error')
    # Processing the call keyword arguments (line 790)
    kwargs_80250 = {}
    # Getting the type of 'TypeError' (line 790)
    TypeError_80248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 790)
    TypeError_call_result_80251 = invoke(stypy.reporting.localization.Localization(__file__, 790, 14), TypeError_80248, *[str_80249], **kwargs_80250)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 790, 8), TypeError_call_result_80251, 'raise parameter', BaseException)
    # SSA join for if statement (line 778)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'spalde(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spalde' in the type store
    # Getting the type of 'stypy_return_type' (line 735)
    stypy_return_type_80252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_80252)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spalde'
    return stypy_return_type_80252

# Assigning a type to the variable 'spalde' (line 735)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 0), 'spalde', spalde)

# Assigning a Dict to a Name (line 795):

# Assigning a Dict to a Name (line 795):

# Obtaining an instance of the builtin type 'dict' (line 795)
dict_80253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 795)
# Adding element type (key, value) (line 795)
str_80254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 17), 'str', 'tx')

# Call to array(...): (line 795)
# Processing the call arguments (line 795)

# Obtaining an instance of the builtin type 'list' (line 795)
list_80256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 795)

# Getting the type of 'float' (line 795)
float_80257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 33), 'float', False)
# Processing the call keyword arguments (line 795)
kwargs_80258 = {}
# Getting the type of 'array' (line 795)
array_80255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 23), 'array', False)
# Calling array(args, kwargs) (line 795)
array_call_result_80259 = invoke(stypy.reporting.localization.Localization(__file__, 795, 23), array_80255, *[list_80256, float_80257], **kwargs_80258)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 16), dict_80253, (str_80254, array_call_result_80259))
# Adding element type (key, value) (line 795)
str_80260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 41), 'str', 'ty')

# Call to array(...): (line 795)
# Processing the call arguments (line 795)

# Obtaining an instance of the builtin type 'list' (line 795)
list_80262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 53), 'list')
# Adding type elements to the builtin type 'list' instance (line 795)

# Getting the type of 'float' (line 795)
float_80263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 57), 'float', False)
# Processing the call keyword arguments (line 795)
kwargs_80264 = {}
# Getting the type of 'array' (line 795)
array_80261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 47), 'array', False)
# Calling array(args, kwargs) (line 795)
array_call_result_80265 = invoke(stypy.reporting.localization.Localization(__file__, 795, 47), array_80261, *[list_80262, float_80263], **kwargs_80264)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 16), dict_80253, (str_80260, array_call_result_80265))
# Adding element type (key, value) (line 795)
str_80266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 17), 'str', 'wrk')

# Call to array(...): (line 796)
# Processing the call arguments (line 796)

# Obtaining an instance of the builtin type 'list' (line 796)
list_80268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 796)

# Getting the type of 'float' (line 796)
float_80269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 34), 'float', False)
# Processing the call keyword arguments (line 796)
kwargs_80270 = {}
# Getting the type of 'array' (line 796)
array_80267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 24), 'array', False)
# Calling array(args, kwargs) (line 796)
array_call_result_80271 = invoke(stypy.reporting.localization.Localization(__file__, 796, 24), array_80267, *[list_80268, float_80269], **kwargs_80270)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 16), dict_80253, (str_80266, array_call_result_80271))
# Adding element type (key, value) (line 795)
str_80272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 42), 'str', 'iwrk')

# Call to array(...): (line 796)
# Processing the call arguments (line 796)

# Obtaining an instance of the builtin type 'list' (line 796)
list_80274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 56), 'list')
# Adding type elements to the builtin type 'list' instance (line 796)

# Getting the type of 'intc' (line 796)
intc_80275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 60), 'intc', False)
# Processing the call keyword arguments (line 796)
kwargs_80276 = {}
# Getting the type of 'array' (line 796)
array_80273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 50), 'array', False)
# Calling array(args, kwargs) (line 796)
array_call_result_80277 = invoke(stypy.reporting.localization.Localization(__file__, 796, 50), array_80273, *[list_80274, intc_80275], **kwargs_80276)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 16), dict_80253, (str_80272, array_call_result_80277))

# Assigning a type to the variable '_surfit_cache' (line 795)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 0), '_surfit_cache', dict_80253)

@norecursion
def bisplrep(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 799)
    None_80278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 24), 'None')
    # Getting the type of 'None' (line 799)
    None_80279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 33), 'None')
    # Getting the type of 'None' (line 799)
    None_80280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 42), 'None')
    # Getting the type of 'None' (line 799)
    None_80281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 51), 'None')
    # Getting the type of 'None' (line 799)
    None_80282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 60), 'None')
    int_80283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 16), 'int')
    int_80284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 22), 'int')
    int_80285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 30), 'int')
    # Getting the type of 'None' (line 800)
    None_80286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 35), 'None')
    float_80287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 45), 'float')
    # Getting the type of 'None' (line 800)
    None_80288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 55), 'None')
    # Getting the type of 'None' (line 800)
    None_80289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 64), 'None')
    int_80290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 25), 'int')
    # Getting the type of 'None' (line 801)
    None_80291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 34), 'None')
    # Getting the type of 'None' (line 801)
    None_80292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 46), 'None')
    int_80293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 58), 'int')
    defaults = [None_80278, None_80279, None_80280, None_80281, None_80282, int_80283, int_80284, int_80285, None_80286, float_80287, None_80288, None_80289, int_80290, None_80291, None_80292, int_80293]
    # Create a new context for function 'bisplrep'
    module_type_store = module_type_store.open_function_context('bisplrep', 799, 0, False)
    
    # Passed parameters checking function
    bisplrep.stypy_localization = localization
    bisplrep.stypy_type_of_self = None
    bisplrep.stypy_type_store = module_type_store
    bisplrep.stypy_function_name = 'bisplrep'
    bisplrep.stypy_param_names_list = ['x', 'y', 'z', 'w', 'xb', 'xe', 'yb', 'ye', 'kx', 'ky', 'task', 's', 'eps', 'tx', 'ty', 'full_output', 'nxest', 'nyest', 'quiet']
    bisplrep.stypy_varargs_param_name = None
    bisplrep.stypy_kwargs_param_name = None
    bisplrep.stypy_call_defaults = defaults
    bisplrep.stypy_call_varargs = varargs
    bisplrep.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bisplrep', ['x', 'y', 'z', 'w', 'xb', 'xe', 'yb', 'ye', 'kx', 'ky', 'task', 's', 'eps', 'tx', 'ty', 'full_output', 'nxest', 'nyest', 'quiet'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bisplrep', localization, ['x', 'y', 'z', 'w', 'xb', 'xe', 'yb', 'ye', 'kx', 'ky', 'task', 's', 'eps', 'tx', 'ty', 'full_output', 'nxest', 'nyest', 'quiet'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bisplrep(...)' code ##################

    str_80294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, (-1)), 'str', '\n    Find a bivariate B-spline representation of a surface.\n\n    Given a set of data points (x[i], y[i], z[i]) representing a surface\n    z=f(x,y), compute a B-spline representation of the surface. Based on\n    the routine SURFIT from FITPACK.\n\n    Parameters\n    ----------\n    x, y, z : ndarray\n        Rank-1 arrays of data points.\n    w : ndarray, optional\n        Rank-1 array of weights. By default ``w=np.ones(len(x))``.\n    xb, xe : float, optional\n        End points of approximation interval in `x`.\n        By default ``xb = x.min(), xe=x.max()``.\n    yb, ye : float, optional\n        End points of approximation interval in `y`.\n        By default ``yb=y.min(), ye = y.max()``.\n    kx, ky : int, optional\n        The degrees of the spline (1 <= kx, ky <= 5).\n        Third order (kx=ky=3) is recommended.\n    task : int, optional\n        If task=0, find knots in x and y and coefficients for a given\n        smoothing factor, s.\n        If task=1, find knots and coefficients for another value of the\n        smoothing factor, s.  bisplrep must have been previously called\n        with task=0 or task=1.\n        If task=-1, find coefficients for a given set of knots tx, ty.\n    s : float, optional\n        A non-negative smoothing factor.  If weights correspond\n        to the inverse of the standard-deviation of the errors in z,\n        then a good s-value should be found in the range\n        ``(m-sqrt(2*m),m+sqrt(2*m))`` where m=len(x).\n    eps : float, optional\n        A threshold for determining the effective rank of an\n        over-determined linear system of equations (0 < eps < 1).\n        `eps` is not likely to need changing.\n    tx, ty : ndarray, optional\n        Rank-1 arrays of the knots of the spline for task=-1\n    full_output : int, optional\n        Non-zero to return optional outputs.\n    nxest, nyest : int, optional\n        Over-estimates of the total number of knots. If None then\n        ``nxest = max(kx+sqrt(m/2),2*kx+3)``,\n        ``nyest = max(ky+sqrt(m/2),2*ky+3)``.\n    quiet : int, optional\n        Non-zero to suppress printing of messages.\n        This parameter is deprecated; use standard Python warning filters\n        instead.\n\n    Returns\n    -------\n    tck : array_like\n        A list [tx, ty, c, kx, ky] containing the knots (tx, ty) and\n        coefficients (c) of the bivariate B-spline representation of the\n        surface along with the degree of the spline.\n    fp : ndarray\n        The weighted sum of squared residuals of the spline approximation.\n    ier : int\n        An integer flag about splrep success.  Success is indicated if\n        ier<=0. If ier in [1,2,3] an error occurred but was not raised.\n        Otherwise an error is raised.\n    msg : str\n        A message corresponding to the integer flag, ier.\n\n    See Also\n    --------\n    splprep, splrep, splint, sproot, splev\n    UnivariateSpline, BivariateSpline\n\n    Notes\n    -----\n    See `bisplev` to evaluate the value of the B-spline given its tck\n    representation.\n\n    References\n    ----------\n    .. [1] Dierckx P.:An algorithm for surface fitting with spline functions\n       Ima J. Numer. Anal. 1 (1981) 267-283.\n    .. [2] Dierckx P.:An algorithm for surface fitting with spline functions\n       report tw50, Dept. Computer Science,K.U.Leuven, 1980.\n    .. [3] Dierckx P.:Curve and surface fitting with splines, Monographs on\n       Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    # Assigning a Call to a Tuple (line 888):
    
    # Assigning a Subscript to a Name (line 888):
    
    # Obtaining the type of the subscript
    int_80295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 4), 'int')
    
    # Call to map(...): (line 888)
    # Processing the call arguments (line 888)
    # Getting the type of 'ravel' (line 888)
    ravel_80297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 18), 'ravel', False)
    
    # Obtaining an instance of the builtin type 'list' (line 888)
    list_80298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 888)
    # Adding element type (line 888)
    # Getting the type of 'x' (line 888)
    x_80299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 26), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80298, x_80299)
    # Adding element type (line 888)
    # Getting the type of 'y' (line 888)
    y_80300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 29), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80298, y_80300)
    # Adding element type (line 888)
    # Getting the type of 'z' (line 888)
    z_80301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 32), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80298, z_80301)
    
    # Processing the call keyword arguments (line 888)
    kwargs_80302 = {}
    # Getting the type of 'map' (line 888)
    map_80296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 14), 'map', False)
    # Calling map(args, kwargs) (line 888)
    map_call_result_80303 = invoke(stypy.reporting.localization.Localization(__file__, 888, 14), map_80296, *[ravel_80297, list_80298], **kwargs_80302)
    
    # Obtaining the member '__getitem__' of a type (line 888)
    getitem___80304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 4), map_call_result_80303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 888)
    subscript_call_result_80305 = invoke(stypy.reporting.localization.Localization(__file__, 888, 4), getitem___80304, int_80295)
    
    # Assigning a type to the variable 'tuple_var_assignment_78312' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'tuple_var_assignment_78312', subscript_call_result_80305)
    
    # Assigning a Subscript to a Name (line 888):
    
    # Obtaining the type of the subscript
    int_80306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 4), 'int')
    
    # Call to map(...): (line 888)
    # Processing the call arguments (line 888)
    # Getting the type of 'ravel' (line 888)
    ravel_80308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 18), 'ravel', False)
    
    # Obtaining an instance of the builtin type 'list' (line 888)
    list_80309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 888)
    # Adding element type (line 888)
    # Getting the type of 'x' (line 888)
    x_80310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 26), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80309, x_80310)
    # Adding element type (line 888)
    # Getting the type of 'y' (line 888)
    y_80311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 29), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80309, y_80311)
    # Adding element type (line 888)
    # Getting the type of 'z' (line 888)
    z_80312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 32), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80309, z_80312)
    
    # Processing the call keyword arguments (line 888)
    kwargs_80313 = {}
    # Getting the type of 'map' (line 888)
    map_80307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 14), 'map', False)
    # Calling map(args, kwargs) (line 888)
    map_call_result_80314 = invoke(stypy.reporting.localization.Localization(__file__, 888, 14), map_80307, *[ravel_80308, list_80309], **kwargs_80313)
    
    # Obtaining the member '__getitem__' of a type (line 888)
    getitem___80315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 4), map_call_result_80314, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 888)
    subscript_call_result_80316 = invoke(stypy.reporting.localization.Localization(__file__, 888, 4), getitem___80315, int_80306)
    
    # Assigning a type to the variable 'tuple_var_assignment_78313' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'tuple_var_assignment_78313', subscript_call_result_80316)
    
    # Assigning a Subscript to a Name (line 888):
    
    # Obtaining the type of the subscript
    int_80317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 4), 'int')
    
    # Call to map(...): (line 888)
    # Processing the call arguments (line 888)
    # Getting the type of 'ravel' (line 888)
    ravel_80319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 18), 'ravel', False)
    
    # Obtaining an instance of the builtin type 'list' (line 888)
    list_80320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 888)
    # Adding element type (line 888)
    # Getting the type of 'x' (line 888)
    x_80321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 26), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80320, x_80321)
    # Adding element type (line 888)
    # Getting the type of 'y' (line 888)
    y_80322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 29), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80320, y_80322)
    # Adding element type (line 888)
    # Getting the type of 'z' (line 888)
    z_80323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 32), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 25), list_80320, z_80323)
    
    # Processing the call keyword arguments (line 888)
    kwargs_80324 = {}
    # Getting the type of 'map' (line 888)
    map_80318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 14), 'map', False)
    # Calling map(args, kwargs) (line 888)
    map_call_result_80325 = invoke(stypy.reporting.localization.Localization(__file__, 888, 14), map_80318, *[ravel_80319, list_80320], **kwargs_80324)
    
    # Obtaining the member '__getitem__' of a type (line 888)
    getitem___80326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 4), map_call_result_80325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 888)
    subscript_call_result_80327 = invoke(stypy.reporting.localization.Localization(__file__, 888, 4), getitem___80326, int_80317)
    
    # Assigning a type to the variable 'tuple_var_assignment_78314' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'tuple_var_assignment_78314', subscript_call_result_80327)
    
    # Assigning a Name to a Name (line 888):
    # Getting the type of 'tuple_var_assignment_78312' (line 888)
    tuple_var_assignment_78312_80328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'tuple_var_assignment_78312')
    # Assigning a type to the variable 'x' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'x', tuple_var_assignment_78312_80328)
    
    # Assigning a Name to a Name (line 888):
    # Getting the type of 'tuple_var_assignment_78313' (line 888)
    tuple_var_assignment_78313_80329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'tuple_var_assignment_78313')
    # Assigning a type to the variable 'y' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 7), 'y', tuple_var_assignment_78313_80329)
    
    # Assigning a Name to a Name (line 888):
    # Getting the type of 'tuple_var_assignment_78314' (line 888)
    tuple_var_assignment_78314_80330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'tuple_var_assignment_78314')
    # Assigning a type to the variable 'z' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 10), 'z', tuple_var_assignment_78314_80330)
    
    # Assigning a Call to a Name (line 889):
    
    # Assigning a Call to a Name (line 889):
    
    # Call to len(...): (line 889)
    # Processing the call arguments (line 889)
    # Getting the type of 'x' (line 889)
    x_80332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 12), 'x', False)
    # Processing the call keyword arguments (line 889)
    kwargs_80333 = {}
    # Getting the type of 'len' (line 889)
    len_80331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 8), 'len', False)
    # Calling len(args, kwargs) (line 889)
    len_call_result_80334 = invoke(stypy.reporting.localization.Localization(__file__, 889, 8), len_80331, *[x_80332], **kwargs_80333)
    
    # Assigning a type to the variable 'm' (line 889)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 4), 'm', len_call_result_80334)
    
    
    
    # Getting the type of 'm' (line 890)
    m_80335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 12), 'm')
    
    # Call to len(...): (line 890)
    # Processing the call arguments (line 890)
    # Getting the type of 'y' (line 890)
    y_80337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 21), 'y', False)
    # Processing the call keyword arguments (line 890)
    kwargs_80338 = {}
    # Getting the type of 'len' (line 890)
    len_80336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 17), 'len', False)
    # Calling len(args, kwargs) (line 890)
    len_call_result_80339 = invoke(stypy.reporting.localization.Localization(__file__, 890, 17), len_80336, *[y_80337], **kwargs_80338)
    
    # Applying the binary operator '==' (line 890)
    result_eq_80340 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 12), '==', m_80335, len_call_result_80339)
    
    # Call to len(...): (line 890)
    # Processing the call arguments (line 890)
    # Getting the type of 'z' (line 890)
    z_80342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 31), 'z', False)
    # Processing the call keyword arguments (line 890)
    kwargs_80343 = {}
    # Getting the type of 'len' (line 890)
    len_80341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 27), 'len', False)
    # Calling len(args, kwargs) (line 890)
    len_call_result_80344 = invoke(stypy.reporting.localization.Localization(__file__, 890, 27), len_80341, *[z_80342], **kwargs_80343)
    
    # Applying the binary operator '==' (line 890)
    result_eq_80345 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 12), '==', len_call_result_80339, len_call_result_80344)
    # Applying the binary operator '&' (line 890)
    result_and__80346 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 12), '&', result_eq_80340, result_eq_80345)
    
    # Applying the 'not' unary operator (line 890)
    result_not__80347 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 7), 'not', result_and__80346)
    
    # Testing the type of an if condition (line 890)
    if_condition_80348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 890, 4), result_not__80347)
    # Assigning a type to the variable 'if_condition_80348' (line 890)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 4), 'if_condition_80348', if_condition_80348)
    # SSA begins for if statement (line 890)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 891)
    # Processing the call arguments (line 891)
    str_80350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 24), 'str', 'len(x)==len(y)==len(z) must hold.')
    # Processing the call keyword arguments (line 891)
    kwargs_80351 = {}
    # Getting the type of 'TypeError' (line 891)
    TypeError_80349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 891)
    TypeError_call_result_80352 = invoke(stypy.reporting.localization.Localization(__file__, 891, 14), TypeError_80349, *[str_80350], **kwargs_80351)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 891, 8), TypeError_call_result_80352, 'raise parameter', BaseException)
    # SSA join for if statement (line 890)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 892)
    # Getting the type of 'w' (line 892)
    w_80353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 7), 'w')
    # Getting the type of 'None' (line 892)
    None_80354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 12), 'None')
    
    (may_be_80355, more_types_in_union_80356) = may_be_none(w_80353, None_80354)

    if may_be_80355:

        if more_types_in_union_80356:
            # Runtime conditional SSA (line 892)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 893):
        
        # Assigning a Call to a Name (line 893):
        
        # Call to ones(...): (line 893)
        # Processing the call arguments (line 893)
        # Getting the type of 'm' (line 893)
        m_80358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 17), 'm', False)
        # Getting the type of 'float' (line 893)
        float_80359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 20), 'float', False)
        # Processing the call keyword arguments (line 893)
        kwargs_80360 = {}
        # Getting the type of 'ones' (line 893)
        ones_80357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 12), 'ones', False)
        # Calling ones(args, kwargs) (line 893)
        ones_call_result_80361 = invoke(stypy.reporting.localization.Localization(__file__, 893, 12), ones_80357, *[m_80358, float_80359], **kwargs_80360)
        
        # Assigning a type to the variable 'w' (line 893)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 8), 'w', ones_call_result_80361)

        if more_types_in_union_80356:
            # Runtime conditional SSA for else branch (line 892)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_80355) or more_types_in_union_80356):
        
        # Assigning a Call to a Name (line 895):
        
        # Assigning a Call to a Name (line 895):
        
        # Call to atleast_1d(...): (line 895)
        # Processing the call arguments (line 895)
        # Getting the type of 'w' (line 895)
        w_80363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 23), 'w', False)
        # Processing the call keyword arguments (line 895)
        kwargs_80364 = {}
        # Getting the type of 'atleast_1d' (line 895)
        atleast_1d_80362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 12), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 895)
        atleast_1d_call_result_80365 = invoke(stypy.reporting.localization.Localization(__file__, 895, 12), atleast_1d_80362, *[w_80363], **kwargs_80364)
        
        # Assigning a type to the variable 'w' (line 895)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 8), 'w', atleast_1d_call_result_80365)

        if (may_be_80355 and more_types_in_union_80356):
            # SSA join for if statement (line 892)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    
    # Call to len(...): (line 896)
    # Processing the call arguments (line 896)
    # Getting the type of 'w' (line 896)
    w_80367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 15), 'w', False)
    # Processing the call keyword arguments (line 896)
    kwargs_80368 = {}
    # Getting the type of 'len' (line 896)
    len_80366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 11), 'len', False)
    # Calling len(args, kwargs) (line 896)
    len_call_result_80369 = invoke(stypy.reporting.localization.Localization(__file__, 896, 11), len_80366, *[w_80367], **kwargs_80368)
    
    # Getting the type of 'm' (line 896)
    m_80370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 21), 'm')
    # Applying the binary operator '==' (line 896)
    result_eq_80371 = python_operator(stypy.reporting.localization.Localization(__file__, 896, 11), '==', len_call_result_80369, m_80370)
    
    # Applying the 'not' unary operator (line 896)
    result_not__80372 = python_operator(stypy.reporting.localization.Localization(__file__, 896, 7), 'not', result_eq_80371)
    
    # Testing the type of an if condition (line 896)
    if_condition_80373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 896, 4), result_not__80372)
    # Assigning a type to the variable 'if_condition_80373' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'if_condition_80373', if_condition_80373)
    # SSA begins for if statement (line 896)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 897)
    # Processing the call arguments (line 897)
    str_80375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 24), 'str', 'len(w)=%d is not equal to m=%d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 897)
    tuple_80376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 897)
    # Adding element type (line 897)
    
    # Call to len(...): (line 897)
    # Processing the call arguments (line 897)
    # Getting the type of 'w' (line 897)
    w_80378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 64), 'w', False)
    # Processing the call keyword arguments (line 897)
    kwargs_80379 = {}
    # Getting the type of 'len' (line 897)
    len_80377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 60), 'len', False)
    # Calling len(args, kwargs) (line 897)
    len_call_result_80380 = invoke(stypy.reporting.localization.Localization(__file__, 897, 60), len_80377, *[w_80378], **kwargs_80379)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 60), tuple_80376, len_call_result_80380)
    # Adding element type (line 897)
    # Getting the type of 'm' (line 897)
    m_80381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 68), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 60), tuple_80376, m_80381)
    
    # Applying the binary operator '%' (line 897)
    result_mod_80382 = python_operator(stypy.reporting.localization.Localization(__file__, 897, 24), '%', str_80375, tuple_80376)
    
    # Processing the call keyword arguments (line 897)
    kwargs_80383 = {}
    # Getting the type of 'TypeError' (line 897)
    TypeError_80374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 897)
    TypeError_call_result_80384 = invoke(stypy.reporting.localization.Localization(__file__, 897, 14), TypeError_80374, *[result_mod_80382], **kwargs_80383)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 897, 8), TypeError_call_result_80384, 'raise parameter', BaseException)
    # SSA join for if statement (line 896)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 898)
    # Getting the type of 'xb' (line 898)
    xb_80385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 7), 'xb')
    # Getting the type of 'None' (line 898)
    None_80386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 13), 'None')
    
    (may_be_80387, more_types_in_union_80388) = may_be_none(xb_80385, None_80386)

    if may_be_80387:

        if more_types_in_union_80388:
            # Runtime conditional SSA (line 898)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 899):
        
        # Assigning a Call to a Name (line 899):
        
        # Call to min(...): (line 899)
        # Processing the call keyword arguments (line 899)
        kwargs_80391 = {}
        # Getting the type of 'x' (line 899)
        x_80389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 13), 'x', False)
        # Obtaining the member 'min' of a type (line 899)
        min_80390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 13), x_80389, 'min')
        # Calling min(args, kwargs) (line 899)
        min_call_result_80392 = invoke(stypy.reporting.localization.Localization(__file__, 899, 13), min_80390, *[], **kwargs_80391)
        
        # Assigning a type to the variable 'xb' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 8), 'xb', min_call_result_80392)

        if more_types_in_union_80388:
            # SSA join for if statement (line 898)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 900)
    # Getting the type of 'xe' (line 900)
    xe_80393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 7), 'xe')
    # Getting the type of 'None' (line 900)
    None_80394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 13), 'None')
    
    (may_be_80395, more_types_in_union_80396) = may_be_none(xe_80393, None_80394)

    if may_be_80395:

        if more_types_in_union_80396:
            # Runtime conditional SSA (line 900)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 901):
        
        # Assigning a Call to a Name (line 901):
        
        # Call to max(...): (line 901)
        # Processing the call keyword arguments (line 901)
        kwargs_80399 = {}
        # Getting the type of 'x' (line 901)
        x_80397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 13), 'x', False)
        # Obtaining the member 'max' of a type (line 901)
        max_80398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 13), x_80397, 'max')
        # Calling max(args, kwargs) (line 901)
        max_call_result_80400 = invoke(stypy.reporting.localization.Localization(__file__, 901, 13), max_80398, *[], **kwargs_80399)
        
        # Assigning a type to the variable 'xe' (line 901)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'xe', max_call_result_80400)

        if more_types_in_union_80396:
            # SSA join for if statement (line 900)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 902)
    # Getting the type of 'yb' (line 902)
    yb_80401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 7), 'yb')
    # Getting the type of 'None' (line 902)
    None_80402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 13), 'None')
    
    (may_be_80403, more_types_in_union_80404) = may_be_none(yb_80401, None_80402)

    if may_be_80403:

        if more_types_in_union_80404:
            # Runtime conditional SSA (line 902)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 903):
        
        # Assigning a Call to a Name (line 903):
        
        # Call to min(...): (line 903)
        # Processing the call keyword arguments (line 903)
        kwargs_80407 = {}
        # Getting the type of 'y' (line 903)
        y_80405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 13), 'y', False)
        # Obtaining the member 'min' of a type (line 903)
        min_80406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 13), y_80405, 'min')
        # Calling min(args, kwargs) (line 903)
        min_call_result_80408 = invoke(stypy.reporting.localization.Localization(__file__, 903, 13), min_80406, *[], **kwargs_80407)
        
        # Assigning a type to the variable 'yb' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 8), 'yb', min_call_result_80408)

        if more_types_in_union_80404:
            # SSA join for if statement (line 902)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 904)
    # Getting the type of 'ye' (line 904)
    ye_80409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 7), 'ye')
    # Getting the type of 'None' (line 904)
    None_80410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 13), 'None')
    
    (may_be_80411, more_types_in_union_80412) = may_be_none(ye_80409, None_80410)

    if may_be_80411:

        if more_types_in_union_80412:
            # Runtime conditional SSA (line 904)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 905):
        
        # Assigning a Call to a Name (line 905):
        
        # Call to max(...): (line 905)
        # Processing the call keyword arguments (line 905)
        kwargs_80415 = {}
        # Getting the type of 'y' (line 905)
        y_80413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 13), 'y', False)
        # Obtaining the member 'max' of a type (line 905)
        max_80414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 13), y_80413, 'max')
        # Calling max(args, kwargs) (line 905)
        max_call_result_80416 = invoke(stypy.reporting.localization.Localization(__file__, 905, 13), max_80414, *[], **kwargs_80415)
        
        # Assigning a type to the variable 'ye' (line 905)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'ye', max_call_result_80416)

        if more_types_in_union_80412:
            # SSA join for if statement (line 904)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    int_80417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 12), 'int')
    # Getting the type of 'task' (line 906)
    task_80418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 18), 'task')
    # Applying the binary operator '<=' (line 906)
    result_le_80419 = python_operator(stypy.reporting.localization.Localization(__file__, 906, 12), '<=', int_80417, task_80418)
    int_80420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 26), 'int')
    # Applying the binary operator '<=' (line 906)
    result_le_80421 = python_operator(stypy.reporting.localization.Localization(__file__, 906, 12), '<=', task_80418, int_80420)
    # Applying the binary operator '&' (line 906)
    result_and__80422 = python_operator(stypy.reporting.localization.Localization(__file__, 906, 12), '&', result_le_80419, result_le_80421)
    
    # Applying the 'not' unary operator (line 906)
    result_not__80423 = python_operator(stypy.reporting.localization.Localization(__file__, 906, 7), 'not', result_and__80422)
    
    # Testing the type of an if condition (line 906)
    if_condition_80424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 906, 4), result_not__80423)
    # Assigning a type to the variable 'if_condition_80424' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 4), 'if_condition_80424', if_condition_80424)
    # SSA begins for if statement (line 906)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 907)
    # Processing the call arguments (line 907)
    str_80426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 24), 'str', 'task must be -1, 0 or 1')
    # Processing the call keyword arguments (line 907)
    kwargs_80427 = {}
    # Getting the type of 'TypeError' (line 907)
    TypeError_80425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 907)
    TypeError_call_result_80428 = invoke(stypy.reporting.localization.Localization(__file__, 907, 14), TypeError_80425, *[str_80426], **kwargs_80427)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 907, 8), TypeError_call_result_80428, 'raise parameter', BaseException)
    # SSA join for if statement (line 906)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 908)
    # Getting the type of 's' (line 908)
    s_80429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 7), 's')
    # Getting the type of 'None' (line 908)
    None_80430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 12), 'None')
    
    (may_be_80431, more_types_in_union_80432) = may_be_none(s_80429, None_80430)

    if may_be_80431:

        if more_types_in_union_80432:
            # Runtime conditional SSA (line 908)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 909):
        
        # Assigning a BinOp to a Name (line 909):
        # Getting the type of 'm' (line 909)
        m_80433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 12), 'm')
        
        # Call to sqrt(...): (line 909)
        # Processing the call arguments (line 909)
        int_80435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 21), 'int')
        # Getting the type of 'm' (line 909)
        m_80436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 23), 'm', False)
        # Applying the binary operator '*' (line 909)
        result_mul_80437 = python_operator(stypy.reporting.localization.Localization(__file__, 909, 21), '*', int_80435, m_80436)
        
        # Processing the call keyword arguments (line 909)
        kwargs_80438 = {}
        # Getting the type of 'sqrt' (line 909)
        sqrt_80434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 909)
        sqrt_call_result_80439 = invoke(stypy.reporting.localization.Localization(__file__, 909, 16), sqrt_80434, *[result_mul_80437], **kwargs_80438)
        
        # Applying the binary operator '-' (line 909)
        result_sub_80440 = python_operator(stypy.reporting.localization.Localization(__file__, 909, 12), '-', m_80433, sqrt_call_result_80439)
        
        # Assigning a type to the variable 's' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 's', result_sub_80440)

        if more_types_in_union_80432:
            # SSA join for if statement (line 908)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'tx' (line 910)
    tx_80441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 7), 'tx')
    # Getting the type of 'None' (line 910)
    None_80442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 13), 'None')
    # Applying the binary operator 'is' (line 910)
    result_is__80443 = python_operator(stypy.reporting.localization.Localization(__file__, 910, 7), 'is', tx_80441, None_80442)
    
    
    # Getting the type of 'task' (line 910)
    task_80444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 22), 'task')
    int_80445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 30), 'int')
    # Applying the binary operator '==' (line 910)
    result_eq_80446 = python_operator(stypy.reporting.localization.Localization(__file__, 910, 22), '==', task_80444, int_80445)
    
    # Applying the binary operator 'and' (line 910)
    result_and_keyword_80447 = python_operator(stypy.reporting.localization.Localization(__file__, 910, 7), 'and', result_is__80443, result_eq_80446)
    
    # Testing the type of an if condition (line 910)
    if_condition_80448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 910, 4), result_and_keyword_80447)
    # Assigning a type to the variable 'if_condition_80448' (line 910)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 4), 'if_condition_80448', if_condition_80448)
    # SSA begins for if statement (line 910)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 911)
    # Processing the call arguments (line 911)
    str_80450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 24), 'str', 'Knots_x must be given for task=-1')
    # Processing the call keyword arguments (line 911)
    kwargs_80451 = {}
    # Getting the type of 'TypeError' (line 911)
    TypeError_80449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 911)
    TypeError_call_result_80452 = invoke(stypy.reporting.localization.Localization(__file__, 911, 14), TypeError_80449, *[str_80450], **kwargs_80451)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 911, 8), TypeError_call_result_80452, 'raise parameter', BaseException)
    # SSA join for if statement (line 910)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 912)
    # Getting the type of 'tx' (line 912)
    tx_80453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 4), 'tx')
    # Getting the type of 'None' (line 912)
    None_80454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 17), 'None')
    
    (may_be_80455, more_types_in_union_80456) = may_not_be_none(tx_80453, None_80454)

    if may_be_80455:

        if more_types_in_union_80456:
            # Runtime conditional SSA (line 912)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 913):
        
        # Assigning a Call to a Subscript (line 913):
        
        # Call to atleast_1d(...): (line 913)
        # Processing the call arguments (line 913)
        # Getting the type of 'tx' (line 913)
        tx_80458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 41), 'tx', False)
        # Processing the call keyword arguments (line 913)
        kwargs_80459 = {}
        # Getting the type of 'atleast_1d' (line 913)
        atleast_1d_80457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 30), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 913)
        atleast_1d_call_result_80460 = invoke(stypy.reporting.localization.Localization(__file__, 913, 30), atleast_1d_80457, *[tx_80458], **kwargs_80459)
        
        # Getting the type of '_surfit_cache' (line 913)
        _surfit_cache_80461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 8), '_surfit_cache')
        str_80462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 22), 'str', 'tx')
        # Storing an element on a container (line 913)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 913, 8), _surfit_cache_80461, (str_80462, atleast_1d_call_result_80460))

        if more_types_in_union_80456:
            # SSA join for if statement (line 912)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 914):
    
    # Assigning a Call to a Name (line 914):
    
    # Call to len(...): (line 914)
    # Processing the call arguments (line 914)
    
    # Obtaining the type of the subscript
    str_80464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 27), 'str', 'tx')
    # Getting the type of '_surfit_cache' (line 914)
    _surfit_cache_80465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 13), '_surfit_cache', False)
    # Obtaining the member '__getitem__' of a type (line 914)
    getitem___80466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 13), _surfit_cache_80465, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 914)
    subscript_call_result_80467 = invoke(stypy.reporting.localization.Localization(__file__, 914, 13), getitem___80466, str_80464)
    
    # Processing the call keyword arguments (line 914)
    kwargs_80468 = {}
    # Getting the type of 'len' (line 914)
    len_80463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 9), 'len', False)
    # Calling len(args, kwargs) (line 914)
    len_call_result_80469 = invoke(stypy.reporting.localization.Localization(__file__, 914, 9), len_80463, *[subscript_call_result_80467], **kwargs_80468)
    
    # Assigning a type to the variable 'nx' (line 914)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 4), 'nx', len_call_result_80469)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ty' (line 915)
    ty_80470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 7), 'ty')
    # Getting the type of 'None' (line 915)
    None_80471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 13), 'None')
    # Applying the binary operator 'is' (line 915)
    result_is__80472 = python_operator(stypy.reporting.localization.Localization(__file__, 915, 7), 'is', ty_80470, None_80471)
    
    
    # Getting the type of 'task' (line 915)
    task_80473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 22), 'task')
    int_80474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 30), 'int')
    # Applying the binary operator '==' (line 915)
    result_eq_80475 = python_operator(stypy.reporting.localization.Localization(__file__, 915, 22), '==', task_80473, int_80474)
    
    # Applying the binary operator 'and' (line 915)
    result_and_keyword_80476 = python_operator(stypy.reporting.localization.Localization(__file__, 915, 7), 'and', result_is__80472, result_eq_80475)
    
    # Testing the type of an if condition (line 915)
    if_condition_80477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 915, 4), result_and_keyword_80476)
    # Assigning a type to the variable 'if_condition_80477' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 4), 'if_condition_80477', if_condition_80477)
    # SSA begins for if statement (line 915)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 916)
    # Processing the call arguments (line 916)
    str_80479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 24), 'str', 'Knots_y must be given for task=-1')
    # Processing the call keyword arguments (line 916)
    kwargs_80480 = {}
    # Getting the type of 'TypeError' (line 916)
    TypeError_80478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 916)
    TypeError_call_result_80481 = invoke(stypy.reporting.localization.Localization(__file__, 916, 14), TypeError_80478, *[str_80479], **kwargs_80480)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 916, 8), TypeError_call_result_80481, 'raise parameter', BaseException)
    # SSA join for if statement (line 915)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 917)
    # Getting the type of 'ty' (line 917)
    ty_80482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 4), 'ty')
    # Getting the type of 'None' (line 917)
    None_80483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 17), 'None')
    
    (may_be_80484, more_types_in_union_80485) = may_not_be_none(ty_80482, None_80483)

    if may_be_80484:

        if more_types_in_union_80485:
            # Runtime conditional SSA (line 917)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 918):
        
        # Assigning a Call to a Subscript (line 918):
        
        # Call to atleast_1d(...): (line 918)
        # Processing the call arguments (line 918)
        # Getting the type of 'ty' (line 918)
        ty_80487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 41), 'ty', False)
        # Processing the call keyword arguments (line 918)
        kwargs_80488 = {}
        # Getting the type of 'atleast_1d' (line 918)
        atleast_1d_80486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 30), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 918)
        atleast_1d_call_result_80489 = invoke(stypy.reporting.localization.Localization(__file__, 918, 30), atleast_1d_80486, *[ty_80487], **kwargs_80488)
        
        # Getting the type of '_surfit_cache' (line 918)
        _surfit_cache_80490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 8), '_surfit_cache')
        str_80491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 22), 'str', 'ty')
        # Storing an element on a container (line 918)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 918, 8), _surfit_cache_80490, (str_80491, atleast_1d_call_result_80489))

        if more_types_in_union_80485:
            # SSA join for if statement (line 917)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 919):
    
    # Assigning a Call to a Name (line 919):
    
    # Call to len(...): (line 919)
    # Processing the call arguments (line 919)
    
    # Obtaining the type of the subscript
    str_80493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 27), 'str', 'ty')
    # Getting the type of '_surfit_cache' (line 919)
    _surfit_cache_80494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 13), '_surfit_cache', False)
    # Obtaining the member '__getitem__' of a type (line 919)
    getitem___80495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 13), _surfit_cache_80494, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 919)
    subscript_call_result_80496 = invoke(stypy.reporting.localization.Localization(__file__, 919, 13), getitem___80495, str_80493)
    
    # Processing the call keyword arguments (line 919)
    kwargs_80497 = {}
    # Getting the type of 'len' (line 919)
    len_80492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 9), 'len', False)
    # Calling len(args, kwargs) (line 919)
    len_call_result_80498 = invoke(stypy.reporting.localization.Localization(__file__, 919, 9), len_80492, *[subscript_call_result_80496], **kwargs_80497)
    
    # Assigning a type to the variable 'ny' (line 919)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 4), 'ny', len_call_result_80498)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'task' (line 920)
    task_80499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 7), 'task')
    int_80500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 15), 'int')
    # Applying the binary operator '==' (line 920)
    result_eq_80501 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 7), '==', task_80499, int_80500)
    
    
    # Getting the type of 'nx' (line 920)
    nx_80502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 22), 'nx')
    int_80503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 27), 'int')
    # Getting the type of 'kx' (line 920)
    kx_80504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 29), 'kx')
    # Applying the binary operator '*' (line 920)
    result_mul_80505 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 27), '*', int_80503, kx_80504)
    
    int_80506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 32), 'int')
    # Applying the binary operator '+' (line 920)
    result_add_80507 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 27), '+', result_mul_80505, int_80506)
    
    # Applying the binary operator '<' (line 920)
    result_lt_80508 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 22), '<', nx_80502, result_add_80507)
    
    # Applying the binary operator 'and' (line 920)
    result_and_keyword_80509 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 7), 'and', result_eq_80501, result_lt_80508)
    
    # Testing the type of an if condition (line 920)
    if_condition_80510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 920, 4), result_and_keyword_80509)
    # Assigning a type to the variable 'if_condition_80510' (line 920)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 4), 'if_condition_80510', if_condition_80510)
    # SSA begins for if statement (line 920)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 921)
    # Processing the call arguments (line 921)
    str_80512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 24), 'str', 'There must be at least 2*kx+2 knots_x for task=-1')
    # Processing the call keyword arguments (line 921)
    kwargs_80513 = {}
    # Getting the type of 'TypeError' (line 921)
    TypeError_80511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 921)
    TypeError_call_result_80514 = invoke(stypy.reporting.localization.Localization(__file__, 921, 14), TypeError_80511, *[str_80512], **kwargs_80513)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 921, 8), TypeError_call_result_80514, 'raise parameter', BaseException)
    # SSA join for if statement (line 920)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'task' (line 922)
    task_80515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 7), 'task')
    int_80516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 15), 'int')
    # Applying the binary operator '==' (line 922)
    result_eq_80517 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 7), '==', task_80515, int_80516)
    
    
    # Getting the type of 'ny' (line 922)
    ny_80518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 22), 'ny')
    int_80519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 27), 'int')
    # Getting the type of 'ky' (line 922)
    ky_80520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 29), 'ky')
    # Applying the binary operator '*' (line 922)
    result_mul_80521 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 27), '*', int_80519, ky_80520)
    
    int_80522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 32), 'int')
    # Applying the binary operator '+' (line 922)
    result_add_80523 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 27), '+', result_mul_80521, int_80522)
    
    # Applying the binary operator '<' (line 922)
    result_lt_80524 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 22), '<', ny_80518, result_add_80523)
    
    # Applying the binary operator 'and' (line 922)
    result_and_keyword_80525 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 7), 'and', result_eq_80517, result_lt_80524)
    
    # Testing the type of an if condition (line 922)
    if_condition_80526 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 922, 4), result_and_keyword_80525)
    # Assigning a type to the variable 'if_condition_80526' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 4), 'if_condition_80526', if_condition_80526)
    # SSA begins for if statement (line 922)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 923)
    # Processing the call arguments (line 923)
    str_80528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 24), 'str', 'There must be at least 2*ky+2 knots_x for task=-1')
    # Processing the call keyword arguments (line 923)
    kwargs_80529 = {}
    # Getting the type of 'TypeError' (line 923)
    TypeError_80527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 923)
    TypeError_call_result_80530 = invoke(stypy.reporting.localization.Localization(__file__, 923, 14), TypeError_80527, *[str_80528], **kwargs_80529)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 923, 8), TypeError_call_result_80530, 'raise parameter', BaseException)
    # SSA join for if statement (line 922)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    int_80531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 13), 'int')
    # Getting the type of 'kx' (line 924)
    kx_80532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 18), 'kx')
    # Applying the binary operator '<=' (line 924)
    result_le_80533 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 13), '<=', int_80531, kx_80532)
    int_80534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 24), 'int')
    # Applying the binary operator '<=' (line 924)
    result_le_80535 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 13), '<=', kx_80532, int_80534)
    # Applying the binary operator '&' (line 924)
    result_and__80536 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 13), '&', result_le_80533, result_le_80535)
    
    
    int_80537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 32), 'int')
    # Getting the type of 'ky' (line 924)
    ky_80538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 37), 'ky')
    # Applying the binary operator '<=' (line 924)
    result_le_80539 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 32), '<=', int_80537, ky_80538)
    int_80540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 43), 'int')
    # Applying the binary operator '<=' (line 924)
    result_le_80541 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 32), '<=', ky_80538, int_80540)
    # Applying the binary operator '&' (line 924)
    result_and__80542 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 32), '&', result_le_80539, result_le_80541)
    
    # Applying the binary operator 'and' (line 924)
    result_and_keyword_80543 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 12), 'and', result_and__80536, result_and__80542)
    
    # Applying the 'not' unary operator (line 924)
    result_not__80544 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 7), 'not', result_and_keyword_80543)
    
    # Testing the type of an if condition (line 924)
    if_condition_80545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 924, 4), result_not__80544)
    # Assigning a type to the variable 'if_condition_80545' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 4), 'if_condition_80545', if_condition_80545)
    # SSA begins for if statement (line 924)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 925)
    # Processing the call arguments (line 925)
    str_80547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 24), 'str', 'Given degree of the spline (kx,ky=%d,%d) is not supported. (1<=k<=5)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 926)
    tuple_80548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 926)
    # Adding element type (line 926)
    # Getting the type of 'kx' (line 926)
    kx_80549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 50), 'kx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 926, 50), tuple_80548, kx_80549)
    # Adding element type (line 926)
    # Getting the type of 'ky' (line 926)
    ky_80550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 54), 'ky', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 926, 50), tuple_80548, ky_80550)
    
    # Applying the binary operator '%' (line 925)
    result_mod_80551 = python_operator(stypy.reporting.localization.Localization(__file__, 925, 24), '%', str_80547, tuple_80548)
    
    # Processing the call keyword arguments (line 925)
    kwargs_80552 = {}
    # Getting the type of 'TypeError' (line 925)
    TypeError_80546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 925)
    TypeError_call_result_80553 = invoke(stypy.reporting.localization.Localization(__file__, 925, 14), TypeError_80546, *[result_mod_80551], **kwargs_80552)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 925, 8), TypeError_call_result_80553, 'raise parameter', BaseException)
    # SSA join for if statement (line 924)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 927)
    m_80554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 7), 'm')
    # Getting the type of 'kx' (line 927)
    kx_80555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 12), 'kx')
    int_80556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 17), 'int')
    # Applying the binary operator '+' (line 927)
    result_add_80557 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 12), '+', kx_80555, int_80556)
    
    # Getting the type of 'ky' (line 927)
    ky_80558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 21), 'ky')
    int_80559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 26), 'int')
    # Applying the binary operator '+' (line 927)
    result_add_80560 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 21), '+', ky_80558, int_80559)
    
    # Applying the binary operator '*' (line 927)
    result_mul_80561 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 11), '*', result_add_80557, result_add_80560)
    
    # Applying the binary operator '<' (line 927)
    result_lt_80562 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 7), '<', m_80554, result_mul_80561)
    
    # Testing the type of an if condition (line 927)
    if_condition_80563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 927, 4), result_lt_80562)
    # Assigning a type to the variable 'if_condition_80563' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 4), 'if_condition_80563', if_condition_80563)
    # SSA begins for if statement (line 927)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 928)
    # Processing the call arguments (line 928)
    str_80565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 24), 'str', 'm >= (kx+1)(ky+1) must hold')
    # Processing the call keyword arguments (line 928)
    kwargs_80566 = {}
    # Getting the type of 'TypeError' (line 928)
    TypeError_80564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 928)
    TypeError_call_result_80567 = invoke(stypy.reporting.localization.Localization(__file__, 928, 14), TypeError_80564, *[str_80565], **kwargs_80566)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 928, 8), TypeError_call_result_80567, 'raise parameter', BaseException)
    # SSA join for if statement (line 927)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 929)
    # Getting the type of 'nxest' (line 929)
    nxest_80568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 7), 'nxest')
    # Getting the type of 'None' (line 929)
    None_80569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 16), 'None')
    
    (may_be_80570, more_types_in_union_80571) = may_be_none(nxest_80568, None_80569)

    if may_be_80570:

        if more_types_in_union_80571:
            # Runtime conditional SSA (line 929)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 930):
        
        # Assigning a Call to a Name (line 930):
        
        # Call to int(...): (line 930)
        # Processing the call arguments (line 930)
        # Getting the type of 'kx' (line 930)
        kx_80573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 20), 'kx', False)
        
        # Call to sqrt(...): (line 930)
        # Processing the call arguments (line 930)
        # Getting the type of 'm' (line 930)
        m_80575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 30), 'm', False)
        int_80576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 32), 'int')
        # Applying the binary operator 'div' (line 930)
        result_div_80577 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 30), 'div', m_80575, int_80576)
        
        # Processing the call keyword arguments (line 930)
        kwargs_80578 = {}
        # Getting the type of 'sqrt' (line 930)
        sqrt_80574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 25), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 930)
        sqrt_call_result_80579 = invoke(stypy.reporting.localization.Localization(__file__, 930, 25), sqrt_80574, *[result_div_80577], **kwargs_80578)
        
        # Applying the binary operator '+' (line 930)
        result_add_80580 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 20), '+', kx_80573, sqrt_call_result_80579)
        
        # Processing the call keyword arguments (line 930)
        kwargs_80581 = {}
        # Getting the type of 'int' (line 930)
        int_80572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 16), 'int', False)
        # Calling int(args, kwargs) (line 930)
        int_call_result_80582 = invoke(stypy.reporting.localization.Localization(__file__, 930, 16), int_80572, *[result_add_80580], **kwargs_80581)
        
        # Assigning a type to the variable 'nxest' (line 930)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 8), 'nxest', int_call_result_80582)

        if more_types_in_union_80571:
            # SSA join for if statement (line 929)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 931)
    # Getting the type of 'nyest' (line 931)
    nyest_80583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 7), 'nyest')
    # Getting the type of 'None' (line 931)
    None_80584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 16), 'None')
    
    (may_be_80585, more_types_in_union_80586) = may_be_none(nyest_80583, None_80584)

    if may_be_80585:

        if more_types_in_union_80586:
            # Runtime conditional SSA (line 931)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 932):
        
        # Assigning a Call to a Name (line 932):
        
        # Call to int(...): (line 932)
        # Processing the call arguments (line 932)
        # Getting the type of 'ky' (line 932)
        ky_80588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 20), 'ky', False)
        
        # Call to sqrt(...): (line 932)
        # Processing the call arguments (line 932)
        # Getting the type of 'm' (line 932)
        m_80590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 30), 'm', False)
        int_80591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 32), 'int')
        # Applying the binary operator 'div' (line 932)
        result_div_80592 = python_operator(stypy.reporting.localization.Localization(__file__, 932, 30), 'div', m_80590, int_80591)
        
        # Processing the call keyword arguments (line 932)
        kwargs_80593 = {}
        # Getting the type of 'sqrt' (line 932)
        sqrt_80589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 25), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 932)
        sqrt_call_result_80594 = invoke(stypy.reporting.localization.Localization(__file__, 932, 25), sqrt_80589, *[result_div_80592], **kwargs_80593)
        
        # Applying the binary operator '+' (line 932)
        result_add_80595 = python_operator(stypy.reporting.localization.Localization(__file__, 932, 20), '+', ky_80588, sqrt_call_result_80594)
        
        # Processing the call keyword arguments (line 932)
        kwargs_80596 = {}
        # Getting the type of 'int' (line 932)
        int_80587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 16), 'int', False)
        # Calling int(args, kwargs) (line 932)
        int_call_result_80597 = invoke(stypy.reporting.localization.Localization(__file__, 932, 16), int_80587, *[result_add_80595], **kwargs_80596)
        
        # Assigning a type to the variable 'nyest' (line 932)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'nyest', int_call_result_80597)

        if more_types_in_union_80586:
            # SSA join for if statement (line 931)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Tuple to a Tuple (line 933):
    
    # Assigning a Call to a Name (line 933):
    
    # Call to max(...): (line 933)
    # Processing the call arguments (line 933)
    # Getting the type of 'nxest' (line 933)
    nxest_80599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 23), 'nxest', False)
    int_80600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 30), 'int')
    # Getting the type of 'kx' (line 933)
    kx_80601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 32), 'kx', False)
    # Applying the binary operator '*' (line 933)
    result_mul_80602 = python_operator(stypy.reporting.localization.Localization(__file__, 933, 30), '*', int_80600, kx_80601)
    
    int_80603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 37), 'int')
    # Applying the binary operator '+' (line 933)
    result_add_80604 = python_operator(stypy.reporting.localization.Localization(__file__, 933, 30), '+', result_mul_80602, int_80603)
    
    # Processing the call keyword arguments (line 933)
    kwargs_80605 = {}
    # Getting the type of 'max' (line 933)
    max_80598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 19), 'max', False)
    # Calling max(args, kwargs) (line 933)
    max_call_result_80606 = invoke(stypy.reporting.localization.Localization(__file__, 933, 19), max_80598, *[nxest_80599, result_add_80604], **kwargs_80605)
    
    # Assigning a type to the variable 'tuple_assignment_78315' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'tuple_assignment_78315', max_call_result_80606)
    
    # Assigning a Call to a Name (line 933):
    
    # Call to max(...): (line 933)
    # Processing the call arguments (line 933)
    # Getting the type of 'nyest' (line 933)
    nyest_80608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 45), 'nyest', False)
    int_80609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 52), 'int')
    # Getting the type of 'ky' (line 933)
    ky_80610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 54), 'ky', False)
    # Applying the binary operator '*' (line 933)
    result_mul_80611 = python_operator(stypy.reporting.localization.Localization(__file__, 933, 52), '*', int_80609, ky_80610)
    
    int_80612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 59), 'int')
    # Applying the binary operator '+' (line 933)
    result_add_80613 = python_operator(stypy.reporting.localization.Localization(__file__, 933, 52), '+', result_mul_80611, int_80612)
    
    # Processing the call keyword arguments (line 933)
    kwargs_80614 = {}
    # Getting the type of 'max' (line 933)
    max_80607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 41), 'max', False)
    # Calling max(args, kwargs) (line 933)
    max_call_result_80615 = invoke(stypy.reporting.localization.Localization(__file__, 933, 41), max_80607, *[nyest_80608, result_add_80613], **kwargs_80614)
    
    # Assigning a type to the variable 'tuple_assignment_78316' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'tuple_assignment_78316', max_call_result_80615)
    
    # Assigning a Name to a Name (line 933):
    # Getting the type of 'tuple_assignment_78315' (line 933)
    tuple_assignment_78315_80616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'tuple_assignment_78315')
    # Assigning a type to the variable 'nxest' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'nxest', tuple_assignment_78315_80616)
    
    # Assigning a Name to a Name (line 933):
    # Getting the type of 'tuple_assignment_78316' (line 933)
    tuple_assignment_78316_80617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'tuple_assignment_78316')
    # Assigning a type to the variable 'nyest' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 11), 'nyest', tuple_assignment_78316_80617)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'task' (line 934)
    task_80618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 7), 'task')
    int_80619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 15), 'int')
    # Applying the binary operator '>=' (line 934)
    result_ge_80620 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 7), '>=', task_80618, int_80619)
    
    
    # Getting the type of 's' (line 934)
    s_80621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 21), 's')
    int_80622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 26), 'int')
    # Applying the binary operator '==' (line 934)
    result_eq_80623 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 21), '==', s_80621, int_80622)
    
    # Applying the binary operator 'and' (line 934)
    result_and_keyword_80624 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 7), 'and', result_ge_80620, result_eq_80623)
    
    # Testing the type of an if condition (line 934)
    if_condition_80625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 934, 4), result_and_keyword_80624)
    # Assigning a type to the variable 'if_condition_80625' (line 934)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 4), 'if_condition_80625', if_condition_80625)
    # SSA begins for if statement (line 934)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 935):
    
    # Assigning a Call to a Name (line 935):
    
    # Call to int(...): (line 935)
    # Processing the call arguments (line 935)
    # Getting the type of 'kx' (line 935)
    kx_80627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 20), 'kx', False)
    
    # Call to sqrt(...): (line 935)
    # Processing the call arguments (line 935)
    int_80629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 30), 'int')
    # Getting the type of 'm' (line 935)
    m_80630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 32), 'm', False)
    # Applying the binary operator '*' (line 935)
    result_mul_80631 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 30), '*', int_80629, m_80630)
    
    # Processing the call keyword arguments (line 935)
    kwargs_80632 = {}
    # Getting the type of 'sqrt' (line 935)
    sqrt_80628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 25), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 935)
    sqrt_call_result_80633 = invoke(stypy.reporting.localization.Localization(__file__, 935, 25), sqrt_80628, *[result_mul_80631], **kwargs_80632)
    
    # Applying the binary operator '+' (line 935)
    result_add_80634 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 20), '+', kx_80627, sqrt_call_result_80633)
    
    # Processing the call keyword arguments (line 935)
    kwargs_80635 = {}
    # Getting the type of 'int' (line 935)
    int_80626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 16), 'int', False)
    # Calling int(args, kwargs) (line 935)
    int_call_result_80636 = invoke(stypy.reporting.localization.Localization(__file__, 935, 16), int_80626, *[result_add_80634], **kwargs_80635)
    
    # Assigning a type to the variable 'nxest' (line 935)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'nxest', int_call_result_80636)
    
    # Assigning a Call to a Name (line 936):
    
    # Assigning a Call to a Name (line 936):
    
    # Call to int(...): (line 936)
    # Processing the call arguments (line 936)
    # Getting the type of 'ky' (line 936)
    ky_80638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 20), 'ky', False)
    
    # Call to sqrt(...): (line 936)
    # Processing the call arguments (line 936)
    int_80640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 30), 'int')
    # Getting the type of 'm' (line 936)
    m_80641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 32), 'm', False)
    # Applying the binary operator '*' (line 936)
    result_mul_80642 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 30), '*', int_80640, m_80641)
    
    # Processing the call keyword arguments (line 936)
    kwargs_80643 = {}
    # Getting the type of 'sqrt' (line 936)
    sqrt_80639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 25), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 936)
    sqrt_call_result_80644 = invoke(stypy.reporting.localization.Localization(__file__, 936, 25), sqrt_80639, *[result_mul_80642], **kwargs_80643)
    
    # Applying the binary operator '+' (line 936)
    result_add_80645 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 20), '+', ky_80638, sqrt_call_result_80644)
    
    # Processing the call keyword arguments (line 936)
    kwargs_80646 = {}
    # Getting the type of 'int' (line 936)
    int_80637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 16), 'int', False)
    # Calling int(args, kwargs) (line 936)
    int_call_result_80647 = invoke(stypy.reporting.localization.Localization(__file__, 936, 16), int_80637, *[result_add_80645], **kwargs_80646)
    
    # Assigning a type to the variable 'nyest' (line 936)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'nyest', int_call_result_80647)
    # SSA join for if statement (line 934)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'task' (line 937)
    task_80648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 7), 'task')
    int_80649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 15), 'int')
    # Applying the binary operator '==' (line 937)
    result_eq_80650 = python_operator(stypy.reporting.localization.Localization(__file__, 937, 7), '==', task_80648, int_80649)
    
    # Testing the type of an if condition (line 937)
    if_condition_80651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 937, 4), result_eq_80650)
    # Assigning a type to the variable 'if_condition_80651' (line 937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 4), 'if_condition_80651', if_condition_80651)
    # SSA begins for if statement (line 937)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 938):
    
    # Assigning a Call to a Subscript (line 938):
    
    # Call to atleast_1d(...): (line 938)
    # Processing the call arguments (line 938)
    # Getting the type of 'tx' (line 938)
    tx_80653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 41), 'tx', False)
    # Processing the call keyword arguments (line 938)
    kwargs_80654 = {}
    # Getting the type of 'atleast_1d' (line 938)
    atleast_1d_80652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 30), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 938)
    atleast_1d_call_result_80655 = invoke(stypy.reporting.localization.Localization(__file__, 938, 30), atleast_1d_80652, *[tx_80653], **kwargs_80654)
    
    # Getting the type of '_surfit_cache' (line 938)
    _surfit_cache_80656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 8), '_surfit_cache')
    str_80657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 22), 'str', 'tx')
    # Storing an element on a container (line 938)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 938, 8), _surfit_cache_80656, (str_80657, atleast_1d_call_result_80655))
    
    # Assigning a Call to a Subscript (line 939):
    
    # Assigning a Call to a Subscript (line 939):
    
    # Call to atleast_1d(...): (line 939)
    # Processing the call arguments (line 939)
    # Getting the type of 'ty' (line 939)
    ty_80659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 41), 'ty', False)
    # Processing the call keyword arguments (line 939)
    kwargs_80660 = {}
    # Getting the type of 'atleast_1d' (line 939)
    atleast_1d_80658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 30), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 939)
    atleast_1d_call_result_80661 = invoke(stypy.reporting.localization.Localization(__file__, 939, 30), atleast_1d_80658, *[ty_80659], **kwargs_80660)
    
    # Getting the type of '_surfit_cache' (line 939)
    _surfit_cache_80662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), '_surfit_cache')
    str_80663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 22), 'str', 'ty')
    # Storing an element on a container (line 939)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 939, 8), _surfit_cache_80662, (str_80663, atleast_1d_call_result_80661))
    # SSA join for if statement (line 937)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 940):
    
    # Assigning a Subscript to a Name (line 940):
    
    # Obtaining the type of the subscript
    str_80664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 27), 'str', 'tx')
    # Getting the type of '_surfit_cache' (line 940)
    _surfit_cache_80665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 13), '_surfit_cache')
    # Obtaining the member '__getitem__' of a type (line 940)
    getitem___80666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 13), _surfit_cache_80665, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 940)
    subscript_call_result_80667 = invoke(stypy.reporting.localization.Localization(__file__, 940, 13), getitem___80666, str_80664)
    
    # Assigning a type to the variable 'tuple_assignment_78317' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 4), 'tuple_assignment_78317', subscript_call_result_80667)
    
    # Assigning a Subscript to a Name (line 940):
    
    # Obtaining the type of the subscript
    str_80668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 48), 'str', 'ty')
    # Getting the type of '_surfit_cache' (line 940)
    _surfit_cache_80669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 34), '_surfit_cache')
    # Obtaining the member '__getitem__' of a type (line 940)
    getitem___80670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 34), _surfit_cache_80669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 940)
    subscript_call_result_80671 = invoke(stypy.reporting.localization.Localization(__file__, 940, 34), getitem___80670, str_80668)
    
    # Assigning a type to the variable 'tuple_assignment_78318' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 4), 'tuple_assignment_78318', subscript_call_result_80671)
    
    # Assigning a Name to a Name (line 940):
    # Getting the type of 'tuple_assignment_78317' (line 940)
    tuple_assignment_78317_80672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 4), 'tuple_assignment_78317')
    # Assigning a type to the variable 'tx' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 4), 'tx', tuple_assignment_78317_80672)
    
    # Assigning a Name to a Name (line 940):
    # Getting the type of 'tuple_assignment_78318' (line 940)
    tuple_assignment_78318_80673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 4), 'tuple_assignment_78318')
    # Assigning a type to the variable 'ty' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 8), 'ty', tuple_assignment_78318_80673)
    
    # Assigning a Subscript to a Name (line 941):
    
    # Assigning a Subscript to a Name (line 941):
    
    # Obtaining the type of the subscript
    str_80674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 24), 'str', 'wrk')
    # Getting the type of '_surfit_cache' (line 941)
    _surfit_cache_80675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 10), '_surfit_cache')
    # Obtaining the member '__getitem__' of a type (line 941)
    getitem___80676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 941, 10), _surfit_cache_80675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 941)
    subscript_call_result_80677 = invoke(stypy.reporting.localization.Localization(__file__, 941, 10), getitem___80676, str_80674)
    
    # Assigning a type to the variable 'wrk' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 4), 'wrk', subscript_call_result_80677)
    
    # Assigning a BinOp to a Name (line 942):
    
    # Assigning a BinOp to a Name (line 942):
    # Getting the type of 'nxest' (line 942)
    nxest_80678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 8), 'nxest')
    # Getting the type of 'kx' (line 942)
    kx_80679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 16), 'kx')
    # Applying the binary operator '-' (line 942)
    result_sub_80680 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 8), '-', nxest_80678, kx_80679)
    
    int_80681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 21), 'int')
    # Applying the binary operator '-' (line 942)
    result_sub_80682 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 19), '-', result_sub_80680, int_80681)
    
    # Assigning a type to the variable 'u' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 4), 'u', result_sub_80682)
    
    # Assigning a BinOp to a Name (line 943):
    
    # Assigning a BinOp to a Name (line 943):
    # Getting the type of 'nyest' (line 943)
    nyest_80683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 8), 'nyest')
    # Getting the type of 'ky' (line 943)
    ky_80684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 16), 'ky')
    # Applying the binary operator '-' (line 943)
    result_sub_80685 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 8), '-', nyest_80683, ky_80684)
    
    int_80686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 21), 'int')
    # Applying the binary operator '-' (line 943)
    result_sub_80687 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 19), '-', result_sub_80685, int_80686)
    
    # Assigning a type to the variable 'v' (line 943)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 4), 'v', result_sub_80687)
    
    # Assigning a BinOp to a Name (line 944):
    
    # Assigning a BinOp to a Name (line 944):
    
    # Call to max(...): (line 944)
    # Processing the call arguments (line 944)
    # Getting the type of 'kx' (line 944)
    kx_80689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 13), 'kx', False)
    # Getting the type of 'ky' (line 944)
    ky_80690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 17), 'ky', False)
    # Processing the call keyword arguments (line 944)
    kwargs_80691 = {}
    # Getting the type of 'max' (line 944)
    max_80688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 9), 'max', False)
    # Calling max(args, kwargs) (line 944)
    max_call_result_80692 = invoke(stypy.reporting.localization.Localization(__file__, 944, 9), max_80688, *[kx_80689, ky_80690], **kwargs_80691)
    
    int_80693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 23), 'int')
    # Applying the binary operator '+' (line 944)
    result_add_80694 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 9), '+', max_call_result_80692, int_80693)
    
    # Assigning a type to the variable 'km' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 4), 'km', result_add_80694)
    
    # Assigning a Call to a Name (line 945):
    
    # Assigning a Call to a Name (line 945):
    
    # Call to max(...): (line 945)
    # Processing the call arguments (line 945)
    # Getting the type of 'nxest' (line 945)
    nxest_80696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 13), 'nxest', False)
    # Getting the type of 'nyest' (line 945)
    nyest_80697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 20), 'nyest', False)
    # Processing the call keyword arguments (line 945)
    kwargs_80698 = {}
    # Getting the type of 'max' (line 945)
    max_80695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 9), 'max', False)
    # Calling max(args, kwargs) (line 945)
    max_call_result_80699 = invoke(stypy.reporting.localization.Localization(__file__, 945, 9), max_80695, *[nxest_80696, nyest_80697], **kwargs_80698)
    
    # Assigning a type to the variable 'ne' (line 945)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 4), 'ne', max_call_result_80699)
    
    # Assigning a Tuple to a Tuple (line 946):
    
    # Assigning a BinOp to a Name (line 946):
    # Getting the type of 'kx' (line 946)
    kx_80700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 13), 'kx')
    # Getting the type of 'v' (line 946)
    v_80701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 16), 'v')
    # Applying the binary operator '*' (line 946)
    result_mul_80702 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 13), '*', kx_80700, v_80701)
    
    # Getting the type of 'ky' (line 946)
    ky_80703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 20), 'ky')
    # Applying the binary operator '+' (line 946)
    result_add_80704 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 13), '+', result_mul_80702, ky_80703)
    
    int_80705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 25), 'int')
    # Applying the binary operator '+' (line 946)
    result_add_80706 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 23), '+', result_add_80704, int_80705)
    
    # Assigning a type to the variable 'tuple_assignment_78319' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'tuple_assignment_78319', result_add_80706)
    
    # Assigning a BinOp to a Name (line 946):
    # Getting the type of 'ky' (line 946)
    ky_80707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 28), 'ky')
    # Getting the type of 'u' (line 946)
    u_80708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 31), 'u')
    # Applying the binary operator '*' (line 946)
    result_mul_80709 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 28), '*', ky_80707, u_80708)
    
    # Getting the type of 'kx' (line 946)
    kx_80710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 35), 'kx')
    # Applying the binary operator '+' (line 946)
    result_add_80711 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 28), '+', result_mul_80709, kx_80710)
    
    int_80712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 40), 'int')
    # Applying the binary operator '+' (line 946)
    result_add_80713 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 38), '+', result_add_80711, int_80712)
    
    # Assigning a type to the variable 'tuple_assignment_78320' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'tuple_assignment_78320', result_add_80713)
    
    # Assigning a Name to a Name (line 946):
    # Getting the type of 'tuple_assignment_78319' (line 946)
    tuple_assignment_78319_80714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'tuple_assignment_78319')
    # Assigning a type to the variable 'bx' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'bx', tuple_assignment_78319_80714)
    
    # Assigning a Name to a Name (line 946):
    # Getting the type of 'tuple_assignment_78320' (line 946)
    tuple_assignment_78320_80715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'tuple_assignment_78320')
    # Assigning a type to the variable 'by' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 8), 'by', tuple_assignment_78320_80715)
    
    # Assigning a Tuple to a Tuple (line 947):
    
    # Assigning a Name to a Name (line 947):
    # Getting the type of 'bx' (line 947)
    bx_80716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 13), 'bx')
    # Assigning a type to the variable 'tuple_assignment_78321' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'tuple_assignment_78321', bx_80716)
    
    # Assigning a BinOp to a Name (line 947):
    # Getting the type of 'bx' (line 947)
    bx_80717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 17), 'bx')
    # Getting the type of 'v' (line 947)
    v_80718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 22), 'v')
    # Applying the binary operator '+' (line 947)
    result_add_80719 = python_operator(stypy.reporting.localization.Localization(__file__, 947, 17), '+', bx_80717, v_80718)
    
    # Getting the type of 'ky' (line 947)
    ky_80720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 26), 'ky')
    # Applying the binary operator '-' (line 947)
    result_sub_80721 = python_operator(stypy.reporting.localization.Localization(__file__, 947, 24), '-', result_add_80719, ky_80720)
    
    # Assigning a type to the variable 'tuple_assignment_78322' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'tuple_assignment_78322', result_sub_80721)
    
    # Assigning a Name to a Name (line 947):
    # Getting the type of 'tuple_assignment_78321' (line 947)
    tuple_assignment_78321_80722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'tuple_assignment_78321')
    # Assigning a type to the variable 'b1' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'b1', tuple_assignment_78321_80722)
    
    # Assigning a Name to a Name (line 947):
    # Getting the type of 'tuple_assignment_78322' (line 947)
    tuple_assignment_78322_80723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'tuple_assignment_78322')
    # Assigning a type to the variable 'b2' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 8), 'b2', tuple_assignment_78322_80723)
    
    
    # Getting the type of 'bx' (line 948)
    bx_80724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 7), 'bx')
    # Getting the type of 'by' (line 948)
    by_80725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 12), 'by')
    # Applying the binary operator '>' (line 948)
    result_gt_80726 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 7), '>', bx_80724, by_80725)
    
    # Testing the type of an if condition (line 948)
    if_condition_80727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 948, 4), result_gt_80726)
    # Assigning a type to the variable 'if_condition_80727' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'if_condition_80727', if_condition_80727)
    # SSA begins for if statement (line 948)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 949):
    
    # Assigning a Name to a Name (line 949):
    # Getting the type of 'by' (line 949)
    by_80728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 17), 'by')
    # Assigning a type to the variable 'tuple_assignment_78323' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 8), 'tuple_assignment_78323', by_80728)
    
    # Assigning a BinOp to a Name (line 949):
    # Getting the type of 'by' (line 949)
    by_80729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 21), 'by')
    # Getting the type of 'u' (line 949)
    u_80730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 26), 'u')
    # Applying the binary operator '+' (line 949)
    result_add_80731 = python_operator(stypy.reporting.localization.Localization(__file__, 949, 21), '+', by_80729, u_80730)
    
    # Getting the type of 'kx' (line 949)
    kx_80732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 30), 'kx')
    # Applying the binary operator '-' (line 949)
    result_sub_80733 = python_operator(stypy.reporting.localization.Localization(__file__, 949, 28), '-', result_add_80731, kx_80732)
    
    # Assigning a type to the variable 'tuple_assignment_78324' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 8), 'tuple_assignment_78324', result_sub_80733)
    
    # Assigning a Name to a Name (line 949):
    # Getting the type of 'tuple_assignment_78323' (line 949)
    tuple_assignment_78323_80734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 8), 'tuple_assignment_78323')
    # Assigning a type to the variable 'b1' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 8), 'b1', tuple_assignment_78323_80734)
    
    # Assigning a Name to a Name (line 949):
    # Getting the type of 'tuple_assignment_78324' (line 949)
    tuple_assignment_78324_80735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 8), 'tuple_assignment_78324')
    # Assigning a type to the variable 'b2' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 12), 'b2', tuple_assignment_78324_80735)
    # SSA join for if statement (line 948)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 950):
    
    # Assigning a Str to a Name (line 950):
    str_80736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 10), 'str', 'Too many data points to interpolate')
    # Assigning a type to the variable 'msg' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'msg', str_80736)
    
    # Assigning a Call to a Name (line 951):
    
    # Assigning a Call to a Name (line 951):
    
    # Call to _intc_overflow(...): (line 951)
    # Processing the call arguments (line 951)
    # Getting the type of 'u' (line 951)
    u_80738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 27), 'u', False)
    # Getting the type of 'v' (line 951)
    v_80739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 29), 'v', False)
    # Applying the binary operator '*' (line 951)
    result_mul_80740 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 27), '*', u_80738, v_80739)
    
    int_80741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 32), 'int')
    # Getting the type of 'b1' (line 951)
    b1_80742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 36), 'b1', False)
    # Applying the binary operator '+' (line 951)
    result_add_80743 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 32), '+', int_80741, b1_80742)
    
    # Getting the type of 'b2' (line 951)
    b2_80744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 41), 'b2', False)
    # Applying the binary operator '+' (line 951)
    result_add_80745 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 39), '+', result_add_80743, b2_80744)
    
    # Applying the binary operator '*' (line 951)
    result_mul_80746 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 30), '*', result_mul_80740, result_add_80745)
    
    int_80747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 27), 'int')
    # Getting the type of 'u' (line 952)
    u_80748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 30), 'u', False)
    # Getting the type of 'v' (line 952)
    v_80749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 34), 'v', False)
    # Applying the binary operator '+' (line 952)
    result_add_80750 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 30), '+', u_80748, v_80749)
    
    # Getting the type of 'km' (line 952)
    km_80751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 38), 'km', False)
    # Getting the type of 'm' (line 952)
    m_80752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 42), 'm', False)
    # Getting the type of 'ne' (line 952)
    ne_80753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 46), 'ne', False)
    # Applying the binary operator '+' (line 952)
    result_add_80754 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 42), '+', m_80752, ne_80753)
    
    # Applying the binary operator '*' (line 952)
    result_mul_80755 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 38), '*', km_80751, result_add_80754)
    
    # Applying the binary operator '+' (line 952)
    result_add_80756 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 36), '+', result_add_80750, result_mul_80755)
    
    # Getting the type of 'ne' (line 952)
    ne_80757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 52), 'ne', False)
    # Applying the binary operator '+' (line 952)
    result_add_80758 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 50), '+', result_add_80756, ne_80757)
    
    # Getting the type of 'kx' (line 952)
    kx_80759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 57), 'kx', False)
    # Applying the binary operator '-' (line 952)
    result_sub_80760 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 55), '-', result_add_80758, kx_80759)
    
    # Getting the type of 'ky' (line 952)
    ky_80761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 62), 'ky', False)
    # Applying the binary operator '-' (line 952)
    result_sub_80762 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 60), '-', result_sub_80760, ky_80761)
    
    # Applying the binary operator '*' (line 952)
    result_mul_80763 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 27), '*', int_80747, result_sub_80762)
    
    # Applying the binary operator '+' (line 951)
    result_add_80764 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 27), '+', result_mul_80746, result_mul_80763)
    
    # Getting the type of 'b2' (line 952)
    b2_80765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 68), 'b2', False)
    # Applying the binary operator '+' (line 952)
    result_add_80766 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 66), '+', result_add_80764, b2_80765)
    
    int_80767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 73), 'int')
    # Applying the binary operator '+' (line 952)
    result_add_80768 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 71), '+', result_add_80766, int_80767)
    
    # Processing the call keyword arguments (line 951)
    # Getting the type of 'msg' (line 953)
    msg_80769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 31), 'msg', False)
    keyword_80770 = msg_80769
    kwargs_80771 = {'msg': keyword_80770}
    # Getting the type of '_intc_overflow' (line 951)
    _intc_overflow_80737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 12), '_intc_overflow', False)
    # Calling _intc_overflow(args, kwargs) (line 951)
    _intc_overflow_call_result_80772 = invoke(stypy.reporting.localization.Localization(__file__, 951, 12), _intc_overflow_80737, *[result_add_80768], **kwargs_80771)
    
    # Assigning a type to the variable 'lwrk1' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 4), 'lwrk1', _intc_overflow_call_result_80772)
    
    # Assigning a Call to a Name (line 954):
    
    # Assigning a Call to a Name (line 954):
    
    # Call to _intc_overflow(...): (line 954)
    # Processing the call arguments (line 954)
    # Getting the type of 'u' (line 954)
    u_80774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 27), 'u', False)
    # Getting the type of 'v' (line 954)
    v_80775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 29), 'v', False)
    # Applying the binary operator '*' (line 954)
    result_mul_80776 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 27), '*', u_80774, v_80775)
    
    # Getting the type of 'b2' (line 954)
    b2_80777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 32), 'b2', False)
    int_80778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 37), 'int')
    # Applying the binary operator '+' (line 954)
    result_add_80779 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 32), '+', b2_80777, int_80778)
    
    # Applying the binary operator '*' (line 954)
    result_mul_80780 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 30), '*', result_mul_80776, result_add_80779)
    
    # Getting the type of 'b2' (line 954)
    b2_80781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 42), 'b2', False)
    # Applying the binary operator '+' (line 954)
    result_add_80782 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 27), '+', result_mul_80780, b2_80781)
    
    # Processing the call keyword arguments (line 954)
    # Getting the type of 'msg' (line 954)
    msg_80783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 50), 'msg', False)
    keyword_80784 = msg_80783
    kwargs_80785 = {'msg': keyword_80784}
    # Getting the type of '_intc_overflow' (line 954)
    _intc_overflow_80773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 12), '_intc_overflow', False)
    # Calling _intc_overflow(args, kwargs) (line 954)
    _intc_overflow_call_result_80786 = invoke(stypy.reporting.localization.Localization(__file__, 954, 12), _intc_overflow_80773, *[result_add_80782], **kwargs_80785)
    
    # Assigning a type to the variable 'lwrk2' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'lwrk2', _intc_overflow_call_result_80786)
    
    # Assigning a Call to a Tuple (line 955):
    
    # Assigning a Subscript to a Name (line 955):
    
    # Obtaining the type of the subscript
    int_80787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 4), 'int')
    
    # Call to _surfit(...): (line 955)
    # Processing the call arguments (line 955)
    # Getting the type of 'x' (line 955)
    x_80790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 36), 'x', False)
    # Getting the type of 'y' (line 955)
    y_80791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 39), 'y', False)
    # Getting the type of 'z' (line 955)
    z_80792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 42), 'z', False)
    # Getting the type of 'w' (line 955)
    w_80793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 45), 'w', False)
    # Getting the type of 'xb' (line 955)
    xb_80794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 48), 'xb', False)
    # Getting the type of 'xe' (line 955)
    xe_80795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 52), 'xe', False)
    # Getting the type of 'yb' (line 955)
    yb_80796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 56), 'yb', False)
    # Getting the type of 'ye' (line 955)
    ye_80797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 60), 'ye', False)
    # Getting the type of 'kx' (line 955)
    kx_80798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 64), 'kx', False)
    # Getting the type of 'ky' (line 955)
    ky_80799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 68), 'ky', False)
    # Getting the type of 'task' (line 956)
    task_80800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 36), 'task', False)
    # Getting the type of 's' (line 956)
    s_80801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 42), 's', False)
    # Getting the type of 'eps' (line 956)
    eps_80802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 45), 'eps', False)
    # Getting the type of 'tx' (line 956)
    tx_80803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 50), 'tx', False)
    # Getting the type of 'ty' (line 956)
    ty_80804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 54), 'ty', False)
    # Getting the type of 'nxest' (line 956)
    nxest_80805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 58), 'nxest', False)
    # Getting the type of 'nyest' (line 956)
    nyest_80806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 65), 'nyest', False)
    # Getting the type of 'wrk' (line 957)
    wrk_80807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 36), 'wrk', False)
    # Getting the type of 'lwrk1' (line 957)
    lwrk1_80808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 41), 'lwrk1', False)
    # Getting the type of 'lwrk2' (line 957)
    lwrk2_80809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 48), 'lwrk2', False)
    # Processing the call keyword arguments (line 955)
    kwargs_80810 = {}
    # Getting the type of '_fitpack' (line 955)
    _fitpack_80788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 19), '_fitpack', False)
    # Obtaining the member '_surfit' of a type (line 955)
    _surfit_80789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 19), _fitpack_80788, '_surfit')
    # Calling _surfit(args, kwargs) (line 955)
    _surfit_call_result_80811 = invoke(stypy.reporting.localization.Localization(__file__, 955, 19), _surfit_80789, *[x_80790, y_80791, z_80792, w_80793, xb_80794, xe_80795, yb_80796, ye_80797, kx_80798, ky_80799, task_80800, s_80801, eps_80802, tx_80803, ty_80804, nxest_80805, nyest_80806, wrk_80807, lwrk1_80808, lwrk2_80809], **kwargs_80810)
    
    # Obtaining the member '__getitem__' of a type (line 955)
    getitem___80812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 4), _surfit_call_result_80811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 955)
    subscript_call_result_80813 = invoke(stypy.reporting.localization.Localization(__file__, 955, 4), getitem___80812, int_80787)
    
    # Assigning a type to the variable 'tuple_var_assignment_78325' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tuple_var_assignment_78325', subscript_call_result_80813)
    
    # Assigning a Subscript to a Name (line 955):
    
    # Obtaining the type of the subscript
    int_80814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 4), 'int')
    
    # Call to _surfit(...): (line 955)
    # Processing the call arguments (line 955)
    # Getting the type of 'x' (line 955)
    x_80817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 36), 'x', False)
    # Getting the type of 'y' (line 955)
    y_80818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 39), 'y', False)
    # Getting the type of 'z' (line 955)
    z_80819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 42), 'z', False)
    # Getting the type of 'w' (line 955)
    w_80820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 45), 'w', False)
    # Getting the type of 'xb' (line 955)
    xb_80821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 48), 'xb', False)
    # Getting the type of 'xe' (line 955)
    xe_80822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 52), 'xe', False)
    # Getting the type of 'yb' (line 955)
    yb_80823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 56), 'yb', False)
    # Getting the type of 'ye' (line 955)
    ye_80824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 60), 'ye', False)
    # Getting the type of 'kx' (line 955)
    kx_80825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 64), 'kx', False)
    # Getting the type of 'ky' (line 955)
    ky_80826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 68), 'ky', False)
    # Getting the type of 'task' (line 956)
    task_80827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 36), 'task', False)
    # Getting the type of 's' (line 956)
    s_80828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 42), 's', False)
    # Getting the type of 'eps' (line 956)
    eps_80829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 45), 'eps', False)
    # Getting the type of 'tx' (line 956)
    tx_80830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 50), 'tx', False)
    # Getting the type of 'ty' (line 956)
    ty_80831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 54), 'ty', False)
    # Getting the type of 'nxest' (line 956)
    nxest_80832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 58), 'nxest', False)
    # Getting the type of 'nyest' (line 956)
    nyest_80833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 65), 'nyest', False)
    # Getting the type of 'wrk' (line 957)
    wrk_80834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 36), 'wrk', False)
    # Getting the type of 'lwrk1' (line 957)
    lwrk1_80835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 41), 'lwrk1', False)
    # Getting the type of 'lwrk2' (line 957)
    lwrk2_80836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 48), 'lwrk2', False)
    # Processing the call keyword arguments (line 955)
    kwargs_80837 = {}
    # Getting the type of '_fitpack' (line 955)
    _fitpack_80815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 19), '_fitpack', False)
    # Obtaining the member '_surfit' of a type (line 955)
    _surfit_80816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 19), _fitpack_80815, '_surfit')
    # Calling _surfit(args, kwargs) (line 955)
    _surfit_call_result_80838 = invoke(stypy.reporting.localization.Localization(__file__, 955, 19), _surfit_80816, *[x_80817, y_80818, z_80819, w_80820, xb_80821, xe_80822, yb_80823, ye_80824, kx_80825, ky_80826, task_80827, s_80828, eps_80829, tx_80830, ty_80831, nxest_80832, nyest_80833, wrk_80834, lwrk1_80835, lwrk2_80836], **kwargs_80837)
    
    # Obtaining the member '__getitem__' of a type (line 955)
    getitem___80839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 4), _surfit_call_result_80838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 955)
    subscript_call_result_80840 = invoke(stypy.reporting.localization.Localization(__file__, 955, 4), getitem___80839, int_80814)
    
    # Assigning a type to the variable 'tuple_var_assignment_78326' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tuple_var_assignment_78326', subscript_call_result_80840)
    
    # Assigning a Subscript to a Name (line 955):
    
    # Obtaining the type of the subscript
    int_80841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 4), 'int')
    
    # Call to _surfit(...): (line 955)
    # Processing the call arguments (line 955)
    # Getting the type of 'x' (line 955)
    x_80844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 36), 'x', False)
    # Getting the type of 'y' (line 955)
    y_80845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 39), 'y', False)
    # Getting the type of 'z' (line 955)
    z_80846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 42), 'z', False)
    # Getting the type of 'w' (line 955)
    w_80847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 45), 'w', False)
    # Getting the type of 'xb' (line 955)
    xb_80848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 48), 'xb', False)
    # Getting the type of 'xe' (line 955)
    xe_80849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 52), 'xe', False)
    # Getting the type of 'yb' (line 955)
    yb_80850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 56), 'yb', False)
    # Getting the type of 'ye' (line 955)
    ye_80851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 60), 'ye', False)
    # Getting the type of 'kx' (line 955)
    kx_80852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 64), 'kx', False)
    # Getting the type of 'ky' (line 955)
    ky_80853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 68), 'ky', False)
    # Getting the type of 'task' (line 956)
    task_80854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 36), 'task', False)
    # Getting the type of 's' (line 956)
    s_80855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 42), 's', False)
    # Getting the type of 'eps' (line 956)
    eps_80856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 45), 'eps', False)
    # Getting the type of 'tx' (line 956)
    tx_80857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 50), 'tx', False)
    # Getting the type of 'ty' (line 956)
    ty_80858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 54), 'ty', False)
    # Getting the type of 'nxest' (line 956)
    nxest_80859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 58), 'nxest', False)
    # Getting the type of 'nyest' (line 956)
    nyest_80860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 65), 'nyest', False)
    # Getting the type of 'wrk' (line 957)
    wrk_80861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 36), 'wrk', False)
    # Getting the type of 'lwrk1' (line 957)
    lwrk1_80862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 41), 'lwrk1', False)
    # Getting the type of 'lwrk2' (line 957)
    lwrk2_80863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 48), 'lwrk2', False)
    # Processing the call keyword arguments (line 955)
    kwargs_80864 = {}
    # Getting the type of '_fitpack' (line 955)
    _fitpack_80842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 19), '_fitpack', False)
    # Obtaining the member '_surfit' of a type (line 955)
    _surfit_80843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 19), _fitpack_80842, '_surfit')
    # Calling _surfit(args, kwargs) (line 955)
    _surfit_call_result_80865 = invoke(stypy.reporting.localization.Localization(__file__, 955, 19), _surfit_80843, *[x_80844, y_80845, z_80846, w_80847, xb_80848, xe_80849, yb_80850, ye_80851, kx_80852, ky_80853, task_80854, s_80855, eps_80856, tx_80857, ty_80858, nxest_80859, nyest_80860, wrk_80861, lwrk1_80862, lwrk2_80863], **kwargs_80864)
    
    # Obtaining the member '__getitem__' of a type (line 955)
    getitem___80866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 4), _surfit_call_result_80865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 955)
    subscript_call_result_80867 = invoke(stypy.reporting.localization.Localization(__file__, 955, 4), getitem___80866, int_80841)
    
    # Assigning a type to the variable 'tuple_var_assignment_78327' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tuple_var_assignment_78327', subscript_call_result_80867)
    
    # Assigning a Subscript to a Name (line 955):
    
    # Obtaining the type of the subscript
    int_80868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 4), 'int')
    
    # Call to _surfit(...): (line 955)
    # Processing the call arguments (line 955)
    # Getting the type of 'x' (line 955)
    x_80871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 36), 'x', False)
    # Getting the type of 'y' (line 955)
    y_80872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 39), 'y', False)
    # Getting the type of 'z' (line 955)
    z_80873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 42), 'z', False)
    # Getting the type of 'w' (line 955)
    w_80874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 45), 'w', False)
    # Getting the type of 'xb' (line 955)
    xb_80875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 48), 'xb', False)
    # Getting the type of 'xe' (line 955)
    xe_80876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 52), 'xe', False)
    # Getting the type of 'yb' (line 955)
    yb_80877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 56), 'yb', False)
    # Getting the type of 'ye' (line 955)
    ye_80878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 60), 'ye', False)
    # Getting the type of 'kx' (line 955)
    kx_80879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 64), 'kx', False)
    # Getting the type of 'ky' (line 955)
    ky_80880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 68), 'ky', False)
    # Getting the type of 'task' (line 956)
    task_80881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 36), 'task', False)
    # Getting the type of 's' (line 956)
    s_80882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 42), 's', False)
    # Getting the type of 'eps' (line 956)
    eps_80883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 45), 'eps', False)
    # Getting the type of 'tx' (line 956)
    tx_80884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 50), 'tx', False)
    # Getting the type of 'ty' (line 956)
    ty_80885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 54), 'ty', False)
    # Getting the type of 'nxest' (line 956)
    nxest_80886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 58), 'nxest', False)
    # Getting the type of 'nyest' (line 956)
    nyest_80887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 65), 'nyest', False)
    # Getting the type of 'wrk' (line 957)
    wrk_80888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 36), 'wrk', False)
    # Getting the type of 'lwrk1' (line 957)
    lwrk1_80889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 41), 'lwrk1', False)
    # Getting the type of 'lwrk2' (line 957)
    lwrk2_80890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 48), 'lwrk2', False)
    # Processing the call keyword arguments (line 955)
    kwargs_80891 = {}
    # Getting the type of '_fitpack' (line 955)
    _fitpack_80869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 19), '_fitpack', False)
    # Obtaining the member '_surfit' of a type (line 955)
    _surfit_80870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 19), _fitpack_80869, '_surfit')
    # Calling _surfit(args, kwargs) (line 955)
    _surfit_call_result_80892 = invoke(stypy.reporting.localization.Localization(__file__, 955, 19), _surfit_80870, *[x_80871, y_80872, z_80873, w_80874, xb_80875, xe_80876, yb_80877, ye_80878, kx_80879, ky_80880, task_80881, s_80882, eps_80883, tx_80884, ty_80885, nxest_80886, nyest_80887, wrk_80888, lwrk1_80889, lwrk2_80890], **kwargs_80891)
    
    # Obtaining the member '__getitem__' of a type (line 955)
    getitem___80893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 4), _surfit_call_result_80892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 955)
    subscript_call_result_80894 = invoke(stypy.reporting.localization.Localization(__file__, 955, 4), getitem___80893, int_80868)
    
    # Assigning a type to the variable 'tuple_var_assignment_78328' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tuple_var_assignment_78328', subscript_call_result_80894)
    
    # Assigning a Name to a Name (line 955):
    # Getting the type of 'tuple_var_assignment_78325' (line 955)
    tuple_var_assignment_78325_80895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tuple_var_assignment_78325')
    # Assigning a type to the variable 'tx' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tx', tuple_var_assignment_78325_80895)
    
    # Assigning a Name to a Name (line 955):
    # Getting the type of 'tuple_var_assignment_78326' (line 955)
    tuple_var_assignment_78326_80896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tuple_var_assignment_78326')
    # Assigning a type to the variable 'ty' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 8), 'ty', tuple_var_assignment_78326_80896)
    
    # Assigning a Name to a Name (line 955):
    # Getting the type of 'tuple_var_assignment_78327' (line 955)
    tuple_var_assignment_78327_80897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tuple_var_assignment_78327')
    # Assigning a type to the variable 'c' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 12), 'c', tuple_var_assignment_78327_80897)
    
    # Assigning a Name to a Name (line 955):
    # Getting the type of 'tuple_var_assignment_78328' (line 955)
    tuple_var_assignment_78328_80898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'tuple_var_assignment_78328')
    # Assigning a type to the variable 'o' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 15), 'o', tuple_var_assignment_78328_80898)
    
    # Assigning a Name to a Subscript (line 958):
    
    # Assigning a Name to a Subscript (line 958):
    # Getting the type of 'tx' (line 958)
    tx_80899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 26), 'tx')
    # Getting the type of '_curfit_cache' (line 958)
    _curfit_cache_80900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 4), '_curfit_cache')
    str_80901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 18), 'str', 'tx')
    # Storing an element on a container (line 958)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 4), _curfit_cache_80900, (str_80901, tx_80899))
    
    # Assigning a Name to a Subscript (line 959):
    
    # Assigning a Name to a Subscript (line 959):
    # Getting the type of 'ty' (line 959)
    ty_80902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 26), 'ty')
    # Getting the type of '_curfit_cache' (line 959)
    _curfit_cache_80903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 4), '_curfit_cache')
    str_80904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 18), 'str', 'ty')
    # Storing an element on a container (line 959)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 959, 4), _curfit_cache_80903, (str_80904, ty_80902))
    
    # Assigning a Subscript to a Subscript (line 960):
    
    # Assigning a Subscript to a Subscript (line 960):
    
    # Obtaining the type of the subscript
    str_80905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 29), 'str', 'wrk')
    # Getting the type of 'o' (line 960)
    o_80906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 27), 'o')
    # Obtaining the member '__getitem__' of a type (line 960)
    getitem___80907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 27), o_80906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 960)
    subscript_call_result_80908 = invoke(stypy.reporting.localization.Localization(__file__, 960, 27), getitem___80907, str_80905)
    
    # Getting the type of '_curfit_cache' (line 960)
    _curfit_cache_80909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), '_curfit_cache')
    str_80910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 18), 'str', 'wrk')
    # Storing an element on a container (line 960)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 4), _curfit_cache_80909, (str_80910, subscript_call_result_80908))
    
    # Assigning a Tuple to a Tuple (line 961):
    
    # Assigning a Subscript to a Name (line 961):
    
    # Obtaining the type of the subscript
    str_80911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 16), 'str', 'ier')
    # Getting the type of 'o' (line 961)
    o_80912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 14), 'o')
    # Obtaining the member '__getitem__' of a type (line 961)
    getitem___80913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 14), o_80912, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 961)
    subscript_call_result_80914 = invoke(stypy.reporting.localization.Localization(__file__, 961, 14), getitem___80913, str_80911)
    
    # Assigning a type to the variable 'tuple_assignment_78329' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'tuple_assignment_78329', subscript_call_result_80914)
    
    # Assigning a Subscript to a Name (line 961):
    
    # Obtaining the type of the subscript
    str_80915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 26), 'str', 'fp')
    # Getting the type of 'o' (line 961)
    o_80916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 24), 'o')
    # Obtaining the member '__getitem__' of a type (line 961)
    getitem___80917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 24), o_80916, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 961)
    subscript_call_result_80918 = invoke(stypy.reporting.localization.Localization(__file__, 961, 24), getitem___80917, str_80915)
    
    # Assigning a type to the variable 'tuple_assignment_78330' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'tuple_assignment_78330', subscript_call_result_80918)
    
    # Assigning a Name to a Name (line 961):
    # Getting the type of 'tuple_assignment_78329' (line 961)
    tuple_assignment_78329_80919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'tuple_assignment_78329')
    # Assigning a type to the variable 'ier' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'ier', tuple_assignment_78329_80919)
    
    # Assigning a Name to a Name (line 961):
    # Getting the type of 'tuple_assignment_78330' (line 961)
    tuple_assignment_78330_80920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'tuple_assignment_78330')
    # Assigning a type to the variable 'fp' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 9), 'fp', tuple_assignment_78330_80920)
    
    # Assigning a List to a Name (line 962):
    
    # Assigning a List to a Name (line 962):
    
    # Obtaining an instance of the builtin type 'list' (line 962)
    list_80921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 962)
    # Adding element type (line 962)
    # Getting the type of 'tx' (line 962)
    tx_80922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 11), 'tx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 10), list_80921, tx_80922)
    # Adding element type (line 962)
    # Getting the type of 'ty' (line 962)
    ty_80923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 15), 'ty')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 10), list_80921, ty_80923)
    # Adding element type (line 962)
    # Getting the type of 'c' (line 962)
    c_80924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 19), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 10), list_80921, c_80924)
    # Adding element type (line 962)
    # Getting the type of 'kx' (line 962)
    kx_80925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 22), 'kx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 10), list_80921, kx_80925)
    # Adding element type (line 962)
    # Getting the type of 'ky' (line 962)
    ky_80926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 26), 'ky')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 10), list_80921, ky_80926)
    
    # Assigning a type to the variable 'tck' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 4), 'tck', list_80921)
    
    # Assigning a Call to a Name (line 964):
    
    # Assigning a Call to a Name (line 964):
    
    # Call to min(...): (line 964)
    # Processing the call arguments (line 964)
    int_80928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 15), 'int')
    
    # Call to max(...): (line 964)
    # Processing the call arguments (line 964)
    int_80930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 23), 'int')
    # Getting the type of 'ier' (line 964)
    ier_80931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 27), 'ier', False)
    # Processing the call keyword arguments (line 964)
    kwargs_80932 = {}
    # Getting the type of 'max' (line 964)
    max_80929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 19), 'max', False)
    # Calling max(args, kwargs) (line 964)
    max_call_result_80933 = invoke(stypy.reporting.localization.Localization(__file__, 964, 19), max_80929, *[int_80930, ier_80931], **kwargs_80932)
    
    # Processing the call keyword arguments (line 964)
    kwargs_80934 = {}
    # Getting the type of 'min' (line 964)
    min_80927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 11), 'min', False)
    # Calling min(args, kwargs) (line 964)
    min_call_result_80935 = invoke(stypy.reporting.localization.Localization(__file__, 964, 11), min_80927, *[int_80928, max_call_result_80933], **kwargs_80934)
    
    # Assigning a type to the variable 'ierm' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 4), 'ierm', min_call_result_80935)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ierm' (line 965)
    ierm_80936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 7), 'ierm')
    int_80937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 15), 'int')
    # Applying the binary operator '<=' (line 965)
    result_le_80938 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 7), '<=', ierm_80936, int_80937)
    
    
    # Getting the type of 'quiet' (line 965)
    quiet_80939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 25), 'quiet')
    # Applying the 'not' unary operator (line 965)
    result_not__80940 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 21), 'not', quiet_80939)
    
    # Applying the binary operator 'and' (line 965)
    result_and_keyword_80941 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 7), 'and', result_le_80938, result_not__80940)
    
    # Testing the type of an if condition (line 965)
    if_condition_80942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 965, 4), result_and_keyword_80941)
    # Assigning a type to the variable 'if_condition_80942' (line 965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 4), 'if_condition_80942', if_condition_80942)
    # SSA begins for if statement (line 965)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 966):
    
    # Assigning a BinOp to a Name (line 966):
    
    # Obtaining the type of the subscript
    int_80943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 33), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ierm' (line 966)
    ierm_80944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 27), 'ierm')
    # Getting the type of '_iermess2' (line 966)
    _iermess2_80945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 17), '_iermess2')
    # Obtaining the member '__getitem__' of a type (line 966)
    getitem___80946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 17), _iermess2_80945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 966)
    subscript_call_result_80947 = invoke(stypy.reporting.localization.Localization(__file__, 966, 17), getitem___80946, ierm_80944)
    
    # Obtaining the member '__getitem__' of a type (line 966)
    getitem___80948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 17), subscript_call_result_80947, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 966)
    subscript_call_result_80949 = invoke(stypy.reporting.localization.Localization(__file__, 966, 17), getitem___80948, int_80943)
    
    str_80950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 17), 'str', '\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 968)
    tuple_80951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 968)
    # Adding element type (line 968)
    # Getting the type of 'kx' (line 968)
    kx_80952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 18), 'kx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 18), tuple_80951, kx_80952)
    # Adding element type (line 968)
    # Getting the type of 'ky' (line 968)
    ky_80953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 22), 'ky')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 18), tuple_80951, ky_80953)
    # Adding element type (line 968)
    
    # Call to len(...): (line 968)
    # Processing the call arguments (line 968)
    # Getting the type of 'tx' (line 968)
    tx_80955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 30), 'tx', False)
    # Processing the call keyword arguments (line 968)
    kwargs_80956 = {}
    # Getting the type of 'len' (line 968)
    len_80954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 26), 'len', False)
    # Calling len(args, kwargs) (line 968)
    len_call_result_80957 = invoke(stypy.reporting.localization.Localization(__file__, 968, 26), len_80954, *[tx_80955], **kwargs_80956)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 18), tuple_80951, len_call_result_80957)
    # Adding element type (line 968)
    
    # Call to len(...): (line 968)
    # Processing the call arguments (line 968)
    # Getting the type of 'ty' (line 968)
    ty_80959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 39), 'ty', False)
    # Processing the call keyword arguments (line 968)
    kwargs_80960 = {}
    # Getting the type of 'len' (line 968)
    len_80958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 35), 'len', False)
    # Calling len(args, kwargs) (line 968)
    len_call_result_80961 = invoke(stypy.reporting.localization.Localization(__file__, 968, 35), len_80958, *[ty_80959], **kwargs_80960)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 18), tuple_80951, len_call_result_80961)
    # Adding element type (line 968)
    # Getting the type of 'm' (line 968)
    m_80962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 44), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 18), tuple_80951, m_80962)
    # Adding element type (line 968)
    # Getting the type of 'fp' (line 968)
    fp_80963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 47), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 18), tuple_80951, fp_80963)
    # Adding element type (line 968)
    # Getting the type of 's' (line 968)
    s_80964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 51), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 968, 18), tuple_80951, s_80964)
    
    # Applying the binary operator '%' (line 967)
    result_mod_80965 = python_operator(stypy.reporting.localization.Localization(__file__, 967, 17), '%', str_80950, tuple_80951)
    
    # Applying the binary operator '+' (line 966)
    result_add_80966 = python_operator(stypy.reporting.localization.Localization(__file__, 966, 17), '+', subscript_call_result_80949, result_mod_80965)
    
    # Assigning a type to the variable '_mess' (line 966)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), '_mess', result_add_80966)
    
    # Call to warn(...): (line 969)
    # Processing the call arguments (line 969)
    
    # Call to RuntimeWarning(...): (line 969)
    # Processing the call arguments (line 969)
    # Getting the type of '_mess' (line 969)
    _mess_80970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 37), '_mess', False)
    # Processing the call keyword arguments (line 969)
    kwargs_80971 = {}
    # Getting the type of 'RuntimeWarning' (line 969)
    RuntimeWarning_80969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 22), 'RuntimeWarning', False)
    # Calling RuntimeWarning(args, kwargs) (line 969)
    RuntimeWarning_call_result_80972 = invoke(stypy.reporting.localization.Localization(__file__, 969, 22), RuntimeWarning_80969, *[_mess_80970], **kwargs_80971)
    
    # Processing the call keyword arguments (line 969)
    kwargs_80973 = {}
    # Getting the type of 'warnings' (line 969)
    warnings_80967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 969)
    warn_80968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 8), warnings_80967, 'warn')
    # Calling warn(args, kwargs) (line 969)
    warn_call_result_80974 = invoke(stypy.reporting.localization.Localization(__file__, 969, 8), warn_80968, *[RuntimeWarning_call_result_80972], **kwargs_80973)
    
    # SSA join for if statement (line 965)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ierm' (line 970)
    ierm_80975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 7), 'ierm')
    int_80976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 14), 'int')
    # Applying the binary operator '>' (line 970)
    result_gt_80977 = python_operator(stypy.reporting.localization.Localization(__file__, 970, 7), '>', ierm_80975, int_80976)
    
    
    # Getting the type of 'full_output' (line 970)
    full_output_80978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 24), 'full_output')
    # Applying the 'not' unary operator (line 970)
    result_not__80979 = python_operator(stypy.reporting.localization.Localization(__file__, 970, 20), 'not', full_output_80978)
    
    # Applying the binary operator 'and' (line 970)
    result_and_keyword_80980 = python_operator(stypy.reporting.localization.Localization(__file__, 970, 7), 'and', result_gt_80977, result_not__80979)
    
    # Testing the type of an if condition (line 970)
    if_condition_80981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 970, 4), result_and_keyword_80980)
    # Assigning a type to the variable 'if_condition_80981' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'if_condition_80981', if_condition_80981)
    # SSA begins for if statement (line 970)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'ier' (line 971)
    ier_80982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 11), 'ier')
    
    # Obtaining an instance of the builtin type 'list' (line 971)
    list_80983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 971)
    # Adding element type (line 971)
    int_80984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 971, 18), list_80983, int_80984)
    # Adding element type (line 971)
    int_80985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 971, 18), list_80983, int_80985)
    # Adding element type (line 971)
    int_80986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 971, 18), list_80983, int_80986)
    # Adding element type (line 971)
    int_80987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 971, 18), list_80983, int_80987)
    # Adding element type (line 971)
    int_80988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 971, 18), list_80983, int_80988)
    
    # Applying the binary operator 'in' (line 971)
    result_contains_80989 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 11), 'in', ier_80982, list_80983)
    
    # Testing the type of an if condition (line 971)
    if_condition_80990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 971, 8), result_contains_80989)
    # Assigning a type to the variable 'if_condition_80990' (line 971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 8), 'if_condition_80990', if_condition_80990)
    # SSA begins for if statement (line 971)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 972):
    
    # Assigning a BinOp to a Name (line 972):
    str_80991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 21), 'str', '\n\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 973)
    tuple_80992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 973)
    # Adding element type (line 973)
    # Getting the type of 'kx' (line 973)
    kx_80993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 22), 'kx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 22), tuple_80992, kx_80993)
    # Adding element type (line 973)
    # Getting the type of 'ky' (line 973)
    ky_80994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 26), 'ky')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 22), tuple_80992, ky_80994)
    # Adding element type (line 973)
    
    # Call to len(...): (line 973)
    # Processing the call arguments (line 973)
    # Getting the type of 'tx' (line 973)
    tx_80996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 34), 'tx', False)
    # Processing the call keyword arguments (line 973)
    kwargs_80997 = {}
    # Getting the type of 'len' (line 973)
    len_80995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 30), 'len', False)
    # Calling len(args, kwargs) (line 973)
    len_call_result_80998 = invoke(stypy.reporting.localization.Localization(__file__, 973, 30), len_80995, *[tx_80996], **kwargs_80997)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 22), tuple_80992, len_call_result_80998)
    # Adding element type (line 973)
    
    # Call to len(...): (line 973)
    # Processing the call arguments (line 973)
    # Getting the type of 'ty' (line 973)
    ty_81000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 43), 'ty', False)
    # Processing the call keyword arguments (line 973)
    kwargs_81001 = {}
    # Getting the type of 'len' (line 973)
    len_80999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 39), 'len', False)
    # Calling len(args, kwargs) (line 973)
    len_call_result_81002 = invoke(stypy.reporting.localization.Localization(__file__, 973, 39), len_80999, *[ty_81000], **kwargs_81001)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 22), tuple_80992, len_call_result_81002)
    # Adding element type (line 973)
    # Getting the type of 'm' (line 973)
    m_81003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 48), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 22), tuple_80992, m_81003)
    # Adding element type (line 973)
    # Getting the type of 'fp' (line 973)
    fp_81004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 51), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 22), tuple_80992, fp_81004)
    # Adding element type (line 973)
    # Getting the type of 's' (line 973)
    s_81005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 55), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 22), tuple_80992, s_81005)
    
    # Applying the binary operator '%' (line 972)
    result_mod_81006 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 21), '%', str_80991, tuple_80992)
    
    # Assigning a type to the variable '_mess' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 12), '_mess', result_mod_81006)
    
    # Call to warn(...): (line 974)
    # Processing the call arguments (line 974)
    
    # Call to RuntimeWarning(...): (line 974)
    # Processing the call arguments (line 974)
    
    # Obtaining the type of the subscript
    int_81010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 57), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ierm' (line 974)
    ierm_81011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 51), 'ierm', False)
    # Getting the type of '_iermess2' (line 974)
    _iermess2_81012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 41), '_iermess2', False)
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___81013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 41), _iermess2_81012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_81014 = invoke(stypy.reporting.localization.Localization(__file__, 974, 41), getitem___81013, ierm_81011)
    
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___81015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 41), subscript_call_result_81014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_81016 = invoke(stypy.reporting.localization.Localization(__file__, 974, 41), getitem___81015, int_81010)
    
    # Getting the type of '_mess' (line 974)
    _mess_81017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 62), '_mess', False)
    # Applying the binary operator '+' (line 974)
    result_add_81018 = python_operator(stypy.reporting.localization.Localization(__file__, 974, 41), '+', subscript_call_result_81016, _mess_81017)
    
    # Processing the call keyword arguments (line 974)
    kwargs_81019 = {}
    # Getting the type of 'RuntimeWarning' (line 974)
    RuntimeWarning_81009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 26), 'RuntimeWarning', False)
    # Calling RuntimeWarning(args, kwargs) (line 974)
    RuntimeWarning_call_result_81020 = invoke(stypy.reporting.localization.Localization(__file__, 974, 26), RuntimeWarning_81009, *[result_add_81018], **kwargs_81019)
    
    # Processing the call keyword arguments (line 974)
    kwargs_81021 = {}
    # Getting the type of 'warnings' (line 974)
    warnings_81007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 974)
    warn_81008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 12), warnings_81007, 'warn')
    # Calling warn(args, kwargs) (line 974)
    warn_call_result_81022 = invoke(stypy.reporting.localization.Localization(__file__, 974, 12), warn_81008, *[RuntimeWarning_call_result_81020], **kwargs_81021)
    
    # SSA branch for the else part of an if statement (line 971)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 976)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to (...): (line 977)
    # Processing the call arguments (line 977)
    
    # Obtaining the type of the subscript
    int_81030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 57), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ierm' (line 977)
    ierm_81031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 51), 'ierm', False)
    # Getting the type of '_iermess2' (line 977)
    _iermess2_81032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 41), '_iermess2', False)
    # Obtaining the member '__getitem__' of a type (line 977)
    getitem___81033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 41), _iermess2_81032, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 977)
    subscript_call_result_81034 = invoke(stypy.reporting.localization.Localization(__file__, 977, 41), getitem___81033, ierm_81031)
    
    # Obtaining the member '__getitem__' of a type (line 977)
    getitem___81035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 41), subscript_call_result_81034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 977)
    subscript_call_result_81036 = invoke(stypy.reporting.localization.Localization(__file__, 977, 41), getitem___81035, int_81030)
    
    # Processing the call keyword arguments (line 977)
    kwargs_81037 = {}
    
    # Obtaining the type of the subscript
    int_81023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 38), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ierm' (line 977)
    ierm_81024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 32), 'ierm', False)
    # Getting the type of '_iermess2' (line 977)
    _iermess2_81025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 22), '_iermess2', False)
    # Obtaining the member '__getitem__' of a type (line 977)
    getitem___81026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 22), _iermess2_81025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 977)
    subscript_call_result_81027 = invoke(stypy.reporting.localization.Localization(__file__, 977, 22), getitem___81026, ierm_81024)
    
    # Obtaining the member '__getitem__' of a type (line 977)
    getitem___81028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 22), subscript_call_result_81027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 977)
    subscript_call_result_81029 = invoke(stypy.reporting.localization.Localization(__file__, 977, 22), getitem___81028, int_81023)
    
    # Calling (args, kwargs) (line 977)
    _call_result_81038 = invoke(stypy.reporting.localization.Localization(__file__, 977, 22), subscript_call_result_81029, *[subscript_call_result_81036], **kwargs_81037)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 977, 16), _call_result_81038, 'raise parameter', BaseException)
    # SSA branch for the except part of a try statement (line 976)
    # SSA branch for the except 'KeyError' branch of a try statement (line 976)
    module_type_store.open_ssa_branch('except')
    
    # Call to (...): (line 979)
    # Processing the call arguments (line 979)
    
    # Obtaining the type of the subscript
    int_81046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 67), 'int')
    
    # Obtaining the type of the subscript
    str_81047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 56), 'str', 'unknown')
    # Getting the type of '_iermess2' (line 979)
    _iermess2_81048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 46), '_iermess2', False)
    # Obtaining the member '__getitem__' of a type (line 979)
    getitem___81049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 46), _iermess2_81048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 979)
    subscript_call_result_81050 = invoke(stypy.reporting.localization.Localization(__file__, 979, 46), getitem___81049, str_81047)
    
    # Obtaining the member '__getitem__' of a type (line 979)
    getitem___81051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 46), subscript_call_result_81050, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 979)
    subscript_call_result_81052 = invoke(stypy.reporting.localization.Localization(__file__, 979, 46), getitem___81051, int_81046)
    
    # Processing the call keyword arguments (line 979)
    kwargs_81053 = {}
    
    # Obtaining the type of the subscript
    int_81039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 43), 'int')
    
    # Obtaining the type of the subscript
    str_81040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 32), 'str', 'unknown')
    # Getting the type of '_iermess2' (line 979)
    _iermess2_81041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 22), '_iermess2', False)
    # Obtaining the member '__getitem__' of a type (line 979)
    getitem___81042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 22), _iermess2_81041, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 979)
    subscript_call_result_81043 = invoke(stypy.reporting.localization.Localization(__file__, 979, 22), getitem___81042, str_81040)
    
    # Obtaining the member '__getitem__' of a type (line 979)
    getitem___81044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 22), subscript_call_result_81043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 979)
    subscript_call_result_81045 = invoke(stypy.reporting.localization.Localization(__file__, 979, 22), getitem___81044, int_81039)
    
    # Calling (args, kwargs) (line 979)
    _call_result_81054 = invoke(stypy.reporting.localization.Localization(__file__, 979, 22), subscript_call_result_81045, *[subscript_call_result_81052], **kwargs_81053)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 979, 16), _call_result_81054, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 976)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 971)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 970)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full_output' (line 980)
    full_output_81055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 7), 'full_output')
    # Testing the type of an if condition (line 980)
    if_condition_81056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 980, 4), full_output_81055)
    # Assigning a type to the variable 'if_condition_81056' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 4), 'if_condition_81056', if_condition_81056)
    # SSA begins for if statement (line 980)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 981)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 982)
    tuple_81057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 982)
    # Adding element type (line 982)
    # Getting the type of 'tck' (line 982)
    tck_81058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 19), 'tck')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 982, 19), tuple_81057, tck_81058)
    # Adding element type (line 982)
    # Getting the type of 'fp' (line 982)
    fp_81059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 24), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 982, 19), tuple_81057, fp_81059)
    # Adding element type (line 982)
    # Getting the type of 'ier' (line 982)
    ier_81060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 28), 'ier')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 982, 19), tuple_81057, ier_81060)
    # Adding element type (line 982)
    
    # Obtaining the type of the subscript
    int_81061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 49), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ierm' (line 982)
    ierm_81062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 43), 'ierm')
    # Getting the type of '_iermess2' (line 982)
    _iermess2_81063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 33), '_iermess2')
    # Obtaining the member '__getitem__' of a type (line 982)
    getitem___81064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 982, 33), _iermess2_81063, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 982)
    subscript_call_result_81065 = invoke(stypy.reporting.localization.Localization(__file__, 982, 33), getitem___81064, ierm_81062)
    
    # Obtaining the member '__getitem__' of a type (line 982)
    getitem___81066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 982, 33), subscript_call_result_81065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 982)
    subscript_call_result_81067 = invoke(stypy.reporting.localization.Localization(__file__, 982, 33), getitem___81066, int_81061)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 982, 19), tuple_81057, subscript_call_result_81067)
    
    # Assigning a type to the variable 'stypy_return_type' (line 982)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 12), 'stypy_return_type', tuple_81057)
    # SSA branch for the except part of a try statement (line 981)
    # SSA branch for the except 'KeyError' branch of a try statement (line 981)
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 984)
    tuple_81068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 984)
    # Adding element type (line 984)
    # Getting the type of 'tck' (line 984)
    tck_81069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 19), 'tck')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 984, 19), tuple_81068, tck_81069)
    # Adding element type (line 984)
    # Getting the type of 'fp' (line 984)
    fp_81070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 24), 'fp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 984, 19), tuple_81068, fp_81070)
    # Adding element type (line 984)
    # Getting the type of 'ier' (line 984)
    ier_81071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 28), 'ier')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 984, 19), tuple_81068, ier_81071)
    # Adding element type (line 984)
    
    # Obtaining the type of the subscript
    int_81072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 54), 'int')
    
    # Obtaining the type of the subscript
    str_81073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 43), 'str', 'unknown')
    # Getting the type of '_iermess2' (line 984)
    _iermess2_81074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 33), '_iermess2')
    # Obtaining the member '__getitem__' of a type (line 984)
    getitem___81075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 33), _iermess2_81074, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 984)
    subscript_call_result_81076 = invoke(stypy.reporting.localization.Localization(__file__, 984, 33), getitem___81075, str_81073)
    
    # Obtaining the member '__getitem__' of a type (line 984)
    getitem___81077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 33), subscript_call_result_81076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 984)
    subscript_call_result_81078 = invoke(stypy.reporting.localization.Localization(__file__, 984, 33), getitem___81077, int_81072)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 984, 19), tuple_81068, subscript_call_result_81078)
    
    # Assigning a type to the variable 'stypy_return_type' (line 984)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 12), 'stypy_return_type', tuple_81068)
    # SSA join for try-except statement (line 981)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 980)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'tck' (line 986)
    tck_81079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 15), 'tck')
    # Assigning a type to the variable 'stypy_return_type' (line 986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 8), 'stypy_return_type', tck_81079)
    # SSA join for if statement (line 980)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'bisplrep(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bisplrep' in the type store
    # Getting the type of 'stypy_return_type' (line 799)
    stypy_return_type_81080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_81080)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bisplrep'
    return stypy_return_type_81080

# Assigning a type to the variable 'bisplrep' (line 799)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 0), 'bisplrep', bisplrep)

@norecursion
def bisplev(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_81081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 26), 'int')
    int_81082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 32), 'int')
    defaults = [int_81081, int_81082]
    # Create a new context for function 'bisplev'
    module_type_store = module_type_store.open_function_context('bisplev', 989, 0, False)
    
    # Passed parameters checking function
    bisplev.stypy_localization = localization
    bisplev.stypy_type_of_self = None
    bisplev.stypy_type_store = module_type_store
    bisplev.stypy_function_name = 'bisplev'
    bisplev.stypy_param_names_list = ['x', 'y', 'tck', 'dx', 'dy']
    bisplev.stypy_varargs_param_name = None
    bisplev.stypy_kwargs_param_name = None
    bisplev.stypy_call_defaults = defaults
    bisplev.stypy_call_varargs = varargs
    bisplev.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bisplev', ['x', 'y', 'tck', 'dx', 'dy'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bisplev', localization, ['x', 'y', 'tck', 'dx', 'dy'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bisplev(...)' code ##################

    str_81083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, (-1)), 'str', '\n    Evaluate a bivariate B-spline and its derivatives.\n\n    Return a rank-2 array of spline function values (or spline derivative\n    values) at points given by the cross-product of the rank-1 arrays `x` and\n    `y`.  In special cases, return an array or just a float if either `x` or\n    `y` or both are floats.  Based on BISPEV from FITPACK.\n\n    Parameters\n    ----------\n    x, y : ndarray\n        Rank-1 arrays specifying the domain over which to evaluate the\n        spline or its derivative.\n    tck : tuple\n        A sequence of length 5 returned by `bisplrep` containing the knot\n        locations, the coefficients, and the degree of the spline:\n        [tx, ty, c, kx, ky].\n    dx, dy : int, optional\n        The orders of the partial derivatives in `x` and `y` respectively.\n\n    Returns\n    -------\n    vals : ndarray\n        The B-spline or its derivative evaluated over the set formed by\n        the cross-product of `x` and `y`.\n\n    See Also\n    --------\n    splprep, splrep, splint, sproot, splev\n    UnivariateSpline, BivariateSpline\n\n    Notes\n    -----\n        See `bisplrep` to generate the `tck` representation.\n\n    References\n    ----------\n    .. [1] Dierckx P. : An algorithm for surface fitting\n       with spline functions\n       Ima J. Numer. Anal. 1 (1981) 267-283.\n    .. [2] Dierckx P. : An algorithm for surface fitting\n       with spline functions\n       report tw50, Dept. Computer Science,K.U.Leuven, 1980.\n    .. [3] Dierckx P. : Curve and surface fitting with splines,\n       Monographs on Numerical Analysis, Oxford University Press, 1993.\n\n    ')
    
    # Assigning a Name to a Tuple (line 1037):
    
    # Assigning a Subscript to a Name (line 1037):
    
    # Obtaining the type of the subscript
    int_81084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 4), 'int')
    # Getting the type of 'tck' (line 1037)
    tck_81085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1037)
    getitem___81086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 4), tck_81085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1037)
    subscript_call_result_81087 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 4), getitem___81086, int_81084)
    
    # Assigning a type to the variable 'tuple_var_assignment_78331' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78331', subscript_call_result_81087)
    
    # Assigning a Subscript to a Name (line 1037):
    
    # Obtaining the type of the subscript
    int_81088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 4), 'int')
    # Getting the type of 'tck' (line 1037)
    tck_81089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1037)
    getitem___81090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 4), tck_81089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1037)
    subscript_call_result_81091 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 4), getitem___81090, int_81088)
    
    # Assigning a type to the variable 'tuple_var_assignment_78332' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78332', subscript_call_result_81091)
    
    # Assigning a Subscript to a Name (line 1037):
    
    # Obtaining the type of the subscript
    int_81092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 4), 'int')
    # Getting the type of 'tck' (line 1037)
    tck_81093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1037)
    getitem___81094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 4), tck_81093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1037)
    subscript_call_result_81095 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 4), getitem___81094, int_81092)
    
    # Assigning a type to the variable 'tuple_var_assignment_78333' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78333', subscript_call_result_81095)
    
    # Assigning a Subscript to a Name (line 1037):
    
    # Obtaining the type of the subscript
    int_81096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 4), 'int')
    # Getting the type of 'tck' (line 1037)
    tck_81097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1037)
    getitem___81098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 4), tck_81097, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1037)
    subscript_call_result_81099 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 4), getitem___81098, int_81096)
    
    # Assigning a type to the variable 'tuple_var_assignment_78334' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78334', subscript_call_result_81099)
    
    # Assigning a Subscript to a Name (line 1037):
    
    # Obtaining the type of the subscript
    int_81100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 4), 'int')
    # Getting the type of 'tck' (line 1037)
    tck_81101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1037)
    getitem___81102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 4), tck_81101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1037)
    subscript_call_result_81103 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 4), getitem___81102, int_81100)
    
    # Assigning a type to the variable 'tuple_var_assignment_78335' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78335', subscript_call_result_81103)
    
    # Assigning a Name to a Name (line 1037):
    # Getting the type of 'tuple_var_assignment_78331' (line 1037)
    tuple_var_assignment_78331_81104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78331')
    # Assigning a type to the variable 'tx' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tx', tuple_var_assignment_78331_81104)
    
    # Assigning a Name to a Name (line 1037):
    # Getting the type of 'tuple_var_assignment_78332' (line 1037)
    tuple_var_assignment_78332_81105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78332')
    # Assigning a type to the variable 'ty' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'ty', tuple_var_assignment_78332_81105)
    
    # Assigning a Name to a Name (line 1037):
    # Getting the type of 'tuple_var_assignment_78333' (line 1037)
    tuple_var_assignment_78333_81106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78333')
    # Assigning a type to the variable 'c' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 12), 'c', tuple_var_assignment_78333_81106)
    
    # Assigning a Name to a Name (line 1037):
    # Getting the type of 'tuple_var_assignment_78334' (line 1037)
    tuple_var_assignment_78334_81107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78334')
    # Assigning a type to the variable 'kx' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 15), 'kx', tuple_var_assignment_78334_81107)
    
    # Assigning a Name to a Name (line 1037):
    # Getting the type of 'tuple_var_assignment_78335' (line 1037)
    tuple_var_assignment_78335_81108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'tuple_var_assignment_78335')
    # Assigning a type to the variable 'ky' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 19), 'ky', tuple_var_assignment_78335_81108)
    
    
    
    int_81109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 12), 'int')
    # Getting the type of 'dx' (line 1038)
    dx_81110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 17), 'dx')
    # Applying the binary operator '<=' (line 1038)
    result_le_81111 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 12), '<=', int_81109, dx_81110)
    # Getting the type of 'kx' (line 1038)
    kx_81112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 22), 'kx')
    # Applying the binary operator '<' (line 1038)
    result_lt_81113 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 12), '<', dx_81110, kx_81112)
    # Applying the binary operator '&' (line 1038)
    result_and__81114 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 12), '&', result_le_81111, result_lt_81113)
    
    # Applying the 'not' unary operator (line 1038)
    result_not__81115 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 7), 'not', result_and__81114)
    
    # Testing the type of an if condition (line 1038)
    if_condition_81116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1038, 4), result_not__81115)
    # Assigning a type to the variable 'if_condition_81116' (line 1038)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1038, 4), 'if_condition_81116', if_condition_81116)
    # SSA begins for if statement (line 1038)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1039)
    # Processing the call arguments (line 1039)
    str_81118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 25), 'str', '0 <= dx = %d < kx = %d must hold')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1039)
    tuple_81119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 63), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1039)
    # Adding element type (line 1039)
    # Getting the type of 'dx' (line 1039)
    dx_81120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 63), 'dx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1039, 63), tuple_81119, dx_81120)
    # Adding element type (line 1039)
    # Getting the type of 'kx' (line 1039)
    kx_81121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 67), 'kx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1039, 63), tuple_81119, kx_81121)
    
    # Applying the binary operator '%' (line 1039)
    result_mod_81122 = python_operator(stypy.reporting.localization.Localization(__file__, 1039, 25), '%', str_81118, tuple_81119)
    
    # Processing the call keyword arguments (line 1039)
    kwargs_81123 = {}
    # Getting the type of 'ValueError' (line 1039)
    ValueError_81117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1039)
    ValueError_call_result_81124 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 14), ValueError_81117, *[result_mod_81122], **kwargs_81123)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1039, 8), ValueError_call_result_81124, 'raise parameter', BaseException)
    # SSA join for if statement (line 1038)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    int_81125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 12), 'int')
    # Getting the type of 'dy' (line 1040)
    dy_81126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 17), 'dy')
    # Applying the binary operator '<=' (line 1040)
    result_le_81127 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 12), '<=', int_81125, dy_81126)
    # Getting the type of 'ky' (line 1040)
    ky_81128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 22), 'ky')
    # Applying the binary operator '<' (line 1040)
    result_lt_81129 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 12), '<', dy_81126, ky_81128)
    # Applying the binary operator '&' (line 1040)
    result_and__81130 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 12), '&', result_le_81127, result_lt_81129)
    
    # Applying the 'not' unary operator (line 1040)
    result_not__81131 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 7), 'not', result_and__81130)
    
    # Testing the type of an if condition (line 1040)
    if_condition_81132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1040, 4), result_not__81131)
    # Assigning a type to the variable 'if_condition_81132' (line 1040)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1040, 4), 'if_condition_81132', if_condition_81132)
    # SSA begins for if statement (line 1040)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1041)
    # Processing the call arguments (line 1041)
    str_81134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 25), 'str', '0 <= dy = %d < ky = %d must hold')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1041)
    tuple_81135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 63), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1041)
    # Adding element type (line 1041)
    # Getting the type of 'dy' (line 1041)
    dy_81136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 63), 'dy', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 63), tuple_81135, dy_81136)
    # Adding element type (line 1041)
    # Getting the type of 'ky' (line 1041)
    ky_81137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 67), 'ky', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 63), tuple_81135, ky_81137)
    
    # Applying the binary operator '%' (line 1041)
    result_mod_81138 = python_operator(stypy.reporting.localization.Localization(__file__, 1041, 25), '%', str_81134, tuple_81135)
    
    # Processing the call keyword arguments (line 1041)
    kwargs_81139 = {}
    # Getting the type of 'ValueError' (line 1041)
    ValueError_81133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1041)
    ValueError_call_result_81140 = invoke(stypy.reporting.localization.Localization(__file__, 1041, 14), ValueError_81133, *[result_mod_81138], **kwargs_81139)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1041, 8), ValueError_call_result_81140, 'raise parameter', BaseException)
    # SSA join for if statement (line 1040)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1042):
    
    # Assigning a Subscript to a Name (line 1042):
    
    # Obtaining the type of the subscript
    int_81141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 4), 'int')
    
    # Call to map(...): (line 1042)
    # Processing the call arguments (line 1042)
    # Getting the type of 'atleast_1d' (line 1042)
    atleast_1d_81143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 15), 'atleast_1d', False)
    
    # Obtaining an instance of the builtin type 'list' (line 1042)
    list_81144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1042)
    # Adding element type (line 1042)
    # Getting the type of 'x' (line 1042)
    x_81145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 27), list_81144, x_81145)
    # Adding element type (line 1042)
    # Getting the type of 'y' (line 1042)
    y_81146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 27), list_81144, y_81146)
    
    # Processing the call keyword arguments (line 1042)
    kwargs_81147 = {}
    # Getting the type of 'map' (line 1042)
    map_81142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 11), 'map', False)
    # Calling map(args, kwargs) (line 1042)
    map_call_result_81148 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 11), map_81142, *[atleast_1d_81143, list_81144], **kwargs_81147)
    
    # Obtaining the member '__getitem__' of a type (line 1042)
    getitem___81149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 4), map_call_result_81148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1042)
    subscript_call_result_81150 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 4), getitem___81149, int_81141)
    
    # Assigning a type to the variable 'tuple_var_assignment_78336' (line 1042)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 4), 'tuple_var_assignment_78336', subscript_call_result_81150)
    
    # Assigning a Subscript to a Name (line 1042):
    
    # Obtaining the type of the subscript
    int_81151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 4), 'int')
    
    # Call to map(...): (line 1042)
    # Processing the call arguments (line 1042)
    # Getting the type of 'atleast_1d' (line 1042)
    atleast_1d_81153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 15), 'atleast_1d', False)
    
    # Obtaining an instance of the builtin type 'list' (line 1042)
    list_81154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1042)
    # Adding element type (line 1042)
    # Getting the type of 'x' (line 1042)
    x_81155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 27), list_81154, x_81155)
    # Adding element type (line 1042)
    # Getting the type of 'y' (line 1042)
    y_81156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 27), list_81154, y_81156)
    
    # Processing the call keyword arguments (line 1042)
    kwargs_81157 = {}
    # Getting the type of 'map' (line 1042)
    map_81152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 11), 'map', False)
    # Calling map(args, kwargs) (line 1042)
    map_call_result_81158 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 11), map_81152, *[atleast_1d_81153, list_81154], **kwargs_81157)
    
    # Obtaining the member '__getitem__' of a type (line 1042)
    getitem___81159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 4), map_call_result_81158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1042)
    subscript_call_result_81160 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 4), getitem___81159, int_81151)
    
    # Assigning a type to the variable 'tuple_var_assignment_78337' (line 1042)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 4), 'tuple_var_assignment_78337', subscript_call_result_81160)
    
    # Assigning a Name to a Name (line 1042):
    # Getting the type of 'tuple_var_assignment_78336' (line 1042)
    tuple_var_assignment_78336_81161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 4), 'tuple_var_assignment_78336')
    # Assigning a type to the variable 'x' (line 1042)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 4), 'x', tuple_var_assignment_78336_81161)
    
    # Assigning a Name to a Name (line 1042):
    # Getting the type of 'tuple_var_assignment_78337' (line 1042)
    tuple_var_assignment_78337_81162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 4), 'tuple_var_assignment_78337')
    # Assigning a type to the variable 'y' (line 1042)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 7), 'y', tuple_var_assignment_78337_81162)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 1043)
    # Processing the call arguments (line 1043)
    # Getting the type of 'x' (line 1043)
    x_81164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 12), 'x', False)
    # Obtaining the member 'shape' of a type (line 1043)
    shape_81165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 12), x_81164, 'shape')
    # Processing the call keyword arguments (line 1043)
    kwargs_81166 = {}
    # Getting the type of 'len' (line 1043)
    len_81163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 8), 'len', False)
    # Calling len(args, kwargs) (line 1043)
    len_call_result_81167 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 8), len_81163, *[shape_81165], **kwargs_81166)
    
    int_81168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 24), 'int')
    # Applying the binary operator '!=' (line 1043)
    result_ne_81169 = python_operator(stypy.reporting.localization.Localization(__file__, 1043, 8), '!=', len_call_result_81167, int_81168)
    
    
    
    # Call to len(...): (line 1043)
    # Processing the call arguments (line 1043)
    # Getting the type of 'y' (line 1043)
    y_81171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 35), 'y', False)
    # Obtaining the member 'shape' of a type (line 1043)
    shape_81172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 35), y_81171, 'shape')
    # Processing the call keyword arguments (line 1043)
    kwargs_81173 = {}
    # Getting the type of 'len' (line 1043)
    len_81170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 31), 'len', False)
    # Calling len(args, kwargs) (line 1043)
    len_call_result_81174 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 31), len_81170, *[shape_81172], **kwargs_81173)
    
    int_81175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 47), 'int')
    # Applying the binary operator '!=' (line 1043)
    result_ne_81176 = python_operator(stypy.reporting.localization.Localization(__file__, 1043, 31), '!=', len_call_result_81174, int_81175)
    
    # Applying the binary operator 'or' (line 1043)
    result_or_keyword_81177 = python_operator(stypy.reporting.localization.Localization(__file__, 1043, 7), 'or', result_ne_81169, result_ne_81176)
    
    # Testing the type of an if condition (line 1043)
    if_condition_81178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1043, 4), result_or_keyword_81177)
    # Assigning a type to the variable 'if_condition_81178' (line 1043)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 4), 'if_condition_81178', if_condition_81178)
    # SSA begins for if statement (line 1043)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1044)
    # Processing the call arguments (line 1044)
    str_81180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 25), 'str', 'First two entries should be rank-1 arrays.')
    # Processing the call keyword arguments (line 1044)
    kwargs_81181 = {}
    # Getting the type of 'ValueError' (line 1044)
    ValueError_81179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1044)
    ValueError_call_result_81182 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 14), ValueError_81179, *[str_81180], **kwargs_81181)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1044, 8), ValueError_call_result_81182, 'raise parameter', BaseException)
    # SSA join for if statement (line 1043)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1045):
    
    # Assigning a Subscript to a Name (line 1045):
    
    # Obtaining the type of the subscript
    int_81183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 4), 'int')
    
    # Call to _bispev(...): (line 1045)
    # Processing the call arguments (line 1045)
    # Getting the type of 'tx' (line 1045)
    tx_81186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 30), 'tx', False)
    # Getting the type of 'ty' (line 1045)
    ty_81187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 34), 'ty', False)
    # Getting the type of 'c' (line 1045)
    c_81188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 38), 'c', False)
    # Getting the type of 'kx' (line 1045)
    kx_81189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 41), 'kx', False)
    # Getting the type of 'ky' (line 1045)
    ky_81190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 45), 'ky', False)
    # Getting the type of 'x' (line 1045)
    x_81191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 49), 'x', False)
    # Getting the type of 'y' (line 1045)
    y_81192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 52), 'y', False)
    # Getting the type of 'dx' (line 1045)
    dx_81193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 55), 'dx', False)
    # Getting the type of 'dy' (line 1045)
    dy_81194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 59), 'dy', False)
    # Processing the call keyword arguments (line 1045)
    kwargs_81195 = {}
    # Getting the type of '_fitpack' (line 1045)
    _fitpack_81184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 13), '_fitpack', False)
    # Obtaining the member '_bispev' of a type (line 1045)
    _bispev_81185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1045, 13), _fitpack_81184, '_bispev')
    # Calling _bispev(args, kwargs) (line 1045)
    _bispev_call_result_81196 = invoke(stypy.reporting.localization.Localization(__file__, 1045, 13), _bispev_81185, *[tx_81186, ty_81187, c_81188, kx_81189, ky_81190, x_81191, y_81192, dx_81193, dy_81194], **kwargs_81195)
    
    # Obtaining the member '__getitem__' of a type (line 1045)
    getitem___81197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1045, 4), _bispev_call_result_81196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1045)
    subscript_call_result_81198 = invoke(stypy.reporting.localization.Localization(__file__, 1045, 4), getitem___81197, int_81183)
    
    # Assigning a type to the variable 'tuple_var_assignment_78338' (line 1045)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'tuple_var_assignment_78338', subscript_call_result_81198)
    
    # Assigning a Subscript to a Name (line 1045):
    
    # Obtaining the type of the subscript
    int_81199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 4), 'int')
    
    # Call to _bispev(...): (line 1045)
    # Processing the call arguments (line 1045)
    # Getting the type of 'tx' (line 1045)
    tx_81202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 30), 'tx', False)
    # Getting the type of 'ty' (line 1045)
    ty_81203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 34), 'ty', False)
    # Getting the type of 'c' (line 1045)
    c_81204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 38), 'c', False)
    # Getting the type of 'kx' (line 1045)
    kx_81205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 41), 'kx', False)
    # Getting the type of 'ky' (line 1045)
    ky_81206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 45), 'ky', False)
    # Getting the type of 'x' (line 1045)
    x_81207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 49), 'x', False)
    # Getting the type of 'y' (line 1045)
    y_81208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 52), 'y', False)
    # Getting the type of 'dx' (line 1045)
    dx_81209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 55), 'dx', False)
    # Getting the type of 'dy' (line 1045)
    dy_81210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 59), 'dy', False)
    # Processing the call keyword arguments (line 1045)
    kwargs_81211 = {}
    # Getting the type of '_fitpack' (line 1045)
    _fitpack_81200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 13), '_fitpack', False)
    # Obtaining the member '_bispev' of a type (line 1045)
    _bispev_81201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1045, 13), _fitpack_81200, '_bispev')
    # Calling _bispev(args, kwargs) (line 1045)
    _bispev_call_result_81212 = invoke(stypy.reporting.localization.Localization(__file__, 1045, 13), _bispev_81201, *[tx_81202, ty_81203, c_81204, kx_81205, ky_81206, x_81207, y_81208, dx_81209, dy_81210], **kwargs_81211)
    
    # Obtaining the member '__getitem__' of a type (line 1045)
    getitem___81213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1045, 4), _bispev_call_result_81212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1045)
    subscript_call_result_81214 = invoke(stypy.reporting.localization.Localization(__file__, 1045, 4), getitem___81213, int_81199)
    
    # Assigning a type to the variable 'tuple_var_assignment_78339' (line 1045)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'tuple_var_assignment_78339', subscript_call_result_81214)
    
    # Assigning a Name to a Name (line 1045):
    # Getting the type of 'tuple_var_assignment_78338' (line 1045)
    tuple_var_assignment_78338_81215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'tuple_var_assignment_78338')
    # Assigning a type to the variable 'z' (line 1045)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'z', tuple_var_assignment_78338_81215)
    
    # Assigning a Name to a Name (line 1045):
    # Getting the type of 'tuple_var_assignment_78339' (line 1045)
    tuple_var_assignment_78339_81216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'tuple_var_assignment_78339')
    # Assigning a type to the variable 'ier' (line 1045)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 7), 'ier', tuple_var_assignment_78339_81216)
    
    
    # Getting the type of 'ier' (line 1046)
    ier_81217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 7), 'ier')
    int_81218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 14), 'int')
    # Applying the binary operator '==' (line 1046)
    result_eq_81219 = python_operator(stypy.reporting.localization.Localization(__file__, 1046, 7), '==', ier_81217, int_81218)
    
    # Testing the type of an if condition (line 1046)
    if_condition_81220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1046, 4), result_eq_81219)
    # Assigning a type to the variable 'if_condition_81220' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 4), 'if_condition_81220', if_condition_81220)
    # SSA begins for if statement (line 1046)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1047)
    # Processing the call arguments (line 1047)
    str_81222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 25), 'str', 'Invalid input data')
    # Processing the call keyword arguments (line 1047)
    kwargs_81223 = {}
    # Getting the type of 'ValueError' (line 1047)
    ValueError_81221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1047)
    ValueError_call_result_81224 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 14), ValueError_81221, *[str_81222], **kwargs_81223)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1047, 8), ValueError_call_result_81224, 'raise parameter', BaseException)
    # SSA join for if statement (line 1046)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ier' (line 1048)
    ier_81225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 7), 'ier')
    # Testing the type of an if condition (line 1048)
    if_condition_81226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1048, 4), ier_81225)
    # Assigning a type to the variable 'if_condition_81226' (line 1048)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1048, 4), 'if_condition_81226', if_condition_81226)
    # SSA begins for if statement (line 1048)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1049)
    # Processing the call arguments (line 1049)
    str_81228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 24), 'str', 'An error occurred')
    # Processing the call keyword arguments (line 1049)
    kwargs_81229 = {}
    # Getting the type of 'TypeError' (line 1049)
    TypeError_81227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1049)
    TypeError_call_result_81230 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 14), TypeError_81227, *[str_81228], **kwargs_81229)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1049, 8), TypeError_call_result_81230, 'raise parameter', BaseException)
    # SSA join for if statement (line 1048)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Attribute (line 1050):
    
    # Assigning a Tuple to a Attribute (line 1050):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1050)
    tuple_81231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1050)
    # Adding element type (line 1050)
    
    # Call to len(...): (line 1050)
    # Processing the call arguments (line 1050)
    # Getting the type of 'x' (line 1050)
    x_81233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 18), 'x', False)
    # Processing the call keyword arguments (line 1050)
    kwargs_81234 = {}
    # Getting the type of 'len' (line 1050)
    len_81232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 14), 'len', False)
    # Calling len(args, kwargs) (line 1050)
    len_call_result_81235 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 14), len_81232, *[x_81233], **kwargs_81234)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1050, 14), tuple_81231, len_call_result_81235)
    # Adding element type (line 1050)
    
    # Call to len(...): (line 1050)
    # Processing the call arguments (line 1050)
    # Getting the type of 'y' (line 1050)
    y_81237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 26), 'y', False)
    # Processing the call keyword arguments (line 1050)
    kwargs_81238 = {}
    # Getting the type of 'len' (line 1050)
    len_81236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 22), 'len', False)
    # Calling len(args, kwargs) (line 1050)
    len_call_result_81239 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 22), len_81236, *[y_81237], **kwargs_81238)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1050, 14), tuple_81231, len_call_result_81239)
    
    # Getting the type of 'z' (line 1050)
    z_81240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 4), 'z')
    # Setting the type of the member 'shape' of a type (line 1050)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 4), z_81240, 'shape', tuple_81231)
    
    
    
    # Call to len(...): (line 1051)
    # Processing the call arguments (line 1051)
    # Getting the type of 'z' (line 1051)
    z_81242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 11), 'z', False)
    # Processing the call keyword arguments (line 1051)
    kwargs_81243 = {}
    # Getting the type of 'len' (line 1051)
    len_81241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 7), 'len', False)
    # Calling len(args, kwargs) (line 1051)
    len_call_result_81244 = invoke(stypy.reporting.localization.Localization(__file__, 1051, 7), len_81241, *[z_81242], **kwargs_81243)
    
    int_81245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, 16), 'int')
    # Applying the binary operator '>' (line 1051)
    result_gt_81246 = python_operator(stypy.reporting.localization.Localization(__file__, 1051, 7), '>', len_call_result_81244, int_81245)
    
    # Testing the type of an if condition (line 1051)
    if_condition_81247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1051, 4), result_gt_81246)
    # Assigning a type to the variable 'if_condition_81247' (line 1051)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1051, 4), 'if_condition_81247', if_condition_81247)
    # SSA begins for if statement (line 1051)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'z' (line 1052)
    z_81248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 15), 'z')
    # Assigning a type to the variable 'stypy_return_type' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'stypy_return_type', z_81248)
    # SSA join for if statement (line 1051)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1053)
    # Processing the call arguments (line 1053)
    
    # Obtaining the type of the subscript
    int_81250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 13), 'int')
    # Getting the type of 'z' (line 1053)
    z_81251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 11), 'z', False)
    # Obtaining the member '__getitem__' of a type (line 1053)
    getitem___81252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 11), z_81251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1053)
    subscript_call_result_81253 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 11), getitem___81252, int_81250)
    
    # Processing the call keyword arguments (line 1053)
    kwargs_81254 = {}
    # Getting the type of 'len' (line 1053)
    len_81249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 7), 'len', False)
    # Calling len(args, kwargs) (line 1053)
    len_call_result_81255 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 7), len_81249, *[subscript_call_result_81253], **kwargs_81254)
    
    int_81256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 19), 'int')
    # Applying the binary operator '>' (line 1053)
    result_gt_81257 = python_operator(stypy.reporting.localization.Localization(__file__, 1053, 7), '>', len_call_result_81255, int_81256)
    
    # Testing the type of an if condition (line 1053)
    if_condition_81258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1053, 4), result_gt_81257)
    # Assigning a type to the variable 'if_condition_81258' (line 1053)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 4), 'if_condition_81258', if_condition_81258)
    # SSA begins for if statement (line 1053)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_81259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1054, 17), 'int')
    # Getting the type of 'z' (line 1054)
    z_81260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 15), 'z')
    # Obtaining the member '__getitem__' of a type (line 1054)
    getitem___81261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 15), z_81260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1054)
    subscript_call_result_81262 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 15), getitem___81261, int_81259)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'stypy_return_type', subscript_call_result_81262)
    # SSA join for if statement (line 1053)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_81263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 16), 'int')
    
    # Obtaining the type of the subscript
    int_81264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 13), 'int')
    # Getting the type of 'z' (line 1055)
    z_81265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 11), 'z')
    # Obtaining the member '__getitem__' of a type (line 1055)
    getitem___81266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 11), z_81265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1055)
    subscript_call_result_81267 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 11), getitem___81266, int_81264)
    
    # Obtaining the member '__getitem__' of a type (line 1055)
    getitem___81268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 11), subscript_call_result_81267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1055)
    subscript_call_result_81269 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 11), getitem___81268, int_81263)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1055)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 4), 'stypy_return_type', subscript_call_result_81269)
    
    # ################# End of 'bisplev(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bisplev' in the type store
    # Getting the type of 'stypy_return_type' (line 989)
    stypy_return_type_81270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_81270)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bisplev'
    return stypy_return_type_81270

# Assigning a type to the variable 'bisplev' (line 989)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 989, 0), 'bisplev', bisplev)

@norecursion
def dblint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dblint'
    module_type_store = module_type_store.open_function_context('dblint', 1058, 0, False)
    
    # Passed parameters checking function
    dblint.stypy_localization = localization
    dblint.stypy_type_of_self = None
    dblint.stypy_type_store = module_type_store
    dblint.stypy_function_name = 'dblint'
    dblint.stypy_param_names_list = ['xa', 'xb', 'ya', 'yb', 'tck']
    dblint.stypy_varargs_param_name = None
    dblint.stypy_kwargs_param_name = None
    dblint.stypy_call_defaults = defaults
    dblint.stypy_call_varargs = varargs
    dblint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dblint', ['xa', 'xb', 'ya', 'yb', 'tck'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dblint', localization, ['xa', 'xb', 'ya', 'yb', 'tck'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dblint(...)' code ##################

    str_81271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, (-1)), 'str', 'Evaluate the integral of a spline over area [xa,xb] x [ya,yb].\n\n    Parameters\n    ----------\n    xa, xb : float\n        The end-points of the x integration interval.\n    ya, yb : float\n        The end-points of the y integration interval.\n    tck : list [tx, ty, c, kx, ky]\n        A sequence of length 5 returned by bisplrep containing the knot\n        locations tx, ty, the coefficients c, and the degrees kx, ky\n        of the spline.\n\n    Returns\n    -------\n    integ : float\n        The value of the resulting integral.\n    ')
    
    # Assigning a Name to a Tuple (line 1077):
    
    # Assigning a Subscript to a Name (line 1077):
    
    # Obtaining the type of the subscript
    int_81272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 4), 'int')
    # Getting the type of 'tck' (line 1077)
    tck_81273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1077)
    getitem___81274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 4), tck_81273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1077)
    subscript_call_result_81275 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 4), getitem___81274, int_81272)
    
    # Assigning a type to the variable 'tuple_var_assignment_78340' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78340', subscript_call_result_81275)
    
    # Assigning a Subscript to a Name (line 1077):
    
    # Obtaining the type of the subscript
    int_81276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 4), 'int')
    # Getting the type of 'tck' (line 1077)
    tck_81277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1077)
    getitem___81278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 4), tck_81277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1077)
    subscript_call_result_81279 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 4), getitem___81278, int_81276)
    
    # Assigning a type to the variable 'tuple_var_assignment_78341' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78341', subscript_call_result_81279)
    
    # Assigning a Subscript to a Name (line 1077):
    
    # Obtaining the type of the subscript
    int_81280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 4), 'int')
    # Getting the type of 'tck' (line 1077)
    tck_81281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1077)
    getitem___81282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 4), tck_81281, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1077)
    subscript_call_result_81283 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 4), getitem___81282, int_81280)
    
    # Assigning a type to the variable 'tuple_var_assignment_78342' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78342', subscript_call_result_81283)
    
    # Assigning a Subscript to a Name (line 1077):
    
    # Obtaining the type of the subscript
    int_81284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 4), 'int')
    # Getting the type of 'tck' (line 1077)
    tck_81285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1077)
    getitem___81286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 4), tck_81285, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1077)
    subscript_call_result_81287 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 4), getitem___81286, int_81284)
    
    # Assigning a type to the variable 'tuple_var_assignment_78343' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78343', subscript_call_result_81287)
    
    # Assigning a Subscript to a Name (line 1077):
    
    # Obtaining the type of the subscript
    int_81288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 4), 'int')
    # Getting the type of 'tck' (line 1077)
    tck_81289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 24), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1077)
    getitem___81290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 4), tck_81289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1077)
    subscript_call_result_81291 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 4), getitem___81290, int_81288)
    
    # Assigning a type to the variable 'tuple_var_assignment_78344' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78344', subscript_call_result_81291)
    
    # Assigning a Name to a Name (line 1077):
    # Getting the type of 'tuple_var_assignment_78340' (line 1077)
    tuple_var_assignment_78340_81292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78340')
    # Assigning a type to the variable 'tx' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tx', tuple_var_assignment_78340_81292)
    
    # Assigning a Name to a Name (line 1077):
    # Getting the type of 'tuple_var_assignment_78341' (line 1077)
    tuple_var_assignment_78341_81293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78341')
    # Assigning a type to the variable 'ty' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 8), 'ty', tuple_var_assignment_78341_81293)
    
    # Assigning a Name to a Name (line 1077):
    # Getting the type of 'tuple_var_assignment_78342' (line 1077)
    tuple_var_assignment_78342_81294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78342')
    # Assigning a type to the variable 'c' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 12), 'c', tuple_var_assignment_78342_81294)
    
    # Assigning a Name to a Name (line 1077):
    # Getting the type of 'tuple_var_assignment_78343' (line 1077)
    tuple_var_assignment_78343_81295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78343')
    # Assigning a type to the variable 'kx' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 15), 'kx', tuple_var_assignment_78343_81295)
    
    # Assigning a Name to a Name (line 1077):
    # Getting the type of 'tuple_var_assignment_78344' (line 1077)
    tuple_var_assignment_78344_81296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'tuple_var_assignment_78344')
    # Assigning a type to the variable 'ky' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 19), 'ky', tuple_var_assignment_78344_81296)
    
    # Call to dblint(...): (line 1078)
    # Processing the call arguments (line 1078)
    # Getting the type of 'tx' (line 1078)
    tx_81299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 27), 'tx', False)
    # Getting the type of 'ty' (line 1078)
    ty_81300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 31), 'ty', False)
    # Getting the type of 'c' (line 1078)
    c_81301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 35), 'c', False)
    # Getting the type of 'kx' (line 1078)
    kx_81302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 38), 'kx', False)
    # Getting the type of 'ky' (line 1078)
    ky_81303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 42), 'ky', False)
    # Getting the type of 'xa' (line 1078)
    xa_81304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 46), 'xa', False)
    # Getting the type of 'xb' (line 1078)
    xb_81305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 50), 'xb', False)
    # Getting the type of 'ya' (line 1078)
    ya_81306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 54), 'ya', False)
    # Getting the type of 'yb' (line 1078)
    yb_81307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 58), 'yb', False)
    # Processing the call keyword arguments (line 1078)
    kwargs_81308 = {}
    # Getting the type of 'dfitpack' (line 1078)
    dfitpack_81297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 11), 'dfitpack', False)
    # Obtaining the member 'dblint' of a type (line 1078)
    dblint_81298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1078, 11), dfitpack_81297, 'dblint')
    # Calling dblint(args, kwargs) (line 1078)
    dblint_call_result_81309 = invoke(stypy.reporting.localization.Localization(__file__, 1078, 11), dblint_81298, *[tx_81299, ty_81300, c_81301, kx_81302, ky_81303, xa_81304, xb_81305, ya_81306, yb_81307], **kwargs_81308)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1078)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'stypy_return_type', dblint_call_result_81309)
    
    # ################# End of 'dblint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dblint' in the type store
    # Getting the type of 'stypy_return_type' (line 1058)
    stypy_return_type_81310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_81310)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dblint'
    return stypy_return_type_81310

# Assigning a type to the variable 'dblint' (line 1058)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 0), 'dblint', dblint)

@norecursion
def insert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_81311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 21), 'int')
    int_81312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 28), 'int')
    defaults = [int_81311, int_81312]
    # Create a new context for function 'insert'
    module_type_store = module_type_store.open_function_context('insert', 1081, 0, False)
    
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

    str_81313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, (-1)), 'str', '\n    Insert knots into a B-spline.\n\n    Given the knots and coefficients of a B-spline representation, create a\n    new B-spline with a knot inserted `m` times at point `x`.\n    This is a wrapper around the FORTRAN routine insert of FITPACK.\n\n    Parameters\n    ----------\n    x (u) : array_like\n        A 1-D point at which to insert a new knot(s).  If `tck` was returned\n        from ``splprep``, then the parameter values, u should be given.\n    tck : tuple\n        A tuple (t,c,k) returned by ``splrep`` or ``splprep`` containing\n        the vector of knots, the B-spline coefficients,\n        and the degree of the spline.\n    m : int, optional\n        The number of times to insert the given knot (its multiplicity).\n        Default is 1.\n    per : int, optional\n        If non-zero, the input spline is considered periodic.\n\n    Returns\n    -------\n    tck : tuple\n        A tuple (t,c,k) containing the vector of knots, the B-spline\n        coefficients, and the degree of the new spline.\n        ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.\n        In case of a periodic spline (``per != 0``) there must be\n        either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``\n        or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.\n\n    Notes\n    -----\n    Based on algorithms from [1]_ and [2]_.\n\n    References\n    ----------\n    .. [1] W. Boehm, "Inserting new knots into b-spline curves.",\n        Computer Aided Design, 12, p.199-201, 1980.\n    .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on\n        Numerical Analysis", Oxford University Press, 1993.\n\n    ')
    
    # Assigning a Name to a Tuple (line 1126):
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_81314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    # Getting the type of 'tck' (line 1126)
    tck_81315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___81316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), tck_81315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_81317 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___81316, int_81314)
    
    # Assigning a type to the variable 'tuple_var_assignment_78345' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_78345', subscript_call_result_81317)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_81318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    # Getting the type of 'tck' (line 1126)
    tck_81319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___81320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), tck_81319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_81321 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___81320, int_81318)
    
    # Assigning a type to the variable 'tuple_var_assignment_78346' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_78346', subscript_call_result_81321)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_81322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    # Getting the type of 'tck' (line 1126)
    tck_81323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___81324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), tck_81323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_81325 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___81324, int_81322)
    
    # Assigning a type to the variable 'tuple_var_assignment_78347' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_78347', subscript_call_result_81325)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_78345' (line 1126)
    tuple_var_assignment_78345_81326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_78345')
    # Assigning a type to the variable 't' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 't', tuple_var_assignment_78345_81326)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_78346' (line 1126)
    tuple_var_assignment_78346_81327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_78346')
    # Assigning a type to the variable 'c' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 7), 'c', tuple_var_assignment_78346_81327)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_78347' (line 1126)
    tuple_var_assignment_78347_81328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_78347')
    # Assigning a type to the variable 'k' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 10), 'k', tuple_var_assignment_78347_81328)
    
    
    # SSA begins for try-except statement (line 1127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    int_81329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1128, 13), 'int')
    
    # Obtaining the type of the subscript
    int_81330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1128, 10), 'int')
    # Getting the type of 'c' (line 1128)
    c_81331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 8), 'c')
    # Obtaining the member '__getitem__' of a type (line 1128)
    getitem___81332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1128, 8), c_81331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1128)
    subscript_call_result_81333 = invoke(stypy.reporting.localization.Localization(__file__, 1128, 8), getitem___81332, int_81330)
    
    # Obtaining the member '__getitem__' of a type (line 1128)
    getitem___81334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1128, 8), subscript_call_result_81333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1128)
    subscript_call_result_81335 = invoke(stypy.reporting.localization.Localization(__file__, 1128, 8), getitem___81334, int_81329)
    
    
    # Assigning a Name to a Name (line 1129):
    
    # Assigning a Name to a Name (line 1129):
    # Getting the type of 'True' (line 1129)
    True_81336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 21), 'True')
    # Assigning a type to the variable 'parametric' (line 1129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 8), 'parametric', True_81336)
    # SSA branch for the except part of a try statement (line 1127)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1127)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 1131):
    
    # Assigning a Name to a Name (line 1131):
    # Getting the type of 'False' (line 1131)
    False_81337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 21), 'False')
    # Assigning a type to the variable 'parametric' (line 1131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1131, 8), 'parametric', False_81337)
    # SSA join for try-except statement (line 1127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'parametric' (line 1132)
    parametric_81338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 7), 'parametric')
    # Testing the type of an if condition (line 1132)
    if_condition_81339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1132, 4), parametric_81338)
    # Assigning a type to the variable 'if_condition_81339' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'if_condition_81339', if_condition_81339)
    # SSA begins for if statement (line 1132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 1133):
    
    # Assigning a List to a Name (line 1133):
    
    # Obtaining an instance of the builtin type 'list' (line 1133)
    list_81340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1133, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1133)
    
    # Assigning a type to the variable 'cc' (line 1133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1133, 8), 'cc', list_81340)
    
    # Getting the type of 'c' (line 1134)
    c_81341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 22), 'c')
    # Testing the type of a for loop iterable (line 1134)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1134, 8), c_81341)
    # Getting the type of the for loop variable (line 1134)
    for_loop_var_81342 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1134, 8), c_81341)
    # Assigning a type to the variable 'c_vals' (line 1134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1134, 8), 'c_vals', for_loop_var_81342)
    # SSA begins for a for statement (line 1134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 1135):
    
    # Assigning a Subscript to a Name (line 1135):
    
    # Obtaining the type of the subscript
    int_81343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 12), 'int')
    
    # Call to insert(...): (line 1135)
    # Processing the call arguments (line 1135)
    # Getting the type of 'x' (line 1135)
    x_81345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 36), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 1135)
    list_81346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1135)
    # Adding element type (line 1135)
    # Getting the type of 't' (line 1135)
    t_81347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 40), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81346, t_81347)
    # Adding element type (line 1135)
    # Getting the type of 'c_vals' (line 1135)
    c_vals_81348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 43), 'c_vals', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81346, c_vals_81348)
    # Adding element type (line 1135)
    # Getting the type of 'k' (line 1135)
    k_81349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 51), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81346, k_81349)
    
    # Getting the type of 'm' (line 1135)
    m_81350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 55), 'm', False)
    # Processing the call keyword arguments (line 1135)
    kwargs_81351 = {}
    # Getting the type of 'insert' (line 1135)
    insert_81344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 29), 'insert', False)
    # Calling insert(args, kwargs) (line 1135)
    insert_call_result_81352 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 29), insert_81344, *[x_81345, list_81346, m_81350], **kwargs_81351)
    
    # Obtaining the member '__getitem__' of a type (line 1135)
    getitem___81353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1135, 12), insert_call_result_81352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1135)
    subscript_call_result_81354 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 12), getitem___81353, int_81343)
    
    # Assigning a type to the variable 'tuple_var_assignment_78348' (line 1135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 12), 'tuple_var_assignment_78348', subscript_call_result_81354)
    
    # Assigning a Subscript to a Name (line 1135):
    
    # Obtaining the type of the subscript
    int_81355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 12), 'int')
    
    # Call to insert(...): (line 1135)
    # Processing the call arguments (line 1135)
    # Getting the type of 'x' (line 1135)
    x_81357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 36), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 1135)
    list_81358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1135)
    # Adding element type (line 1135)
    # Getting the type of 't' (line 1135)
    t_81359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 40), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81358, t_81359)
    # Adding element type (line 1135)
    # Getting the type of 'c_vals' (line 1135)
    c_vals_81360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 43), 'c_vals', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81358, c_vals_81360)
    # Adding element type (line 1135)
    # Getting the type of 'k' (line 1135)
    k_81361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 51), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81358, k_81361)
    
    # Getting the type of 'm' (line 1135)
    m_81362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 55), 'm', False)
    # Processing the call keyword arguments (line 1135)
    kwargs_81363 = {}
    # Getting the type of 'insert' (line 1135)
    insert_81356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 29), 'insert', False)
    # Calling insert(args, kwargs) (line 1135)
    insert_call_result_81364 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 29), insert_81356, *[x_81357, list_81358, m_81362], **kwargs_81363)
    
    # Obtaining the member '__getitem__' of a type (line 1135)
    getitem___81365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1135, 12), insert_call_result_81364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1135)
    subscript_call_result_81366 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 12), getitem___81365, int_81355)
    
    # Assigning a type to the variable 'tuple_var_assignment_78349' (line 1135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 12), 'tuple_var_assignment_78349', subscript_call_result_81366)
    
    # Assigning a Subscript to a Name (line 1135):
    
    # Obtaining the type of the subscript
    int_81367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 12), 'int')
    
    # Call to insert(...): (line 1135)
    # Processing the call arguments (line 1135)
    # Getting the type of 'x' (line 1135)
    x_81369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 36), 'x', False)
    
    # Obtaining an instance of the builtin type 'list' (line 1135)
    list_81370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1135)
    # Adding element type (line 1135)
    # Getting the type of 't' (line 1135)
    t_81371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 40), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81370, t_81371)
    # Adding element type (line 1135)
    # Getting the type of 'c_vals' (line 1135)
    c_vals_81372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 43), 'c_vals', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81370, c_vals_81372)
    # Adding element type (line 1135)
    # Getting the type of 'k' (line 1135)
    k_81373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 51), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 39), list_81370, k_81373)
    
    # Getting the type of 'm' (line 1135)
    m_81374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 55), 'm', False)
    # Processing the call keyword arguments (line 1135)
    kwargs_81375 = {}
    # Getting the type of 'insert' (line 1135)
    insert_81368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 29), 'insert', False)
    # Calling insert(args, kwargs) (line 1135)
    insert_call_result_81376 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 29), insert_81368, *[x_81369, list_81370, m_81374], **kwargs_81375)
    
    # Obtaining the member '__getitem__' of a type (line 1135)
    getitem___81377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1135, 12), insert_call_result_81376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1135)
    subscript_call_result_81378 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 12), getitem___81377, int_81367)
    
    # Assigning a type to the variable 'tuple_var_assignment_78350' (line 1135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 12), 'tuple_var_assignment_78350', subscript_call_result_81378)
    
    # Assigning a Name to a Name (line 1135):
    # Getting the type of 'tuple_var_assignment_78348' (line 1135)
    tuple_var_assignment_78348_81379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 12), 'tuple_var_assignment_78348')
    # Assigning a type to the variable 'tt' (line 1135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 12), 'tt', tuple_var_assignment_78348_81379)
    
    # Assigning a Name to a Name (line 1135):
    # Getting the type of 'tuple_var_assignment_78349' (line 1135)
    tuple_var_assignment_78349_81380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 12), 'tuple_var_assignment_78349')
    # Assigning a type to the variable 'cc_val' (line 1135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 16), 'cc_val', tuple_var_assignment_78349_81380)
    
    # Assigning a Name to a Name (line 1135):
    # Getting the type of 'tuple_var_assignment_78350' (line 1135)
    tuple_var_assignment_78350_81381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 12), 'tuple_var_assignment_78350')
    # Assigning a type to the variable 'kk' (line 1135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 24), 'kk', tuple_var_assignment_78350_81381)
    
    # Call to append(...): (line 1136)
    # Processing the call arguments (line 1136)
    # Getting the type of 'cc_val' (line 1136)
    cc_val_81384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 22), 'cc_val', False)
    # Processing the call keyword arguments (line 1136)
    kwargs_81385 = {}
    # Getting the type of 'cc' (line 1136)
    cc_81382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 12), 'cc', False)
    # Obtaining the member 'append' of a type (line 1136)
    append_81383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1136, 12), cc_81382, 'append')
    # Calling append(args, kwargs) (line 1136)
    append_call_result_81386 = invoke(stypy.reporting.localization.Localization(__file__, 1136, 12), append_81383, *[cc_val_81384], **kwargs_81385)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1137)
    tuple_81387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1137)
    # Adding element type (line 1137)
    # Getting the type of 'tt' (line 1137)
    tt_81388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 16), 'tt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 16), tuple_81387, tt_81388)
    # Adding element type (line 1137)
    # Getting the type of 'cc' (line 1137)
    cc_81389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 20), 'cc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 16), tuple_81387, cc_81389)
    # Adding element type (line 1137)
    # Getting the type of 'kk' (line 1137)
    kk_81390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 24), 'kk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 16), tuple_81387, kk_81390)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1137, 8), 'stypy_return_type', tuple_81387)
    # SSA branch for the else part of an if statement (line 1132)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 1139):
    
    # Assigning a Subscript to a Name (line 1139):
    
    # Obtaining the type of the subscript
    int_81391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 8), 'int')
    
    # Call to _insert(...): (line 1139)
    # Processing the call arguments (line 1139)
    # Getting the type of 'per' (line 1139)
    per_81394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 39), 'per', False)
    # Getting the type of 't' (line 1139)
    t_81395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 44), 't', False)
    # Getting the type of 'c' (line 1139)
    c_81396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 47), 'c', False)
    # Getting the type of 'k' (line 1139)
    k_81397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 50), 'k', False)
    # Getting the type of 'x' (line 1139)
    x_81398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 53), 'x', False)
    # Getting the type of 'm' (line 1139)
    m_81399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 56), 'm', False)
    # Processing the call keyword arguments (line 1139)
    kwargs_81400 = {}
    # Getting the type of '_fitpack' (line 1139)
    _fitpack_81392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 22), '_fitpack', False)
    # Obtaining the member '_insert' of a type (line 1139)
    _insert_81393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 22), _fitpack_81392, '_insert')
    # Calling _insert(args, kwargs) (line 1139)
    _insert_call_result_81401 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 22), _insert_81393, *[per_81394, t_81395, c_81396, k_81397, x_81398, m_81399], **kwargs_81400)
    
    # Obtaining the member '__getitem__' of a type (line 1139)
    getitem___81402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 8), _insert_call_result_81401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1139)
    subscript_call_result_81403 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 8), getitem___81402, int_81391)
    
    # Assigning a type to the variable 'tuple_var_assignment_78351' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'tuple_var_assignment_78351', subscript_call_result_81403)
    
    # Assigning a Subscript to a Name (line 1139):
    
    # Obtaining the type of the subscript
    int_81404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 8), 'int')
    
    # Call to _insert(...): (line 1139)
    # Processing the call arguments (line 1139)
    # Getting the type of 'per' (line 1139)
    per_81407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 39), 'per', False)
    # Getting the type of 't' (line 1139)
    t_81408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 44), 't', False)
    # Getting the type of 'c' (line 1139)
    c_81409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 47), 'c', False)
    # Getting the type of 'k' (line 1139)
    k_81410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 50), 'k', False)
    # Getting the type of 'x' (line 1139)
    x_81411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 53), 'x', False)
    # Getting the type of 'm' (line 1139)
    m_81412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 56), 'm', False)
    # Processing the call keyword arguments (line 1139)
    kwargs_81413 = {}
    # Getting the type of '_fitpack' (line 1139)
    _fitpack_81405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 22), '_fitpack', False)
    # Obtaining the member '_insert' of a type (line 1139)
    _insert_81406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 22), _fitpack_81405, '_insert')
    # Calling _insert(args, kwargs) (line 1139)
    _insert_call_result_81414 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 22), _insert_81406, *[per_81407, t_81408, c_81409, k_81410, x_81411, m_81412], **kwargs_81413)
    
    # Obtaining the member '__getitem__' of a type (line 1139)
    getitem___81415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 8), _insert_call_result_81414, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1139)
    subscript_call_result_81416 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 8), getitem___81415, int_81404)
    
    # Assigning a type to the variable 'tuple_var_assignment_78352' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'tuple_var_assignment_78352', subscript_call_result_81416)
    
    # Assigning a Subscript to a Name (line 1139):
    
    # Obtaining the type of the subscript
    int_81417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 8), 'int')
    
    # Call to _insert(...): (line 1139)
    # Processing the call arguments (line 1139)
    # Getting the type of 'per' (line 1139)
    per_81420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 39), 'per', False)
    # Getting the type of 't' (line 1139)
    t_81421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 44), 't', False)
    # Getting the type of 'c' (line 1139)
    c_81422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 47), 'c', False)
    # Getting the type of 'k' (line 1139)
    k_81423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 50), 'k', False)
    # Getting the type of 'x' (line 1139)
    x_81424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 53), 'x', False)
    # Getting the type of 'm' (line 1139)
    m_81425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 56), 'm', False)
    # Processing the call keyword arguments (line 1139)
    kwargs_81426 = {}
    # Getting the type of '_fitpack' (line 1139)
    _fitpack_81418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 22), '_fitpack', False)
    # Obtaining the member '_insert' of a type (line 1139)
    _insert_81419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 22), _fitpack_81418, '_insert')
    # Calling _insert(args, kwargs) (line 1139)
    _insert_call_result_81427 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 22), _insert_81419, *[per_81420, t_81421, c_81422, k_81423, x_81424, m_81425], **kwargs_81426)
    
    # Obtaining the member '__getitem__' of a type (line 1139)
    getitem___81428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 8), _insert_call_result_81427, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1139)
    subscript_call_result_81429 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 8), getitem___81428, int_81417)
    
    # Assigning a type to the variable 'tuple_var_assignment_78353' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'tuple_var_assignment_78353', subscript_call_result_81429)
    
    # Assigning a Name to a Name (line 1139):
    # Getting the type of 'tuple_var_assignment_78351' (line 1139)
    tuple_var_assignment_78351_81430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'tuple_var_assignment_78351')
    # Assigning a type to the variable 'tt' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'tt', tuple_var_assignment_78351_81430)
    
    # Assigning a Name to a Name (line 1139):
    # Getting the type of 'tuple_var_assignment_78352' (line 1139)
    tuple_var_assignment_78352_81431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'tuple_var_assignment_78352')
    # Assigning a type to the variable 'cc' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 12), 'cc', tuple_var_assignment_78352_81431)
    
    # Assigning a Name to a Name (line 1139):
    # Getting the type of 'tuple_var_assignment_78353' (line 1139)
    tuple_var_assignment_78353_81432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'tuple_var_assignment_78353')
    # Assigning a type to the variable 'ier' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 16), 'ier', tuple_var_assignment_78353_81432)
    
    
    # Getting the type of 'ier' (line 1140)
    ier_81433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 11), 'ier')
    int_81434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, 18), 'int')
    # Applying the binary operator '==' (line 1140)
    result_eq_81435 = python_operator(stypy.reporting.localization.Localization(__file__, 1140, 11), '==', ier_81433, int_81434)
    
    # Testing the type of an if condition (line 1140)
    if_condition_81436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1140, 8), result_eq_81435)
    # Assigning a type to the variable 'if_condition_81436' (line 1140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1140, 8), 'if_condition_81436', if_condition_81436)
    # SSA begins for if statement (line 1140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1141)
    # Processing the call arguments (line 1141)
    str_81438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1141, 29), 'str', 'Invalid input data')
    # Processing the call keyword arguments (line 1141)
    kwargs_81439 = {}
    # Getting the type of 'ValueError' (line 1141)
    ValueError_81437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1141)
    ValueError_call_result_81440 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 18), ValueError_81437, *[str_81438], **kwargs_81439)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1141, 12), ValueError_call_result_81440, 'raise parameter', BaseException)
    # SSA join for if statement (line 1140)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ier' (line 1142)
    ier_81441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 11), 'ier')
    # Testing the type of an if condition (line 1142)
    if_condition_81442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1142, 8), ier_81441)
    # Assigning a type to the variable 'if_condition_81442' (line 1142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 8), 'if_condition_81442', if_condition_81442)
    # SSA begins for if statement (line 1142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1143)
    # Processing the call arguments (line 1143)
    str_81444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1143, 28), 'str', 'An error occurred')
    # Processing the call keyword arguments (line 1143)
    kwargs_81445 = {}
    # Getting the type of 'TypeError' (line 1143)
    TypeError_81443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1143)
    TypeError_call_result_81446 = invoke(stypy.reporting.localization.Localization(__file__, 1143, 18), TypeError_81443, *[str_81444], **kwargs_81445)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1143, 12), TypeError_call_result_81446, 'raise parameter', BaseException)
    # SSA join for if statement (line 1142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1144)
    tuple_81447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1144)
    # Adding element type (line 1144)
    # Getting the type of 'tt' (line 1144)
    tt_81448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 16), 'tt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1144, 16), tuple_81447, tt_81448)
    # Adding element type (line 1144)
    # Getting the type of 'cc' (line 1144)
    cc_81449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 20), 'cc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1144, 16), tuple_81447, cc_81449)
    # Adding element type (line 1144)
    # Getting the type of 'k' (line 1144)
    k_81450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 24), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1144, 16), tuple_81447, k_81450)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 8), 'stypy_return_type', tuple_81447)
    # SSA join for if statement (line 1132)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'insert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'insert' in the type store
    # Getting the type of 'stypy_return_type' (line 1081)
    stypy_return_type_81451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_81451)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'insert'
    return stypy_return_type_81451

# Assigning a type to the variable 'insert' (line 1081)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1081, 0), 'insert', insert)

@norecursion
def splder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_81452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1147, 18), 'int')
    defaults = [int_81452]
    # Create a new context for function 'splder'
    module_type_store = module_type_store.open_function_context('splder', 1147, 0, False)
    
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

    str_81453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1193, (-1)), 'str', "\n    Compute the spline representation of the derivative of a given spline\n\n    Parameters\n    ----------\n    tck : tuple of (t, c, k)\n        Spline whose derivative to compute\n    n : int, optional\n        Order of derivative to evaluate. Default: 1\n\n    Returns\n    -------\n    tck_der : tuple of (t2, c2, k2)\n        Spline of order k2=k-n representing the derivative\n        of the input spline.\n\n    Notes\n    -----\n\n    .. versionadded:: 0.13.0\n\n    See Also\n    --------\n    splantider, splev, spalde\n\n    Examples\n    --------\n    This can be used for finding maxima of a curve:\n\n    >>> from scipy.interpolate import splrep, splder, sproot\n    >>> x = np.linspace(0, 10, 70)\n    >>> y = np.sin(x)\n    >>> spl = splrep(x, y, k=4)\n\n    Now, differentiate the spline and find the zeros of the\n    derivative. (NB: `sproot` only works for order 3 splines, so we\n    fit an order 4 spline):\n\n    >>> dspl = splder(spl)\n    >>> sproot(dspl) / np.pi\n    array([ 0.50000001,  1.5       ,  2.49999998])\n\n    This agrees well with roots :math:`\\pi/2 + n\\pi` of\n    :math:`\\cos(x) = \\sin'(x)`.\n\n    ")
    
    
    # Getting the type of 'n' (line 1194)
    n_81454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 7), 'n')
    int_81455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1194, 11), 'int')
    # Applying the binary operator '<' (line 1194)
    result_lt_81456 = python_operator(stypy.reporting.localization.Localization(__file__, 1194, 7), '<', n_81454, int_81455)
    
    # Testing the type of an if condition (line 1194)
    if_condition_81457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1194, 4), result_lt_81456)
    # Assigning a type to the variable 'if_condition_81457' (line 1194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 4), 'if_condition_81457', if_condition_81457)
    # SSA begins for if statement (line 1194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to splantider(...): (line 1195)
    # Processing the call arguments (line 1195)
    # Getting the type of 'tck' (line 1195)
    tck_81459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 26), 'tck', False)
    
    # Getting the type of 'n' (line 1195)
    n_81460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 32), 'n', False)
    # Applying the 'usub' unary operator (line 1195)
    result___neg___81461 = python_operator(stypy.reporting.localization.Localization(__file__, 1195, 31), 'usub', n_81460)
    
    # Processing the call keyword arguments (line 1195)
    kwargs_81462 = {}
    # Getting the type of 'splantider' (line 1195)
    splantider_81458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 15), 'splantider', False)
    # Calling splantider(args, kwargs) (line 1195)
    splantider_call_result_81463 = invoke(stypy.reporting.localization.Localization(__file__, 1195, 15), splantider_81458, *[tck_81459, result___neg___81461], **kwargs_81462)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 8), 'stypy_return_type', splantider_call_result_81463)
    # SSA join for if statement (line 1194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1197):
    
    # Assigning a Subscript to a Name (line 1197):
    
    # Obtaining the type of the subscript
    int_81464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 4), 'int')
    # Getting the type of 'tck' (line 1197)
    tck_81465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1197)
    getitem___81466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1197, 4), tck_81465, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1197)
    subscript_call_result_81467 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 4), getitem___81466, int_81464)
    
    # Assigning a type to the variable 'tuple_var_assignment_78354' (line 1197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 4), 'tuple_var_assignment_78354', subscript_call_result_81467)
    
    # Assigning a Subscript to a Name (line 1197):
    
    # Obtaining the type of the subscript
    int_81468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 4), 'int')
    # Getting the type of 'tck' (line 1197)
    tck_81469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1197)
    getitem___81470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1197, 4), tck_81469, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1197)
    subscript_call_result_81471 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 4), getitem___81470, int_81468)
    
    # Assigning a type to the variable 'tuple_var_assignment_78355' (line 1197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 4), 'tuple_var_assignment_78355', subscript_call_result_81471)
    
    # Assigning a Subscript to a Name (line 1197):
    
    # Obtaining the type of the subscript
    int_81472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 4), 'int')
    # Getting the type of 'tck' (line 1197)
    tck_81473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1197)
    getitem___81474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1197, 4), tck_81473, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1197)
    subscript_call_result_81475 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 4), getitem___81474, int_81472)
    
    # Assigning a type to the variable 'tuple_var_assignment_78356' (line 1197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 4), 'tuple_var_assignment_78356', subscript_call_result_81475)
    
    # Assigning a Name to a Name (line 1197):
    # Getting the type of 'tuple_var_assignment_78354' (line 1197)
    tuple_var_assignment_78354_81476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 4), 'tuple_var_assignment_78354')
    # Assigning a type to the variable 't' (line 1197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 4), 't', tuple_var_assignment_78354_81476)
    
    # Assigning a Name to a Name (line 1197):
    # Getting the type of 'tuple_var_assignment_78355' (line 1197)
    tuple_var_assignment_78355_81477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 4), 'tuple_var_assignment_78355')
    # Assigning a type to the variable 'c' (line 1197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 7), 'c', tuple_var_assignment_78355_81477)
    
    # Assigning a Name to a Name (line 1197):
    # Getting the type of 'tuple_var_assignment_78356' (line 1197)
    tuple_var_assignment_78356_81478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 4), 'tuple_var_assignment_78356')
    # Assigning a type to the variable 'k' (line 1197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 10), 'k', tuple_var_assignment_78356_81478)
    
    
    # Getting the type of 'n' (line 1199)
    n_81479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 7), 'n')
    # Getting the type of 'k' (line 1199)
    k_81480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 11), 'k')
    # Applying the binary operator '>' (line 1199)
    result_gt_81481 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 7), '>', n_81479, k_81480)
    
    # Testing the type of an if condition (line 1199)
    if_condition_81482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1199, 4), result_gt_81481)
    # Assigning a type to the variable 'if_condition_81482' (line 1199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1199, 4), 'if_condition_81482', if_condition_81482)
    # SSA begins for if statement (line 1199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1200)
    # Processing the call arguments (line 1200)
    str_81484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1200, 26), 'str', 'Order of derivative (n = %r) must be <= order of spline (k = %r)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1201)
    tuple_81485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1201, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1201)
    # Adding element type (line 1201)
    # Getting the type of 'n' (line 1201)
    n_81486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 57), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1201, 57), tuple_81485, n_81486)
    # Adding element type (line 1201)
    
    # Obtaining the type of the subscript
    int_81487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1201, 64), 'int')
    # Getting the type of 'tck' (line 1201)
    tck_81488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 60), 'tck', False)
    # Obtaining the member '__getitem__' of a type (line 1201)
    getitem___81489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1201, 60), tck_81488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1201)
    subscript_call_result_81490 = invoke(stypy.reporting.localization.Localization(__file__, 1201, 60), getitem___81489, int_81487)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1201, 57), tuple_81485, subscript_call_result_81490)
    
    # Applying the binary operator '%' (line 1200)
    result_mod_81491 = python_operator(stypy.reporting.localization.Localization(__file__, 1200, 25), '%', str_81484, tuple_81485)
    
    # Processing the call keyword arguments (line 1200)
    kwargs_81492 = {}
    # Getting the type of 'ValueError' (line 1200)
    ValueError_81483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1200)
    ValueError_call_result_81493 = invoke(stypy.reporting.localization.Localization(__file__, 1200, 14), ValueError_81483, *[result_mod_81491], **kwargs_81492)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1200, 8), ValueError_call_result_81493, 'raise parameter', BaseException)
    # SSA join for if statement (line 1199)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1204):
    
    # Assigning a BinOp to a Name (line 1204):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1204)
    tuple_81494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1204, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1204)
    # Adding element type (line 1204)
    
    # Call to slice(...): (line 1204)
    # Processing the call arguments (line 1204)
    # Getting the type of 'None' (line 1204)
    None_81496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 16), 'None', False)
    # Processing the call keyword arguments (line 1204)
    kwargs_81497 = {}
    # Getting the type of 'slice' (line 1204)
    slice_81495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 10), 'slice', False)
    # Calling slice(args, kwargs) (line 1204)
    slice_call_result_81498 = invoke(stypy.reporting.localization.Localization(__file__, 1204, 10), slice_81495, *[None_81496], **kwargs_81497)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1204, 10), tuple_81494, slice_call_result_81498)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1204)
    tuple_81499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1204, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1204)
    # Adding element type (line 1204)
    # Getting the type of 'None' (line 1204)
    None_81500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 28), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1204, 28), tuple_81499, None_81500)
    
    
    # Call to len(...): (line 1204)
    # Processing the call arguments (line 1204)
    
    # Obtaining the type of the subscript
    int_81502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1204, 47), 'int')
    slice_81503 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1204, 39), int_81502, None, None)
    # Getting the type of 'c' (line 1204)
    c_81504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 39), 'c', False)
    # Obtaining the member 'shape' of a type (line 1204)
    shape_81505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1204, 39), c_81504, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1204)
    getitem___81506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1204, 39), shape_81505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1204)
    subscript_call_result_81507 = invoke(stypy.reporting.localization.Localization(__file__, 1204, 39), getitem___81506, slice_81503)
    
    # Processing the call keyword arguments (line 1204)
    kwargs_81508 = {}
    # Getting the type of 'len' (line 1204)
    len_81501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 35), 'len', False)
    # Calling len(args, kwargs) (line 1204)
    len_call_result_81509 = invoke(stypy.reporting.localization.Localization(__file__, 1204, 35), len_81501, *[subscript_call_result_81507], **kwargs_81508)
    
    # Applying the binary operator '*' (line 1204)
    result_mul_81510 = python_operator(stypy.reporting.localization.Localization(__file__, 1204, 27), '*', tuple_81499, len_call_result_81509)
    
    # Applying the binary operator '+' (line 1204)
    result_add_81511 = python_operator(stypy.reporting.localization.Localization(__file__, 1204, 9), '+', tuple_81494, result_mul_81510)
    
    # Assigning a type to the variable 'sh' (line 1204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1204, 4), 'sh', result_add_81511)
    
    # Call to errstate(...): (line 1206)
    # Processing the call keyword arguments (line 1206)
    str_81514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1206, 29), 'str', 'raise')
    keyword_81515 = str_81514
    str_81516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1206, 45), 'str', 'raise')
    keyword_81517 = str_81516
    kwargs_81518 = {'divide': keyword_81517, 'invalid': keyword_81515}
    # Getting the type of 'np' (line 1206)
    np_81512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 1206)
    errstate_81513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1206, 9), np_81512, 'errstate')
    # Calling errstate(args, kwargs) (line 1206)
    errstate_call_result_81519 = invoke(stypy.reporting.localization.Localization(__file__, 1206, 9), errstate_81513, *[], **kwargs_81518)
    
    with_81520 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 1206, 9), errstate_call_result_81519, 'with parameter', '__enter__', '__exit__')

    if with_81520:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 1206)
        enter___81521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1206, 9), errstate_call_result_81519, '__enter__')
        with_enter_81522 = invoke(stypy.reporting.localization.Localization(__file__, 1206, 9), enter___81521)
        
        
        # SSA begins for try-except statement (line 1207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Call to range(...): (line 1208)
        # Processing the call arguments (line 1208)
        # Getting the type of 'n' (line 1208)
        n_81524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 27), 'n', False)
        # Processing the call keyword arguments (line 1208)
        kwargs_81525 = {}
        # Getting the type of 'range' (line 1208)
        range_81523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 21), 'range', False)
        # Calling range(args, kwargs) (line 1208)
        range_call_result_81526 = invoke(stypy.reporting.localization.Localization(__file__, 1208, 21), range_81523, *[n_81524], **kwargs_81525)
        
        # Testing the type of a for loop iterable (line 1208)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1208, 12), range_call_result_81526)
        # Getting the type of the for loop variable (line 1208)
        for_loop_var_81527 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1208, 12), range_call_result_81526)
        # Assigning a type to the variable 'j' (line 1208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 12), 'j', for_loop_var_81527)
        # SSA begins for a for statement (line 1208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 1213):
        
        # Assigning a BinOp to a Name (line 1213):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 1213)
        k_81528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 23), 'k')
        int_81529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 25), 'int')
        # Applying the binary operator '+' (line 1213)
        result_add_81530 = python_operator(stypy.reporting.localization.Localization(__file__, 1213, 23), '+', k_81528, int_81529)
        
        int_81531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 27), 'int')
        slice_81532 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1213, 21), result_add_81530, int_81531, None)
        # Getting the type of 't' (line 1213)
        t_81533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 21), 't')
        # Obtaining the member '__getitem__' of a type (line 1213)
        getitem___81534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1213, 21), t_81533, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1213)
        subscript_call_result_81535 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 21), getitem___81534, slice_81532)
        
        
        # Obtaining the type of the subscript
        int_81536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 35), 'int')
        
        # Getting the type of 'k' (line 1213)
        k_81537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 38), 'k')
        # Applying the 'usub' unary operator (line 1213)
        result___neg___81538 = python_operator(stypy.reporting.localization.Localization(__file__, 1213, 37), 'usub', k_81537)
        
        int_81539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 40), 'int')
        # Applying the binary operator '-' (line 1213)
        result_sub_81540 = python_operator(stypy.reporting.localization.Localization(__file__, 1213, 37), '-', result___neg___81538, int_81539)
        
        slice_81541 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1213, 33), int_81536, result_sub_81540, None)
        # Getting the type of 't' (line 1213)
        t_81542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 33), 't')
        # Obtaining the member '__getitem__' of a type (line 1213)
        getitem___81543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1213, 33), t_81542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1213)
        subscript_call_result_81544 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 33), getitem___81543, slice_81541)
        
        # Applying the binary operator '-' (line 1213)
        result_sub_81545 = python_operator(stypy.reporting.localization.Localization(__file__, 1213, 21), '-', subscript_call_result_81535, subscript_call_result_81544)
        
        # Assigning a type to the variable 'dt' (line 1213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 16), 'dt', result_sub_81545)
        
        # Assigning a Subscript to a Name (line 1214):
        
        # Assigning a Subscript to a Name (line 1214):
        
        # Obtaining the type of the subscript
        # Getting the type of 'sh' (line 1214)
        sh_81546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 24), 'sh')
        # Getting the type of 'dt' (line 1214)
        dt_81547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 21), 'dt')
        # Obtaining the member '__getitem__' of a type (line 1214)
        getitem___81548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1214, 21), dt_81547, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1214)
        subscript_call_result_81549 = invoke(stypy.reporting.localization.Localization(__file__, 1214, 21), getitem___81548, sh_81546)
        
        # Assigning a type to the variable 'dt' (line 1214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1214, 16), 'dt', subscript_call_result_81549)
        
        # Assigning a BinOp to a Name (line 1216):
        
        # Assigning a BinOp to a Name (line 1216):
        
        # Obtaining the type of the subscript
        int_81550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 23), 'int')
        int_81551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 25), 'int')
        # Getting the type of 'k' (line 1216)
        k_81552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 28), 'k')
        # Applying the binary operator '-' (line 1216)
        result_sub_81553 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 25), '-', int_81551, k_81552)
        
        slice_81554 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1216, 21), int_81550, result_sub_81553, None)
        # Getting the type of 'c' (line 1216)
        c_81555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 21), 'c')
        # Obtaining the member '__getitem__' of a type (line 1216)
        getitem___81556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1216, 21), c_81555, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1216)
        subscript_call_result_81557 = invoke(stypy.reporting.localization.Localization(__file__, 1216, 21), getitem___81556, slice_81554)
        
        
        # Obtaining the type of the subscript
        int_81558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 36), 'int')
        # Getting the type of 'k' (line 1216)
        k_81559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 39), 'k')
        # Applying the binary operator '-' (line 1216)
        result_sub_81560 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 36), '-', int_81558, k_81559)
        
        slice_81561 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1216, 33), None, result_sub_81560, None)
        # Getting the type of 'c' (line 1216)
        c_81562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 33), 'c')
        # Obtaining the member '__getitem__' of a type (line 1216)
        getitem___81563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1216, 33), c_81562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1216)
        subscript_call_result_81564 = invoke(stypy.reporting.localization.Localization(__file__, 1216, 33), getitem___81563, slice_81561)
        
        # Applying the binary operator '-' (line 1216)
        result_sub_81565 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 21), '-', subscript_call_result_81557, subscript_call_result_81564)
        
        # Getting the type of 'k' (line 1216)
        k_81566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 45), 'k')
        # Applying the binary operator '*' (line 1216)
        result_mul_81567 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 20), '*', result_sub_81565, k_81566)
        
        # Getting the type of 'dt' (line 1216)
        dt_81568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 49), 'dt')
        # Applying the binary operator 'div' (line 1216)
        result_div_81569 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 47), 'div', result_mul_81567, dt_81568)
        
        # Assigning a type to the variable 'c' (line 1216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1216, 16), 'c', result_div_81569)
        
        # Assigning a Subscript to a Name (line 1219):
        
        # Assigning a Subscript to a Name (line 1219):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 1219)
        tuple_81570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1219)
        # Adding element type (line 1219)
        # Getting the type of 'c' (line 1219)
        c_81571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 26), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1219, 26), tuple_81570, c_81571)
        # Adding element type (line 1219)
        
        # Call to zeros(...): (line 1219)
        # Processing the call arguments (line 1219)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1219)
        tuple_81574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1219)
        # Adding element type (line 1219)
        # Getting the type of 'k' (line 1219)
        k_81575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 39), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1219, 39), tuple_81574, k_81575)
        
        
        # Obtaining the type of the subscript
        int_81576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 53), 'int')
        slice_81577 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1219, 45), int_81576, None, None)
        # Getting the type of 'c' (line 1219)
        c_81578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 45), 'c', False)
        # Obtaining the member 'shape' of a type (line 1219)
        shape_81579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 45), c_81578, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1219)
        getitem___81580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 45), shape_81579, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1219)
        subscript_call_result_81581 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 45), getitem___81580, slice_81577)
        
        # Applying the binary operator '+' (line 1219)
        result_add_81582 = python_operator(stypy.reporting.localization.Localization(__file__, 1219, 38), '+', tuple_81574, subscript_call_result_81581)
        
        # Processing the call keyword arguments (line 1219)
        kwargs_81583 = {}
        # Getting the type of 'np' (line 1219)
        np_81572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 29), 'np', False)
        # Obtaining the member 'zeros' of a type (line 1219)
        zeros_81573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 29), np_81572, 'zeros')
        # Calling zeros(args, kwargs) (line 1219)
        zeros_call_result_81584 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 29), zeros_81573, *[result_add_81582], **kwargs_81583)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1219, 26), tuple_81570, zeros_call_result_81584)
        
        # Getting the type of 'np' (line 1219)
        np_81585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 20), 'np')
        # Obtaining the member 'r_' of a type (line 1219)
        r__81586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 20), np_81585, 'r_')
        # Obtaining the member '__getitem__' of a type (line 1219)
        getitem___81587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 20), r__81586, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1219)
        subscript_call_result_81588 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 20), getitem___81587, tuple_81570)
        
        # Assigning a type to the variable 'c' (line 1219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1219, 16), 'c', subscript_call_result_81588)
        
        # Assigning a Subscript to a Name (line 1221):
        
        # Assigning a Subscript to a Name (line 1221):
        
        # Obtaining the type of the subscript
        int_81589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 22), 'int')
        int_81590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 24), 'int')
        slice_81591 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1221, 20), int_81589, int_81590, None)
        # Getting the type of 't' (line 1221)
        t_81592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 20), 't')
        # Obtaining the member '__getitem__' of a type (line 1221)
        getitem___81593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 20), t_81592, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1221)
        subscript_call_result_81594 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 20), getitem___81593, slice_81591)
        
        # Assigning a type to the variable 't' (line 1221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 16), 't', subscript_call_result_81594)
        
        # Getting the type of 'k' (line 1222)
        k_81595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 16), 'k')
        int_81596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1222, 21), 'int')
        # Applying the binary operator '-=' (line 1222)
        result_isub_81597 = python_operator(stypy.reporting.localization.Localization(__file__, 1222, 16), '-=', k_81595, int_81596)
        # Assigning a type to the variable 'k' (line 1222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 16), 'k', result_isub_81597)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 1207)
        # SSA branch for the except 'FloatingPointError' branch of a try statement (line 1207)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 1224)
        # Processing the call arguments (line 1224)
        str_81599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 30), 'str', 'The spline has internal repeated knots and is not differentiable %d times')
        # Getting the type of 'n' (line 1225)
        n_81600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 70), 'n', False)
        # Applying the binary operator '%' (line 1224)
        result_mod_81601 = python_operator(stypy.reporting.localization.Localization(__file__, 1224, 29), '%', str_81599, n_81600)
        
        # Processing the call keyword arguments (line 1224)
        kwargs_81602 = {}
        # Getting the type of 'ValueError' (line 1224)
        ValueError_81598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1224)
        ValueError_call_result_81603 = invoke(stypy.reporting.localization.Localization(__file__, 1224, 18), ValueError_81598, *[result_mod_81601], **kwargs_81602)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1224, 12), ValueError_call_result_81603, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 1207)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 1206)
        exit___81604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1206, 9), errstate_call_result_81519, '__exit__')
        with_exit_81605 = invoke(stypy.reporting.localization.Localization(__file__, 1206, 9), exit___81604, None, None, None)

    
    # Obtaining an instance of the builtin type 'tuple' (line 1227)
    tuple_81606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1227)
    # Adding element type (line 1227)
    # Getting the type of 't' (line 1227)
    t_81607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 11), 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1227, 11), tuple_81606, t_81607)
    # Adding element type (line 1227)
    # Getting the type of 'c' (line 1227)
    c_81608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 14), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1227, 11), tuple_81606, c_81608)
    # Adding element type (line 1227)
    # Getting the type of 'k' (line 1227)
    k_81609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 17), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1227, 11), tuple_81606, k_81609)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 4), 'stypy_return_type', tuple_81606)
    
    # ################# End of 'splder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splder' in the type store
    # Getting the type of 'stypy_return_type' (line 1147)
    stypy_return_type_81610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_81610)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splder'
    return stypy_return_type_81610

# Assigning a type to the variable 'splder' (line 1147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1147, 0), 'splder', splder)

@norecursion
def splantider(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_81611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1230, 22), 'int')
    defaults = [int_81611]
    # Create a new context for function 'splantider'
    module_type_store = module_type_store.open_function_context('splantider', 1230, 0, False)
    
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

    str_81612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, (-1)), 'str', '\n    Compute the spline for the antiderivative (integral) of a given spline.\n\n    Parameters\n    ----------\n    tck : tuple of (t, c, k)\n        Spline whose antiderivative to compute\n    n : int, optional\n        Order of antiderivative to evaluate. Default: 1\n\n    Returns\n    -------\n    tck_ader : tuple of (t2, c2, k2)\n        Spline of order k2=k+n representing the antiderivative of the input\n        spline.\n\n    See Also\n    --------\n    splder, splev, spalde\n\n    Notes\n    -----\n    The `splder` function is the inverse operation of this function.\n    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo\n    rounding error.\n\n    .. versionadded:: 0.13.0\n\n    Examples\n    --------\n    >>> from scipy.interpolate import splrep, splder, splantider, splev\n    >>> x = np.linspace(0, np.pi/2, 70)\n    >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)\n    >>> spl = splrep(x, y)\n\n    The derivative is the inverse operation of the antiderivative,\n    although some floating point error accumulates:\n\n    >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))\n    (array(2.1565429877197317), array(2.1565429877201865))\n\n    Antiderivative can be used to evaluate definite integrals:\n\n    >>> ispl = splantider(spl)\n    >>> splev(np.pi/2, ispl) - splev(0, ispl)\n    2.2572053588768486\n\n    This is indeed an approximation to the complete elliptic integral\n    :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:\n\n    >>> from scipy.special import ellipk\n    >>> ellipk(0.8)\n    2.2572053268208538\n\n    ')
    
    
    # Getting the type of 'n' (line 1286)
    n_81613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 7), 'n')
    int_81614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1286, 11), 'int')
    # Applying the binary operator '<' (line 1286)
    result_lt_81615 = python_operator(stypy.reporting.localization.Localization(__file__, 1286, 7), '<', n_81613, int_81614)
    
    # Testing the type of an if condition (line 1286)
    if_condition_81616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1286, 4), result_lt_81615)
    # Assigning a type to the variable 'if_condition_81616' (line 1286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1286, 4), 'if_condition_81616', if_condition_81616)
    # SSA begins for if statement (line 1286)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to splder(...): (line 1287)
    # Processing the call arguments (line 1287)
    # Getting the type of 'tck' (line 1287)
    tck_81618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 22), 'tck', False)
    
    # Getting the type of 'n' (line 1287)
    n_81619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 28), 'n', False)
    # Applying the 'usub' unary operator (line 1287)
    result___neg___81620 = python_operator(stypy.reporting.localization.Localization(__file__, 1287, 27), 'usub', n_81619)
    
    # Processing the call keyword arguments (line 1287)
    kwargs_81621 = {}
    # Getting the type of 'splder' (line 1287)
    splder_81617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 15), 'splder', False)
    # Calling splder(args, kwargs) (line 1287)
    splder_call_result_81622 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 15), splder_81617, *[tck_81618, result___neg___81620], **kwargs_81621)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 8), 'stypy_return_type', splder_call_result_81622)
    # SSA join for if statement (line 1286)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1289):
    
    # Assigning a Subscript to a Name (line 1289):
    
    # Obtaining the type of the subscript
    int_81623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 4), 'int')
    # Getting the type of 'tck' (line 1289)
    tck_81624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1289)
    getitem___81625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 4), tck_81624, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1289)
    subscript_call_result_81626 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 4), getitem___81625, int_81623)
    
    # Assigning a type to the variable 'tuple_var_assignment_78357' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'tuple_var_assignment_78357', subscript_call_result_81626)
    
    # Assigning a Subscript to a Name (line 1289):
    
    # Obtaining the type of the subscript
    int_81627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 4), 'int')
    # Getting the type of 'tck' (line 1289)
    tck_81628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1289)
    getitem___81629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 4), tck_81628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1289)
    subscript_call_result_81630 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 4), getitem___81629, int_81627)
    
    # Assigning a type to the variable 'tuple_var_assignment_78358' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'tuple_var_assignment_78358', subscript_call_result_81630)
    
    # Assigning a Subscript to a Name (line 1289):
    
    # Obtaining the type of the subscript
    int_81631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 4), 'int')
    # Getting the type of 'tck' (line 1289)
    tck_81632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 1289)
    getitem___81633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 4), tck_81632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1289)
    subscript_call_result_81634 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 4), getitem___81633, int_81631)
    
    # Assigning a type to the variable 'tuple_var_assignment_78359' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'tuple_var_assignment_78359', subscript_call_result_81634)
    
    # Assigning a Name to a Name (line 1289):
    # Getting the type of 'tuple_var_assignment_78357' (line 1289)
    tuple_var_assignment_78357_81635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'tuple_var_assignment_78357')
    # Assigning a type to the variable 't' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 't', tuple_var_assignment_78357_81635)
    
    # Assigning a Name to a Name (line 1289):
    # Getting the type of 'tuple_var_assignment_78358' (line 1289)
    tuple_var_assignment_78358_81636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'tuple_var_assignment_78358')
    # Assigning a type to the variable 'c' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 7), 'c', tuple_var_assignment_78358_81636)
    
    # Assigning a Name to a Name (line 1289):
    # Getting the type of 'tuple_var_assignment_78359' (line 1289)
    tuple_var_assignment_78359_81637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'tuple_var_assignment_78359')
    # Assigning a type to the variable 'k' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 10), 'k', tuple_var_assignment_78359_81637)
    
    # Assigning a BinOp to a Name (line 1292):
    
    # Assigning a BinOp to a Name (line 1292):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1292)
    tuple_81638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1292)
    # Adding element type (line 1292)
    
    # Call to slice(...): (line 1292)
    # Processing the call arguments (line 1292)
    # Getting the type of 'None' (line 1292)
    None_81640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 16), 'None', False)
    # Processing the call keyword arguments (line 1292)
    kwargs_81641 = {}
    # Getting the type of 'slice' (line 1292)
    slice_81639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 10), 'slice', False)
    # Calling slice(args, kwargs) (line 1292)
    slice_call_result_81642 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 10), slice_81639, *[None_81640], **kwargs_81641)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 10), tuple_81638, slice_call_result_81642)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1292)
    tuple_81643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1292)
    # Adding element type (line 1292)
    # Getting the type of 'None' (line 1292)
    None_81644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 27), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 27), tuple_81643, None_81644)
    
    
    # Call to len(...): (line 1292)
    # Processing the call arguments (line 1292)
    
    # Obtaining the type of the subscript
    int_81646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 46), 'int')
    slice_81647 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1292, 38), int_81646, None, None)
    # Getting the type of 'c' (line 1292)
    c_81648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 38), 'c', False)
    # Obtaining the member 'shape' of a type (line 1292)
    shape_81649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1292, 38), c_81648, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1292)
    getitem___81650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1292, 38), shape_81649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1292)
    subscript_call_result_81651 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 38), getitem___81650, slice_81647)
    
    # Processing the call keyword arguments (line 1292)
    kwargs_81652 = {}
    # Getting the type of 'len' (line 1292)
    len_81645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 34), 'len', False)
    # Calling len(args, kwargs) (line 1292)
    len_call_result_81653 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 34), len_81645, *[subscript_call_result_81651], **kwargs_81652)
    
    # Applying the binary operator '*' (line 1292)
    result_mul_81654 = python_operator(stypy.reporting.localization.Localization(__file__, 1292, 26), '*', tuple_81643, len_call_result_81653)
    
    # Applying the binary operator '+' (line 1292)
    result_add_81655 = python_operator(stypy.reporting.localization.Localization(__file__, 1292, 9), '+', tuple_81638, result_mul_81654)
    
    # Assigning a type to the variable 'sh' (line 1292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'sh', result_add_81655)
    
    
    # Call to range(...): (line 1294)
    # Processing the call arguments (line 1294)
    # Getting the type of 'n' (line 1294)
    n_81657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 19), 'n', False)
    # Processing the call keyword arguments (line 1294)
    kwargs_81658 = {}
    # Getting the type of 'range' (line 1294)
    range_81656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 13), 'range', False)
    # Calling range(args, kwargs) (line 1294)
    range_call_result_81659 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 13), range_81656, *[n_81657], **kwargs_81658)
    
    # Testing the type of a for loop iterable (line 1294)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1294, 4), range_call_result_81659)
    # Getting the type of the for loop variable (line 1294)
    for_loop_var_81660 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1294, 4), range_call_result_81659)
    # Assigning a type to the variable 'j' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'j', for_loop_var_81660)
    # SSA begins for a for statement (line 1294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 1298):
    
    # Assigning a BinOp to a Name (line 1298):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 1298)
    k_81661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 15), 'k')
    int_81662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1298, 17), 'int')
    # Applying the binary operator '+' (line 1298)
    result_add_81663 = python_operator(stypy.reporting.localization.Localization(__file__, 1298, 15), '+', k_81661, int_81662)
    
    slice_81664 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1298, 13), result_add_81663, None, None)
    # Getting the type of 't' (line 1298)
    t_81665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 13), 't')
    # Obtaining the member '__getitem__' of a type (line 1298)
    getitem___81666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 13), t_81665, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1298)
    subscript_call_result_81667 = invoke(stypy.reporting.localization.Localization(__file__, 1298, 13), getitem___81666, slice_81664)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'k' (line 1298)
    k_81668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 27), 'k')
    # Applying the 'usub' unary operator (line 1298)
    result___neg___81669 = python_operator(stypy.reporting.localization.Localization(__file__, 1298, 26), 'usub', k_81668)
    
    int_81670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1298, 29), 'int')
    # Applying the binary operator '-' (line 1298)
    result_sub_81671 = python_operator(stypy.reporting.localization.Localization(__file__, 1298, 26), '-', result___neg___81669, int_81670)
    
    slice_81672 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1298, 23), None, result_sub_81671, None)
    # Getting the type of 't' (line 1298)
    t_81673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 23), 't')
    # Obtaining the member '__getitem__' of a type (line 1298)
    getitem___81674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 23), t_81673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1298)
    subscript_call_result_81675 = invoke(stypy.reporting.localization.Localization(__file__, 1298, 23), getitem___81674, slice_81672)
    
    # Applying the binary operator '-' (line 1298)
    result_sub_81676 = python_operator(stypy.reporting.localization.Localization(__file__, 1298, 13), '-', subscript_call_result_81667, subscript_call_result_81675)
    
    # Assigning a type to the variable 'dt' (line 1298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1298, 8), 'dt', result_sub_81676)
    
    # Assigning a Subscript to a Name (line 1299):
    
    # Assigning a Subscript to a Name (line 1299):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sh' (line 1299)
    sh_81677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 16), 'sh')
    # Getting the type of 'dt' (line 1299)
    dt_81678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 13), 'dt')
    # Obtaining the member '__getitem__' of a type (line 1299)
    getitem___81679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1299, 13), dt_81678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1299)
    subscript_call_result_81680 = invoke(stypy.reporting.localization.Localization(__file__, 1299, 13), getitem___81679, sh_81677)
    
    # Assigning a type to the variable 'dt' (line 1299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1299, 8), 'dt', subscript_call_result_81680)
    
    # Assigning a BinOp to a Name (line 1301):
    
    # Assigning a BinOp to a Name (line 1301):
    
    # Call to cumsum(...): (line 1301)
    # Processing the call arguments (line 1301)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'k' (line 1301)
    k_81683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 26), 'k', False)
    # Applying the 'usub' unary operator (line 1301)
    result___neg___81684 = python_operator(stypy.reporting.localization.Localization(__file__, 1301, 25), 'usub', k_81683)
    
    int_81685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1301, 28), 'int')
    # Applying the binary operator '-' (line 1301)
    result_sub_81686 = python_operator(stypy.reporting.localization.Localization(__file__, 1301, 25), '-', result___neg___81684, int_81685)
    
    slice_81687 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1301, 22), None, result_sub_81686, None)
    # Getting the type of 'c' (line 1301)
    c_81688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 22), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1301)
    getitem___81689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 22), c_81688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1301)
    subscript_call_result_81690 = invoke(stypy.reporting.localization.Localization(__file__, 1301, 22), getitem___81689, slice_81687)
    
    # Getting the type of 'dt' (line 1301)
    dt_81691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 33), 'dt', False)
    # Applying the binary operator '*' (line 1301)
    result_mul_81692 = python_operator(stypy.reporting.localization.Localization(__file__, 1301, 22), '*', subscript_call_result_81690, dt_81691)
    
    # Processing the call keyword arguments (line 1301)
    int_81693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1301, 42), 'int')
    keyword_81694 = int_81693
    kwargs_81695 = {'axis': keyword_81694}
    # Getting the type of 'np' (line 1301)
    np_81681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 12), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 1301)
    cumsum_81682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 12), np_81681, 'cumsum')
    # Calling cumsum(args, kwargs) (line 1301)
    cumsum_call_result_81696 = invoke(stypy.reporting.localization.Localization(__file__, 1301, 12), cumsum_81682, *[result_mul_81692], **kwargs_81695)
    
    # Getting the type of 'k' (line 1301)
    k_81697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 48), 'k')
    int_81698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1301, 52), 'int')
    # Applying the binary operator '+' (line 1301)
    result_add_81699 = python_operator(stypy.reporting.localization.Localization(__file__, 1301, 48), '+', k_81697, int_81698)
    
    # Applying the binary operator 'div' (line 1301)
    result_div_81700 = python_operator(stypy.reporting.localization.Localization(__file__, 1301, 12), 'div', cumsum_call_result_81696, result_add_81699)
    
    # Assigning a type to the variable 'c' (line 1301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1301, 8), 'c', result_div_81700)
    
    # Assigning a Subscript to a Name (line 1302):
    
    # Assigning a Subscript to a Name (line 1302):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 1302)
    tuple_81701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1302, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1302)
    # Adding element type (line 1302)
    
    # Call to zeros(...): (line 1302)
    # Processing the call arguments (line 1302)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1302)
    tuple_81704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1302, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1302)
    # Adding element type (line 1302)
    int_81705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1302, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 28), tuple_81704, int_81705)
    
    
    # Obtaining the type of the subscript
    int_81706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1302, 42), 'int')
    slice_81707 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1302, 34), int_81706, None, None)
    # Getting the type of 'c' (line 1302)
    c_81708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 34), 'c', False)
    # Obtaining the member 'shape' of a type (line 1302)
    shape_81709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 34), c_81708, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1302)
    getitem___81710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 34), shape_81709, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1302)
    subscript_call_result_81711 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 34), getitem___81710, slice_81707)
    
    # Applying the binary operator '+' (line 1302)
    result_add_81712 = python_operator(stypy.reporting.localization.Localization(__file__, 1302, 27), '+', tuple_81704, subscript_call_result_81711)
    
    # Processing the call keyword arguments (line 1302)
    kwargs_81713 = {}
    # Getting the type of 'np' (line 1302)
    np_81702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 18), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1302)
    zeros_81703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 18), np_81702, 'zeros')
    # Calling zeros(args, kwargs) (line 1302)
    zeros_call_result_81714 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 18), zeros_81703, *[result_add_81712], **kwargs_81713)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 18), tuple_81701, zeros_call_result_81714)
    # Adding element type (line 1302)
    # Getting the type of 'c' (line 1303)
    c_81715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 18), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 18), tuple_81701, c_81715)
    # Adding element type (line 1302)
    
    # Obtaining an instance of the builtin type 'list' (line 1304)
    list_81716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1304, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1304)
    # Adding element type (line 1304)
    
    # Obtaining the type of the subscript
    int_81717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1304, 21), 'int')
    # Getting the type of 'c' (line 1304)
    c_81718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 19), 'c')
    # Obtaining the member '__getitem__' of a type (line 1304)
    getitem___81719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 19), c_81718, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1304)
    subscript_call_result_81720 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 19), getitem___81719, int_81717)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1304, 18), list_81716, subscript_call_result_81720)
    
    # Getting the type of 'k' (line 1304)
    k_81721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 29), 'k')
    int_81722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1304, 31), 'int')
    # Applying the binary operator '+' (line 1304)
    result_add_81723 = python_operator(stypy.reporting.localization.Localization(__file__, 1304, 29), '+', k_81721, int_81722)
    
    # Applying the binary operator '*' (line 1304)
    result_mul_81724 = python_operator(stypy.reporting.localization.Localization(__file__, 1304, 18), '*', list_81716, result_add_81723)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 18), tuple_81701, result_mul_81724)
    
    # Getting the type of 'np' (line 1302)
    np_81725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 12), 'np')
    # Obtaining the member 'r_' of a type (line 1302)
    r__81726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 12), np_81725, 'r_')
    # Obtaining the member '__getitem__' of a type (line 1302)
    getitem___81727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 12), r__81726, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1302)
    subscript_call_result_81728 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 12), getitem___81727, tuple_81701)
    
    # Assigning a type to the variable 'c' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 8), 'c', subscript_call_result_81728)
    
    # Assigning a Subscript to a Name (line 1306):
    
    # Assigning a Subscript to a Name (line 1306):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 1306)
    tuple_81729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1306, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1306)
    # Adding element type (line 1306)
    
    # Obtaining the type of the subscript
    int_81730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1306, 20), 'int')
    # Getting the type of 't' (line 1306)
    t_81731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 18), 't')
    # Obtaining the member '__getitem__' of a type (line 1306)
    getitem___81732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1306, 18), t_81731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1306)
    subscript_call_result_81733 = invoke(stypy.reporting.localization.Localization(__file__, 1306, 18), getitem___81732, int_81730)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 18), tuple_81729, subscript_call_result_81733)
    # Adding element type (line 1306)
    # Getting the type of 't' (line 1306)
    t_81734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 24), 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 18), tuple_81729, t_81734)
    # Adding element type (line 1306)
    
    # Obtaining the type of the subscript
    int_81735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1306, 29), 'int')
    # Getting the type of 't' (line 1306)
    t_81736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 27), 't')
    # Obtaining the member '__getitem__' of a type (line 1306)
    getitem___81737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1306, 27), t_81736, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1306)
    subscript_call_result_81738 = invoke(stypy.reporting.localization.Localization(__file__, 1306, 27), getitem___81737, int_81735)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 18), tuple_81729, subscript_call_result_81738)
    
    # Getting the type of 'np' (line 1306)
    np_81739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 12), 'np')
    # Obtaining the member 'r_' of a type (line 1306)
    r__81740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1306, 12), np_81739, 'r_')
    # Obtaining the member '__getitem__' of a type (line 1306)
    getitem___81741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1306, 12), r__81740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1306)
    subscript_call_result_81742 = invoke(stypy.reporting.localization.Localization(__file__, 1306, 12), getitem___81741, tuple_81729)
    
    # Assigning a type to the variable 't' (line 1306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1306, 8), 't', subscript_call_result_81742)
    
    # Getting the type of 'k' (line 1307)
    k_81743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1307, 8), 'k')
    int_81744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1307, 13), 'int')
    # Applying the binary operator '+=' (line 1307)
    result_iadd_81745 = python_operator(stypy.reporting.localization.Localization(__file__, 1307, 8), '+=', k_81743, int_81744)
    # Assigning a type to the variable 'k' (line 1307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1307, 8), 'k', result_iadd_81745)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1309)
    tuple_81746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1309)
    # Adding element type (line 1309)
    # Getting the type of 't' (line 1309)
    t_81747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 11), 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1309, 11), tuple_81746, t_81747)
    # Adding element type (line 1309)
    # Getting the type of 'c' (line 1309)
    c_81748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 14), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1309, 11), tuple_81746, c_81748)
    # Adding element type (line 1309)
    # Getting the type of 'k' (line 1309)
    k_81749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 17), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1309, 11), tuple_81746, k_81749)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1309, 4), 'stypy_return_type', tuple_81746)
    
    # ################# End of 'splantider(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splantider' in the type store
    # Getting the type of 'stypy_return_type' (line 1230)
    stypy_return_type_81750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_81750)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splantider'
    return stypy_return_type_81750

# Assigning a type to the variable 'splantider' (line 1230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 0), 'splantider', splantider)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
