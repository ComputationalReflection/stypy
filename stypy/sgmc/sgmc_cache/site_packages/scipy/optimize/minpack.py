
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import warnings
4: from . import _minpack
5: 
6: import numpy as np
7: from numpy import (atleast_1d, dot, take, triu, shape, eye,
8:                    transpose, zeros, product, greater, array,
9:                    all, where, isscalar, asarray, inf, abs,
10:                    finfo, inexact, issubdtype, dtype)
11: from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError
12: from scipy._lib._util import _asarray_validated, _lazywhere
13: from .optimize import OptimizeResult, _check_unknown_options, OptimizeWarning
14: from ._lsq import least_squares
15: from ._lsq.common import make_strictly_feasible
16: from ._lsq.least_squares import prepare_bounds
17: 
18: 
19: error = _minpack.error
20: 
21: __all__ = ['fsolve', 'leastsq', 'fixed_point', 'curve_fit']
22: 
23: 
24: def _check_func(checker, argname, thefunc, x0, args, numinputs,
25:                 output_shape=None):
26:     res = atleast_1d(thefunc(*((x0[:numinputs],) + args)))
27:     if (output_shape is not None) and (shape(res) != output_shape):
28:         if (output_shape[0] != 1):
29:             if len(output_shape) > 1:
30:                 if output_shape[1] == 1:
31:                     return shape(res)
32:             msg = "%s: there is a mismatch between the input and output " \
33:                   "shape of the '%s' argument" % (checker, argname)
34:             func_name = getattr(thefunc, '__name__', None)
35:             if func_name:
36:                 msg += " '%s'." % func_name
37:             else:
38:                 msg += "."
39:             msg += 'Shape should be %s but it is %s.' % (output_shape, shape(res))
40:             raise TypeError(msg)
41:     if issubdtype(res.dtype, inexact):
42:         dt = res.dtype
43:     else:
44:         dt = dtype(float)
45:     return shape(res), dt
46: 
47: 
48: def fsolve(func, x0, args=(), fprime=None, full_output=0,
49:            col_deriv=0, xtol=1.49012e-8, maxfev=0, band=None,
50:            epsfcn=None, factor=100, diag=None):
51:     '''
52:     Find the roots of a function.
53: 
54:     Return the roots of the (non-linear) equations defined by
55:     ``func(x) = 0`` given a starting estimate.
56: 
57:     Parameters
58:     ----------
59:     func : callable ``f(x, *args)``
60:         A function that takes at least one (possibly vector) argument.
61:     x0 : ndarray
62:         The starting estimate for the roots of ``func(x) = 0``.
63:     args : tuple, optional
64:         Any extra arguments to `func`.
65:     fprime : callable ``f(x, *args)``, optional
66:         A function to compute the Jacobian of `func` with derivatives
67:         across the rows. By default, the Jacobian will be estimated.
68:     full_output : bool, optional
69:         If True, return optional outputs.
70:     col_deriv : bool, optional
71:         Specify whether the Jacobian function computes derivatives down
72:         the columns (faster, because there is no transpose operation).
73:     xtol : float, optional
74:         The calculation will terminate if the relative error between two
75:         consecutive iterates is at most `xtol`.
76:     maxfev : int, optional
77:         The maximum number of calls to the function. If zero, then
78:         ``100*(N+1)`` is the maximum where N is the number of elements
79:         in `x0`.
80:     band : tuple, optional
81:         If set to a two-sequence containing the number of sub- and
82:         super-diagonals within the band of the Jacobi matrix, the
83:         Jacobi matrix is considered banded (only for ``fprime=None``).
84:     epsfcn : float, optional
85:         A suitable step length for the forward-difference
86:         approximation of the Jacobian (for ``fprime=None``). If
87:         `epsfcn` is less than the machine precision, it is assumed
88:         that the relative errors in the functions are of the order of
89:         the machine precision.
90:     factor : float, optional
91:         A parameter determining the initial step bound
92:         (``factor * || diag * x||``).  Should be in the interval
93:         ``(0.1, 100)``.
94:     diag : sequence, optional
95:         N positive entries that serve as a scale factors for the
96:         variables.
97: 
98:     Returns
99:     -------
100:     x : ndarray
101:         The solution (or the result of the last iteration for
102:         an unsuccessful call).
103:     infodict : dict
104:         A dictionary of optional outputs with the keys:
105: 
106:         ``nfev``
107:             number of function calls
108:         ``njev``
109:             number of Jacobian calls
110:         ``fvec``
111:             function evaluated at the output
112:         ``fjac``
113:             the orthogonal matrix, q, produced by the QR
114:             factorization of the final approximate Jacobian
115:             matrix, stored column wise
116:         ``r``
117:             upper triangular matrix produced by QR factorization
118:             of the same matrix
119:         ``qtf``
120:             the vector ``(transpose(q) * fvec)``
121: 
122:     ier : int
123:         An integer flag.  Set to 1 if a solution was found, otherwise refer
124:         to `mesg` for more information.
125:     mesg : str
126:         If no solution is found, `mesg` details the cause of failure.
127: 
128:     See Also
129:     --------
130:     root : Interface to root finding algorithms for multivariate
131:     functions. See the 'hybr' `method` in particular.
132: 
133:     Notes
134:     -----
135:     ``fsolve`` is a wrapper around MINPACK's hybrd and hybrj algorithms.
136: 
137:     '''
138:     options = {'col_deriv': col_deriv,
139:                'xtol': xtol,
140:                'maxfev': maxfev,
141:                'band': band,
142:                'eps': epsfcn,
143:                'factor': factor,
144:                'diag': diag}
145: 
146:     res = _root_hybr(func, x0, args, jac=fprime, **options)
147:     if full_output:
148:         x = res['x']
149:         info = dict((k, res.get(k))
150:                     for k in ('nfev', 'njev', 'fjac', 'r', 'qtf') if k in res)
151:         info['fvec'] = res['fun']
152:         return x, info, res['status'], res['message']
153:     else:
154:         status = res['status']
155:         msg = res['message']
156:         if status == 0:
157:             raise TypeError(msg)
158:         elif status == 1:
159:             pass
160:         elif status in [2, 3, 4, 5]:
161:             warnings.warn(msg, RuntimeWarning)
162:         else:
163:             raise TypeError(msg)
164:         return res['x']
165: 
166: 
167: def _root_hybr(func, x0, args=(), jac=None,
168:                col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, eps=None,
169:                factor=100, diag=None, **unknown_options):
170:     '''
171:     Find the roots of a multivariate function using MINPACK's hybrd and
172:     hybrj routines (modified Powell method).
173: 
174:     Options
175:     -------
176:     col_deriv : bool
177:         Specify whether the Jacobian function computes derivatives down
178:         the columns (faster, because there is no transpose operation).
179:     xtol : float
180:         The calculation will terminate if the relative error between two
181:         consecutive iterates is at most `xtol`.
182:     maxfev : int
183:         The maximum number of calls to the function. If zero, then
184:         ``100*(N+1)`` is the maximum where N is the number of elements
185:         in `x0`.
186:     band : tuple
187:         If set to a two-sequence containing the number of sub- and
188:         super-diagonals within the band of the Jacobi matrix, the
189:         Jacobi matrix is considered banded (only for ``fprime=None``).
190:     eps : float
191:         A suitable step length for the forward-difference
192:         approximation of the Jacobian (for ``fprime=None``). If
193:         `eps` is less than the machine precision, it is assumed
194:         that the relative errors in the functions are of the order of
195:         the machine precision.
196:     factor : float
197:         A parameter determining the initial step bound
198:         (``factor * || diag * x||``).  Should be in the interval
199:         ``(0.1, 100)``.
200:     diag : sequence
201:         N positive entries that serve as a scale factors for the
202:         variables.
203: 
204:     '''
205:     _check_unknown_options(unknown_options)
206:     epsfcn = eps
207: 
208:     x0 = asarray(x0).flatten()
209:     n = len(x0)
210:     if not isinstance(args, tuple):
211:         args = (args,)
212:     shape, dtype = _check_func('fsolve', 'func', func, x0, args, n, (n,))
213:     if epsfcn is None:
214:         epsfcn = finfo(dtype).eps
215:     Dfun = jac
216:     if Dfun is None:
217:         if band is None:
218:             ml, mu = -10, -10
219:         else:
220:             ml, mu = band[:2]
221:         if maxfev == 0:
222:             maxfev = 200 * (n + 1)
223:         retval = _minpack._hybrd(func, x0, args, 1, xtol, maxfev,
224:                                  ml, mu, epsfcn, factor, diag)
225:     else:
226:         _check_func('fsolve', 'fprime', Dfun, x0, args, n, (n, n))
227:         if (maxfev == 0):
228:             maxfev = 100 * (n + 1)
229:         retval = _minpack._hybrj(func, Dfun, x0, args, 1,
230:                                  col_deriv, xtol, maxfev, factor, diag)
231: 
232:     x, status = retval[0], retval[-1]
233: 
234:     errors = {0: "Improper input parameters were entered.",
235:               1: "The solution converged.",
236:               2: "The number of calls to function has "
237:                   "reached maxfev = %d." % maxfev,
238:               3: "xtol=%f is too small, no further improvement "
239:                   "in the approximate\n  solution "
240:                   "is possible." % xtol,
241:               4: "The iteration is not making good progress, as measured "
242:                   "by the \n  improvement from the last five "
243:                   "Jacobian evaluations.",
244:               5: "The iteration is not making good progress, "
245:                   "as measured by the \n  improvement from the last "
246:                   "ten iterations.",
247:               'unknown': "An error occurred."}
248: 
249:     info = retval[1]
250:     info['fun'] = info.pop('fvec')
251:     sol = OptimizeResult(x=x, success=(status == 1), status=status)
252:     sol.update(info)
253:     try:
254:         sol['message'] = errors[status]
255:     except KeyError:
256:         info['message'] = errors['unknown']
257: 
258:     return sol
259: 
260: 
261: def leastsq(func, x0, args=(), Dfun=None, full_output=0,
262:             col_deriv=0, ftol=1.49012e-8, xtol=1.49012e-8,
263:             gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
264:     '''
265:     Minimize the sum of squares of a set of equations.
266: 
267:     ::
268: 
269:         x = arg min(sum(func(y)**2,axis=0))
270:                  y
271: 
272:     Parameters
273:     ----------
274:     func : callable
275:         should take at least one (possibly length N vector) argument and
276:         returns M floating point numbers. It must not return NaNs or
277:         fitting might fail.
278:     x0 : ndarray
279:         The starting estimate for the minimization.
280:     args : tuple, optional
281:         Any extra arguments to func are placed in this tuple.
282:     Dfun : callable, optional
283:         A function or method to compute the Jacobian of func with derivatives
284:         across the rows. If this is None, the Jacobian will be estimated.
285:     full_output : bool, optional
286:         non-zero to return all optional outputs.
287:     col_deriv : bool, optional
288:         non-zero to specify that the Jacobian function computes derivatives
289:         down the columns (faster, because there is no transpose operation).
290:     ftol : float, optional
291:         Relative error desired in the sum of squares.
292:     xtol : float, optional
293:         Relative error desired in the approximate solution.
294:     gtol : float, optional
295:         Orthogonality desired between the function vector and the columns of
296:         the Jacobian.
297:     maxfev : int, optional
298:         The maximum number of calls to the function. If `Dfun` is provided
299:         then the default `maxfev` is 100*(N+1) where N is the number of elements
300:         in x0, otherwise the default `maxfev` is 200*(N+1).
301:     epsfcn : float, optional
302:         A variable used in determining a suitable step length for the forward-
303:         difference approximation of the Jacobian (for Dfun=None).
304:         Normally the actual step length will be sqrt(epsfcn)*x
305:         If epsfcn is less than the machine precision, it is assumed that the
306:         relative errors are of the order of the machine precision.
307:     factor : float, optional
308:         A parameter determining the initial step bound
309:         (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
310:     diag : sequence, optional
311:         N positive entries that serve as a scale factors for the variables.
312: 
313:     Returns
314:     -------
315:     x : ndarray
316:         The solution (or the result of the last iteration for an unsuccessful
317:         call).
318:     cov_x : ndarray
319:         Uses the fjac and ipvt optional outputs to construct an
320:         estimate of the jacobian around the solution. None if a
321:         singular matrix encountered (indicates very flat curvature in
322:         some direction).  This matrix must be multiplied by the
323:         residual variance to get the covariance of the
324:         parameter estimates -- see curve_fit.
325:     infodict : dict
326:         a dictionary of optional outputs with the key s:
327: 
328:         ``nfev``
329:             The number of function calls
330:         ``fvec``
331:             The function evaluated at the output
332:         ``fjac``
333:             A permutation of the R matrix of a QR
334:             factorization of the final approximate
335:             Jacobian matrix, stored column wise.
336:             Together with ipvt, the covariance of the
337:             estimate can be approximated.
338:         ``ipvt``
339:             An integer array of length N which defines
340:             a permutation matrix, p, such that
341:             fjac*p = q*r, where r is upper triangular
342:             with diagonal elements of nonincreasing
343:             magnitude. Column j of p is column ipvt(j)
344:             of the identity matrix.
345:         ``qtf``
346:             The vector (transpose(q) * fvec).
347: 
348:     mesg : str
349:         A string message giving information about the cause of failure.
350:     ier : int
351:         An integer flag.  If it is equal to 1, 2, 3 or 4, the solution was
352:         found.  Otherwise, the solution was not found. In either case, the
353:         optional output variable 'mesg' gives more information.
354: 
355:     Notes
356:     -----
357:     "leastsq" is a wrapper around MINPACK's lmdif and lmder algorithms.
358: 
359:     cov_x is a Jacobian approximation to the Hessian of the least squares
360:     objective function.
361:     This approximation assumes that the objective function is based on the
362:     difference between some observed target data (ydata) and a (non-linear)
363:     function of the parameters `f(xdata, params)` ::
364: 
365:            func(params) = ydata - f(xdata, params)
366: 
367:     so that the objective function is ::
368: 
369:            min   sum((ydata - f(xdata, params))**2, axis=0)
370:          params
371: 
372:     '''
373:     x0 = asarray(x0).flatten()
374:     n = len(x0)
375:     if not isinstance(args, tuple):
376:         args = (args,)
377:     shape, dtype = _check_func('leastsq', 'func', func, x0, args, n)
378:     m = shape[0]
379:     if n > m:
380:         raise TypeError('Improper input: N=%s must not exceed M=%s' % (n, m))
381:     if epsfcn is None:
382:         epsfcn = finfo(dtype).eps
383:     if Dfun is None:
384:         if maxfev == 0:
385:             maxfev = 200*(n + 1)
386:         retval = _minpack._lmdif(func, x0, args, full_output, ftol, xtol,
387:                                  gtol, maxfev, epsfcn, factor, diag)
388:     else:
389:         if col_deriv:
390:             _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (n, m))
391:         else:
392:             _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (m, n))
393:         if maxfev == 0:
394:             maxfev = 100 * (n + 1)
395:         retval = _minpack._lmder(func, Dfun, x0, args, full_output, col_deriv,
396:                                  ftol, xtol, gtol, maxfev, factor, diag)
397: 
398:     errors = {0: ["Improper input parameters.", TypeError],
399:               1: ["Both actual and predicted relative reductions "
400:                   "in the sum of squares\n  are at most %f" % ftol, None],
401:               2: ["The relative error between two consecutive "
402:                   "iterates is at most %f" % xtol, None],
403:               3: ["Both actual and predicted relative reductions in "
404:                   "the sum of squares\n  are at most %f and the "
405:                   "relative error between two consecutive "
406:                   "iterates is at \n  most %f" % (ftol, xtol), None],
407:               4: ["The cosine of the angle between func(x) and any "
408:                   "column of the\n  Jacobian is at most %f in "
409:                   "absolute value" % gtol, None],
410:               5: ["Number of calls to function has reached "
411:                   "maxfev = %d." % maxfev, ValueError],
412:               6: ["ftol=%f is too small, no further reduction "
413:                   "in the sum of squares\n  is possible.''' % ftol,
414:                   ValueError],
415:               7: ["xtol=%f is too small, no further improvement in "
416:                   "the approximate\n  solution is possible." % xtol,
417:                   ValueError],
418:               8: ["gtol=%f is too small, func(x) is orthogonal to the "
419:                   "columns of\n  the Jacobian to machine "
420:                   "precision." % gtol, ValueError],
421:               'unknown': ["Unknown error.", TypeError]}
422: 
423:     info = retval[-1]    # The FORTRAN return value
424: 
425:     if info not in [1, 2, 3, 4] and not full_output:
426:         if info in [5, 6, 7, 8]:
427:             warnings.warn(errors[info][0], RuntimeWarning)
428:         else:
429:             try:
430:                 raise errors[info][1](errors[info][0])
431:             except KeyError:
432:                 raise errors['unknown'][1](errors['unknown'][0])
433: 
434:     mesg = errors[info][0]
435:     if full_output:
436:         cov_x = None
437:         if info in [1, 2, 3, 4]:
438:             from numpy.dual import inv
439:             perm = take(eye(n), retval[1]['ipvt'] - 1, 0)
440:             r = triu(transpose(retval[1]['fjac'])[:n, :])
441:             R = dot(r, perm)
442:             try:
443:                 cov_x = inv(dot(transpose(R), R))
444:             except (LinAlgError, ValueError):
445:                 pass
446:         return (retval[0], cov_x) + retval[1:-1] + (mesg, info)
447:     else:
448:         return (retval[0], info)
449: 
450: 
451: def _wrap_func(func, xdata, ydata, transform):
452:     if transform is None:
453:         def func_wrapped(params):
454:             return func(xdata, *params) - ydata
455:     elif transform.ndim == 1:
456:         def func_wrapped(params):
457:             return transform * (func(xdata, *params) - ydata)
458:     else:
459:         # Chisq = (y - yd)^T C^{-1} (y-yd)
460:         # transform = L such that C = L L^T
461:         # C^{-1} = L^{-T} L^{-1}
462:         # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
463:         # Define (y-yd)' = L^{-1} (y-yd)
464:         # by solving
465:         # L (y-yd)' = (y-yd)
466:         # and minimize (y-yd)'^T (y-yd)'
467:         def func_wrapped(params):
468:             return solve_triangular(transform, func(xdata, *params) - ydata, lower=True)
469:     return func_wrapped
470: 
471: 
472: def _wrap_jac(jac, xdata, transform):
473:     if transform is None:
474:         def jac_wrapped(params):
475:             return jac(xdata, *params)
476:     elif transform.ndim == 1:
477:         def jac_wrapped(params):
478:             return transform[:, np.newaxis] * np.asarray(jac(xdata, *params))
479:     else:
480:         def jac_wrapped(params):
481:             return solve_triangular(transform, np.asarray(jac(xdata, *params)), lower=True)
482:     return jac_wrapped
483: 
484: 
485: def _initialize_feasible(lb, ub):
486:     p0 = np.ones_like(lb)
487:     lb_finite = np.isfinite(lb)
488:     ub_finite = np.isfinite(ub)
489: 
490:     mask = lb_finite & ub_finite
491:     p0[mask] = 0.5 * (lb[mask] + ub[mask])
492: 
493:     mask = lb_finite & ~ub_finite
494:     p0[mask] = lb[mask] + 1
495: 
496:     mask = ~lb_finite & ub_finite
497:     p0[mask] = ub[mask] - 1
498: 
499:     return p0
500: 
501: 
502: def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
503:               check_finite=True, bounds=(-np.inf, np.inf), method=None,
504:               jac=None, **kwargs):
505:     '''
506:     Use non-linear least squares to fit a function, f, to data.
507: 
508:     Assumes ``ydata = f(xdata, *params) + eps``
509: 
510:     Parameters
511:     ----------
512:     f : callable
513:         The model function, f(x, ...).  It must take the independent
514:         variable as the first argument and the parameters to fit as
515:         separate remaining arguments.
516:     xdata : An M-length sequence or an (k,M)-shaped array for functions with k predictors
517:         The independent variable where the data is measured.
518:     ydata : M-length sequence
519:         The dependent data --- nominally f(xdata, ...)
520:     p0 : None, scalar, or N-length sequence, optional
521:         Initial guess for the parameters.  If None, then the initial
522:         values will all be 1 (if the number of parameters for the function
523:         can be determined using introspection, otherwise a ValueError
524:         is raised).
525:     sigma : None or M-length sequence or MxM array, optional
526:         Determines the uncertainty in `ydata`. If we define residuals as
527:         ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
528:         depends on its number of dimensions:
529: 
530:             - A 1-d `sigma` should contain values of standard deviations of
531:               errors in `ydata`. In this case, the optimized function is
532:               ``chisq = sum((r / sigma) ** 2)``.
533: 
534:             - A 2-d `sigma` should contain the covariance matrix of
535:               errors in `ydata`. In this case, the optimized function is
536:               ``chisq = r.T @ inv(sigma) @ r``.
537: 
538:               .. versionadded:: 0.19
539: 
540:         None (default) is equivalent of 1-d `sigma` filled with ones.
541:     absolute_sigma : bool, optional
542:         If True, `sigma` is used in an absolute sense and the estimated parameter
543:         covariance `pcov` reflects these absolute values.
544: 
545:         If False, only the relative magnitudes of the `sigma` values matter.
546:         The returned parameter covariance matrix `pcov` is based on scaling
547:         `sigma` by a constant factor. This constant is set by demanding that the
548:         reduced `chisq` for the optimal parameters `popt` when using the
549:         *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
550:         match the sample variance of the residuals after the fit.
551:         Mathematically,
552:         ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
553:     check_finite : bool, optional
554:         If True, check that the input arrays do not contain nans of infs,
555:         and raise a ValueError if they do. Setting this parameter to
556:         False may silently produce nonsensical results if the input arrays
557:         do contain nans. Default is True.
558:     bounds : 2-tuple of array_like, optional
559:         Lower and upper bounds on independent variables. Defaults to no bounds.
560:         Each element of the tuple must be either an array with the length equal
561:         to the number of parameters, or a scalar (in which case the bound is
562:         taken to be the same for all parameters.) Use ``np.inf`` with an
563:         appropriate sign to disable bounds on all or some parameters.
564: 
565:         .. versionadded:: 0.17
566:     method : {'lm', 'trf', 'dogbox'}, optional
567:         Method to use for optimization.  See `least_squares` for more details.
568:         Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
569:         provided. The method 'lm' won't work when the number of observations
570:         is less than the number of variables, use 'trf' or 'dogbox' in this
571:         case.
572: 
573:         .. versionadded:: 0.17
574:     jac : callable, string or None, optional
575:         Function with signature ``jac(x, ...)`` which computes the Jacobian
576:         matrix of the model function with respect to parameters as a dense
577:         array_like structure. It will be scaled according to provided `sigma`.
578:         If None (default), the Jacobian will be estimated numerically.
579:         String keywords for 'trf' and 'dogbox' methods can be used to select
580:         a finite difference scheme, see `least_squares`.
581: 
582:         .. versionadded:: 0.18
583:     kwargs
584:         Keyword arguments passed to `leastsq` for ``method='lm'`` or
585:         `least_squares` otherwise.
586: 
587:     Returns
588:     -------
589:     popt : array
590:         Optimal values for the parameters so that the sum of the squared
591:         residuals of ``f(xdata, *popt) - ydata`` is minimized
592:     pcov : 2d array
593:         The estimated covariance of popt. The diagonals provide the variance
594:         of the parameter estimate. To compute one standard deviation errors
595:         on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
596: 
597:         How the `sigma` parameter affects the estimated covariance
598:         depends on `absolute_sigma` argument, as described above.
599: 
600:         If the Jacobian matrix at the solution doesn't have a full rank, then
601:         'lm' method returns a matrix filled with ``np.inf``, on the other hand
602:         'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
603:         the covariance matrix.
604: 
605:     Raises
606:     ------
607:     ValueError
608:         if either `ydata` or `xdata` contain NaNs, or if incompatible options
609:         are used.
610: 
611:     RuntimeError
612:         if the least-squares minimization fails.
613: 
614:     OptimizeWarning
615:         if covariance of the parameters can not be estimated.
616: 
617:     See Also
618:     --------
619:     least_squares : Minimize the sum of squares of nonlinear functions.
620:     scipy.stats.linregress : Calculate a linear least squares regression for
621:                              two sets of measurements.
622: 
623:     Notes
624:     -----
625:     With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm
626:     through `leastsq`. Note that this algorithm can only deal with
627:     unconstrained problems.
628: 
629:     Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to
630:     the docstring of `least_squares` for more information.
631: 
632:     Examples
633:     --------
634:     >>> import numpy as np
635:     >>> import matplotlib.pyplot as plt
636:     >>> from scipy.optimize import curve_fit
637: 
638:     >>> def func(x, a, b, c):
639:     ...     return a * np.exp(-b * x) + c
640: 
641:     Define the data to be fit with some noise:
642: 
643:     >>> xdata = np.linspace(0, 4, 50)
644:     >>> y = func(xdata, 2.5, 1.3, 0.5)
645:     >>> np.random.seed(1729)
646:     >>> y_noise = 0.2 * np.random.normal(size=xdata.size)
647:     >>> ydata = y + y_noise
648:     >>> plt.plot(xdata, ydata, 'b-', label='data')
649: 
650:     Fit for the parameters a, b, c of the function `func`:
651: 
652:     >>> popt, pcov = curve_fit(func, xdata, ydata)
653:     >>> popt
654:     array([ 2.55423706,  1.35190947,  0.47450618])
655:     >>> plt.plot(xdata, func(xdata, *popt), 'r-',
656:     ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
657: 
658:     Constrain the optimization to the region of ``0 <= a <= 3``,
659:     ``0 <= b <= 1`` and ``0 <= c <= 0.5``:
660: 
661:     >>> popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
662:     >>> popt
663:     array([ 2.43708906,  1.        ,  0.35015434])
664:     >>> plt.plot(xdata, func(xdata, *popt), 'g--',
665:     ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
666: 
667:     >>> plt.xlabel('x')
668:     >>> plt.ylabel('y')
669:     >>> plt.legend()
670:     >>> plt.show()
671: 
672:     '''
673:     if p0 is None:
674:         # determine number of parameters by inspecting the function
675:         from scipy._lib._util import getargspec_no_self as _getargspec
676:         args, varargs, varkw, defaults = _getargspec(f)
677:         if len(args) < 2:
678:             raise ValueError("Unable to determine number of fit parameters.")
679:         n = len(args) - 1
680:     else:
681:         p0 = np.atleast_1d(p0)
682:         n = p0.size
683: 
684:     lb, ub = prepare_bounds(bounds, n)
685:     if p0 is None:
686:         p0 = _initialize_feasible(lb, ub)
687: 
688:     bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
689:     if method is None:
690:         if bounded_problem:
691:             method = 'trf'
692:         else:
693:             method = 'lm'
694: 
695:     if method == 'lm' and bounded_problem:
696:         raise ValueError("Method 'lm' only works for unconstrained problems. "
697:                          "Use 'trf' or 'dogbox' instead.")
698: 
699:     # NaNs can not be handled
700:     if check_finite:
701:         ydata = np.asarray_chkfinite(ydata)
702:     else:
703:         ydata = np.asarray(ydata)
704: 
705:     if isinstance(xdata, (list, tuple, np.ndarray)):
706:         # `xdata` is passed straight to the user-defined `f`, so allow
707:         # non-array_like `xdata`.
708:         if check_finite:
709:             xdata = np.asarray_chkfinite(xdata)
710:         else:
711:             xdata = np.asarray(xdata)
712: 
713:     # Determine type of sigma
714:     if sigma is not None:
715:         sigma = np.asarray(sigma)
716: 
717:         # if 1-d, sigma are errors, define transform = 1/sigma
718:         if sigma.shape == (ydata.size, ):
719:             transform = 1.0 / sigma
720:         # if 2-d, sigma is the covariance matrix,
721:         # define transform = L such that L L^T = C
722:         elif sigma.shape == (ydata.size, ydata.size):
723:             try:
724:                 # scipy.linalg.cholesky requires lower=True to return L L^T = A
725:                 transform = cholesky(sigma, lower=True)
726:             except LinAlgError:
727:                 raise ValueError("`sigma` must be positive definite.")
728:         else:
729:             raise ValueError("`sigma` has incorrect shape.")
730:     else:
731:         transform = None
732: 
733:     func = _wrap_func(f, xdata, ydata, transform)
734:     if callable(jac):
735:         jac = _wrap_jac(jac, xdata, transform)
736:     elif jac is None and method != 'lm':
737:         jac = '2-point'
738: 
739:     if method == 'lm':
740:         # Remove full_output from kwargs, otherwise we're passing it in twice.
741:         return_full = kwargs.pop('full_output', False)
742:         res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
743:         popt, pcov, infodict, errmsg, ier = res
744:         cost = np.sum(infodict['fvec'] ** 2)
745:         if ier not in [1, 2, 3, 4]:
746:             raise RuntimeError("Optimal parameters not found: " + errmsg)
747:     else:
748:         # Rename maxfev (leastsq) to max_nfev (least_squares), if specified.
749:         if 'max_nfev' not in kwargs:
750:             kwargs['max_nfev'] = kwargs.pop('maxfev', None)
751: 
752:         res = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
753:                             **kwargs)
754: 
755:         if not res.success:
756:             raise RuntimeError("Optimal parameters not found: " + res.message)
757: 
758:         cost = 2 * res.cost  # res.cost is half sum of squares!
759:         popt = res.x
760: 
761:         # Do Moore-Penrose inverse discarding zero singular values.
762:         _, s, VT = svd(res.jac, full_matrices=False)
763:         threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
764:         s = s[s > threshold]
765:         VT = VT[:s.size]
766:         pcov = np.dot(VT.T / s**2, VT)
767:         return_full = False
768: 
769:     warn_cov = False
770:     if pcov is None:
771:         # indeterminate covariance
772:         pcov = zeros((len(popt), len(popt)), dtype=float)
773:         pcov.fill(inf)
774:         warn_cov = True
775:     elif not absolute_sigma:
776:         if ydata.size > p0.size:
777:             s_sq = cost / (ydata.size - p0.size)
778:             pcov = pcov * s_sq
779:         else:
780:             pcov.fill(inf)
781:             warn_cov = True
782: 
783:     if warn_cov:
784:         warnings.warn('Covariance of the parameters could not be estimated',
785:                       category=OptimizeWarning)
786: 
787:     if return_full:
788:         return popt, pcov, infodict, errmsg, ier
789:     else:
790:         return popt, pcov
791: 
792: 
793: def check_gradient(fcn, Dfcn, x0, args=(), col_deriv=0):
794:     '''Perform a simple check on the gradient for correctness.
795: 
796:     '''
797: 
798:     x = atleast_1d(x0)
799:     n = len(x)
800:     x = x.reshape((n,))
801:     fvec = atleast_1d(fcn(x, *args))
802:     m = len(fvec)
803:     fvec = fvec.reshape((m,))
804:     ldfjac = m
805:     fjac = atleast_1d(Dfcn(x, *args))
806:     fjac = fjac.reshape((m, n))
807:     if col_deriv == 0:
808:         fjac = transpose(fjac)
809: 
810:     xp = zeros((n,), float)
811:     err = zeros((m,), float)
812:     fvecp = None
813:     _minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 1, err)
814: 
815:     fvecp = atleast_1d(fcn(xp, *args))
816:     fvecp = fvecp.reshape((m,))
817:     _minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 2, err)
818: 
819:     good = (product(greater(err, 0.5), axis=0))
820: 
821:     return (good, err)
822: 
823: 
824: def _del2(p0, p1, d):
825:     return p0 - np.square(p1 - p0) / d
826: 
827: 
828: def _relerr(actual, desired):
829:     return (actual - desired) / desired
830: 
831: 
832: def _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel):
833:     p0 = x0
834:     for i in range(maxiter):
835:         p1 = func(p0, *args)
836:         if use_accel:
837:             p2 = func(p1, *args)
838:             d = p2 - 2.0 * p1 + p0
839:             p = _lazywhere(d != 0, (p0, p1, d), f=_del2, fillvalue=p2)
840:         else:
841:             p = p1
842:         relerr = _lazywhere(p0 != 0, (p, p0), f=_relerr, fillvalue=p)
843:         if np.all(np.abs(relerr) < xtol):
844:             return p
845:         p0 = p
846:     msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
847:     raise RuntimeError(msg)
848: 
849: 
850: def fixed_point(func, x0, args=(), xtol=1e-8, maxiter=500, method='del2'):
851:     '''
852:     Find a fixed point of the function.
853: 
854:     Given a function of one or more variables and a starting point, find a
855:     fixed-point of the function: i.e. where ``func(x0) == x0``.
856: 
857:     Parameters
858:     ----------
859:     func : function
860:         Function to evaluate.
861:     x0 : array_like
862:         Fixed point of function.
863:     args : tuple, optional
864:         Extra arguments to `func`.
865:     xtol : float, optional
866:         Convergence tolerance, defaults to 1e-08.
867:     maxiter : int, optional
868:         Maximum number of iterations, defaults to 500.
869:     method : {"del2", "iteration"}, optional
870:         Method of finding the fixed-point, defaults to "del2"
871:         which uses Steffensen's Method with Aitken's ``Del^2``
872:         convergence acceleration [1]_. The "iteration" method simply iterates
873:         the function until convergence is detected, without attempting to
874:         accelerate the convergence.
875: 
876:     References
877:     ----------
878:     .. [1] Burden, Faires, "Numerical Analysis", 5th edition, pg. 80
879: 
880:     Examples
881:     --------
882:     >>> from scipy import optimize
883:     >>> def func(x, c1, c2):
884:     ...    return np.sqrt(c1/(x+c2))
885:     >>> c1 = np.array([10,12.])
886:     >>> c2 = np.array([3, 5.])
887:     >>> optimize.fixed_point(func, [1.2, 1.3], args=(c1,c2))
888:     array([ 1.4920333 ,  1.37228132])
889: 
890:     '''
891:     use_accel = {'del2': True, 'iteration': False}[method]
892:     x0 = _asarray_validated(x0, as_inexact=True)
893:     return _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel)
894: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.optimize import _minpack' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171070 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize')

if (type(import_171070) is not StypyTypeError):

    if (import_171070 != 'pyd_module'):
        __import__(import_171070)
        sys_modules_171071 = sys.modules[import_171070]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', sys_modules_171071.module_type_store, module_type_store, ['_minpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_171071, sys_modules_171071.module_type_store, module_type_store)
    else:
        from scipy.optimize import _minpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', None, module_type_store, ['_minpack'], [_minpack])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', import_171070)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_171072) is not StypyTypeError):

    if (import_171072 != 'pyd_module'):
        __import__(import_171072)
        sys_modules_171073 = sys.modules[import_171072]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_171073.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_171072)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy import atleast_1d, dot, take, triu, shape, eye, transpose, zeros, product, greater, array, all, where, isscalar, asarray, inf, abs, finfo, inexact, issubdtype, dtype' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_171074) is not StypyTypeError):

    if (import_171074 != 'pyd_module'):
        __import__(import_171074)
        sys_modules_171075 = sys.modules[import_171074]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', sys_modules_171075.module_type_store, module_type_store, ['atleast_1d', 'dot', 'take', 'triu', 'shape', 'eye', 'transpose', 'zeros', 'product', 'greater', 'array', 'all', 'where', 'isscalar', 'asarray', 'inf', 'abs', 'finfo', 'inexact', 'issubdtype', 'dtype'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_171075, sys_modules_171075.module_type_store, module_type_store)
    else:
        from numpy import atleast_1d, dot, take, triu, shape, eye, transpose, zeros, product, greater, array, all, where, isscalar, asarray, inf, abs, finfo, inexact, issubdtype, dtype

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', None, module_type_store, ['atleast_1d', 'dot', 'take', 'triu', 'shape', 'eye', 'transpose', 'zeros', 'product', 'greater', 'array', 'all', 'where', 'isscalar', 'asarray', 'inf', 'abs', 'finfo', 'inexact', 'issubdtype', 'dtype'], [atleast_1d, dot, take, triu, shape, eye, transpose, zeros, product, greater, array, all, where, isscalar, asarray, inf, abs, finfo, inexact, issubdtype, dtype])

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_171074)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171076 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg')

if (type(import_171076) is not StypyTypeError):

    if (import_171076 != 'pyd_module'):
        __import__(import_171076)
        sys_modules_171077 = sys.modules[import_171076]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', sys_modules_171077.module_type_store, module_type_store, ['svd', 'cholesky', 'solve_triangular', 'LinAlgError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_171077, sys_modules_171077.module_type_store, module_type_store)
    else:
        from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', None, module_type_store, ['svd', 'cholesky', 'solve_triangular', 'LinAlgError'], [svd, cholesky, solve_triangular, LinAlgError])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', import_171076)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy._lib._util import _asarray_validated, _lazywhere' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171078 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._util')

if (type(import_171078) is not StypyTypeError):

    if (import_171078 != 'pyd_module'):
        __import__(import_171078)
        sys_modules_171079 = sys.modules[import_171078]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._util', sys_modules_171079.module_type_store, module_type_store, ['_asarray_validated', '_lazywhere'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_171079, sys_modules_171079.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _asarray_validated, _lazywhere

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._util', None, module_type_store, ['_asarray_validated', '_lazywhere'], [_asarray_validated, _lazywhere])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib._util', import_171078)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.optimize.optimize import OptimizeResult, _check_unknown_options, OptimizeWarning' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171080 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.optimize.optimize')

if (type(import_171080) is not StypyTypeError):

    if (import_171080 != 'pyd_module'):
        __import__(import_171080)
        sys_modules_171081 = sys.modules[import_171080]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.optimize.optimize', sys_modules_171081.module_type_store, module_type_store, ['OptimizeResult', '_check_unknown_options', 'OptimizeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_171081, sys_modules_171081.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import OptimizeResult, _check_unknown_options, OptimizeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.optimize.optimize', None, module_type_store, ['OptimizeResult', '_check_unknown_options', 'OptimizeWarning'], [OptimizeResult, _check_unknown_options, OptimizeWarning])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.optimize.optimize', import_171080)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.optimize._lsq import least_squares' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171082 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.optimize._lsq')

if (type(import_171082) is not StypyTypeError):

    if (import_171082 != 'pyd_module'):
        __import__(import_171082)
        sys_modules_171083 = sys.modules[import_171082]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.optimize._lsq', sys_modules_171083.module_type_store, module_type_store, ['least_squares'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_171083, sys_modules_171083.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq import least_squares

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.optimize._lsq', None, module_type_store, ['least_squares'], [least_squares])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.optimize._lsq', import_171082)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.optimize._lsq.common import make_strictly_feasible' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171084 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize._lsq.common')

if (type(import_171084) is not StypyTypeError):

    if (import_171084 != 'pyd_module'):
        __import__(import_171084)
        sys_modules_171085 = sys.modules[import_171084]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize._lsq.common', sys_modules_171085.module_type_store, module_type_store, ['make_strictly_feasible'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_171085, sys_modules_171085.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.common import make_strictly_feasible

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize._lsq.common', None, module_type_store, ['make_strictly_feasible'], [make_strictly_feasible])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.common' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.optimize._lsq.common', import_171084)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.optimize._lsq.least_squares import prepare_bounds' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_171086 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.least_squares')

if (type(import_171086) is not StypyTypeError):

    if (import_171086 != 'pyd_module'):
        __import__(import_171086)
        sys_modules_171087 = sys.modules[import_171086]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.least_squares', sys_modules_171087.module_type_store, module_type_store, ['prepare_bounds'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_171087, sys_modules_171087.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.least_squares import prepare_bounds

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.least_squares', None, module_type_store, ['prepare_bounds'], [prepare_bounds])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.least_squares' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize._lsq.least_squares', import_171086)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a Attribute to a Name (line 19):

# Assigning a Attribute to a Name (line 19):
# Getting the type of '_minpack' (line 19)
_minpack_171088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), '_minpack')
# Obtaining the member 'error' of a type (line 19)
error_171089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), _minpack_171088, 'error')
# Assigning a type to the variable 'error' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'error', error_171089)

# Assigning a List to a Name (line 21):

# Assigning a List to a Name (line 21):
__all__ = ['fsolve', 'leastsq', 'fixed_point', 'curve_fit']
module_type_store.set_exportable_members(['fsolve', 'leastsq', 'fixed_point', 'curve_fit'])

# Obtaining an instance of the builtin type 'list' (line 21)
list_171090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_171091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'fsolve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_171090, str_171091)
# Adding element type (line 21)
str_171092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'str', 'leastsq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_171090, str_171092)
# Adding element type (line 21)
str_171093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'str', 'fixed_point')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_171090, str_171093)
# Adding element type (line 21)
str_171094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 47), 'str', 'curve_fit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_171090, str_171094)

# Assigning a type to the variable '__all__' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '__all__', list_171090)

@norecursion
def _check_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 25)
    None_171095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'None')
    defaults = [None_171095]
    # Create a new context for function '_check_func'
    module_type_store = module_type_store.open_function_context('_check_func', 24, 0, False)
    
    # Passed parameters checking function
    _check_func.stypy_localization = localization
    _check_func.stypy_type_of_self = None
    _check_func.stypy_type_store = module_type_store
    _check_func.stypy_function_name = '_check_func'
    _check_func.stypy_param_names_list = ['checker', 'argname', 'thefunc', 'x0', 'args', 'numinputs', 'output_shape']
    _check_func.stypy_varargs_param_name = None
    _check_func.stypy_kwargs_param_name = None
    _check_func.stypy_call_defaults = defaults
    _check_func.stypy_call_varargs = varargs
    _check_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_func', ['checker', 'argname', 'thefunc', 'x0', 'args', 'numinputs', 'output_shape'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_func', localization, ['checker', 'argname', 'thefunc', 'x0', 'args', 'numinputs', 'output_shape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_func(...)' code ##################

    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to atleast_1d(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to thefunc(...): (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_171098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    
    # Obtaining the type of the subscript
    # Getting the type of 'numinputs' (line 26)
    numinputs_171099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'numinputs', False)
    slice_171100 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 26, 32), None, numinputs_171099, None)
    # Getting the type of 'x0' (line 26)
    x0_171101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 32), 'x0', False)
    # Obtaining the member '__getitem__' of a type (line 26)
    getitem___171102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 32), x0_171101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 26)
    subscript_call_result_171103 = invoke(stypy.reporting.localization.Localization(__file__, 26, 32), getitem___171102, slice_171100)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 32), tuple_171098, subscript_call_result_171103)
    
    # Getting the type of 'args' (line 26)
    args_171104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 51), 'args', False)
    # Applying the binary operator '+' (line 26)
    result_add_171105 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 31), '+', tuple_171098, args_171104)
    
    # Processing the call keyword arguments (line 26)
    kwargs_171106 = {}
    # Getting the type of 'thefunc' (line 26)
    thefunc_171097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'thefunc', False)
    # Calling thefunc(args, kwargs) (line 26)
    thefunc_call_result_171107 = invoke(stypy.reporting.localization.Localization(__file__, 26, 21), thefunc_171097, *[result_add_171105], **kwargs_171106)
    
    # Processing the call keyword arguments (line 26)
    kwargs_171108 = {}
    # Getting the type of 'atleast_1d' (line 26)
    atleast_1d_171096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 26)
    atleast_1d_call_result_171109 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), atleast_1d_171096, *[thefunc_call_result_171107], **kwargs_171108)
    
    # Assigning a type to the variable 'res' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'res', atleast_1d_call_result_171109)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'output_shape' (line 27)
    output_shape_171110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'output_shape')
    # Getting the type of 'None' (line 27)
    None_171111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'None')
    # Applying the binary operator 'isnot' (line 27)
    result_is_not_171112 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 8), 'isnot', output_shape_171110, None_171111)
    
    
    
    # Call to shape(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'res' (line 27)
    res_171114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 45), 'res', False)
    # Processing the call keyword arguments (line 27)
    kwargs_171115 = {}
    # Getting the type of 'shape' (line 27)
    shape_171113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 39), 'shape', False)
    # Calling shape(args, kwargs) (line 27)
    shape_call_result_171116 = invoke(stypy.reporting.localization.Localization(__file__, 27, 39), shape_171113, *[res_171114], **kwargs_171115)
    
    # Getting the type of 'output_shape' (line 27)
    output_shape_171117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 53), 'output_shape')
    # Applying the binary operator '!=' (line 27)
    result_ne_171118 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 39), '!=', shape_call_result_171116, output_shape_171117)
    
    # Applying the binary operator 'and' (line 27)
    result_and_keyword_171119 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 7), 'and', result_is_not_171112, result_ne_171118)
    
    # Testing the type of an if condition (line 27)
    if_condition_171120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), result_and_keyword_171119)
    # Assigning a type to the variable 'if_condition_171120' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_171120', if_condition_171120)
    # SSA begins for if statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_171121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
    # Getting the type of 'output_shape' (line 28)
    output_shape_171122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'output_shape')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___171123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), output_shape_171122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_171124 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), getitem___171123, int_171121)
    
    int_171125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'int')
    # Applying the binary operator '!=' (line 28)
    result_ne_171126 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 12), '!=', subscript_call_result_171124, int_171125)
    
    # Testing the type of an if condition (line 28)
    if_condition_171127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 8), result_ne_171126)
    # Assigning a type to the variable 'if_condition_171127' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'if_condition_171127', if_condition_171127)
    # SSA begins for if statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to len(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'output_shape' (line 29)
    output_shape_171129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'output_shape', False)
    # Processing the call keyword arguments (line 29)
    kwargs_171130 = {}
    # Getting the type of 'len' (line 29)
    len_171128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'len', False)
    # Calling len(args, kwargs) (line 29)
    len_call_result_171131 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), len_171128, *[output_shape_171129], **kwargs_171130)
    
    int_171132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 35), 'int')
    # Applying the binary operator '>' (line 29)
    result_gt_171133 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), '>', len_call_result_171131, int_171132)
    
    # Testing the type of an if condition (line 29)
    if_condition_171134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 12), result_gt_171133)
    # Assigning a type to the variable 'if_condition_171134' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'if_condition_171134', if_condition_171134)
    # SSA begins for if statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_171135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 32), 'int')
    # Getting the type of 'output_shape' (line 30)
    output_shape_171136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'output_shape')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___171137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 19), output_shape_171136, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_171138 = invoke(stypy.reporting.localization.Localization(__file__, 30, 19), getitem___171137, int_171135)
    
    int_171139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'int')
    # Applying the binary operator '==' (line 30)
    result_eq_171140 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 19), '==', subscript_call_result_171138, int_171139)
    
    # Testing the type of an if condition (line 30)
    if_condition_171141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 16), result_eq_171140)
    # Assigning a type to the variable 'if_condition_171141' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'if_condition_171141', if_condition_171141)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to shape(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'res' (line 31)
    res_171143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'res', False)
    # Processing the call keyword arguments (line 31)
    kwargs_171144 = {}
    # Getting the type of 'shape' (line 31)
    shape_171142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'shape', False)
    # Calling shape(args, kwargs) (line 31)
    shape_call_result_171145 = invoke(stypy.reporting.localization.Localization(__file__, 31, 27), shape_171142, *[res_171143], **kwargs_171144)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'stypy_return_type', shape_call_result_171145)
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 29)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 32):
    
    # Assigning a BinOp to a Name (line 32):
    str_171146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'str', "%s: there is a mismatch between the input and output shape of the '%s' argument")
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_171147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    # Getting the type of 'checker' (line 33)
    checker_171148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 50), 'checker')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 50), tuple_171147, checker_171148)
    # Adding element type (line 33)
    # Getting the type of 'argname' (line 33)
    argname_171149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 59), 'argname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 50), tuple_171147, argname_171149)
    
    # Applying the binary operator '%' (line 32)
    result_mod_171150 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 18), '%', str_171146, tuple_171147)
    
    # Assigning a type to the variable 'msg' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'msg', result_mod_171150)
    
    # Assigning a Call to a Name (line 34):
    
    # Assigning a Call to a Name (line 34):
    
    # Call to getattr(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'thefunc' (line 34)
    thefunc_171152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'thefunc', False)
    str_171153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 41), 'str', '__name__')
    # Getting the type of 'None' (line 34)
    None_171154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 53), 'None', False)
    # Processing the call keyword arguments (line 34)
    kwargs_171155 = {}
    # Getting the type of 'getattr' (line 34)
    getattr_171151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'getattr', False)
    # Calling getattr(args, kwargs) (line 34)
    getattr_call_result_171156 = invoke(stypy.reporting.localization.Localization(__file__, 34, 24), getattr_171151, *[thefunc_171152, str_171153, None_171154], **kwargs_171155)
    
    # Assigning a type to the variable 'func_name' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'func_name', getattr_call_result_171156)
    
    # Getting the type of 'func_name' (line 35)
    func_name_171157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'func_name')
    # Testing the type of an if condition (line 35)
    if_condition_171158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), func_name_171157)
    # Assigning a type to the variable 'if_condition_171158' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_171158', if_condition_171158)
    # SSA begins for if statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'msg' (line 36)
    msg_171159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'msg')
    str_171160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', " '%s'.")
    # Getting the type of 'func_name' (line 36)
    func_name_171161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'func_name')
    # Applying the binary operator '%' (line 36)
    result_mod_171162 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), '%', str_171160, func_name_171161)
    
    # Applying the binary operator '+=' (line 36)
    result_iadd_171163 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 16), '+=', msg_171159, result_mod_171162)
    # Assigning a type to the variable 'msg' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'msg', result_iadd_171163)
    
    # SSA branch for the else part of an if statement (line 35)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'msg' (line 38)
    msg_171164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'msg')
    str_171165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'str', '.')
    # Applying the binary operator '+=' (line 38)
    result_iadd_171166 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 16), '+=', msg_171164, str_171165)
    # Assigning a type to the variable 'msg' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'msg', result_iadd_171166)
    
    # SSA join for if statement (line 35)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'msg' (line 39)
    msg_171167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'msg')
    str_171168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'str', 'Shape should be %s but it is %s.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 39)
    tuple_171169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'output_shape' (line 39)
    output_shape_171170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 57), 'output_shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 57), tuple_171169, output_shape_171170)
    # Adding element type (line 39)
    
    # Call to shape(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'res' (line 39)
    res_171172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 77), 'res', False)
    # Processing the call keyword arguments (line 39)
    kwargs_171173 = {}
    # Getting the type of 'shape' (line 39)
    shape_171171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 71), 'shape', False)
    # Calling shape(args, kwargs) (line 39)
    shape_call_result_171174 = invoke(stypy.reporting.localization.Localization(__file__, 39, 71), shape_171171, *[res_171172], **kwargs_171173)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 57), tuple_171169, shape_call_result_171174)
    
    # Applying the binary operator '%' (line 39)
    result_mod_171175 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 19), '%', str_171168, tuple_171169)
    
    # Applying the binary operator '+=' (line 39)
    result_iadd_171176 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 12), '+=', msg_171167, result_mod_171175)
    # Assigning a type to the variable 'msg' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'msg', result_iadd_171176)
    
    
    # Call to TypeError(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'msg' (line 40)
    msg_171178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'msg', False)
    # Processing the call keyword arguments (line 40)
    kwargs_171179 = {}
    # Getting the type of 'TypeError' (line 40)
    TypeError_171177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 40)
    TypeError_call_result_171180 = invoke(stypy.reporting.localization.Localization(__file__, 40, 18), TypeError_171177, *[msg_171178], **kwargs_171179)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 40, 12), TypeError_call_result_171180, 'raise parameter', BaseException)
    # SSA join for if statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issubdtype(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'res' (line 41)
    res_171182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'res', False)
    # Obtaining the member 'dtype' of a type (line 41)
    dtype_171183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 18), res_171182, 'dtype')
    # Getting the type of 'inexact' (line 41)
    inexact_171184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 29), 'inexact', False)
    # Processing the call keyword arguments (line 41)
    kwargs_171185 = {}
    # Getting the type of 'issubdtype' (line 41)
    issubdtype_171181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'issubdtype', False)
    # Calling issubdtype(args, kwargs) (line 41)
    issubdtype_call_result_171186 = invoke(stypy.reporting.localization.Localization(__file__, 41, 7), issubdtype_171181, *[dtype_171183, inexact_171184], **kwargs_171185)
    
    # Testing the type of an if condition (line 41)
    if_condition_171187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), issubdtype_call_result_171186)
    # Assigning a type to the variable 'if_condition_171187' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_171187', if_condition_171187)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 42):
    
    # Assigning a Attribute to a Name (line 42):
    # Getting the type of 'res' (line 42)
    res_171188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'res')
    # Obtaining the member 'dtype' of a type (line 42)
    dtype_171189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), res_171188, 'dtype')
    # Assigning a type to the variable 'dt' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'dt', dtype_171189)
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to dtype(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'float' (line 44)
    float_171191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'float', False)
    # Processing the call keyword arguments (line 44)
    kwargs_171192 = {}
    # Getting the type of 'dtype' (line 44)
    dtype_171190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'dtype', False)
    # Calling dtype(args, kwargs) (line 44)
    dtype_call_result_171193 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), dtype_171190, *[float_171191], **kwargs_171192)
    
    # Assigning a type to the variable 'dt' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'dt', dtype_call_result_171193)
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_171194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    
    # Call to shape(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'res' (line 45)
    res_171196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'res', False)
    # Processing the call keyword arguments (line 45)
    kwargs_171197 = {}
    # Getting the type of 'shape' (line 45)
    shape_171195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'shape', False)
    # Calling shape(args, kwargs) (line 45)
    shape_call_result_171198 = invoke(stypy.reporting.localization.Localization(__file__, 45, 11), shape_171195, *[res_171196], **kwargs_171197)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 11), tuple_171194, shape_call_result_171198)
    # Adding element type (line 45)
    # Getting the type of 'dt' (line 45)
    dt_171199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 11), tuple_171194, dt_171199)
    
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type', tuple_171194)
    
    # ################# End of '_check_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_func' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_171200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171200)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_func'
    return stypy_return_type_171200

# Assigning a type to the variable '_check_func' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '_check_func', _check_func)

@norecursion
def fsolve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_171201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    
    # Getting the type of 'None' (line 48)
    None_171202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'None')
    int_171203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 55), 'int')
    int_171204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'int')
    float_171205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'float')
    int_171206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 48), 'int')
    # Getting the type of 'None' (line 49)
    None_171207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 56), 'None')
    # Getting the type of 'None' (line 50)
    None_171208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'None')
    int_171209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 31), 'int')
    # Getting the type of 'None' (line 50)
    None_171210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), 'None')
    defaults = [tuple_171201, None_171202, int_171203, int_171204, float_171205, int_171206, None_171207, None_171208, int_171209, None_171210]
    # Create a new context for function 'fsolve'
    module_type_store = module_type_store.open_function_context('fsolve', 48, 0, False)
    
    # Passed parameters checking function
    fsolve.stypy_localization = localization
    fsolve.stypy_type_of_self = None
    fsolve.stypy_type_store = module_type_store
    fsolve.stypy_function_name = 'fsolve'
    fsolve.stypy_param_names_list = ['func', 'x0', 'args', 'fprime', 'full_output', 'col_deriv', 'xtol', 'maxfev', 'band', 'epsfcn', 'factor', 'diag']
    fsolve.stypy_varargs_param_name = None
    fsolve.stypy_kwargs_param_name = None
    fsolve.stypy_call_defaults = defaults
    fsolve.stypy_call_varargs = varargs
    fsolve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fsolve', ['func', 'x0', 'args', 'fprime', 'full_output', 'col_deriv', 'xtol', 'maxfev', 'band', 'epsfcn', 'factor', 'diag'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fsolve', localization, ['func', 'x0', 'args', 'fprime', 'full_output', 'col_deriv', 'xtol', 'maxfev', 'band', 'epsfcn', 'factor', 'diag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fsolve(...)' code ##################

    str_171211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, (-1)), 'str', "\n    Find the roots of a function.\n\n    Return the roots of the (non-linear) equations defined by\n    ``func(x) = 0`` given a starting estimate.\n\n    Parameters\n    ----------\n    func : callable ``f(x, *args)``\n        A function that takes at least one (possibly vector) argument.\n    x0 : ndarray\n        The starting estimate for the roots of ``func(x) = 0``.\n    args : tuple, optional\n        Any extra arguments to `func`.\n    fprime : callable ``f(x, *args)``, optional\n        A function to compute the Jacobian of `func` with derivatives\n        across the rows. By default, the Jacobian will be estimated.\n    full_output : bool, optional\n        If True, return optional outputs.\n    col_deriv : bool, optional\n        Specify whether the Jacobian function computes derivatives down\n        the columns (faster, because there is no transpose operation).\n    xtol : float, optional\n        The calculation will terminate if the relative error between two\n        consecutive iterates is at most `xtol`.\n    maxfev : int, optional\n        The maximum number of calls to the function. If zero, then\n        ``100*(N+1)`` is the maximum where N is the number of elements\n        in `x0`.\n    band : tuple, optional\n        If set to a two-sequence containing the number of sub- and\n        super-diagonals within the band of the Jacobi matrix, the\n        Jacobi matrix is considered banded (only for ``fprime=None``).\n    epsfcn : float, optional\n        A suitable step length for the forward-difference\n        approximation of the Jacobian (for ``fprime=None``). If\n        `epsfcn` is less than the machine precision, it is assumed\n        that the relative errors in the functions are of the order of\n        the machine precision.\n    factor : float, optional\n        A parameter determining the initial step bound\n        (``factor * || diag * x||``).  Should be in the interval\n        ``(0.1, 100)``.\n    diag : sequence, optional\n        N positive entries that serve as a scale factors for the\n        variables.\n\n    Returns\n    -------\n    x : ndarray\n        The solution (or the result of the last iteration for\n        an unsuccessful call).\n    infodict : dict\n        A dictionary of optional outputs with the keys:\n\n        ``nfev``\n            number of function calls\n        ``njev``\n            number of Jacobian calls\n        ``fvec``\n            function evaluated at the output\n        ``fjac``\n            the orthogonal matrix, q, produced by the QR\n            factorization of the final approximate Jacobian\n            matrix, stored column wise\n        ``r``\n            upper triangular matrix produced by QR factorization\n            of the same matrix\n        ``qtf``\n            the vector ``(transpose(q) * fvec)``\n\n    ier : int\n        An integer flag.  Set to 1 if a solution was found, otherwise refer\n        to `mesg` for more information.\n    mesg : str\n        If no solution is found, `mesg` details the cause of failure.\n\n    See Also\n    --------\n    root : Interface to root finding algorithms for multivariate\n    functions. See the 'hybr' `method` in particular.\n\n    Notes\n    -----\n    ``fsolve`` is a wrapper around MINPACK's hybrd and hybrj algorithms.\n\n    ")
    
    # Assigning a Dict to a Name (line 138):
    
    # Assigning a Dict to a Name (line 138):
    
    # Obtaining an instance of the builtin type 'dict' (line 138)
    dict_171212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 138)
    # Adding element type (key, value) (line 138)
    str_171213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 15), 'str', 'col_deriv')
    # Getting the type of 'col_deriv' (line 138)
    col_deriv_171214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'col_deriv')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 14), dict_171212, (str_171213, col_deriv_171214))
    # Adding element type (key, value) (line 138)
    str_171215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 15), 'str', 'xtol')
    # Getting the type of 'xtol' (line 139)
    xtol_171216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'xtol')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 14), dict_171212, (str_171215, xtol_171216))
    # Adding element type (key, value) (line 138)
    str_171217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 15), 'str', 'maxfev')
    # Getting the type of 'maxfev' (line 140)
    maxfev_171218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'maxfev')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 14), dict_171212, (str_171217, maxfev_171218))
    # Adding element type (key, value) (line 138)
    str_171219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'str', 'band')
    # Getting the type of 'band' (line 141)
    band_171220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'band')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 14), dict_171212, (str_171219, band_171220))
    # Adding element type (key, value) (line 138)
    str_171221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'str', 'eps')
    # Getting the type of 'epsfcn' (line 142)
    epsfcn_171222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), 'epsfcn')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 14), dict_171212, (str_171221, epsfcn_171222))
    # Adding element type (key, value) (line 138)
    str_171223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'str', 'factor')
    # Getting the type of 'factor' (line 143)
    factor_171224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'factor')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 14), dict_171212, (str_171223, factor_171224))
    # Adding element type (key, value) (line 138)
    str_171225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'str', 'diag')
    # Getting the type of 'diag' (line 144)
    diag_171226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), 'diag')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 14), dict_171212, (str_171225, diag_171226))
    
    # Assigning a type to the variable 'options' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'options', dict_171212)
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to _root_hybr(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'func' (line 146)
    func_171228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 21), 'func', False)
    # Getting the type of 'x0' (line 146)
    x0_171229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'x0', False)
    # Getting the type of 'args' (line 146)
    args_171230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 31), 'args', False)
    # Processing the call keyword arguments (line 146)
    # Getting the type of 'fprime' (line 146)
    fprime_171231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 41), 'fprime', False)
    keyword_171232 = fprime_171231
    # Getting the type of 'options' (line 146)
    options_171233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 51), 'options', False)
    kwargs_171234 = {'options_171233': options_171233, 'jac': keyword_171232}
    # Getting the type of '_root_hybr' (line 146)
    _root_hybr_171227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 10), '_root_hybr', False)
    # Calling _root_hybr(args, kwargs) (line 146)
    _root_hybr_call_result_171235 = invoke(stypy.reporting.localization.Localization(__file__, 146, 10), _root_hybr_171227, *[func_171228, x0_171229, args_171230], **kwargs_171234)
    
    # Assigning a type to the variable 'res' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'res', _root_hybr_call_result_171235)
    
    # Getting the type of 'full_output' (line 147)
    full_output_171236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'full_output')
    # Testing the type of an if condition (line 147)
    if_condition_171237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), full_output_171236)
    # Assigning a type to the variable 'if_condition_171237' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_171237', if_condition_171237)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 148):
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    str_171238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 16), 'str', 'x')
    # Getting the type of 'res' (line 148)
    res_171239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'res')
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___171240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), res_171239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_171241 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), getitem___171240, str_171238)
    
    # Assigning a type to the variable 'x' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'x', subscript_call_result_171241)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to dict(...): (line 149)
    # Processing the call arguments (line 149)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 149, 20, True)
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'tuple' (line 150)
    tuple_171253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 150)
    # Adding element type (line 150)
    str_171254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 30), 'str', 'nfev')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 30), tuple_171253, str_171254)
    # Adding element type (line 150)
    str_171255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 38), 'str', 'njev')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 30), tuple_171253, str_171255)
    # Adding element type (line 150)
    str_171256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 46), 'str', 'fjac')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 30), tuple_171253, str_171256)
    # Adding element type (line 150)
    str_171257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 54), 'str', 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 30), tuple_171253, str_171257)
    # Adding element type (line 150)
    str_171258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 59), 'str', 'qtf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 30), tuple_171253, str_171258)
    
    comprehension_171259 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 20), tuple_171253)
    # Assigning a type to the variable 'k' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'k', comprehension_171259)
    
    # Getting the type of 'k' (line 150)
    k_171250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 69), 'k', False)
    # Getting the type of 'res' (line 150)
    res_171251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 74), 'res', False)
    # Applying the binary operator 'in' (line 150)
    result_contains_171252 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 69), 'in', k_171250, res_171251)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_171243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    # Getting the type of 'k' (line 149)
    k_171244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 21), tuple_171243, k_171244)
    # Adding element type (line 149)
    
    # Call to get(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'k' (line 149)
    k_171247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'k', False)
    # Processing the call keyword arguments (line 149)
    kwargs_171248 = {}
    # Getting the type of 'res' (line 149)
    res_171245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'res', False)
    # Obtaining the member 'get' of a type (line 149)
    get_171246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), res_171245, 'get')
    # Calling get(args, kwargs) (line 149)
    get_call_result_171249 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), get_171246, *[k_171247], **kwargs_171248)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 21), tuple_171243, get_call_result_171249)
    
    list_171260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 20), list_171260, tuple_171243)
    # Processing the call keyword arguments (line 149)
    kwargs_171261 = {}
    # Getting the type of 'dict' (line 149)
    dict_171242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'dict', False)
    # Calling dict(args, kwargs) (line 149)
    dict_call_result_171262 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), dict_171242, *[list_171260], **kwargs_171261)
    
    # Assigning a type to the variable 'info' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'info', dict_call_result_171262)
    
    # Assigning a Subscript to a Subscript (line 151):
    
    # Assigning a Subscript to a Subscript (line 151):
    
    # Obtaining the type of the subscript
    str_171263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 27), 'str', 'fun')
    # Getting the type of 'res' (line 151)
    res_171264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'res')
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___171265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 23), res_171264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_171266 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), getitem___171265, str_171263)
    
    # Getting the type of 'info' (line 151)
    info_171267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'info')
    str_171268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 13), 'str', 'fvec')
    # Storing an element on a container (line 151)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 8), info_171267, (str_171268, subscript_call_result_171266))
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_171269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    # Getting the type of 'x' (line 152)
    x_171270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_171269, x_171270)
    # Adding element type (line 152)
    # Getting the type of 'info' (line 152)
    info_171271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_171269, info_171271)
    # Adding element type (line 152)
    
    # Obtaining the type of the subscript
    str_171272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'str', 'status')
    # Getting the type of 'res' (line 152)
    res_171273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'res')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___171274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 24), res_171273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_171275 = invoke(stypy.reporting.localization.Localization(__file__, 152, 24), getitem___171274, str_171272)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_171269, subscript_call_result_171275)
    # Adding element type (line 152)
    
    # Obtaining the type of the subscript
    str_171276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 43), 'str', 'message')
    # Getting the type of 'res' (line 152)
    res_171277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'res')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___171278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 39), res_171277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_171279 = invoke(stypy.reporting.localization.Localization(__file__, 152, 39), getitem___171278, str_171276)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_171269, subscript_call_result_171279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'stypy_return_type', tuple_171269)
    # SSA branch for the else part of an if statement (line 147)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 154):
    
    # Assigning a Subscript to a Name (line 154):
    
    # Obtaining the type of the subscript
    str_171280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'str', 'status')
    # Getting the type of 'res' (line 154)
    res_171281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'res')
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___171282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 17), res_171281, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_171283 = invoke(stypy.reporting.localization.Localization(__file__, 154, 17), getitem___171282, str_171280)
    
    # Assigning a type to the variable 'status' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'status', subscript_call_result_171283)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    str_171284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'str', 'message')
    # Getting the type of 'res' (line 155)
    res_171285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'res')
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___171286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 14), res_171285, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_171287 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), getitem___171286, str_171284)
    
    # Assigning a type to the variable 'msg' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'msg', subscript_call_result_171287)
    
    
    # Getting the type of 'status' (line 156)
    status_171288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'status')
    int_171289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'int')
    # Applying the binary operator '==' (line 156)
    result_eq_171290 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), '==', status_171288, int_171289)
    
    # Testing the type of an if condition (line 156)
    if_condition_171291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_eq_171290)
    # Assigning a type to the variable 'if_condition_171291' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_171291', if_condition_171291)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'msg' (line 157)
    msg_171293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'msg', False)
    # Processing the call keyword arguments (line 157)
    kwargs_171294 = {}
    # Getting the type of 'TypeError' (line 157)
    TypeError_171292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 157)
    TypeError_call_result_171295 = invoke(stypy.reporting.localization.Localization(__file__, 157, 18), TypeError_171292, *[msg_171293], **kwargs_171294)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 157, 12), TypeError_call_result_171295, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 156)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'status' (line 158)
    status_171296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 13), 'status')
    int_171297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'int')
    # Applying the binary operator '==' (line 158)
    result_eq_171298 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 13), '==', status_171296, int_171297)
    
    # Testing the type of an if condition (line 158)
    if_condition_171299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 13), result_eq_171298)
    # Assigning a type to the variable 'if_condition_171299' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 13), 'if_condition_171299', if_condition_171299)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 158)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'status' (line 160)
    status_171300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'status')
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_171301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    # Adding element type (line 160)
    int_171302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 23), list_171301, int_171302)
    # Adding element type (line 160)
    int_171303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 23), list_171301, int_171303)
    # Adding element type (line 160)
    int_171304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 23), list_171301, int_171304)
    # Adding element type (line 160)
    int_171305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 23), list_171301, int_171305)
    
    # Applying the binary operator 'in' (line 160)
    result_contains_171306 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 13), 'in', status_171300, list_171301)
    
    # Testing the type of an if condition (line 160)
    if_condition_171307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 13), result_contains_171306)
    # Assigning a type to the variable 'if_condition_171307' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'if_condition_171307', if_condition_171307)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'msg' (line 161)
    msg_171310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'msg', False)
    # Getting the type of 'RuntimeWarning' (line 161)
    RuntimeWarning_171311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 161)
    kwargs_171312 = {}
    # Getting the type of 'warnings' (line 161)
    warnings_171308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 161)
    warn_171309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), warnings_171308, 'warn')
    # Calling warn(args, kwargs) (line 161)
    warn_call_result_171313 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), warn_171309, *[msg_171310, RuntimeWarning_171311], **kwargs_171312)
    
    # SSA branch for the else part of an if statement (line 160)
    module_type_store.open_ssa_branch('else')
    
    # Call to TypeError(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'msg' (line 163)
    msg_171315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'msg', False)
    # Processing the call keyword arguments (line 163)
    kwargs_171316 = {}
    # Getting the type of 'TypeError' (line 163)
    TypeError_171314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 163)
    TypeError_call_result_171317 = invoke(stypy.reporting.localization.Localization(__file__, 163, 18), TypeError_171314, *[msg_171315], **kwargs_171316)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 163, 12), TypeError_call_result_171317, 'raise parameter', BaseException)
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    str_171318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 19), 'str', 'x')
    # Getting the type of 'res' (line 164)
    res_171319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'res')
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___171320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 15), res_171319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_171321 = invoke(stypy.reporting.localization.Localization(__file__, 164, 15), getitem___171320, str_171318)
    
    # Assigning a type to the variable 'stypy_return_type' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', subscript_call_result_171321)
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'fsolve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fsolve' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_171322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171322)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fsolve'
    return stypy_return_type_171322

# Assigning a type to the variable 'fsolve' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'fsolve', fsolve)

@norecursion
def _root_hybr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_171323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    
    # Getting the type of 'None' (line 167)
    None_171324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 38), 'None')
    int_171325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 25), 'int')
    float_171326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 33), 'float')
    int_171327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 53), 'int')
    # Getting the type of 'None' (line 168)
    None_171328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 61), 'None')
    # Getting the type of 'None' (line 168)
    None_171329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 71), 'None')
    int_171330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'int')
    # Getting the type of 'None' (line 169)
    None_171331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 32), 'None')
    defaults = [tuple_171323, None_171324, int_171325, float_171326, int_171327, None_171328, None_171329, int_171330, None_171331]
    # Create a new context for function '_root_hybr'
    module_type_store = module_type_store.open_function_context('_root_hybr', 167, 0, False)
    
    # Passed parameters checking function
    _root_hybr.stypy_localization = localization
    _root_hybr.stypy_type_of_self = None
    _root_hybr.stypy_type_store = module_type_store
    _root_hybr.stypy_function_name = '_root_hybr'
    _root_hybr.stypy_param_names_list = ['func', 'x0', 'args', 'jac', 'col_deriv', 'xtol', 'maxfev', 'band', 'eps', 'factor', 'diag']
    _root_hybr.stypy_varargs_param_name = None
    _root_hybr.stypy_kwargs_param_name = 'unknown_options'
    _root_hybr.stypy_call_defaults = defaults
    _root_hybr.stypy_call_varargs = varargs
    _root_hybr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_hybr', ['func', 'x0', 'args', 'jac', 'col_deriv', 'xtol', 'maxfev', 'band', 'eps', 'factor', 'diag'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_hybr', localization, ['func', 'x0', 'args', 'jac', 'col_deriv', 'xtol', 'maxfev', 'band', 'eps', 'factor', 'diag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_hybr(...)' code ##################

    str_171332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'str', "\n    Find the roots of a multivariate function using MINPACK's hybrd and\n    hybrj routines (modified Powell method).\n\n    Options\n    -------\n    col_deriv : bool\n        Specify whether the Jacobian function computes derivatives down\n        the columns (faster, because there is no transpose operation).\n    xtol : float\n        The calculation will terminate if the relative error between two\n        consecutive iterates is at most `xtol`.\n    maxfev : int\n        The maximum number of calls to the function. If zero, then\n        ``100*(N+1)`` is the maximum where N is the number of elements\n        in `x0`.\n    band : tuple\n        If set to a two-sequence containing the number of sub- and\n        super-diagonals within the band of the Jacobi matrix, the\n        Jacobi matrix is considered banded (only for ``fprime=None``).\n    eps : float\n        A suitable step length for the forward-difference\n        approximation of the Jacobian (for ``fprime=None``). If\n        `eps` is less than the machine precision, it is assumed\n        that the relative errors in the functions are of the order of\n        the machine precision.\n    factor : float\n        A parameter determining the initial step bound\n        (``factor * || diag * x||``).  Should be in the interval\n        ``(0.1, 100)``.\n    diag : sequence\n        N positive entries that serve as a scale factors for the\n        variables.\n\n    ")
    
    # Call to _check_unknown_options(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'unknown_options' (line 205)
    unknown_options_171334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 205)
    kwargs_171335 = {}
    # Getting the type of '_check_unknown_options' (line 205)
    _check_unknown_options_171333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 205)
    _check_unknown_options_call_result_171336 = invoke(stypy.reporting.localization.Localization(__file__, 205, 4), _check_unknown_options_171333, *[unknown_options_171334], **kwargs_171335)
    
    
    # Assigning a Name to a Name (line 206):
    
    # Assigning a Name to a Name (line 206):
    # Getting the type of 'eps' (line 206)
    eps_171337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'eps')
    # Assigning a type to the variable 'epsfcn' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'epsfcn', eps_171337)
    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to flatten(...): (line 208)
    # Processing the call keyword arguments (line 208)
    kwargs_171343 = {}
    
    # Call to asarray(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'x0' (line 208)
    x0_171339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), 'x0', False)
    # Processing the call keyword arguments (line 208)
    kwargs_171340 = {}
    # Getting the type of 'asarray' (line 208)
    asarray_171338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 9), 'asarray', False)
    # Calling asarray(args, kwargs) (line 208)
    asarray_call_result_171341 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), asarray_171338, *[x0_171339], **kwargs_171340)
    
    # Obtaining the member 'flatten' of a type (line 208)
    flatten_171342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 9), asarray_call_result_171341, 'flatten')
    # Calling flatten(args, kwargs) (line 208)
    flatten_call_result_171344 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), flatten_171342, *[], **kwargs_171343)
    
    # Assigning a type to the variable 'x0' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'x0', flatten_call_result_171344)
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to len(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'x0' (line 209)
    x0_171346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'x0', False)
    # Processing the call keyword arguments (line 209)
    kwargs_171347 = {}
    # Getting the type of 'len' (line 209)
    len_171345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'len', False)
    # Calling len(args, kwargs) (line 209)
    len_call_result_171348 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), len_171345, *[x0_171346], **kwargs_171347)
    
    # Assigning a type to the variable 'n' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'n', len_call_result_171348)
    
    # Type idiom detected: calculating its left and rigth part (line 210)
    # Getting the type of 'tuple' (line 210)
    tuple_171349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 28), 'tuple')
    # Getting the type of 'args' (line 210)
    args_171350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'args')
    
    (may_be_171351, more_types_in_union_171352) = may_not_be_subtype(tuple_171349, args_171350)

    if may_be_171351:

        if more_types_in_union_171352:
            # Runtime conditional SSA (line 210)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'args', remove_subtype_from_union(args_171350, tuple))
        
        # Assigning a Tuple to a Name (line 211):
        
        # Assigning a Tuple to a Name (line 211):
        
        # Obtaining an instance of the builtin type 'tuple' (line 211)
        tuple_171353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 211)
        # Adding element type (line 211)
        # Getting the type of 'args' (line 211)
        args_171354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 16), tuple_171353, args_171354)
        
        # Assigning a type to the variable 'args' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'args', tuple_171353)

        if more_types_in_union_171352:
            # SSA join for if statement (line 210)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 212):
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_171355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 4), 'int')
    
    # Call to _check_func(...): (line 212)
    # Processing the call arguments (line 212)
    str_171357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 31), 'str', 'fsolve')
    str_171358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 41), 'str', 'func')
    # Getting the type of 'func' (line 212)
    func_171359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 49), 'func', False)
    # Getting the type of 'x0' (line 212)
    x0_171360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 55), 'x0', False)
    # Getting the type of 'args' (line 212)
    args_171361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 59), 'args', False)
    # Getting the type of 'n' (line 212)
    n_171362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 65), 'n', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 212)
    tuple_171363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 212)
    # Adding element type (line 212)
    # Getting the type of 'n' (line 212)
    n_171364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 69), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 69), tuple_171363, n_171364)
    
    # Processing the call keyword arguments (line 212)
    kwargs_171365 = {}
    # Getting the type of '_check_func' (line 212)
    _check_func_171356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), '_check_func', False)
    # Calling _check_func(args, kwargs) (line 212)
    _check_func_call_result_171366 = invoke(stypy.reporting.localization.Localization(__file__, 212, 19), _check_func_171356, *[str_171357, str_171358, func_171359, x0_171360, args_171361, n_171362, tuple_171363], **kwargs_171365)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___171367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 4), _check_func_call_result_171366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_171368 = invoke(stypy.reporting.localization.Localization(__file__, 212, 4), getitem___171367, int_171355)
    
    # Assigning a type to the variable 'tuple_var_assignment_171046' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'tuple_var_assignment_171046', subscript_call_result_171368)
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_171369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 4), 'int')
    
    # Call to _check_func(...): (line 212)
    # Processing the call arguments (line 212)
    str_171371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 31), 'str', 'fsolve')
    str_171372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 41), 'str', 'func')
    # Getting the type of 'func' (line 212)
    func_171373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 49), 'func', False)
    # Getting the type of 'x0' (line 212)
    x0_171374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 55), 'x0', False)
    # Getting the type of 'args' (line 212)
    args_171375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 59), 'args', False)
    # Getting the type of 'n' (line 212)
    n_171376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 65), 'n', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 212)
    tuple_171377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 212)
    # Adding element type (line 212)
    # Getting the type of 'n' (line 212)
    n_171378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 69), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 69), tuple_171377, n_171378)
    
    # Processing the call keyword arguments (line 212)
    kwargs_171379 = {}
    # Getting the type of '_check_func' (line 212)
    _check_func_171370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), '_check_func', False)
    # Calling _check_func(args, kwargs) (line 212)
    _check_func_call_result_171380 = invoke(stypy.reporting.localization.Localization(__file__, 212, 19), _check_func_171370, *[str_171371, str_171372, func_171373, x0_171374, args_171375, n_171376, tuple_171377], **kwargs_171379)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___171381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 4), _check_func_call_result_171380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_171382 = invoke(stypy.reporting.localization.Localization(__file__, 212, 4), getitem___171381, int_171369)
    
    # Assigning a type to the variable 'tuple_var_assignment_171047' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'tuple_var_assignment_171047', subscript_call_result_171382)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_171046' (line 212)
    tuple_var_assignment_171046_171383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'tuple_var_assignment_171046')
    # Assigning a type to the variable 'shape' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'shape', tuple_var_assignment_171046_171383)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_171047' (line 212)
    tuple_var_assignment_171047_171384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'tuple_var_assignment_171047')
    # Assigning a type to the variable 'dtype' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'dtype', tuple_var_assignment_171047_171384)
    
    # Type idiom detected: calculating its left and rigth part (line 213)
    # Getting the type of 'epsfcn' (line 213)
    epsfcn_171385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 7), 'epsfcn')
    # Getting the type of 'None' (line 213)
    None_171386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'None')
    
    (may_be_171387, more_types_in_union_171388) = may_be_none(epsfcn_171385, None_171386)

    if may_be_171387:

        if more_types_in_union_171388:
            # Runtime conditional SSA (line 213)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 214):
        
        # Assigning a Attribute to a Name (line 214):
        
        # Call to finfo(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'dtype' (line 214)
        dtype_171390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'dtype', False)
        # Processing the call keyword arguments (line 214)
        kwargs_171391 = {}
        # Getting the type of 'finfo' (line 214)
        finfo_171389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 17), 'finfo', False)
        # Calling finfo(args, kwargs) (line 214)
        finfo_call_result_171392 = invoke(stypy.reporting.localization.Localization(__file__, 214, 17), finfo_171389, *[dtype_171390], **kwargs_171391)
        
        # Obtaining the member 'eps' of a type (line 214)
        eps_171393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 17), finfo_call_result_171392, 'eps')
        # Assigning a type to the variable 'epsfcn' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'epsfcn', eps_171393)

        if more_types_in_union_171388:
            # SSA join for if statement (line 213)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 215):
    
    # Assigning a Name to a Name (line 215):
    # Getting the type of 'jac' (line 215)
    jac_171394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'jac')
    # Assigning a type to the variable 'Dfun' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'Dfun', jac_171394)
    
    # Type idiom detected: calculating its left and rigth part (line 216)
    # Getting the type of 'Dfun' (line 216)
    Dfun_171395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 7), 'Dfun')
    # Getting the type of 'None' (line 216)
    None_171396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'None')
    
    (may_be_171397, more_types_in_union_171398) = may_be_none(Dfun_171395, None_171396)

    if may_be_171397:

        if more_types_in_union_171398:
            # Runtime conditional SSA (line 216)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 217)
        # Getting the type of 'band' (line 217)
        band_171399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'band')
        # Getting the type of 'None' (line 217)
        None_171400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'None')
        
        (may_be_171401, more_types_in_union_171402) = may_be_none(band_171399, None_171400)

        if may_be_171401:

            if more_types_in_union_171402:
                # Runtime conditional SSA (line 217)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Tuple to a Tuple (line 218):
            
            # Assigning a Num to a Name (line 218):
            int_171403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'int')
            # Assigning a type to the variable 'tuple_assignment_171048' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'tuple_assignment_171048', int_171403)
            
            # Assigning a Num to a Name (line 218):
            int_171404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 26), 'int')
            # Assigning a type to the variable 'tuple_assignment_171049' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'tuple_assignment_171049', int_171404)
            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'tuple_assignment_171048' (line 218)
            tuple_assignment_171048_171405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'tuple_assignment_171048')
            # Assigning a type to the variable 'ml' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'ml', tuple_assignment_171048_171405)
            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'tuple_assignment_171049' (line 218)
            tuple_assignment_171049_171406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'tuple_assignment_171049')
            # Assigning a type to the variable 'mu' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'mu', tuple_assignment_171049_171406)

            if more_types_in_union_171402:
                # Runtime conditional SSA for else branch (line 217)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_171401) or more_types_in_union_171402):
            
            # Assigning a Subscript to a Tuple (line 220):
            
            # Assigning a Subscript to a Name (line 220):
            
            # Obtaining the type of the subscript
            int_171407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 12), 'int')
            
            # Obtaining the type of the subscript
            int_171408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 27), 'int')
            slice_171409 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 220, 21), None, int_171408, None)
            # Getting the type of 'band' (line 220)
            band_171410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'band')
            # Obtaining the member '__getitem__' of a type (line 220)
            getitem___171411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), band_171410, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 220)
            subscript_call_result_171412 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), getitem___171411, slice_171409)
            
            # Obtaining the member '__getitem__' of a type (line 220)
            getitem___171413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), subscript_call_result_171412, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 220)
            subscript_call_result_171414 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), getitem___171413, int_171407)
            
            # Assigning a type to the variable 'tuple_var_assignment_171050' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'tuple_var_assignment_171050', subscript_call_result_171414)
            
            # Assigning a Subscript to a Name (line 220):
            
            # Obtaining the type of the subscript
            int_171415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 12), 'int')
            
            # Obtaining the type of the subscript
            int_171416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 27), 'int')
            slice_171417 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 220, 21), None, int_171416, None)
            # Getting the type of 'band' (line 220)
            band_171418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'band')
            # Obtaining the member '__getitem__' of a type (line 220)
            getitem___171419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), band_171418, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 220)
            subscript_call_result_171420 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), getitem___171419, slice_171417)
            
            # Obtaining the member '__getitem__' of a type (line 220)
            getitem___171421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), subscript_call_result_171420, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 220)
            subscript_call_result_171422 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), getitem___171421, int_171415)
            
            # Assigning a type to the variable 'tuple_var_assignment_171051' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'tuple_var_assignment_171051', subscript_call_result_171422)
            
            # Assigning a Name to a Name (line 220):
            # Getting the type of 'tuple_var_assignment_171050' (line 220)
            tuple_var_assignment_171050_171423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'tuple_var_assignment_171050')
            # Assigning a type to the variable 'ml' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'ml', tuple_var_assignment_171050_171423)
            
            # Assigning a Name to a Name (line 220):
            # Getting the type of 'tuple_var_assignment_171051' (line 220)
            tuple_var_assignment_171051_171424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'tuple_var_assignment_171051')
            # Assigning a type to the variable 'mu' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'mu', tuple_var_assignment_171051_171424)

            if (may_be_171401 and more_types_in_union_171402):
                # SSA join for if statement (line 217)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'maxfev' (line 221)
        maxfev_171425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'maxfev')
        int_171426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 21), 'int')
        # Applying the binary operator '==' (line 221)
        result_eq_171427 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 11), '==', maxfev_171425, int_171426)
        
        # Testing the type of an if condition (line 221)
        if_condition_171428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), result_eq_171427)
        # Assigning a type to the variable 'if_condition_171428' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_171428', if_condition_171428)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 222):
        
        # Assigning a BinOp to a Name (line 222):
        int_171429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 21), 'int')
        # Getting the type of 'n' (line 222)
        n_171430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'n')
        int_171431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 32), 'int')
        # Applying the binary operator '+' (line 222)
        result_add_171432 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 28), '+', n_171430, int_171431)
        
        # Applying the binary operator '*' (line 222)
        result_mul_171433 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 21), '*', int_171429, result_add_171432)
        
        # Assigning a type to the variable 'maxfev' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'maxfev', result_mul_171433)
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to _hybrd(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'func' (line 223)
        func_171436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'func', False)
        # Getting the type of 'x0' (line 223)
        x0_171437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 39), 'x0', False)
        # Getting the type of 'args' (line 223)
        args_171438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 43), 'args', False)
        int_171439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 49), 'int')
        # Getting the type of 'xtol' (line 223)
        xtol_171440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 52), 'xtol', False)
        # Getting the type of 'maxfev' (line 223)
        maxfev_171441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 58), 'maxfev', False)
        # Getting the type of 'ml' (line 224)
        ml_171442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 33), 'ml', False)
        # Getting the type of 'mu' (line 224)
        mu_171443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 37), 'mu', False)
        # Getting the type of 'epsfcn' (line 224)
        epsfcn_171444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 41), 'epsfcn', False)
        # Getting the type of 'factor' (line 224)
        factor_171445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 49), 'factor', False)
        # Getting the type of 'diag' (line 224)
        diag_171446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'diag', False)
        # Processing the call keyword arguments (line 223)
        kwargs_171447 = {}
        # Getting the type of '_minpack' (line 223)
        _minpack_171434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), '_minpack', False)
        # Obtaining the member '_hybrd' of a type (line 223)
        _hybrd_171435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 17), _minpack_171434, '_hybrd')
        # Calling _hybrd(args, kwargs) (line 223)
        _hybrd_call_result_171448 = invoke(stypy.reporting.localization.Localization(__file__, 223, 17), _hybrd_171435, *[func_171436, x0_171437, args_171438, int_171439, xtol_171440, maxfev_171441, ml_171442, mu_171443, epsfcn_171444, factor_171445, diag_171446], **kwargs_171447)
        
        # Assigning a type to the variable 'retval' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'retval', _hybrd_call_result_171448)

        if more_types_in_union_171398:
            # Runtime conditional SSA for else branch (line 216)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_171397) or more_types_in_union_171398):
        
        # Call to _check_func(...): (line 226)
        # Processing the call arguments (line 226)
        str_171450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 20), 'str', 'fsolve')
        str_171451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 30), 'str', 'fprime')
        # Getting the type of 'Dfun' (line 226)
        Dfun_171452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'Dfun', False)
        # Getting the type of 'x0' (line 226)
        x0_171453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 46), 'x0', False)
        # Getting the type of 'args' (line 226)
        args_171454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 50), 'args', False)
        # Getting the type of 'n' (line 226)
        n_171455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 56), 'n', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_171456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'n' (line 226)
        n_171457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 60), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 60), tuple_171456, n_171457)
        # Adding element type (line 226)
        # Getting the type of 'n' (line 226)
        n_171458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 63), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 60), tuple_171456, n_171458)
        
        # Processing the call keyword arguments (line 226)
        kwargs_171459 = {}
        # Getting the type of '_check_func' (line 226)
        _check_func_171449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), '_check_func', False)
        # Calling _check_func(args, kwargs) (line 226)
        _check_func_call_result_171460 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), _check_func_171449, *[str_171450, str_171451, Dfun_171452, x0_171453, args_171454, n_171455, tuple_171456], **kwargs_171459)
        
        
        
        # Getting the type of 'maxfev' (line 227)
        maxfev_171461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'maxfev')
        int_171462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 22), 'int')
        # Applying the binary operator '==' (line 227)
        result_eq_171463 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 12), '==', maxfev_171461, int_171462)
        
        # Testing the type of an if condition (line 227)
        if_condition_171464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_eq_171463)
        # Assigning a type to the variable 'if_condition_171464' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_171464', if_condition_171464)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 228):
        
        # Assigning a BinOp to a Name (line 228):
        int_171465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 21), 'int')
        # Getting the type of 'n' (line 228)
        n_171466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 28), 'n')
        int_171467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 32), 'int')
        # Applying the binary operator '+' (line 228)
        result_add_171468 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 28), '+', n_171466, int_171467)
        
        # Applying the binary operator '*' (line 228)
        result_mul_171469 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 21), '*', int_171465, result_add_171468)
        
        # Assigning a type to the variable 'maxfev' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'maxfev', result_mul_171469)
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to _hybrj(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'func' (line 229)
        func_171472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 33), 'func', False)
        # Getting the type of 'Dfun' (line 229)
        Dfun_171473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 39), 'Dfun', False)
        # Getting the type of 'x0' (line 229)
        x0_171474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 45), 'x0', False)
        # Getting the type of 'args' (line 229)
        args_171475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 49), 'args', False)
        int_171476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 55), 'int')
        # Getting the type of 'col_deriv' (line 230)
        col_deriv_171477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 33), 'col_deriv', False)
        # Getting the type of 'xtol' (line 230)
        xtol_171478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 44), 'xtol', False)
        # Getting the type of 'maxfev' (line 230)
        maxfev_171479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 50), 'maxfev', False)
        # Getting the type of 'factor' (line 230)
        factor_171480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 58), 'factor', False)
        # Getting the type of 'diag' (line 230)
        diag_171481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 66), 'diag', False)
        # Processing the call keyword arguments (line 229)
        kwargs_171482 = {}
        # Getting the type of '_minpack' (line 229)
        _minpack_171470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), '_minpack', False)
        # Obtaining the member '_hybrj' of a type (line 229)
        _hybrj_171471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 17), _minpack_171470, '_hybrj')
        # Calling _hybrj(args, kwargs) (line 229)
        _hybrj_call_result_171483 = invoke(stypy.reporting.localization.Localization(__file__, 229, 17), _hybrj_171471, *[func_171472, Dfun_171473, x0_171474, args_171475, int_171476, col_deriv_171477, xtol_171478, maxfev_171479, factor_171480, diag_171481], **kwargs_171482)
        
        # Assigning a type to the variable 'retval' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'retval', _hybrj_call_result_171483)

        if (may_be_171397 and more_types_in_union_171398):
            # SSA join for if statement (line 216)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Tuple to a Tuple (line 232):
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_171484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 23), 'int')
    # Getting the type of 'retval' (line 232)
    retval_171485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'retval')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___171486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 16), retval_171485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_171487 = invoke(stypy.reporting.localization.Localization(__file__, 232, 16), getitem___171486, int_171484)
    
    # Assigning a type to the variable 'tuple_assignment_171052' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_assignment_171052', subscript_call_result_171487)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_171488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 34), 'int')
    # Getting the type of 'retval' (line 232)
    retval_171489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'retval')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___171490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 27), retval_171489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_171491 = invoke(stypy.reporting.localization.Localization(__file__, 232, 27), getitem___171490, int_171488)
    
    # Assigning a type to the variable 'tuple_assignment_171053' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_assignment_171053', subscript_call_result_171491)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_assignment_171052' (line 232)
    tuple_assignment_171052_171492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_assignment_171052')
    # Assigning a type to the variable 'x' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'x', tuple_assignment_171052_171492)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_assignment_171053' (line 232)
    tuple_assignment_171053_171493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_assignment_171053')
    # Assigning a type to the variable 'status' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 7), 'status', tuple_assignment_171053_171493)
    
    # Assigning a Dict to a Name (line 234):
    
    # Assigning a Dict to a Name (line 234):
    
    # Obtaining an instance of the builtin type 'dict' (line 234)
    dict_171494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 234)
    # Adding element type (key, value) (line 234)
    int_171495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 14), 'int')
    str_171496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 17), 'str', 'Improper input parameters were entered.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), dict_171494, (int_171495, str_171496))
    # Adding element type (key, value) (line 234)
    int_171497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 14), 'int')
    str_171498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 17), 'str', 'The solution converged.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), dict_171494, (int_171497, str_171498))
    # Adding element type (key, value) (line 234)
    int_171499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 14), 'int')
    str_171500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 17), 'str', 'The number of calls to function has reached maxfev = %d.')
    # Getting the type of 'maxfev' (line 237)
    maxfev_171501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 43), 'maxfev')
    # Applying the binary operator '%' (line 236)
    result_mod_171502 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 17), '%', str_171500, maxfev_171501)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), dict_171494, (int_171499, result_mod_171502))
    # Adding element type (key, value) (line 234)
    int_171503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 14), 'int')
    str_171504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 17), 'str', 'xtol=%f is too small, no further improvement in the approximate\n  solution is possible.')
    # Getting the type of 'xtol' (line 240)
    xtol_171505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 35), 'xtol')
    # Applying the binary operator '%' (line 238)
    result_mod_171506 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 17), '%', str_171504, xtol_171505)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), dict_171494, (int_171503, result_mod_171506))
    # Adding element type (key, value) (line 234)
    int_171507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 14), 'int')
    str_171508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 17), 'str', 'The iteration is not making good progress, as measured by the \n  improvement from the last five Jacobian evaluations.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), dict_171494, (int_171507, str_171508))
    # Adding element type (key, value) (line 234)
    int_171509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 14), 'int')
    str_171510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 17), 'str', 'The iteration is not making good progress, as measured by the \n  improvement from the last ten iterations.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), dict_171494, (int_171509, str_171510))
    # Adding element type (key, value) (line 234)
    str_171511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 14), 'str', 'unknown')
    str_171512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 25), 'str', 'An error occurred.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), dict_171494, (str_171511, str_171512))
    
    # Assigning a type to the variable 'errors' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'errors', dict_171494)
    
    # Assigning a Subscript to a Name (line 249):
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    int_171513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 18), 'int')
    # Getting the type of 'retval' (line 249)
    retval_171514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'retval')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___171515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 11), retval_171514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_171516 = invoke(stypy.reporting.localization.Localization(__file__, 249, 11), getitem___171515, int_171513)
    
    # Assigning a type to the variable 'info' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'info', subscript_call_result_171516)
    
    # Assigning a Call to a Subscript (line 250):
    
    # Assigning a Call to a Subscript (line 250):
    
    # Call to pop(...): (line 250)
    # Processing the call arguments (line 250)
    str_171519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 27), 'str', 'fvec')
    # Processing the call keyword arguments (line 250)
    kwargs_171520 = {}
    # Getting the type of 'info' (line 250)
    info_171517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 18), 'info', False)
    # Obtaining the member 'pop' of a type (line 250)
    pop_171518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 18), info_171517, 'pop')
    # Calling pop(args, kwargs) (line 250)
    pop_call_result_171521 = invoke(stypy.reporting.localization.Localization(__file__, 250, 18), pop_171518, *[str_171519], **kwargs_171520)
    
    # Getting the type of 'info' (line 250)
    info_171522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'info')
    str_171523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 9), 'str', 'fun')
    # Storing an element on a container (line 250)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 4), info_171522, (str_171523, pop_call_result_171521))
    
    # Assigning a Call to a Name (line 251):
    
    # Assigning a Call to a Name (line 251):
    
    # Call to OptimizeResult(...): (line 251)
    # Processing the call keyword arguments (line 251)
    # Getting the type of 'x' (line 251)
    x_171525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'x', False)
    keyword_171526 = x_171525
    
    # Getting the type of 'status' (line 251)
    status_171527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 39), 'status', False)
    int_171528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 49), 'int')
    # Applying the binary operator '==' (line 251)
    result_eq_171529 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 39), '==', status_171527, int_171528)
    
    keyword_171530 = result_eq_171529
    # Getting the type of 'status' (line 251)
    status_171531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 60), 'status', False)
    keyword_171532 = status_171531
    kwargs_171533 = {'status': keyword_171532, 'x': keyword_171526, 'success': keyword_171530}
    # Getting the type of 'OptimizeResult' (line 251)
    OptimizeResult_171524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 10), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 251)
    OptimizeResult_call_result_171534 = invoke(stypy.reporting.localization.Localization(__file__, 251, 10), OptimizeResult_171524, *[], **kwargs_171533)
    
    # Assigning a type to the variable 'sol' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'sol', OptimizeResult_call_result_171534)
    
    # Call to update(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'info' (line 252)
    info_171537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'info', False)
    # Processing the call keyword arguments (line 252)
    kwargs_171538 = {}
    # Getting the type of 'sol' (line 252)
    sol_171535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'sol', False)
    # Obtaining the member 'update' of a type (line 252)
    update_171536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 4), sol_171535, 'update')
    # Calling update(args, kwargs) (line 252)
    update_call_result_171539 = invoke(stypy.reporting.localization.Localization(__file__, 252, 4), update_171536, *[info_171537], **kwargs_171538)
    
    
    
    # SSA begins for try-except statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Subscript (line 254):
    
    # Assigning a Subscript to a Subscript (line 254):
    
    # Obtaining the type of the subscript
    # Getting the type of 'status' (line 254)
    status_171540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 32), 'status')
    # Getting the type of 'errors' (line 254)
    errors_171541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 'errors')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___171542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 25), errors_171541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_171543 = invoke(stypy.reporting.localization.Localization(__file__, 254, 25), getitem___171542, status_171540)
    
    # Getting the type of 'sol' (line 254)
    sol_171544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'sol')
    str_171545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 12), 'str', 'message')
    # Storing an element on a container (line 254)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 8), sol_171544, (str_171545, subscript_call_result_171543))
    # SSA branch for the except part of a try statement (line 253)
    # SSA branch for the except 'KeyError' branch of a try statement (line 253)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Subscript to a Subscript (line 256):
    
    # Assigning a Subscript to a Subscript (line 256):
    
    # Obtaining the type of the subscript
    str_171546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 33), 'str', 'unknown')
    # Getting the type of 'errors' (line 256)
    errors_171547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'errors')
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___171548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 26), errors_171547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_171549 = invoke(stypy.reporting.localization.Localization(__file__, 256, 26), getitem___171548, str_171546)
    
    # Getting the type of 'info' (line 256)
    info_171550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'info')
    str_171551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 13), 'str', 'message')
    # Storing an element on a container (line 256)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 8), info_171550, (str_171551, subscript_call_result_171549))
    # SSA join for try-except statement (line 253)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'sol' (line 258)
    sol_171552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'sol')
    # Assigning a type to the variable 'stypy_return_type' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type', sol_171552)
    
    # ################# End of '_root_hybr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_hybr' in the type store
    # Getting the type of 'stypy_return_type' (line 167)
    stypy_return_type_171553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171553)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_hybr'
    return stypy_return_type_171553

# Assigning a type to the variable '_root_hybr' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), '_root_hybr', _root_hybr)

@norecursion
def leastsq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 261)
    tuple_171554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 261)
    
    # Getting the type of 'None' (line 261)
    None_171555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 36), 'None')
    int_171556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 54), 'int')
    int_171557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 22), 'int')
    float_171558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 30), 'float')
    float_171559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 47), 'float')
    float_171560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 17), 'float')
    int_171561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 29), 'int')
    # Getting the type of 'None' (line 263)
    None_171562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 39), 'None')
    int_171563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 52), 'int')
    # Getting the type of 'None' (line 263)
    None_171564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 62), 'None')
    defaults = [tuple_171554, None_171555, int_171556, int_171557, float_171558, float_171559, float_171560, int_171561, None_171562, int_171563, None_171564]
    # Create a new context for function 'leastsq'
    module_type_store = module_type_store.open_function_context('leastsq', 261, 0, False)
    
    # Passed parameters checking function
    leastsq.stypy_localization = localization
    leastsq.stypy_type_of_self = None
    leastsq.stypy_type_store = module_type_store
    leastsq.stypy_function_name = 'leastsq'
    leastsq.stypy_param_names_list = ['func', 'x0', 'args', 'Dfun', 'full_output', 'col_deriv', 'ftol', 'xtol', 'gtol', 'maxfev', 'epsfcn', 'factor', 'diag']
    leastsq.stypy_varargs_param_name = None
    leastsq.stypy_kwargs_param_name = None
    leastsq.stypy_call_defaults = defaults
    leastsq.stypy_call_varargs = varargs
    leastsq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'leastsq', ['func', 'x0', 'args', 'Dfun', 'full_output', 'col_deriv', 'ftol', 'xtol', 'gtol', 'maxfev', 'epsfcn', 'factor', 'diag'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'leastsq', localization, ['func', 'x0', 'args', 'Dfun', 'full_output', 'col_deriv', 'ftol', 'xtol', 'gtol', 'maxfev', 'epsfcn', 'factor', 'diag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'leastsq(...)' code ##################

    str_171565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, (-1)), 'str', '\n    Minimize the sum of squares of a set of equations.\n\n    ::\n\n        x = arg min(sum(func(y)**2,axis=0))\n                 y\n\n    Parameters\n    ----------\n    func : callable\n        should take at least one (possibly length N vector) argument and\n        returns M floating point numbers. It must not return NaNs or\n        fitting might fail.\n    x0 : ndarray\n        The starting estimate for the minimization.\n    args : tuple, optional\n        Any extra arguments to func are placed in this tuple.\n    Dfun : callable, optional\n        A function or method to compute the Jacobian of func with derivatives\n        across the rows. If this is None, the Jacobian will be estimated.\n    full_output : bool, optional\n        non-zero to return all optional outputs.\n    col_deriv : bool, optional\n        non-zero to specify that the Jacobian function computes derivatives\n        down the columns (faster, because there is no transpose operation).\n    ftol : float, optional\n        Relative error desired in the sum of squares.\n    xtol : float, optional\n        Relative error desired in the approximate solution.\n    gtol : float, optional\n        Orthogonality desired between the function vector and the columns of\n        the Jacobian.\n    maxfev : int, optional\n        The maximum number of calls to the function. If `Dfun` is provided\n        then the default `maxfev` is 100*(N+1) where N is the number of elements\n        in x0, otherwise the default `maxfev` is 200*(N+1).\n    epsfcn : float, optional\n        A variable used in determining a suitable step length for the forward-\n        difference approximation of the Jacobian (for Dfun=None).\n        Normally the actual step length will be sqrt(epsfcn)*x\n        If epsfcn is less than the machine precision, it is assumed that the\n        relative errors are of the order of the machine precision.\n    factor : float, optional\n        A parameter determining the initial step bound\n        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.\n    diag : sequence, optional\n        N positive entries that serve as a scale factors for the variables.\n\n    Returns\n    -------\n    x : ndarray\n        The solution (or the result of the last iteration for an unsuccessful\n        call).\n    cov_x : ndarray\n        Uses the fjac and ipvt optional outputs to construct an\n        estimate of the jacobian around the solution. None if a\n        singular matrix encountered (indicates very flat curvature in\n        some direction).  This matrix must be multiplied by the\n        residual variance to get the covariance of the\n        parameter estimates -- see curve_fit.\n    infodict : dict\n        a dictionary of optional outputs with the key s:\n\n        ``nfev``\n            The number of function calls\n        ``fvec``\n            The function evaluated at the output\n        ``fjac``\n            A permutation of the R matrix of a QR\n            factorization of the final approximate\n            Jacobian matrix, stored column wise.\n            Together with ipvt, the covariance of the\n            estimate can be approximated.\n        ``ipvt``\n            An integer array of length N which defines\n            a permutation matrix, p, such that\n            fjac*p = q*r, where r is upper triangular\n            with diagonal elements of nonincreasing\n            magnitude. Column j of p is column ipvt(j)\n            of the identity matrix.\n        ``qtf``\n            The vector (transpose(q) * fvec).\n\n    mesg : str\n        A string message giving information about the cause of failure.\n    ier : int\n        An integer flag.  If it is equal to 1, 2, 3 or 4, the solution was\n        found.  Otherwise, the solution was not found. In either case, the\n        optional output variable \'mesg\' gives more information.\n\n    Notes\n    -----\n    "leastsq" is a wrapper around MINPACK\'s lmdif and lmder algorithms.\n\n    cov_x is a Jacobian approximation to the Hessian of the least squares\n    objective function.\n    This approximation assumes that the objective function is based on the\n    difference between some observed target data (ydata) and a (non-linear)\n    function of the parameters `f(xdata, params)` ::\n\n           func(params) = ydata - f(xdata, params)\n\n    so that the objective function is ::\n\n           min   sum((ydata - f(xdata, params))**2, axis=0)\n         params\n\n    ')
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to flatten(...): (line 373)
    # Processing the call keyword arguments (line 373)
    kwargs_171571 = {}
    
    # Call to asarray(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'x0' (line 373)
    x0_171567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'x0', False)
    # Processing the call keyword arguments (line 373)
    kwargs_171568 = {}
    # Getting the type of 'asarray' (line 373)
    asarray_171566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 9), 'asarray', False)
    # Calling asarray(args, kwargs) (line 373)
    asarray_call_result_171569 = invoke(stypy.reporting.localization.Localization(__file__, 373, 9), asarray_171566, *[x0_171567], **kwargs_171568)
    
    # Obtaining the member 'flatten' of a type (line 373)
    flatten_171570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 9), asarray_call_result_171569, 'flatten')
    # Calling flatten(args, kwargs) (line 373)
    flatten_call_result_171572 = invoke(stypy.reporting.localization.Localization(__file__, 373, 9), flatten_171570, *[], **kwargs_171571)
    
    # Assigning a type to the variable 'x0' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'x0', flatten_call_result_171572)
    
    # Assigning a Call to a Name (line 374):
    
    # Assigning a Call to a Name (line 374):
    
    # Call to len(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'x0' (line 374)
    x0_171574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'x0', False)
    # Processing the call keyword arguments (line 374)
    kwargs_171575 = {}
    # Getting the type of 'len' (line 374)
    len_171573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'len', False)
    # Calling len(args, kwargs) (line 374)
    len_call_result_171576 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), len_171573, *[x0_171574], **kwargs_171575)
    
    # Assigning a type to the variable 'n' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'n', len_call_result_171576)
    
    # Type idiom detected: calculating its left and rigth part (line 375)
    # Getting the type of 'tuple' (line 375)
    tuple_171577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 28), 'tuple')
    # Getting the type of 'args' (line 375)
    args_171578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 22), 'args')
    
    (may_be_171579, more_types_in_union_171580) = may_not_be_subtype(tuple_171577, args_171578)

    if may_be_171579:

        if more_types_in_union_171580:
            # Runtime conditional SSA (line 375)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'args', remove_subtype_from_union(args_171578, tuple))
        
        # Assigning a Tuple to a Name (line 376):
        
        # Assigning a Tuple to a Name (line 376):
        
        # Obtaining an instance of the builtin type 'tuple' (line 376)
        tuple_171581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 376)
        # Adding element type (line 376)
        # Getting the type of 'args' (line 376)
        args_171582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 16), tuple_171581, args_171582)
        
        # Assigning a type to the variable 'args' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'args', tuple_171581)

        if more_types_in_union_171580:
            # SSA join for if statement (line 375)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 377):
    
    # Assigning a Subscript to a Name (line 377):
    
    # Obtaining the type of the subscript
    int_171583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 4), 'int')
    
    # Call to _check_func(...): (line 377)
    # Processing the call arguments (line 377)
    str_171585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 31), 'str', 'leastsq')
    str_171586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 42), 'str', 'func')
    # Getting the type of 'func' (line 377)
    func_171587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 50), 'func', False)
    # Getting the type of 'x0' (line 377)
    x0_171588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 56), 'x0', False)
    # Getting the type of 'args' (line 377)
    args_171589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 60), 'args', False)
    # Getting the type of 'n' (line 377)
    n_171590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 66), 'n', False)
    # Processing the call keyword arguments (line 377)
    kwargs_171591 = {}
    # Getting the type of '_check_func' (line 377)
    _check_func_171584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), '_check_func', False)
    # Calling _check_func(args, kwargs) (line 377)
    _check_func_call_result_171592 = invoke(stypy.reporting.localization.Localization(__file__, 377, 19), _check_func_171584, *[str_171585, str_171586, func_171587, x0_171588, args_171589, n_171590], **kwargs_171591)
    
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___171593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 4), _check_func_call_result_171592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 377)
    subscript_call_result_171594 = invoke(stypy.reporting.localization.Localization(__file__, 377, 4), getitem___171593, int_171583)
    
    # Assigning a type to the variable 'tuple_var_assignment_171054' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'tuple_var_assignment_171054', subscript_call_result_171594)
    
    # Assigning a Subscript to a Name (line 377):
    
    # Obtaining the type of the subscript
    int_171595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 4), 'int')
    
    # Call to _check_func(...): (line 377)
    # Processing the call arguments (line 377)
    str_171597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 31), 'str', 'leastsq')
    str_171598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 42), 'str', 'func')
    # Getting the type of 'func' (line 377)
    func_171599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 50), 'func', False)
    # Getting the type of 'x0' (line 377)
    x0_171600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 56), 'x0', False)
    # Getting the type of 'args' (line 377)
    args_171601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 60), 'args', False)
    # Getting the type of 'n' (line 377)
    n_171602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 66), 'n', False)
    # Processing the call keyword arguments (line 377)
    kwargs_171603 = {}
    # Getting the type of '_check_func' (line 377)
    _check_func_171596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), '_check_func', False)
    # Calling _check_func(args, kwargs) (line 377)
    _check_func_call_result_171604 = invoke(stypy.reporting.localization.Localization(__file__, 377, 19), _check_func_171596, *[str_171597, str_171598, func_171599, x0_171600, args_171601, n_171602], **kwargs_171603)
    
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___171605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 4), _check_func_call_result_171604, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 377)
    subscript_call_result_171606 = invoke(stypy.reporting.localization.Localization(__file__, 377, 4), getitem___171605, int_171595)
    
    # Assigning a type to the variable 'tuple_var_assignment_171055' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'tuple_var_assignment_171055', subscript_call_result_171606)
    
    # Assigning a Name to a Name (line 377):
    # Getting the type of 'tuple_var_assignment_171054' (line 377)
    tuple_var_assignment_171054_171607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'tuple_var_assignment_171054')
    # Assigning a type to the variable 'shape' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'shape', tuple_var_assignment_171054_171607)
    
    # Assigning a Name to a Name (line 377):
    # Getting the type of 'tuple_var_assignment_171055' (line 377)
    tuple_var_assignment_171055_171608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'tuple_var_assignment_171055')
    # Assigning a type to the variable 'dtype' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), 'dtype', tuple_var_assignment_171055_171608)
    
    # Assigning a Subscript to a Name (line 378):
    
    # Assigning a Subscript to a Name (line 378):
    
    # Obtaining the type of the subscript
    int_171609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 14), 'int')
    # Getting the type of 'shape' (line 378)
    shape_171610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'shape')
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___171611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), shape_171610, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 378)
    subscript_call_result_171612 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), getitem___171611, int_171609)
    
    # Assigning a type to the variable 'm' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'm', subscript_call_result_171612)
    
    
    # Getting the type of 'n' (line 379)
    n_171613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 7), 'n')
    # Getting the type of 'm' (line 379)
    m_171614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'm')
    # Applying the binary operator '>' (line 379)
    result_gt_171615 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 7), '>', n_171613, m_171614)
    
    # Testing the type of an if condition (line 379)
    if_condition_171616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 4), result_gt_171615)
    # Assigning a type to the variable 'if_condition_171616' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'if_condition_171616', if_condition_171616)
    # SSA begins for if statement (line 379)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 380)
    # Processing the call arguments (line 380)
    str_171618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 24), 'str', 'Improper input: N=%s must not exceed M=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 380)
    tuple_171619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 71), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 380)
    # Adding element type (line 380)
    # Getting the type of 'n' (line 380)
    n_171620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 71), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 71), tuple_171619, n_171620)
    # Adding element type (line 380)
    # Getting the type of 'm' (line 380)
    m_171621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 74), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 71), tuple_171619, m_171621)
    
    # Applying the binary operator '%' (line 380)
    result_mod_171622 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 24), '%', str_171618, tuple_171619)
    
    # Processing the call keyword arguments (line 380)
    kwargs_171623 = {}
    # Getting the type of 'TypeError' (line 380)
    TypeError_171617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 380)
    TypeError_call_result_171624 = invoke(stypy.reporting.localization.Localization(__file__, 380, 14), TypeError_171617, *[result_mod_171622], **kwargs_171623)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 380, 8), TypeError_call_result_171624, 'raise parameter', BaseException)
    # SSA join for if statement (line 379)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 381)
    # Getting the type of 'epsfcn' (line 381)
    epsfcn_171625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 7), 'epsfcn')
    # Getting the type of 'None' (line 381)
    None_171626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 17), 'None')
    
    (may_be_171627, more_types_in_union_171628) = may_be_none(epsfcn_171625, None_171626)

    if may_be_171627:

        if more_types_in_union_171628:
            # Runtime conditional SSA (line 381)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 382):
        
        # Assigning a Attribute to a Name (line 382):
        
        # Call to finfo(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'dtype' (line 382)
        dtype_171630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'dtype', False)
        # Processing the call keyword arguments (line 382)
        kwargs_171631 = {}
        # Getting the type of 'finfo' (line 382)
        finfo_171629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 17), 'finfo', False)
        # Calling finfo(args, kwargs) (line 382)
        finfo_call_result_171632 = invoke(stypy.reporting.localization.Localization(__file__, 382, 17), finfo_171629, *[dtype_171630], **kwargs_171631)
        
        # Obtaining the member 'eps' of a type (line 382)
        eps_171633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 17), finfo_call_result_171632, 'eps')
        # Assigning a type to the variable 'epsfcn' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'epsfcn', eps_171633)

        if more_types_in_union_171628:
            # SSA join for if statement (line 381)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 383)
    # Getting the type of 'Dfun' (line 383)
    Dfun_171634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 7), 'Dfun')
    # Getting the type of 'None' (line 383)
    None_171635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'None')
    
    (may_be_171636, more_types_in_union_171637) = may_be_none(Dfun_171634, None_171635)

    if may_be_171636:

        if more_types_in_union_171637:
            # Runtime conditional SSA (line 383)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'maxfev' (line 384)
        maxfev_171638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 11), 'maxfev')
        int_171639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 21), 'int')
        # Applying the binary operator '==' (line 384)
        result_eq_171640 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 11), '==', maxfev_171638, int_171639)
        
        # Testing the type of an if condition (line 384)
        if_condition_171641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 8), result_eq_171640)
        # Assigning a type to the variable 'if_condition_171641' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'if_condition_171641', if_condition_171641)
        # SSA begins for if statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 385):
        
        # Assigning a BinOp to a Name (line 385):
        int_171642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 21), 'int')
        # Getting the type of 'n' (line 385)
        n_171643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 26), 'n')
        int_171644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 30), 'int')
        # Applying the binary operator '+' (line 385)
        result_add_171645 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 26), '+', n_171643, int_171644)
        
        # Applying the binary operator '*' (line 385)
        result_mul_171646 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 21), '*', int_171642, result_add_171645)
        
        # Assigning a type to the variable 'maxfev' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'maxfev', result_mul_171646)
        # SSA join for if statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to _lmdif(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'func' (line 386)
        func_171649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 33), 'func', False)
        # Getting the type of 'x0' (line 386)
        x0_171650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 39), 'x0', False)
        # Getting the type of 'args' (line 386)
        args_171651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 43), 'args', False)
        # Getting the type of 'full_output' (line 386)
        full_output_171652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 49), 'full_output', False)
        # Getting the type of 'ftol' (line 386)
        ftol_171653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 62), 'ftol', False)
        # Getting the type of 'xtol' (line 386)
        xtol_171654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 68), 'xtol', False)
        # Getting the type of 'gtol' (line 387)
        gtol_171655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 33), 'gtol', False)
        # Getting the type of 'maxfev' (line 387)
        maxfev_171656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 39), 'maxfev', False)
        # Getting the type of 'epsfcn' (line 387)
        epsfcn_171657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 47), 'epsfcn', False)
        # Getting the type of 'factor' (line 387)
        factor_171658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 55), 'factor', False)
        # Getting the type of 'diag' (line 387)
        diag_171659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 63), 'diag', False)
        # Processing the call keyword arguments (line 386)
        kwargs_171660 = {}
        # Getting the type of '_minpack' (line 386)
        _minpack_171647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), '_minpack', False)
        # Obtaining the member '_lmdif' of a type (line 386)
        _lmdif_171648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 17), _minpack_171647, '_lmdif')
        # Calling _lmdif(args, kwargs) (line 386)
        _lmdif_call_result_171661 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), _lmdif_171648, *[func_171649, x0_171650, args_171651, full_output_171652, ftol_171653, xtol_171654, gtol_171655, maxfev_171656, epsfcn_171657, factor_171658, diag_171659], **kwargs_171660)
        
        # Assigning a type to the variable 'retval' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'retval', _lmdif_call_result_171661)

        if more_types_in_union_171637:
            # Runtime conditional SSA for else branch (line 383)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_171636) or more_types_in_union_171637):
        
        # Getting the type of 'col_deriv' (line 389)
        col_deriv_171662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'col_deriv')
        # Testing the type of an if condition (line 389)
        if_condition_171663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), col_deriv_171662)
        # Assigning a type to the variable 'if_condition_171663' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_171663', if_condition_171663)
        # SSA begins for if statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _check_func(...): (line 390)
        # Processing the call arguments (line 390)
        str_171665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 24), 'str', 'leastsq')
        str_171666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 35), 'str', 'Dfun')
        # Getting the type of 'Dfun' (line 390)
        Dfun_171667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 43), 'Dfun', False)
        # Getting the type of 'x0' (line 390)
        x0_171668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 49), 'x0', False)
        # Getting the type of 'args' (line 390)
        args_171669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 53), 'args', False)
        # Getting the type of 'n' (line 390)
        n_171670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 59), 'n', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 390)
        tuple_171671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 390)
        # Adding element type (line 390)
        # Getting the type of 'n' (line 390)
        n_171672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 63), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 63), tuple_171671, n_171672)
        # Adding element type (line 390)
        # Getting the type of 'm' (line 390)
        m_171673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 66), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 63), tuple_171671, m_171673)
        
        # Processing the call keyword arguments (line 390)
        kwargs_171674 = {}
        # Getting the type of '_check_func' (line 390)
        _check_func_171664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), '_check_func', False)
        # Calling _check_func(args, kwargs) (line 390)
        _check_func_call_result_171675 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), _check_func_171664, *[str_171665, str_171666, Dfun_171667, x0_171668, args_171669, n_171670, tuple_171671], **kwargs_171674)
        
        # SSA branch for the else part of an if statement (line 389)
        module_type_store.open_ssa_branch('else')
        
        # Call to _check_func(...): (line 392)
        # Processing the call arguments (line 392)
        str_171677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 24), 'str', 'leastsq')
        str_171678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 35), 'str', 'Dfun')
        # Getting the type of 'Dfun' (line 392)
        Dfun_171679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 43), 'Dfun', False)
        # Getting the type of 'x0' (line 392)
        x0_171680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 49), 'x0', False)
        # Getting the type of 'args' (line 392)
        args_171681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 53), 'args', False)
        # Getting the type of 'n' (line 392)
        n_171682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 59), 'n', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 392)
        tuple_171683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 392)
        # Adding element type (line 392)
        # Getting the type of 'm' (line 392)
        m_171684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 63), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 63), tuple_171683, m_171684)
        # Adding element type (line 392)
        # Getting the type of 'n' (line 392)
        n_171685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 66), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 63), tuple_171683, n_171685)
        
        # Processing the call keyword arguments (line 392)
        kwargs_171686 = {}
        # Getting the type of '_check_func' (line 392)
        _check_func_171676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), '_check_func', False)
        # Calling _check_func(args, kwargs) (line 392)
        _check_func_call_result_171687 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), _check_func_171676, *[str_171677, str_171678, Dfun_171679, x0_171680, args_171681, n_171682, tuple_171683], **kwargs_171686)
        
        # SSA join for if statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'maxfev' (line 393)
        maxfev_171688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'maxfev')
        int_171689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 21), 'int')
        # Applying the binary operator '==' (line 393)
        result_eq_171690 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 11), '==', maxfev_171688, int_171689)
        
        # Testing the type of an if condition (line 393)
        if_condition_171691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 8), result_eq_171690)
        # Assigning a type to the variable 'if_condition_171691' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'if_condition_171691', if_condition_171691)
        # SSA begins for if statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 394):
        
        # Assigning a BinOp to a Name (line 394):
        int_171692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 21), 'int')
        # Getting the type of 'n' (line 394)
        n_171693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 28), 'n')
        int_171694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 32), 'int')
        # Applying the binary operator '+' (line 394)
        result_add_171695 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 28), '+', n_171693, int_171694)
        
        # Applying the binary operator '*' (line 394)
        result_mul_171696 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 21), '*', int_171692, result_add_171695)
        
        # Assigning a type to the variable 'maxfev' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'maxfev', result_mul_171696)
        # SSA join for if statement (line 393)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to _lmder(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'func' (line 395)
        func_171699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 33), 'func', False)
        # Getting the type of 'Dfun' (line 395)
        Dfun_171700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 39), 'Dfun', False)
        # Getting the type of 'x0' (line 395)
        x0_171701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 45), 'x0', False)
        # Getting the type of 'args' (line 395)
        args_171702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 49), 'args', False)
        # Getting the type of 'full_output' (line 395)
        full_output_171703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 55), 'full_output', False)
        # Getting the type of 'col_deriv' (line 395)
        col_deriv_171704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 68), 'col_deriv', False)
        # Getting the type of 'ftol' (line 396)
        ftol_171705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 33), 'ftol', False)
        # Getting the type of 'xtol' (line 396)
        xtol_171706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 39), 'xtol', False)
        # Getting the type of 'gtol' (line 396)
        gtol_171707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 45), 'gtol', False)
        # Getting the type of 'maxfev' (line 396)
        maxfev_171708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 51), 'maxfev', False)
        # Getting the type of 'factor' (line 396)
        factor_171709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 59), 'factor', False)
        # Getting the type of 'diag' (line 396)
        diag_171710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 67), 'diag', False)
        # Processing the call keyword arguments (line 395)
        kwargs_171711 = {}
        # Getting the type of '_minpack' (line 395)
        _minpack_171697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 17), '_minpack', False)
        # Obtaining the member '_lmder' of a type (line 395)
        _lmder_171698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 17), _minpack_171697, '_lmder')
        # Calling _lmder(args, kwargs) (line 395)
        _lmder_call_result_171712 = invoke(stypy.reporting.localization.Localization(__file__, 395, 17), _lmder_171698, *[func_171699, Dfun_171700, x0_171701, args_171702, full_output_171703, col_deriv_171704, ftol_171705, xtol_171706, gtol_171707, maxfev_171708, factor_171709, diag_171710], **kwargs_171711)
        
        # Assigning a type to the variable 'retval' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'retval', _lmder_call_result_171712)

        if (may_be_171636 and more_types_in_union_171637):
            # SSA join for if statement (line 383)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Dict to a Name (line 398):
    
    # Assigning a Dict to a Name (line 398):
    
    # Obtaining an instance of the builtin type 'dict' (line 398)
    dict_171713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 398)
    # Adding element type (key, value) (line 398)
    int_171714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 398)
    list_171715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 398)
    # Adding element type (line 398)
    str_171716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 18), 'str', 'Improper input parameters.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 17), list_171715, str_171716)
    # Adding element type (line 398)
    # Getting the type of 'TypeError' (line 398)
    TypeError_171717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 48), 'TypeError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 17), list_171715, TypeError_171717)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171714, list_171715))
    # Adding element type (key, value) (line 398)
    int_171718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 399)
    list_171719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 399)
    # Adding element type (line 399)
    str_171720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 18), 'str', 'Both actual and predicted relative reductions in the sum of squares\n  are at most %f')
    # Getting the type of 'ftol' (line 400)
    ftol_171721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 62), 'ftol')
    # Applying the binary operator '%' (line 399)
    result_mod_171722 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 18), '%', str_171720, ftol_171721)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 17), list_171719, result_mod_171722)
    # Adding element type (line 399)
    # Getting the type of 'None' (line 400)
    None_171723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 68), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 17), list_171719, None_171723)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171718, list_171719))
    # Adding element type (key, value) (line 398)
    int_171724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 401)
    list_171725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 401)
    # Adding element type (line 401)
    str_171726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 18), 'str', 'The relative error between two consecutive iterates is at most %f')
    # Getting the type of 'xtol' (line 402)
    xtol_171727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 45), 'xtol')
    # Applying the binary operator '%' (line 401)
    result_mod_171728 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 18), '%', str_171726, xtol_171727)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 17), list_171725, result_mod_171728)
    # Adding element type (line 401)
    # Getting the type of 'None' (line 402)
    None_171729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 51), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 17), list_171725, None_171729)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171724, list_171725))
    # Adding element type (key, value) (line 398)
    int_171730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 403)
    list_171731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 403)
    # Adding element type (line 403)
    str_171732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 18), 'str', 'Both actual and predicted relative reductions in the sum of squares\n  are at most %f and the relative error between two consecutive iterates is at \n  most %f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 406)
    tuple_171733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 406)
    # Adding element type (line 406)
    # Getting the type of 'ftol' (line 406)
    ftol_171734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 50), 'ftol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 50), tuple_171733, ftol_171734)
    # Adding element type (line 406)
    # Getting the type of 'xtol' (line 406)
    xtol_171735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 56), 'xtol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 50), tuple_171733, xtol_171735)
    
    # Applying the binary operator '%' (line 403)
    result_mod_171736 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 18), '%', str_171732, tuple_171733)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 17), list_171731, result_mod_171736)
    # Adding element type (line 403)
    # Getting the type of 'None' (line 406)
    None_171737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 63), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 17), list_171731, None_171737)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171730, list_171731))
    # Adding element type (key, value) (line 398)
    int_171738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 407)
    list_171739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 407)
    # Adding element type (line 407)
    str_171740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 18), 'str', 'The cosine of the angle between func(x) and any column of the\n  Jacobian is at most %f in absolute value')
    # Getting the type of 'gtol' (line 409)
    gtol_171741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 37), 'gtol')
    # Applying the binary operator '%' (line 407)
    result_mod_171742 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 18), '%', str_171740, gtol_171741)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 17), list_171739, result_mod_171742)
    # Adding element type (line 407)
    # Getting the type of 'None' (line 409)
    None_171743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 43), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 17), list_171739, None_171743)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171738, list_171739))
    # Adding element type (key, value) (line 398)
    int_171744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 410)
    list_171745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 410)
    # Adding element type (line 410)
    str_171746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 18), 'str', 'Number of calls to function has reached maxfev = %d.')
    # Getting the type of 'maxfev' (line 411)
    maxfev_171747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 35), 'maxfev')
    # Applying the binary operator '%' (line 410)
    result_mod_171748 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 18), '%', str_171746, maxfev_171747)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 17), list_171745, result_mod_171748)
    # Adding element type (line 410)
    # Getting the type of 'ValueError' (line 411)
    ValueError_171749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 43), 'ValueError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 17), list_171745, ValueError_171749)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171744, list_171745))
    # Adding element type (key, value) (line 398)
    int_171750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 412)
    list_171751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 412)
    # Adding element type (line 412)
    str_171752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 18), 'str', 'ftol=%f is too small, no further reduction in the sum of squares\n  is possible.')
    # Getting the type of 'ftol' (line 413)
    ftol_171753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 62), 'ftol')
    # Applying the binary operator '%' (line 412)
    result_mod_171754 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 18), '%', str_171752, ftol_171753)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 17), list_171751, result_mod_171754)
    # Adding element type (line 412)
    # Getting the type of 'ValueError' (line 414)
    ValueError_171755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 18), 'ValueError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 17), list_171751, ValueError_171755)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171750, list_171751))
    # Adding element type (key, value) (line 398)
    int_171756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 415)
    list_171757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 415)
    # Adding element type (line 415)
    str_171758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 18), 'str', 'xtol=%f is too small, no further improvement in the approximate\n  solution is possible.')
    # Getting the type of 'xtol' (line 416)
    xtol_171759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 63), 'xtol')
    # Applying the binary operator '%' (line 415)
    result_mod_171760 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 18), '%', str_171758, xtol_171759)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 17), list_171757, result_mod_171760)
    # Adding element type (line 415)
    # Getting the type of 'ValueError' (line 417)
    ValueError_171761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'ValueError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 17), list_171757, ValueError_171761)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171756, list_171757))
    # Adding element type (key, value) (line 398)
    int_171762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 14), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 418)
    list_171763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 418)
    # Adding element type (line 418)
    str_171764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 18), 'str', 'gtol=%f is too small, func(x) is orthogonal to the columns of\n  the Jacobian to machine precision.')
    # Getting the type of 'gtol' (line 420)
    gtol_171765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 33), 'gtol')
    # Applying the binary operator '%' (line 418)
    result_mod_171766 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 18), '%', str_171764, gtol_171765)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 17), list_171763, result_mod_171766)
    # Adding element type (line 418)
    # Getting the type of 'ValueError' (line 420)
    ValueError_171767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 39), 'ValueError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 17), list_171763, ValueError_171767)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (int_171762, list_171763))
    # Adding element type (key, value) (line 398)
    str_171768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 14), 'str', 'unknown')
    
    # Obtaining an instance of the builtin type 'list' (line 421)
    list_171769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 421)
    # Adding element type (line 421)
    str_171770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 26), 'str', 'Unknown error.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 25), list_171769, str_171770)
    # Adding element type (line 421)
    # Getting the type of 'TypeError' (line 421)
    TypeError_171771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 44), 'TypeError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 25), list_171769, TypeError_171771)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 13), dict_171713, (str_171768, list_171769))
    
    # Assigning a type to the variable 'errors' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'errors', dict_171713)
    
    # Assigning a Subscript to a Name (line 423):
    
    # Assigning a Subscript to a Name (line 423):
    
    # Obtaining the type of the subscript
    int_171772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 18), 'int')
    # Getting the type of 'retval' (line 423)
    retval_171773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'retval')
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___171774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 11), retval_171773, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_171775 = invoke(stypy.reporting.localization.Localization(__file__, 423, 11), getitem___171774, int_171772)
    
    # Assigning a type to the variable 'info' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'info', subscript_call_result_171775)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 425)
    info_171776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 7), 'info')
    
    # Obtaining an instance of the builtin type 'list' (line 425)
    list_171777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 425)
    # Adding element type (line 425)
    int_171778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 19), list_171777, int_171778)
    # Adding element type (line 425)
    int_171779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 19), list_171777, int_171779)
    # Adding element type (line 425)
    int_171780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 19), list_171777, int_171780)
    # Adding element type (line 425)
    int_171781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 19), list_171777, int_171781)
    
    # Applying the binary operator 'notin' (line 425)
    result_contains_171782 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 7), 'notin', info_171776, list_171777)
    
    
    # Getting the type of 'full_output' (line 425)
    full_output_171783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 40), 'full_output')
    # Applying the 'not' unary operator (line 425)
    result_not__171784 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 36), 'not', full_output_171783)
    
    # Applying the binary operator 'and' (line 425)
    result_and_keyword_171785 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 7), 'and', result_contains_171782, result_not__171784)
    
    # Testing the type of an if condition (line 425)
    if_condition_171786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 4), result_and_keyword_171785)
    # Assigning a type to the variable 'if_condition_171786' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'if_condition_171786', if_condition_171786)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'info' (line 426)
    info_171787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 11), 'info')
    
    # Obtaining an instance of the builtin type 'list' (line 426)
    list_171788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 426)
    # Adding element type (line 426)
    int_171789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_171788, int_171789)
    # Adding element type (line 426)
    int_171790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_171788, int_171790)
    # Adding element type (line 426)
    int_171791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_171788, int_171791)
    # Adding element type (line 426)
    int_171792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), list_171788, int_171792)
    
    # Applying the binary operator 'in' (line 426)
    result_contains_171793 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 11), 'in', info_171787, list_171788)
    
    # Testing the type of an if condition (line 426)
    if_condition_171794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 8), result_contains_171793)
    # Assigning a type to the variable 'if_condition_171794' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'if_condition_171794', if_condition_171794)
    # SSA begins for if statement (line 426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 427)
    # Processing the call arguments (line 427)
    
    # Obtaining the type of the subscript
    int_171797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 39), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'info' (line 427)
    info_171798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 33), 'info', False)
    # Getting the type of 'errors' (line 427)
    errors_171799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 26), 'errors', False)
    # Obtaining the member '__getitem__' of a type (line 427)
    getitem___171800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 26), errors_171799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 427)
    subscript_call_result_171801 = invoke(stypy.reporting.localization.Localization(__file__, 427, 26), getitem___171800, info_171798)
    
    # Obtaining the member '__getitem__' of a type (line 427)
    getitem___171802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 26), subscript_call_result_171801, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 427)
    subscript_call_result_171803 = invoke(stypy.reporting.localization.Localization(__file__, 427, 26), getitem___171802, int_171797)
    
    # Getting the type of 'RuntimeWarning' (line 427)
    RuntimeWarning_171804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 43), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 427)
    kwargs_171805 = {}
    # Getting the type of 'warnings' (line 427)
    warnings_171795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 427)
    warn_171796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 12), warnings_171795, 'warn')
    # Calling warn(args, kwargs) (line 427)
    warn_call_result_171806 = invoke(stypy.reporting.localization.Localization(__file__, 427, 12), warn_171796, *[subscript_call_result_171803, RuntimeWarning_171804], **kwargs_171805)
    
    # SSA branch for the else part of an if statement (line 426)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 429)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to (...): (line 430)
    # Processing the call arguments (line 430)
    
    # Obtaining the type of the subscript
    int_171814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 51), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'info' (line 430)
    info_171815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 45), 'info', False)
    # Getting the type of 'errors' (line 430)
    errors_171816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 38), 'errors', False)
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___171817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 38), errors_171816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_171818 = invoke(stypy.reporting.localization.Localization(__file__, 430, 38), getitem___171817, info_171815)
    
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___171819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 38), subscript_call_result_171818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_171820 = invoke(stypy.reporting.localization.Localization(__file__, 430, 38), getitem___171819, int_171814)
    
    # Processing the call keyword arguments (line 430)
    kwargs_171821 = {}
    
    # Obtaining the type of the subscript
    int_171807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 35), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'info' (line 430)
    info_171808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 29), 'info', False)
    # Getting the type of 'errors' (line 430)
    errors_171809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 22), 'errors', False)
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___171810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 22), errors_171809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_171811 = invoke(stypy.reporting.localization.Localization(__file__, 430, 22), getitem___171810, info_171808)
    
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___171812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 22), subscript_call_result_171811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_171813 = invoke(stypy.reporting.localization.Localization(__file__, 430, 22), getitem___171812, int_171807)
    
    # Calling (args, kwargs) (line 430)
    _call_result_171822 = invoke(stypy.reporting.localization.Localization(__file__, 430, 22), subscript_call_result_171813, *[subscript_call_result_171820], **kwargs_171821)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 430, 16), _call_result_171822, 'raise parameter', BaseException)
    # SSA branch for the except part of a try statement (line 429)
    # SSA branch for the except 'KeyError' branch of a try statement (line 429)
    module_type_store.open_ssa_branch('except')
    
    # Call to (...): (line 432)
    # Processing the call arguments (line 432)
    
    # Obtaining the type of the subscript
    int_171830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 61), 'int')
    
    # Obtaining the type of the subscript
    str_171831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 50), 'str', 'unknown')
    # Getting the type of 'errors' (line 432)
    errors_171832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 43), 'errors', False)
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___171833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 43), errors_171832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_171834 = invoke(stypy.reporting.localization.Localization(__file__, 432, 43), getitem___171833, str_171831)
    
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___171835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 43), subscript_call_result_171834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_171836 = invoke(stypy.reporting.localization.Localization(__file__, 432, 43), getitem___171835, int_171830)
    
    # Processing the call keyword arguments (line 432)
    kwargs_171837 = {}
    
    # Obtaining the type of the subscript
    int_171823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 40), 'int')
    
    # Obtaining the type of the subscript
    str_171824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 29), 'str', 'unknown')
    # Getting the type of 'errors' (line 432)
    errors_171825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 22), 'errors', False)
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___171826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 22), errors_171825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_171827 = invoke(stypy.reporting.localization.Localization(__file__, 432, 22), getitem___171826, str_171824)
    
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___171828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 22), subscript_call_result_171827, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_171829 = invoke(stypy.reporting.localization.Localization(__file__, 432, 22), getitem___171828, int_171823)
    
    # Calling (args, kwargs) (line 432)
    _call_result_171838 = invoke(stypy.reporting.localization.Localization(__file__, 432, 22), subscript_call_result_171829, *[subscript_call_result_171836], **kwargs_171837)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 432, 16), _call_result_171838, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 429)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 426)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 434):
    
    # Assigning a Subscript to a Name (line 434):
    
    # Obtaining the type of the subscript
    int_171839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 24), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'info' (line 434)
    info_171840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 18), 'info')
    # Getting the type of 'errors' (line 434)
    errors_171841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 11), 'errors')
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___171842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 11), errors_171841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_171843 = invoke(stypy.reporting.localization.Localization(__file__, 434, 11), getitem___171842, info_171840)
    
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___171844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 11), subscript_call_result_171843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_171845 = invoke(stypy.reporting.localization.Localization(__file__, 434, 11), getitem___171844, int_171839)
    
    # Assigning a type to the variable 'mesg' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'mesg', subscript_call_result_171845)
    
    # Getting the type of 'full_output' (line 435)
    full_output_171846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 7), 'full_output')
    # Testing the type of an if condition (line 435)
    if_condition_171847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 4), full_output_171846)
    # Assigning a type to the variable 'if_condition_171847' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'if_condition_171847', if_condition_171847)
    # SSA begins for if statement (line 435)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 436):
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'None' (line 436)
    None_171848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'None')
    # Assigning a type to the variable 'cov_x' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'cov_x', None_171848)
    
    
    # Getting the type of 'info' (line 437)
    info_171849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'info')
    
    # Obtaining an instance of the builtin type 'list' (line 437)
    list_171850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 437)
    # Adding element type (line 437)
    int_171851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), list_171850, int_171851)
    # Adding element type (line 437)
    int_171852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), list_171850, int_171852)
    # Adding element type (line 437)
    int_171853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), list_171850, int_171853)
    # Adding element type (line 437)
    int_171854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 19), list_171850, int_171854)
    
    # Applying the binary operator 'in' (line 437)
    result_contains_171855 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 11), 'in', info_171849, list_171850)
    
    # Testing the type of an if condition (line 437)
    if_condition_171856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 8), result_contains_171855)
    # Assigning a type to the variable 'if_condition_171856' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'if_condition_171856', if_condition_171856)
    # SSA begins for if statement (line 437)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 438, 12))
    
    # 'from numpy.dual import inv' statement (line 438)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
    import_171857 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 438, 12), 'numpy.dual')

    if (type(import_171857) is not StypyTypeError):

        if (import_171857 != 'pyd_module'):
            __import__(import_171857)
            sys_modules_171858 = sys.modules[import_171857]
            import_from_module(stypy.reporting.localization.Localization(__file__, 438, 12), 'numpy.dual', sys_modules_171858.module_type_store, module_type_store, ['inv'])
            nest_module(stypy.reporting.localization.Localization(__file__, 438, 12), __file__, sys_modules_171858, sys_modules_171858.module_type_store, module_type_store)
        else:
            from numpy.dual import inv

            import_from_module(stypy.reporting.localization.Localization(__file__, 438, 12), 'numpy.dual', None, module_type_store, ['inv'], [inv])

    else:
        # Assigning a type to the variable 'numpy.dual' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'numpy.dual', import_171857)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')
    
    
    # Assigning a Call to a Name (line 439):
    
    # Assigning a Call to a Name (line 439):
    
    # Call to take(...): (line 439)
    # Processing the call arguments (line 439)
    
    # Call to eye(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'n' (line 439)
    n_171861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 28), 'n', False)
    # Processing the call keyword arguments (line 439)
    kwargs_171862 = {}
    # Getting the type of 'eye' (line 439)
    eye_171860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'eye', False)
    # Calling eye(args, kwargs) (line 439)
    eye_call_result_171863 = invoke(stypy.reporting.localization.Localization(__file__, 439, 24), eye_171860, *[n_171861], **kwargs_171862)
    
    
    # Obtaining the type of the subscript
    str_171864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 42), 'str', 'ipvt')
    
    # Obtaining the type of the subscript
    int_171865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 39), 'int')
    # Getting the type of 'retval' (line 439)
    retval_171866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 32), 'retval', False)
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___171867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 32), retval_171866, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_171868 = invoke(stypy.reporting.localization.Localization(__file__, 439, 32), getitem___171867, int_171865)
    
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___171869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 32), subscript_call_result_171868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_171870 = invoke(stypy.reporting.localization.Localization(__file__, 439, 32), getitem___171869, str_171864)
    
    int_171871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 52), 'int')
    # Applying the binary operator '-' (line 439)
    result_sub_171872 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 32), '-', subscript_call_result_171870, int_171871)
    
    int_171873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 55), 'int')
    # Processing the call keyword arguments (line 439)
    kwargs_171874 = {}
    # Getting the type of 'take' (line 439)
    take_171859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 19), 'take', False)
    # Calling take(args, kwargs) (line 439)
    take_call_result_171875 = invoke(stypy.reporting.localization.Localization(__file__, 439, 19), take_171859, *[eye_call_result_171863, result_sub_171872, int_171873], **kwargs_171874)
    
    # Assigning a type to the variable 'perm' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'perm', take_call_result_171875)
    
    # Assigning a Call to a Name (line 440):
    
    # Assigning a Call to a Name (line 440):
    
    # Call to triu(...): (line 440)
    # Processing the call arguments (line 440)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 440)
    n_171877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 51), 'n', False)
    slice_171878 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 440, 21), None, n_171877, None)
    slice_171879 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 440, 21), None, None, None)
    
    # Call to transpose(...): (line 440)
    # Processing the call arguments (line 440)
    
    # Obtaining the type of the subscript
    str_171881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 41), 'str', 'fjac')
    
    # Obtaining the type of the subscript
    int_171882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 38), 'int')
    # Getting the type of 'retval' (line 440)
    retval_171883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 31), 'retval', False)
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___171884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 31), retval_171883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_171885 = invoke(stypy.reporting.localization.Localization(__file__, 440, 31), getitem___171884, int_171882)
    
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___171886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 31), subscript_call_result_171885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_171887 = invoke(stypy.reporting.localization.Localization(__file__, 440, 31), getitem___171886, str_171881)
    
    # Processing the call keyword arguments (line 440)
    kwargs_171888 = {}
    # Getting the type of 'transpose' (line 440)
    transpose_171880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 21), 'transpose', False)
    # Calling transpose(args, kwargs) (line 440)
    transpose_call_result_171889 = invoke(stypy.reporting.localization.Localization(__file__, 440, 21), transpose_171880, *[subscript_call_result_171887], **kwargs_171888)
    
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___171890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 21), transpose_call_result_171889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_171891 = invoke(stypy.reporting.localization.Localization(__file__, 440, 21), getitem___171890, (slice_171878, slice_171879))
    
    # Processing the call keyword arguments (line 440)
    kwargs_171892 = {}
    # Getting the type of 'triu' (line 440)
    triu_171876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'triu', False)
    # Calling triu(args, kwargs) (line 440)
    triu_call_result_171893 = invoke(stypy.reporting.localization.Localization(__file__, 440, 16), triu_171876, *[subscript_call_result_171891], **kwargs_171892)
    
    # Assigning a type to the variable 'r' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'r', triu_call_result_171893)
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to dot(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'r' (line 441)
    r_171895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'r', False)
    # Getting the type of 'perm' (line 441)
    perm_171896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'perm', False)
    # Processing the call keyword arguments (line 441)
    kwargs_171897 = {}
    # Getting the type of 'dot' (line 441)
    dot_171894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'dot', False)
    # Calling dot(args, kwargs) (line 441)
    dot_call_result_171898 = invoke(stypy.reporting.localization.Localization(__file__, 441, 16), dot_171894, *[r_171895, perm_171896], **kwargs_171897)
    
    # Assigning a type to the variable 'R' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'R', dot_call_result_171898)
    
    
    # SSA begins for try-except statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 443):
    
    # Assigning a Call to a Name (line 443):
    
    # Call to inv(...): (line 443)
    # Processing the call arguments (line 443)
    
    # Call to dot(...): (line 443)
    # Processing the call arguments (line 443)
    
    # Call to transpose(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'R' (line 443)
    R_171902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 42), 'R', False)
    # Processing the call keyword arguments (line 443)
    kwargs_171903 = {}
    # Getting the type of 'transpose' (line 443)
    transpose_171901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 32), 'transpose', False)
    # Calling transpose(args, kwargs) (line 443)
    transpose_call_result_171904 = invoke(stypy.reporting.localization.Localization(__file__, 443, 32), transpose_171901, *[R_171902], **kwargs_171903)
    
    # Getting the type of 'R' (line 443)
    R_171905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 46), 'R', False)
    # Processing the call keyword arguments (line 443)
    kwargs_171906 = {}
    # Getting the type of 'dot' (line 443)
    dot_171900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 28), 'dot', False)
    # Calling dot(args, kwargs) (line 443)
    dot_call_result_171907 = invoke(stypy.reporting.localization.Localization(__file__, 443, 28), dot_171900, *[transpose_call_result_171904, R_171905], **kwargs_171906)
    
    # Processing the call keyword arguments (line 443)
    kwargs_171908 = {}
    # Getting the type of 'inv' (line 443)
    inv_171899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 24), 'inv', False)
    # Calling inv(args, kwargs) (line 443)
    inv_call_result_171909 = invoke(stypy.reporting.localization.Localization(__file__, 443, 24), inv_171899, *[dot_call_result_171907], **kwargs_171908)
    
    # Assigning a type to the variable 'cov_x' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 16), 'cov_x', inv_call_result_171909)
    # SSA branch for the except part of a try statement (line 442)
    # SSA branch for the except 'Tuple' branch of a try statement (line 442)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 437)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 446)
    tuple_171910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 446)
    # Adding element type (line 446)
    
    # Obtaining the type of the subscript
    int_171911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 23), 'int')
    # Getting the type of 'retval' (line 446)
    retval_171912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'retval')
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___171913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 16), retval_171912, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_171914 = invoke(stypy.reporting.localization.Localization(__file__, 446, 16), getitem___171913, int_171911)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 16), tuple_171910, subscript_call_result_171914)
    # Adding element type (line 446)
    # Getting the type of 'cov_x' (line 446)
    cov_x_171915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 27), 'cov_x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 16), tuple_171910, cov_x_171915)
    
    
    # Obtaining the type of the subscript
    int_171916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 43), 'int')
    int_171917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 45), 'int')
    slice_171918 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 446, 36), int_171916, int_171917, None)
    # Getting the type of 'retval' (line 446)
    retval_171919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 36), 'retval')
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___171920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 36), retval_171919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_171921 = invoke(stypy.reporting.localization.Localization(__file__, 446, 36), getitem___171920, slice_171918)
    
    # Applying the binary operator '+' (line 446)
    result_add_171922 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 15), '+', tuple_171910, subscript_call_result_171921)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 446)
    tuple_171923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 446)
    # Adding element type (line 446)
    # Getting the type of 'mesg' (line 446)
    mesg_171924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 52), 'mesg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 52), tuple_171923, mesg_171924)
    # Adding element type (line 446)
    # Getting the type of 'info' (line 446)
    info_171925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 58), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 52), tuple_171923, info_171925)
    
    # Applying the binary operator '+' (line 446)
    result_add_171926 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 49), '+', result_add_171922, tuple_171923)
    
    # Assigning a type to the variable 'stypy_return_type' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'stypy_return_type', result_add_171926)
    # SSA branch for the else part of an if statement (line 435)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 448)
    tuple_171927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 448)
    # Adding element type (line 448)
    
    # Obtaining the type of the subscript
    int_171928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 23), 'int')
    # Getting the type of 'retval' (line 448)
    retval_171929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'retval')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___171930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 16), retval_171929, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_171931 = invoke(stypy.reporting.localization.Localization(__file__, 448, 16), getitem___171930, int_171928)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 16), tuple_171927, subscript_call_result_171931)
    # Adding element type (line 448)
    # Getting the type of 'info' (line 448)
    info_171932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 27), 'info')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 16), tuple_171927, info_171932)
    
    # Assigning a type to the variable 'stypy_return_type' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'stypy_return_type', tuple_171927)
    # SSA join for if statement (line 435)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'leastsq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'leastsq' in the type store
    # Getting the type of 'stypy_return_type' (line 261)
    stypy_return_type_171933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171933)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'leastsq'
    return stypy_return_type_171933

# Assigning a type to the variable 'leastsq' (line 261)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'leastsq', leastsq)

@norecursion
def _wrap_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_wrap_func'
    module_type_store = module_type_store.open_function_context('_wrap_func', 451, 0, False)
    
    # Passed parameters checking function
    _wrap_func.stypy_localization = localization
    _wrap_func.stypy_type_of_self = None
    _wrap_func.stypy_type_store = module_type_store
    _wrap_func.stypy_function_name = '_wrap_func'
    _wrap_func.stypy_param_names_list = ['func', 'xdata', 'ydata', 'transform']
    _wrap_func.stypy_varargs_param_name = None
    _wrap_func.stypy_kwargs_param_name = None
    _wrap_func.stypy_call_defaults = defaults
    _wrap_func.stypy_call_varargs = varargs
    _wrap_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_wrap_func', ['func', 'xdata', 'ydata', 'transform'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_wrap_func', localization, ['func', 'xdata', 'ydata', 'transform'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_wrap_func(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 452)
    # Getting the type of 'transform' (line 452)
    transform_171934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 7), 'transform')
    # Getting the type of 'None' (line 452)
    None_171935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 20), 'None')
    
    (may_be_171936, more_types_in_union_171937) = may_be_none(transform_171934, None_171935)

    if may_be_171936:

        if more_types_in_union_171937:
            # Runtime conditional SSA (line 452)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def func_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func_wrapped'
            module_type_store = module_type_store.open_function_context('func_wrapped', 453, 8, False)
            
            # Passed parameters checking function
            func_wrapped.stypy_localization = localization
            func_wrapped.stypy_type_of_self = None
            func_wrapped.stypy_type_store = module_type_store
            func_wrapped.stypy_function_name = 'func_wrapped'
            func_wrapped.stypy_param_names_list = ['params']
            func_wrapped.stypy_varargs_param_name = None
            func_wrapped.stypy_kwargs_param_name = None
            func_wrapped.stypy_call_defaults = defaults
            func_wrapped.stypy_call_varargs = varargs
            func_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func_wrapped', ['params'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func_wrapped', localization, ['params'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func_wrapped(...)' code ##################

            
            # Call to func(...): (line 454)
            # Processing the call arguments (line 454)
            # Getting the type of 'xdata' (line 454)
            xdata_171939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 'xdata', False)
            # Getting the type of 'params' (line 454)
            params_171940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 32), 'params', False)
            # Processing the call keyword arguments (line 454)
            kwargs_171941 = {}
            # Getting the type of 'func' (line 454)
            func_171938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 19), 'func', False)
            # Calling func(args, kwargs) (line 454)
            func_call_result_171942 = invoke(stypy.reporting.localization.Localization(__file__, 454, 19), func_171938, *[xdata_171939, params_171940], **kwargs_171941)
            
            # Getting the type of 'ydata' (line 454)
            ydata_171943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 42), 'ydata')
            # Applying the binary operator '-' (line 454)
            result_sub_171944 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 19), '-', func_call_result_171942, ydata_171943)
            
            # Assigning a type to the variable 'stypy_return_type' (line 454)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'stypy_return_type', result_sub_171944)
            
            # ################# End of 'func_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 453)
            stypy_return_type_171945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_171945)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func_wrapped'
            return stypy_return_type_171945

        # Assigning a type to the variable 'func_wrapped' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'func_wrapped', func_wrapped)

        if more_types_in_union_171937:
            # Runtime conditional SSA for else branch (line 452)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_171936) or more_types_in_union_171937):
        
        
        # Getting the type of 'transform' (line 455)
        transform_171946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 9), 'transform')
        # Obtaining the member 'ndim' of a type (line 455)
        ndim_171947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 9), transform_171946, 'ndim')
        int_171948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 27), 'int')
        # Applying the binary operator '==' (line 455)
        result_eq_171949 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 9), '==', ndim_171947, int_171948)
        
        # Testing the type of an if condition (line 455)
        if_condition_171950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 9), result_eq_171949)
        # Assigning a type to the variable 'if_condition_171950' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 9), 'if_condition_171950', if_condition_171950)
        # SSA begins for if statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def func_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func_wrapped'
            module_type_store = module_type_store.open_function_context('func_wrapped', 456, 8, False)
            
            # Passed parameters checking function
            func_wrapped.stypy_localization = localization
            func_wrapped.stypy_type_of_self = None
            func_wrapped.stypy_type_store = module_type_store
            func_wrapped.stypy_function_name = 'func_wrapped'
            func_wrapped.stypy_param_names_list = ['params']
            func_wrapped.stypy_varargs_param_name = None
            func_wrapped.stypy_kwargs_param_name = None
            func_wrapped.stypy_call_defaults = defaults
            func_wrapped.stypy_call_varargs = varargs
            func_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func_wrapped', ['params'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func_wrapped', localization, ['params'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func_wrapped(...)' code ##################

            # Getting the type of 'transform' (line 457)
            transform_171951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'transform')
            
            # Call to func(...): (line 457)
            # Processing the call arguments (line 457)
            # Getting the type of 'xdata' (line 457)
            xdata_171953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 37), 'xdata', False)
            # Getting the type of 'params' (line 457)
            params_171954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 45), 'params', False)
            # Processing the call keyword arguments (line 457)
            kwargs_171955 = {}
            # Getting the type of 'func' (line 457)
            func_171952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 32), 'func', False)
            # Calling func(args, kwargs) (line 457)
            func_call_result_171956 = invoke(stypy.reporting.localization.Localization(__file__, 457, 32), func_171952, *[xdata_171953, params_171954], **kwargs_171955)
            
            # Getting the type of 'ydata' (line 457)
            ydata_171957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 55), 'ydata')
            # Applying the binary operator '-' (line 457)
            result_sub_171958 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 32), '-', func_call_result_171956, ydata_171957)
            
            # Applying the binary operator '*' (line 457)
            result_mul_171959 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 19), '*', transform_171951, result_sub_171958)
            
            # Assigning a type to the variable 'stypy_return_type' (line 457)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'stypy_return_type', result_mul_171959)
            
            # ################# End of 'func_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 456)
            stypy_return_type_171960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_171960)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func_wrapped'
            return stypy_return_type_171960

        # Assigning a type to the variable 'func_wrapped' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'func_wrapped', func_wrapped)
        # SSA branch for the else part of an if statement (line 455)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def func_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func_wrapped'
            module_type_store = module_type_store.open_function_context('func_wrapped', 467, 8, False)
            
            # Passed parameters checking function
            func_wrapped.stypy_localization = localization
            func_wrapped.stypy_type_of_self = None
            func_wrapped.stypy_type_store = module_type_store
            func_wrapped.stypy_function_name = 'func_wrapped'
            func_wrapped.stypy_param_names_list = ['params']
            func_wrapped.stypy_varargs_param_name = None
            func_wrapped.stypy_kwargs_param_name = None
            func_wrapped.stypy_call_defaults = defaults
            func_wrapped.stypy_call_varargs = varargs
            func_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func_wrapped', ['params'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func_wrapped', localization, ['params'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func_wrapped(...)' code ##################

            
            # Call to solve_triangular(...): (line 468)
            # Processing the call arguments (line 468)
            # Getting the type of 'transform' (line 468)
            transform_171962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 36), 'transform', False)
            
            # Call to func(...): (line 468)
            # Processing the call arguments (line 468)
            # Getting the type of 'xdata' (line 468)
            xdata_171964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 52), 'xdata', False)
            # Getting the type of 'params' (line 468)
            params_171965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 60), 'params', False)
            # Processing the call keyword arguments (line 468)
            kwargs_171966 = {}
            # Getting the type of 'func' (line 468)
            func_171963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 47), 'func', False)
            # Calling func(args, kwargs) (line 468)
            func_call_result_171967 = invoke(stypy.reporting.localization.Localization(__file__, 468, 47), func_171963, *[xdata_171964, params_171965], **kwargs_171966)
            
            # Getting the type of 'ydata' (line 468)
            ydata_171968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 70), 'ydata', False)
            # Applying the binary operator '-' (line 468)
            result_sub_171969 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 47), '-', func_call_result_171967, ydata_171968)
            
            # Processing the call keyword arguments (line 468)
            # Getting the type of 'True' (line 468)
            True_171970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 83), 'True', False)
            keyword_171971 = True_171970
            kwargs_171972 = {'lower': keyword_171971}
            # Getting the type of 'solve_triangular' (line 468)
            solve_triangular_171961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 19), 'solve_triangular', False)
            # Calling solve_triangular(args, kwargs) (line 468)
            solve_triangular_call_result_171973 = invoke(stypy.reporting.localization.Localization(__file__, 468, 19), solve_triangular_171961, *[transform_171962, result_sub_171969], **kwargs_171972)
            
            # Assigning a type to the variable 'stypy_return_type' (line 468)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'stypy_return_type', solve_triangular_call_result_171973)
            
            # ################# End of 'func_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 467)
            stypy_return_type_171974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_171974)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func_wrapped'
            return stypy_return_type_171974

        # Assigning a type to the variable 'func_wrapped' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'func_wrapped', func_wrapped)
        # SSA join for if statement (line 455)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_171936 and more_types_in_union_171937):
            # SSA join for if statement (line 452)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'func_wrapped' (line 469)
    func_wrapped_171975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 11), 'func_wrapped')
    # Assigning a type to the variable 'stypy_return_type' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'stypy_return_type', func_wrapped_171975)
    
    # ################# End of '_wrap_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_wrap_func' in the type store
    # Getting the type of 'stypy_return_type' (line 451)
    stypy_return_type_171976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_171976)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_wrap_func'
    return stypy_return_type_171976

# Assigning a type to the variable '_wrap_func' (line 451)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), '_wrap_func', _wrap_func)

@norecursion
def _wrap_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_wrap_jac'
    module_type_store = module_type_store.open_function_context('_wrap_jac', 472, 0, False)
    
    # Passed parameters checking function
    _wrap_jac.stypy_localization = localization
    _wrap_jac.stypy_type_of_self = None
    _wrap_jac.stypy_type_store = module_type_store
    _wrap_jac.stypy_function_name = '_wrap_jac'
    _wrap_jac.stypy_param_names_list = ['jac', 'xdata', 'transform']
    _wrap_jac.stypy_varargs_param_name = None
    _wrap_jac.stypy_kwargs_param_name = None
    _wrap_jac.stypy_call_defaults = defaults
    _wrap_jac.stypy_call_varargs = varargs
    _wrap_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_wrap_jac', ['jac', 'xdata', 'transform'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_wrap_jac', localization, ['jac', 'xdata', 'transform'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_wrap_jac(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 473)
    # Getting the type of 'transform' (line 473)
    transform_171977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 7), 'transform')
    # Getting the type of 'None' (line 473)
    None_171978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'None')
    
    (may_be_171979, more_types_in_union_171980) = may_be_none(transform_171977, None_171978)

    if may_be_171979:

        if more_types_in_union_171980:
            # Runtime conditional SSA (line 473)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def jac_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'jac_wrapped'
            module_type_store = module_type_store.open_function_context('jac_wrapped', 474, 8, False)
            
            # Passed parameters checking function
            jac_wrapped.stypy_localization = localization
            jac_wrapped.stypy_type_of_self = None
            jac_wrapped.stypy_type_store = module_type_store
            jac_wrapped.stypy_function_name = 'jac_wrapped'
            jac_wrapped.stypy_param_names_list = ['params']
            jac_wrapped.stypy_varargs_param_name = None
            jac_wrapped.stypy_kwargs_param_name = None
            jac_wrapped.stypy_call_defaults = defaults
            jac_wrapped.stypy_call_varargs = varargs
            jac_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['params'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'jac_wrapped', localization, ['params'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'jac_wrapped(...)' code ##################

            
            # Call to jac(...): (line 475)
            # Processing the call arguments (line 475)
            # Getting the type of 'xdata' (line 475)
            xdata_171982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 23), 'xdata', False)
            # Getting the type of 'params' (line 475)
            params_171983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 31), 'params', False)
            # Processing the call keyword arguments (line 475)
            kwargs_171984 = {}
            # Getting the type of 'jac' (line 475)
            jac_171981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'jac', False)
            # Calling jac(args, kwargs) (line 475)
            jac_call_result_171985 = invoke(stypy.reporting.localization.Localization(__file__, 475, 19), jac_171981, *[xdata_171982, params_171983], **kwargs_171984)
            
            # Assigning a type to the variable 'stypy_return_type' (line 475)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'stypy_return_type', jac_call_result_171985)
            
            # ################# End of 'jac_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'jac_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 474)
            stypy_return_type_171986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_171986)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'jac_wrapped'
            return stypy_return_type_171986

        # Assigning a type to the variable 'jac_wrapped' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'jac_wrapped', jac_wrapped)

        if more_types_in_union_171980:
            # Runtime conditional SSA for else branch (line 473)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_171979) or more_types_in_union_171980):
        
        
        # Getting the type of 'transform' (line 476)
        transform_171987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 9), 'transform')
        # Obtaining the member 'ndim' of a type (line 476)
        ndim_171988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 9), transform_171987, 'ndim')
        int_171989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 27), 'int')
        # Applying the binary operator '==' (line 476)
        result_eq_171990 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 9), '==', ndim_171988, int_171989)
        
        # Testing the type of an if condition (line 476)
        if_condition_171991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 476, 9), result_eq_171990)
        # Assigning a type to the variable 'if_condition_171991' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 9), 'if_condition_171991', if_condition_171991)
        # SSA begins for if statement (line 476)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def jac_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'jac_wrapped'
            module_type_store = module_type_store.open_function_context('jac_wrapped', 477, 8, False)
            
            # Passed parameters checking function
            jac_wrapped.stypy_localization = localization
            jac_wrapped.stypy_type_of_self = None
            jac_wrapped.stypy_type_store = module_type_store
            jac_wrapped.stypy_function_name = 'jac_wrapped'
            jac_wrapped.stypy_param_names_list = ['params']
            jac_wrapped.stypy_varargs_param_name = None
            jac_wrapped.stypy_kwargs_param_name = None
            jac_wrapped.stypy_call_defaults = defaults
            jac_wrapped.stypy_call_varargs = varargs
            jac_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['params'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'jac_wrapped', localization, ['params'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'jac_wrapped(...)' code ##################

            
            # Obtaining the type of the subscript
            slice_171992 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 478, 19), None, None, None)
            # Getting the type of 'np' (line 478)
            np_171993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'np')
            # Obtaining the member 'newaxis' of a type (line 478)
            newaxis_171994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 32), np_171993, 'newaxis')
            # Getting the type of 'transform' (line 478)
            transform_171995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 19), 'transform')
            # Obtaining the member '__getitem__' of a type (line 478)
            getitem___171996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 19), transform_171995, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 478)
            subscript_call_result_171997 = invoke(stypy.reporting.localization.Localization(__file__, 478, 19), getitem___171996, (slice_171992, newaxis_171994))
            
            
            # Call to asarray(...): (line 478)
            # Processing the call arguments (line 478)
            
            # Call to jac(...): (line 478)
            # Processing the call arguments (line 478)
            # Getting the type of 'xdata' (line 478)
            xdata_172001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 61), 'xdata', False)
            # Getting the type of 'params' (line 478)
            params_172002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 69), 'params', False)
            # Processing the call keyword arguments (line 478)
            kwargs_172003 = {}
            # Getting the type of 'jac' (line 478)
            jac_172000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 57), 'jac', False)
            # Calling jac(args, kwargs) (line 478)
            jac_call_result_172004 = invoke(stypy.reporting.localization.Localization(__file__, 478, 57), jac_172000, *[xdata_172001, params_172002], **kwargs_172003)
            
            # Processing the call keyword arguments (line 478)
            kwargs_172005 = {}
            # Getting the type of 'np' (line 478)
            np_171998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 46), 'np', False)
            # Obtaining the member 'asarray' of a type (line 478)
            asarray_171999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 46), np_171998, 'asarray')
            # Calling asarray(args, kwargs) (line 478)
            asarray_call_result_172006 = invoke(stypy.reporting.localization.Localization(__file__, 478, 46), asarray_171999, *[jac_call_result_172004], **kwargs_172005)
            
            # Applying the binary operator '*' (line 478)
            result_mul_172007 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 19), '*', subscript_call_result_171997, asarray_call_result_172006)
            
            # Assigning a type to the variable 'stypy_return_type' (line 478)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'stypy_return_type', result_mul_172007)
            
            # ################# End of 'jac_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'jac_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 477)
            stypy_return_type_172008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_172008)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'jac_wrapped'
            return stypy_return_type_172008

        # Assigning a type to the variable 'jac_wrapped' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'jac_wrapped', jac_wrapped)
        # SSA branch for the else part of an if statement (line 476)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def jac_wrapped(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'jac_wrapped'
            module_type_store = module_type_store.open_function_context('jac_wrapped', 480, 8, False)
            
            # Passed parameters checking function
            jac_wrapped.stypy_localization = localization
            jac_wrapped.stypy_type_of_self = None
            jac_wrapped.stypy_type_store = module_type_store
            jac_wrapped.stypy_function_name = 'jac_wrapped'
            jac_wrapped.stypy_param_names_list = ['params']
            jac_wrapped.stypy_varargs_param_name = None
            jac_wrapped.stypy_kwargs_param_name = None
            jac_wrapped.stypy_call_defaults = defaults
            jac_wrapped.stypy_call_varargs = varargs
            jac_wrapped.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'jac_wrapped', ['params'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'jac_wrapped', localization, ['params'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'jac_wrapped(...)' code ##################

            
            # Call to solve_triangular(...): (line 481)
            # Processing the call arguments (line 481)
            # Getting the type of 'transform' (line 481)
            transform_172010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 36), 'transform', False)
            
            # Call to asarray(...): (line 481)
            # Processing the call arguments (line 481)
            
            # Call to jac(...): (line 481)
            # Processing the call arguments (line 481)
            # Getting the type of 'xdata' (line 481)
            xdata_172014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 62), 'xdata', False)
            # Getting the type of 'params' (line 481)
            params_172015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 70), 'params', False)
            # Processing the call keyword arguments (line 481)
            kwargs_172016 = {}
            # Getting the type of 'jac' (line 481)
            jac_172013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 58), 'jac', False)
            # Calling jac(args, kwargs) (line 481)
            jac_call_result_172017 = invoke(stypy.reporting.localization.Localization(__file__, 481, 58), jac_172013, *[xdata_172014, params_172015], **kwargs_172016)
            
            # Processing the call keyword arguments (line 481)
            kwargs_172018 = {}
            # Getting the type of 'np' (line 481)
            np_172011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 47), 'np', False)
            # Obtaining the member 'asarray' of a type (line 481)
            asarray_172012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 47), np_172011, 'asarray')
            # Calling asarray(args, kwargs) (line 481)
            asarray_call_result_172019 = invoke(stypy.reporting.localization.Localization(__file__, 481, 47), asarray_172012, *[jac_call_result_172017], **kwargs_172018)
            
            # Processing the call keyword arguments (line 481)
            # Getting the type of 'True' (line 481)
            True_172020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 86), 'True', False)
            keyword_172021 = True_172020
            kwargs_172022 = {'lower': keyword_172021}
            # Getting the type of 'solve_triangular' (line 481)
            solve_triangular_172009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 19), 'solve_triangular', False)
            # Calling solve_triangular(args, kwargs) (line 481)
            solve_triangular_call_result_172023 = invoke(stypy.reporting.localization.Localization(__file__, 481, 19), solve_triangular_172009, *[transform_172010, asarray_call_result_172019], **kwargs_172022)
            
            # Assigning a type to the variable 'stypy_return_type' (line 481)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'stypy_return_type', solve_triangular_call_result_172023)
            
            # ################# End of 'jac_wrapped(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'jac_wrapped' in the type store
            # Getting the type of 'stypy_return_type' (line 480)
            stypy_return_type_172024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_172024)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'jac_wrapped'
            return stypy_return_type_172024

        # Assigning a type to the variable 'jac_wrapped' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'jac_wrapped', jac_wrapped)
        # SSA join for if statement (line 476)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_171979 and more_types_in_union_171980):
            # SSA join for if statement (line 473)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'jac_wrapped' (line 482)
    jac_wrapped_172025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 11), 'jac_wrapped')
    # Assigning a type to the variable 'stypy_return_type' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'stypy_return_type', jac_wrapped_172025)
    
    # ################# End of '_wrap_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_wrap_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 472)
    stypy_return_type_172026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172026)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_wrap_jac'
    return stypy_return_type_172026

# Assigning a type to the variable '_wrap_jac' (line 472)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 0), '_wrap_jac', _wrap_jac)

@norecursion
def _initialize_feasible(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_initialize_feasible'
    module_type_store = module_type_store.open_function_context('_initialize_feasible', 485, 0, False)
    
    # Passed parameters checking function
    _initialize_feasible.stypy_localization = localization
    _initialize_feasible.stypy_type_of_self = None
    _initialize_feasible.stypy_type_store = module_type_store
    _initialize_feasible.stypy_function_name = '_initialize_feasible'
    _initialize_feasible.stypy_param_names_list = ['lb', 'ub']
    _initialize_feasible.stypy_varargs_param_name = None
    _initialize_feasible.stypy_kwargs_param_name = None
    _initialize_feasible.stypy_call_defaults = defaults
    _initialize_feasible.stypy_call_varargs = varargs
    _initialize_feasible.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_initialize_feasible', ['lb', 'ub'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_initialize_feasible', localization, ['lb', 'ub'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_initialize_feasible(...)' code ##################

    
    # Assigning a Call to a Name (line 486):
    
    # Assigning a Call to a Name (line 486):
    
    # Call to ones_like(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'lb' (line 486)
    lb_172029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 22), 'lb', False)
    # Processing the call keyword arguments (line 486)
    kwargs_172030 = {}
    # Getting the type of 'np' (line 486)
    np_172027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 9), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 486)
    ones_like_172028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 9), np_172027, 'ones_like')
    # Calling ones_like(args, kwargs) (line 486)
    ones_like_call_result_172031 = invoke(stypy.reporting.localization.Localization(__file__, 486, 9), ones_like_172028, *[lb_172029], **kwargs_172030)
    
    # Assigning a type to the variable 'p0' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'p0', ones_like_call_result_172031)
    
    # Assigning a Call to a Name (line 487):
    
    # Assigning a Call to a Name (line 487):
    
    # Call to isfinite(...): (line 487)
    # Processing the call arguments (line 487)
    # Getting the type of 'lb' (line 487)
    lb_172034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 28), 'lb', False)
    # Processing the call keyword arguments (line 487)
    kwargs_172035 = {}
    # Getting the type of 'np' (line 487)
    np_172032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 487)
    isfinite_172033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 16), np_172032, 'isfinite')
    # Calling isfinite(args, kwargs) (line 487)
    isfinite_call_result_172036 = invoke(stypy.reporting.localization.Localization(__file__, 487, 16), isfinite_172033, *[lb_172034], **kwargs_172035)
    
    # Assigning a type to the variable 'lb_finite' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'lb_finite', isfinite_call_result_172036)
    
    # Assigning a Call to a Name (line 488):
    
    # Assigning a Call to a Name (line 488):
    
    # Call to isfinite(...): (line 488)
    # Processing the call arguments (line 488)
    # Getting the type of 'ub' (line 488)
    ub_172039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 28), 'ub', False)
    # Processing the call keyword arguments (line 488)
    kwargs_172040 = {}
    # Getting the type of 'np' (line 488)
    np_172037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 16), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 488)
    isfinite_172038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 16), np_172037, 'isfinite')
    # Calling isfinite(args, kwargs) (line 488)
    isfinite_call_result_172041 = invoke(stypy.reporting.localization.Localization(__file__, 488, 16), isfinite_172038, *[ub_172039], **kwargs_172040)
    
    # Assigning a type to the variable 'ub_finite' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'ub_finite', isfinite_call_result_172041)
    
    # Assigning a BinOp to a Name (line 490):
    
    # Assigning a BinOp to a Name (line 490):
    # Getting the type of 'lb_finite' (line 490)
    lb_finite_172042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'lb_finite')
    # Getting the type of 'ub_finite' (line 490)
    ub_finite_172043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'ub_finite')
    # Applying the binary operator '&' (line 490)
    result_and__172044 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), '&', lb_finite_172042, ub_finite_172043)
    
    # Assigning a type to the variable 'mask' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'mask', result_and__172044)
    
    # Assigning a BinOp to a Subscript (line 491):
    
    # Assigning a BinOp to a Subscript (line 491):
    float_172045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 15), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 491)
    mask_172046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 25), 'mask')
    # Getting the type of 'lb' (line 491)
    lb_172047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 22), 'lb')
    # Obtaining the member '__getitem__' of a type (line 491)
    getitem___172048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 22), lb_172047, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 491)
    subscript_call_result_172049 = invoke(stypy.reporting.localization.Localization(__file__, 491, 22), getitem___172048, mask_172046)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 491)
    mask_172050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 36), 'mask')
    # Getting the type of 'ub' (line 491)
    ub_172051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 33), 'ub')
    # Obtaining the member '__getitem__' of a type (line 491)
    getitem___172052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 33), ub_172051, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 491)
    subscript_call_result_172053 = invoke(stypy.reporting.localization.Localization(__file__, 491, 33), getitem___172052, mask_172050)
    
    # Applying the binary operator '+' (line 491)
    result_add_172054 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 22), '+', subscript_call_result_172049, subscript_call_result_172053)
    
    # Applying the binary operator '*' (line 491)
    result_mul_172055 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 15), '*', float_172045, result_add_172054)
    
    # Getting the type of 'p0' (line 491)
    p0_172056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'p0')
    # Getting the type of 'mask' (line 491)
    mask_172057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 7), 'mask')
    # Storing an element on a container (line 491)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 4), p0_172056, (mask_172057, result_mul_172055))
    
    # Assigning a BinOp to a Name (line 493):
    
    # Assigning a BinOp to a Name (line 493):
    # Getting the type of 'lb_finite' (line 493)
    lb_finite_172058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'lb_finite')
    
    # Getting the type of 'ub_finite' (line 493)
    ub_finite_172059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 24), 'ub_finite')
    # Applying the '~' unary operator (line 493)
    result_inv_172060 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 23), '~', ub_finite_172059)
    
    # Applying the binary operator '&' (line 493)
    result_and__172061 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 11), '&', lb_finite_172058, result_inv_172060)
    
    # Assigning a type to the variable 'mask' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'mask', result_and__172061)
    
    # Assigning a BinOp to a Subscript (line 494):
    
    # Assigning a BinOp to a Subscript (line 494):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 494)
    mask_172062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 18), 'mask')
    # Getting the type of 'lb' (line 494)
    lb_172063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'lb')
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___172064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 15), lb_172063, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 494)
    subscript_call_result_172065 = invoke(stypy.reporting.localization.Localization(__file__, 494, 15), getitem___172064, mask_172062)
    
    int_172066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 26), 'int')
    # Applying the binary operator '+' (line 494)
    result_add_172067 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 15), '+', subscript_call_result_172065, int_172066)
    
    # Getting the type of 'p0' (line 494)
    p0_172068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'p0')
    # Getting the type of 'mask' (line 494)
    mask_172069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 7), 'mask')
    # Storing an element on a container (line 494)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 4), p0_172068, (mask_172069, result_add_172067))
    
    # Assigning a BinOp to a Name (line 496):
    
    # Assigning a BinOp to a Name (line 496):
    
    # Getting the type of 'lb_finite' (line 496)
    lb_finite_172070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'lb_finite')
    # Applying the '~' unary operator (line 496)
    result_inv_172071 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 11), '~', lb_finite_172070)
    
    # Getting the type of 'ub_finite' (line 496)
    ub_finite_172072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 24), 'ub_finite')
    # Applying the binary operator '&' (line 496)
    result_and__172073 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 11), '&', result_inv_172071, ub_finite_172072)
    
    # Assigning a type to the variable 'mask' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'mask', result_and__172073)
    
    # Assigning a BinOp to a Subscript (line 497):
    
    # Assigning a BinOp to a Subscript (line 497):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 497)
    mask_172074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 18), 'mask')
    # Getting the type of 'ub' (line 497)
    ub_172075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), 'ub')
    # Obtaining the member '__getitem__' of a type (line 497)
    getitem___172076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 15), ub_172075, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 497)
    subscript_call_result_172077 = invoke(stypy.reporting.localization.Localization(__file__, 497, 15), getitem___172076, mask_172074)
    
    int_172078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 26), 'int')
    # Applying the binary operator '-' (line 497)
    result_sub_172079 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 15), '-', subscript_call_result_172077, int_172078)
    
    # Getting the type of 'p0' (line 497)
    p0_172080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'p0')
    # Getting the type of 'mask' (line 497)
    mask_172081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 7), 'mask')
    # Storing an element on a container (line 497)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 4), p0_172080, (mask_172081, result_sub_172079))
    # Getting the type of 'p0' (line 499)
    p0_172082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'p0')
    # Assigning a type to the variable 'stypy_return_type' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'stypy_return_type', p0_172082)
    
    # ################# End of '_initialize_feasible(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_initialize_feasible' in the type store
    # Getting the type of 'stypy_return_type' (line 485)
    stypy_return_type_172083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172083)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_initialize_feasible'
    return stypy_return_type_172083

# Assigning a type to the variable '_initialize_feasible' (line 485)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), '_initialize_feasible', _initialize_feasible)

@norecursion
def curve_fit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 502)
    None_172084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 34), 'None')
    # Getting the type of 'None' (line 502)
    None_172085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 46), 'None')
    # Getting the type of 'False' (line 502)
    False_172086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 67), 'False')
    # Getting the type of 'True' (line 503)
    True_172087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 27), 'True')
    
    # Obtaining an instance of the builtin type 'tuple' (line 503)
    tuple_172088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 503)
    # Adding element type (line 503)
    
    # Getting the type of 'np' (line 503)
    np_172089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 42), 'np')
    # Obtaining the member 'inf' of a type (line 503)
    inf_172090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 42), np_172089, 'inf')
    # Applying the 'usub' unary operator (line 503)
    result___neg___172091 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 41), 'usub', inf_172090)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 41), tuple_172088, result___neg___172091)
    # Adding element type (line 503)
    # Getting the type of 'np' (line 503)
    np_172092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 50), 'np')
    # Obtaining the member 'inf' of a type (line 503)
    inf_172093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 50), np_172092, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 41), tuple_172088, inf_172093)
    
    # Getting the type of 'None' (line 503)
    None_172094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 66), 'None')
    # Getting the type of 'None' (line 504)
    None_172095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 18), 'None')
    defaults = [None_172084, None_172085, False_172086, True_172087, tuple_172088, None_172094, None_172095]
    # Create a new context for function 'curve_fit'
    module_type_store = module_type_store.open_function_context('curve_fit', 502, 0, False)
    
    # Passed parameters checking function
    curve_fit.stypy_localization = localization
    curve_fit.stypy_type_of_self = None
    curve_fit.stypy_type_store = module_type_store
    curve_fit.stypy_function_name = 'curve_fit'
    curve_fit.stypy_param_names_list = ['f', 'xdata', 'ydata', 'p0', 'sigma', 'absolute_sigma', 'check_finite', 'bounds', 'method', 'jac']
    curve_fit.stypy_varargs_param_name = None
    curve_fit.stypy_kwargs_param_name = 'kwargs'
    curve_fit.stypy_call_defaults = defaults
    curve_fit.stypy_call_varargs = varargs
    curve_fit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'curve_fit', ['f', 'xdata', 'ydata', 'p0', 'sigma', 'absolute_sigma', 'check_finite', 'bounds', 'method', 'jac'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'curve_fit', localization, ['f', 'xdata', 'ydata', 'p0', 'sigma', 'absolute_sigma', 'check_finite', 'bounds', 'method', 'jac'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'curve_fit(...)' code ##################

    str_172096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, (-1)), 'str', "\n    Use non-linear least squares to fit a function, f, to data.\n\n    Assumes ``ydata = f(xdata, *params) + eps``\n\n    Parameters\n    ----------\n    f : callable\n        The model function, f(x, ...).  It must take the independent\n        variable as the first argument and the parameters to fit as\n        separate remaining arguments.\n    xdata : An M-length sequence or an (k,M)-shaped array for functions with k predictors\n        The independent variable where the data is measured.\n    ydata : M-length sequence\n        The dependent data --- nominally f(xdata, ...)\n    p0 : None, scalar, or N-length sequence, optional\n        Initial guess for the parameters.  If None, then the initial\n        values will all be 1 (if the number of parameters for the function\n        can be determined using introspection, otherwise a ValueError\n        is raised).\n    sigma : None or M-length sequence or MxM array, optional\n        Determines the uncertainty in `ydata`. If we define residuals as\n        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`\n        depends on its number of dimensions:\n\n            - A 1-d `sigma` should contain values of standard deviations of\n              errors in `ydata`. In this case, the optimized function is\n              ``chisq = sum((r / sigma) ** 2)``.\n\n            - A 2-d `sigma` should contain the covariance matrix of\n              errors in `ydata`. In this case, the optimized function is\n              ``chisq = r.T @ inv(sigma) @ r``.\n\n              .. versionadded:: 0.19\n\n        None (default) is equivalent of 1-d `sigma` filled with ones.\n    absolute_sigma : bool, optional\n        If True, `sigma` is used in an absolute sense and the estimated parameter\n        covariance `pcov` reflects these absolute values.\n\n        If False, only the relative magnitudes of the `sigma` values matter.\n        The returned parameter covariance matrix `pcov` is based on scaling\n        `sigma` by a constant factor. This constant is set by demanding that the\n        reduced `chisq` for the optimal parameters `popt` when using the\n        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to\n        match the sample variance of the residuals after the fit.\n        Mathematically,\n        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``\n    check_finite : bool, optional\n        If True, check that the input arrays do not contain nans of infs,\n        and raise a ValueError if they do. Setting this parameter to\n        False may silently produce nonsensical results if the input arrays\n        do contain nans. Default is True.\n    bounds : 2-tuple of array_like, optional\n        Lower and upper bounds on independent variables. Defaults to no bounds.\n        Each element of the tuple must be either an array with the length equal\n        to the number of parameters, or a scalar (in which case the bound is\n        taken to be the same for all parameters.) Use ``np.inf`` with an\n        appropriate sign to disable bounds on all or some parameters.\n\n        .. versionadded:: 0.17\n    method : {'lm', 'trf', 'dogbox'}, optional\n        Method to use for optimization.  See `least_squares` for more details.\n        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are\n        provided. The method 'lm' won't work when the number of observations\n        is less than the number of variables, use 'trf' or 'dogbox' in this\n        case.\n\n        .. versionadded:: 0.17\n    jac : callable, string or None, optional\n        Function with signature ``jac(x, ...)`` which computes the Jacobian\n        matrix of the model function with respect to parameters as a dense\n        array_like structure. It will be scaled according to provided `sigma`.\n        If None (default), the Jacobian will be estimated numerically.\n        String keywords for 'trf' and 'dogbox' methods can be used to select\n        a finite difference scheme, see `least_squares`.\n\n        .. versionadded:: 0.18\n    kwargs\n        Keyword arguments passed to `leastsq` for ``method='lm'`` or\n        `least_squares` otherwise.\n\n    Returns\n    -------\n    popt : array\n        Optimal values for the parameters so that the sum of the squared\n        residuals of ``f(xdata, *popt) - ydata`` is minimized\n    pcov : 2d array\n        The estimated covariance of popt. The diagonals provide the variance\n        of the parameter estimate. To compute one standard deviation errors\n        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.\n\n        How the `sigma` parameter affects the estimated covariance\n        depends on `absolute_sigma` argument, as described above.\n\n        If the Jacobian matrix at the solution doesn't have a full rank, then\n        'lm' method returns a matrix filled with ``np.inf``, on the other hand\n        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute\n        the covariance matrix.\n\n    Raises\n    ------\n    ValueError\n        if either `ydata` or `xdata` contain NaNs, or if incompatible options\n        are used.\n\n    RuntimeError\n        if the least-squares minimization fails.\n\n    OptimizeWarning\n        if covariance of the parameters can not be estimated.\n\n    See Also\n    --------\n    least_squares : Minimize the sum of squares of nonlinear functions.\n    scipy.stats.linregress : Calculate a linear least squares regression for\n                             two sets of measurements.\n\n    Notes\n    -----\n    With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm\n    through `leastsq`. Note that this algorithm can only deal with\n    unconstrained problems.\n\n    Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to\n    the docstring of `least_squares` for more information.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.optimize import curve_fit\n\n    >>> def func(x, a, b, c):\n    ...     return a * np.exp(-b * x) + c\n\n    Define the data to be fit with some noise:\n\n    >>> xdata = np.linspace(0, 4, 50)\n    >>> y = func(xdata, 2.5, 1.3, 0.5)\n    >>> np.random.seed(1729)\n    >>> y_noise = 0.2 * np.random.normal(size=xdata.size)\n    >>> ydata = y + y_noise\n    >>> plt.plot(xdata, ydata, 'b-', label='data')\n\n    Fit for the parameters a, b, c of the function `func`:\n\n    >>> popt, pcov = curve_fit(func, xdata, ydata)\n    >>> popt\n    array([ 2.55423706,  1.35190947,  0.47450618])\n    >>> plt.plot(xdata, func(xdata, *popt), 'r-',\n    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))\n\n    Constrain the optimization to the region of ``0 <= a <= 3``,\n    ``0 <= b <= 1`` and ``0 <= c <= 0.5``:\n\n    >>> popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))\n    >>> popt\n    array([ 2.43708906,  1.        ,  0.35015434])\n    >>> plt.plot(xdata, func(xdata, *popt), 'g--',\n    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))\n\n    >>> plt.xlabel('x')\n    >>> plt.ylabel('y')\n    >>> plt.legend()\n    >>> plt.show()\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 673)
    # Getting the type of 'p0' (line 673)
    p0_172097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 7), 'p0')
    # Getting the type of 'None' (line 673)
    None_172098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 13), 'None')
    
    (may_be_172099, more_types_in_union_172100) = may_be_none(p0_172097, None_172098)

    if may_be_172099:

        if more_types_in_union_172100:
            # Runtime conditional SSA (line 673)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 675, 8))
        
        # 'from scipy._lib._util import _getargspec' statement (line 675)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
        import_172101 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 675, 8), 'scipy._lib._util')

        if (type(import_172101) is not StypyTypeError):

            if (import_172101 != 'pyd_module'):
                __import__(import_172101)
                sys_modules_172102 = sys.modules[import_172101]
                import_from_module(stypy.reporting.localization.Localization(__file__, 675, 8), 'scipy._lib._util', sys_modules_172102.module_type_store, module_type_store, ['getargspec_no_self'])
                nest_module(stypy.reporting.localization.Localization(__file__, 675, 8), __file__, sys_modules_172102, sys_modules_172102.module_type_store, module_type_store)
            else:
                from scipy._lib._util import getargspec_no_self as _getargspec

                import_from_module(stypy.reporting.localization.Localization(__file__, 675, 8), 'scipy._lib._util', None, module_type_store, ['getargspec_no_self'], [_getargspec])

        else:
            # Assigning a type to the variable 'scipy._lib._util' (line 675)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'scipy._lib._util', import_172101)

        # Adding an alias
        module_type_store.add_alias('_getargspec', 'getargspec_no_self')
        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')
        
        
        # Assigning a Call to a Tuple (line 676):
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_172103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 8), 'int')
        
        # Call to _getargspec(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'f' (line 676)
        f_172105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 53), 'f', False)
        # Processing the call keyword arguments (line 676)
        kwargs_172106 = {}
        # Getting the type of '_getargspec' (line 676)
        _getargspec_172104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 41), '_getargspec', False)
        # Calling _getargspec(args, kwargs) (line 676)
        _getargspec_call_result_172107 = invoke(stypy.reporting.localization.Localization(__file__, 676, 41), _getargspec_172104, *[f_172105], **kwargs_172106)
        
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___172108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 8), _getargspec_call_result_172107, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_172109 = invoke(stypy.reporting.localization.Localization(__file__, 676, 8), getitem___172108, int_172103)
        
        # Assigning a type to the variable 'tuple_var_assignment_171056' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'tuple_var_assignment_171056', subscript_call_result_172109)
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_172110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 8), 'int')
        
        # Call to _getargspec(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'f' (line 676)
        f_172112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 53), 'f', False)
        # Processing the call keyword arguments (line 676)
        kwargs_172113 = {}
        # Getting the type of '_getargspec' (line 676)
        _getargspec_172111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 41), '_getargspec', False)
        # Calling _getargspec(args, kwargs) (line 676)
        _getargspec_call_result_172114 = invoke(stypy.reporting.localization.Localization(__file__, 676, 41), _getargspec_172111, *[f_172112], **kwargs_172113)
        
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___172115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 8), _getargspec_call_result_172114, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_172116 = invoke(stypy.reporting.localization.Localization(__file__, 676, 8), getitem___172115, int_172110)
        
        # Assigning a type to the variable 'tuple_var_assignment_171057' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'tuple_var_assignment_171057', subscript_call_result_172116)
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_172117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 8), 'int')
        
        # Call to _getargspec(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'f' (line 676)
        f_172119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 53), 'f', False)
        # Processing the call keyword arguments (line 676)
        kwargs_172120 = {}
        # Getting the type of '_getargspec' (line 676)
        _getargspec_172118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 41), '_getargspec', False)
        # Calling _getargspec(args, kwargs) (line 676)
        _getargspec_call_result_172121 = invoke(stypy.reporting.localization.Localization(__file__, 676, 41), _getargspec_172118, *[f_172119], **kwargs_172120)
        
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___172122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 8), _getargspec_call_result_172121, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_172123 = invoke(stypy.reporting.localization.Localization(__file__, 676, 8), getitem___172122, int_172117)
        
        # Assigning a type to the variable 'tuple_var_assignment_171058' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'tuple_var_assignment_171058', subscript_call_result_172123)
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_172124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 8), 'int')
        
        # Call to _getargspec(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'f' (line 676)
        f_172126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 53), 'f', False)
        # Processing the call keyword arguments (line 676)
        kwargs_172127 = {}
        # Getting the type of '_getargspec' (line 676)
        _getargspec_172125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 41), '_getargspec', False)
        # Calling _getargspec(args, kwargs) (line 676)
        _getargspec_call_result_172128 = invoke(stypy.reporting.localization.Localization(__file__, 676, 41), _getargspec_172125, *[f_172126], **kwargs_172127)
        
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___172129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 8), _getargspec_call_result_172128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_172130 = invoke(stypy.reporting.localization.Localization(__file__, 676, 8), getitem___172129, int_172124)
        
        # Assigning a type to the variable 'tuple_var_assignment_171059' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'tuple_var_assignment_171059', subscript_call_result_172130)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_171056' (line 676)
        tuple_var_assignment_171056_172131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'tuple_var_assignment_171056')
        # Assigning a type to the variable 'args' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'args', tuple_var_assignment_171056_172131)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_171057' (line 676)
        tuple_var_assignment_171057_172132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'tuple_var_assignment_171057')
        # Assigning a type to the variable 'varargs' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 14), 'varargs', tuple_var_assignment_171057_172132)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_171058' (line 676)
        tuple_var_assignment_171058_172133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'tuple_var_assignment_171058')
        # Assigning a type to the variable 'varkw' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 23), 'varkw', tuple_var_assignment_171058_172133)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_171059' (line 676)
        tuple_var_assignment_171059_172134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'tuple_var_assignment_171059')
        # Assigning a type to the variable 'defaults' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 30), 'defaults', tuple_var_assignment_171059_172134)
        
        
        
        # Call to len(...): (line 677)
        # Processing the call arguments (line 677)
        # Getting the type of 'args' (line 677)
        args_172136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 15), 'args', False)
        # Processing the call keyword arguments (line 677)
        kwargs_172137 = {}
        # Getting the type of 'len' (line 677)
        len_172135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 11), 'len', False)
        # Calling len(args, kwargs) (line 677)
        len_call_result_172138 = invoke(stypy.reporting.localization.Localization(__file__, 677, 11), len_172135, *[args_172136], **kwargs_172137)
        
        int_172139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 23), 'int')
        # Applying the binary operator '<' (line 677)
        result_lt_172140 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 11), '<', len_call_result_172138, int_172139)
        
        # Testing the type of an if condition (line 677)
        if_condition_172141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 677, 8), result_lt_172140)
        # Assigning a type to the variable 'if_condition_172141' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'if_condition_172141', if_condition_172141)
        # SSA begins for if statement (line 677)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 678)
        # Processing the call arguments (line 678)
        str_172143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 29), 'str', 'Unable to determine number of fit parameters.')
        # Processing the call keyword arguments (line 678)
        kwargs_172144 = {}
        # Getting the type of 'ValueError' (line 678)
        ValueError_172142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 678)
        ValueError_call_result_172145 = invoke(stypy.reporting.localization.Localization(__file__, 678, 18), ValueError_172142, *[str_172143], **kwargs_172144)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 678, 12), ValueError_call_result_172145, 'raise parameter', BaseException)
        # SSA join for if statement (line 677)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 679):
        
        # Assigning a BinOp to a Name (line 679):
        
        # Call to len(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'args' (line 679)
        args_172147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'args', False)
        # Processing the call keyword arguments (line 679)
        kwargs_172148 = {}
        # Getting the type of 'len' (line 679)
        len_172146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'len', False)
        # Calling len(args, kwargs) (line 679)
        len_call_result_172149 = invoke(stypy.reporting.localization.Localization(__file__, 679, 12), len_172146, *[args_172147], **kwargs_172148)
        
        int_172150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 24), 'int')
        # Applying the binary operator '-' (line 679)
        result_sub_172151 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 12), '-', len_call_result_172149, int_172150)
        
        # Assigning a type to the variable 'n' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), 'n', result_sub_172151)

        if more_types_in_union_172100:
            # Runtime conditional SSA for else branch (line 673)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_172099) or more_types_in_union_172100):
        
        # Assigning a Call to a Name (line 681):
        
        # Assigning a Call to a Name (line 681):
        
        # Call to atleast_1d(...): (line 681)
        # Processing the call arguments (line 681)
        # Getting the type of 'p0' (line 681)
        p0_172154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 27), 'p0', False)
        # Processing the call keyword arguments (line 681)
        kwargs_172155 = {}
        # Getting the type of 'np' (line 681)
        np_172152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 13), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 681)
        atleast_1d_172153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 13), np_172152, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 681)
        atleast_1d_call_result_172156 = invoke(stypy.reporting.localization.Localization(__file__, 681, 13), atleast_1d_172153, *[p0_172154], **kwargs_172155)
        
        # Assigning a type to the variable 'p0' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'p0', atleast_1d_call_result_172156)
        
        # Assigning a Attribute to a Name (line 682):
        
        # Assigning a Attribute to a Name (line 682):
        # Getting the type of 'p0' (line 682)
        p0_172157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 12), 'p0')
        # Obtaining the member 'size' of a type (line 682)
        size_172158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 12), p0_172157, 'size')
        # Assigning a type to the variable 'n' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'n', size_172158)

        if (may_be_172099 and more_types_in_union_172100):
            # SSA join for if statement (line 673)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 684):
    
    # Assigning a Subscript to a Name (line 684):
    
    # Obtaining the type of the subscript
    int_172159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 4), 'int')
    
    # Call to prepare_bounds(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 'bounds' (line 684)
    bounds_172161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 28), 'bounds', False)
    # Getting the type of 'n' (line 684)
    n_172162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 36), 'n', False)
    # Processing the call keyword arguments (line 684)
    kwargs_172163 = {}
    # Getting the type of 'prepare_bounds' (line 684)
    prepare_bounds_172160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 13), 'prepare_bounds', False)
    # Calling prepare_bounds(args, kwargs) (line 684)
    prepare_bounds_call_result_172164 = invoke(stypy.reporting.localization.Localization(__file__, 684, 13), prepare_bounds_172160, *[bounds_172161, n_172162], **kwargs_172163)
    
    # Obtaining the member '__getitem__' of a type (line 684)
    getitem___172165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 4), prepare_bounds_call_result_172164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 684)
    subscript_call_result_172166 = invoke(stypy.reporting.localization.Localization(__file__, 684, 4), getitem___172165, int_172159)
    
    # Assigning a type to the variable 'tuple_var_assignment_171060' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'tuple_var_assignment_171060', subscript_call_result_172166)
    
    # Assigning a Subscript to a Name (line 684):
    
    # Obtaining the type of the subscript
    int_172167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 4), 'int')
    
    # Call to prepare_bounds(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 'bounds' (line 684)
    bounds_172169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 28), 'bounds', False)
    # Getting the type of 'n' (line 684)
    n_172170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 36), 'n', False)
    # Processing the call keyword arguments (line 684)
    kwargs_172171 = {}
    # Getting the type of 'prepare_bounds' (line 684)
    prepare_bounds_172168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 13), 'prepare_bounds', False)
    # Calling prepare_bounds(args, kwargs) (line 684)
    prepare_bounds_call_result_172172 = invoke(stypy.reporting.localization.Localization(__file__, 684, 13), prepare_bounds_172168, *[bounds_172169, n_172170], **kwargs_172171)
    
    # Obtaining the member '__getitem__' of a type (line 684)
    getitem___172173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 4), prepare_bounds_call_result_172172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 684)
    subscript_call_result_172174 = invoke(stypy.reporting.localization.Localization(__file__, 684, 4), getitem___172173, int_172167)
    
    # Assigning a type to the variable 'tuple_var_assignment_171061' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'tuple_var_assignment_171061', subscript_call_result_172174)
    
    # Assigning a Name to a Name (line 684):
    # Getting the type of 'tuple_var_assignment_171060' (line 684)
    tuple_var_assignment_171060_172175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'tuple_var_assignment_171060')
    # Assigning a type to the variable 'lb' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'lb', tuple_var_assignment_171060_172175)
    
    # Assigning a Name to a Name (line 684):
    # Getting the type of 'tuple_var_assignment_171061' (line 684)
    tuple_var_assignment_171061_172176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'tuple_var_assignment_171061')
    # Assigning a type to the variable 'ub' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'ub', tuple_var_assignment_171061_172176)
    
    # Type idiom detected: calculating its left and rigth part (line 685)
    # Getting the type of 'p0' (line 685)
    p0_172177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 7), 'p0')
    # Getting the type of 'None' (line 685)
    None_172178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 13), 'None')
    
    (may_be_172179, more_types_in_union_172180) = may_be_none(p0_172177, None_172178)

    if may_be_172179:

        if more_types_in_union_172180:
            # Runtime conditional SSA (line 685)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 686):
        
        # Assigning a Call to a Name (line 686):
        
        # Call to _initialize_feasible(...): (line 686)
        # Processing the call arguments (line 686)
        # Getting the type of 'lb' (line 686)
        lb_172182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 34), 'lb', False)
        # Getting the type of 'ub' (line 686)
        ub_172183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 38), 'ub', False)
        # Processing the call keyword arguments (line 686)
        kwargs_172184 = {}
        # Getting the type of '_initialize_feasible' (line 686)
        _initialize_feasible_172181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 13), '_initialize_feasible', False)
        # Calling _initialize_feasible(args, kwargs) (line 686)
        _initialize_feasible_call_result_172185 = invoke(stypy.reporting.localization.Localization(__file__, 686, 13), _initialize_feasible_172181, *[lb_172182, ub_172183], **kwargs_172184)
        
        # Assigning a type to the variable 'p0' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'p0', _initialize_feasible_call_result_172185)

        if more_types_in_union_172180:
            # SSA join for if statement (line 685)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 688):
    
    # Assigning a Call to a Name (line 688):
    
    # Call to any(...): (line 688)
    # Processing the call arguments (line 688)
    
    # Getting the type of 'lb' (line 688)
    lb_172188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 30), 'lb', False)
    
    # Getting the type of 'np' (line 688)
    np_172189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 36), 'np', False)
    # Obtaining the member 'inf' of a type (line 688)
    inf_172190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 36), np_172189, 'inf')
    # Applying the 'usub' unary operator (line 688)
    result___neg___172191 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 35), 'usub', inf_172190)
    
    # Applying the binary operator '>' (line 688)
    result_gt_172192 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 30), '>', lb_172188, result___neg___172191)
    
    
    # Getting the type of 'ub' (line 688)
    ub_172193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 47), 'ub', False)
    # Getting the type of 'np' (line 688)
    np_172194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 52), 'np', False)
    # Obtaining the member 'inf' of a type (line 688)
    inf_172195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 52), np_172194, 'inf')
    # Applying the binary operator '<' (line 688)
    result_lt_172196 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 47), '<', ub_172193, inf_172195)
    
    # Applying the binary operator '|' (line 688)
    result_or__172197 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 29), '|', result_gt_172192, result_lt_172196)
    
    # Processing the call keyword arguments (line 688)
    kwargs_172198 = {}
    # Getting the type of 'np' (line 688)
    np_172186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 22), 'np', False)
    # Obtaining the member 'any' of a type (line 688)
    any_172187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 22), np_172186, 'any')
    # Calling any(args, kwargs) (line 688)
    any_call_result_172199 = invoke(stypy.reporting.localization.Localization(__file__, 688, 22), any_172187, *[result_or__172197], **kwargs_172198)
    
    # Assigning a type to the variable 'bounded_problem' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'bounded_problem', any_call_result_172199)
    
    # Type idiom detected: calculating its left and rigth part (line 689)
    # Getting the type of 'method' (line 689)
    method_172200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 7), 'method')
    # Getting the type of 'None' (line 689)
    None_172201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 17), 'None')
    
    (may_be_172202, more_types_in_union_172203) = may_be_none(method_172200, None_172201)

    if may_be_172202:

        if more_types_in_union_172203:
            # Runtime conditional SSA (line 689)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'bounded_problem' (line 690)
        bounded_problem_172204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 11), 'bounded_problem')
        # Testing the type of an if condition (line 690)
        if_condition_172205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 690, 8), bounded_problem_172204)
        # Assigning a type to the variable 'if_condition_172205' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'if_condition_172205', if_condition_172205)
        # SSA begins for if statement (line 690)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 691):
        
        # Assigning a Str to a Name (line 691):
        str_172206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 21), 'str', 'trf')
        # Assigning a type to the variable 'method' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'method', str_172206)
        # SSA branch for the else part of an if statement (line 690)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 693):
        
        # Assigning a Str to a Name (line 693):
        str_172207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 21), 'str', 'lm')
        # Assigning a type to the variable 'method' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 12), 'method', str_172207)
        # SSA join for if statement (line 690)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_172203:
            # SSA join for if statement (line 689)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 695)
    method_172208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 7), 'method')
    str_172209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 17), 'str', 'lm')
    # Applying the binary operator '==' (line 695)
    result_eq_172210 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 7), '==', method_172208, str_172209)
    
    # Getting the type of 'bounded_problem' (line 695)
    bounded_problem_172211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 26), 'bounded_problem')
    # Applying the binary operator 'and' (line 695)
    result_and_keyword_172212 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 7), 'and', result_eq_172210, bounded_problem_172211)
    
    # Testing the type of an if condition (line 695)
    if_condition_172213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 695, 4), result_and_keyword_172212)
    # Assigning a type to the variable 'if_condition_172213' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'if_condition_172213', if_condition_172213)
    # SSA begins for if statement (line 695)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 696)
    # Processing the call arguments (line 696)
    str_172215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 25), 'str', "Method 'lm' only works for unconstrained problems. Use 'trf' or 'dogbox' instead.")
    # Processing the call keyword arguments (line 696)
    kwargs_172216 = {}
    # Getting the type of 'ValueError' (line 696)
    ValueError_172214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 696)
    ValueError_call_result_172217 = invoke(stypy.reporting.localization.Localization(__file__, 696, 14), ValueError_172214, *[str_172215], **kwargs_172216)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 696, 8), ValueError_call_result_172217, 'raise parameter', BaseException)
    # SSA join for if statement (line 695)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'check_finite' (line 700)
    check_finite_172218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 7), 'check_finite')
    # Testing the type of an if condition (line 700)
    if_condition_172219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 700, 4), check_finite_172218)
    # Assigning a type to the variable 'if_condition_172219' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'if_condition_172219', if_condition_172219)
    # SSA begins for if statement (line 700)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 701):
    
    # Assigning a Call to a Name (line 701):
    
    # Call to asarray_chkfinite(...): (line 701)
    # Processing the call arguments (line 701)
    # Getting the type of 'ydata' (line 701)
    ydata_172222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 37), 'ydata', False)
    # Processing the call keyword arguments (line 701)
    kwargs_172223 = {}
    # Getting the type of 'np' (line 701)
    np_172220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 701)
    asarray_chkfinite_172221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 16), np_172220, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 701)
    asarray_chkfinite_call_result_172224 = invoke(stypy.reporting.localization.Localization(__file__, 701, 16), asarray_chkfinite_172221, *[ydata_172222], **kwargs_172223)
    
    # Assigning a type to the variable 'ydata' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'ydata', asarray_chkfinite_call_result_172224)
    # SSA branch for the else part of an if statement (line 700)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 703):
    
    # Assigning a Call to a Name (line 703):
    
    # Call to asarray(...): (line 703)
    # Processing the call arguments (line 703)
    # Getting the type of 'ydata' (line 703)
    ydata_172227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 27), 'ydata', False)
    # Processing the call keyword arguments (line 703)
    kwargs_172228 = {}
    # Getting the type of 'np' (line 703)
    np_172225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 16), 'np', False)
    # Obtaining the member 'asarray' of a type (line 703)
    asarray_172226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 16), np_172225, 'asarray')
    # Calling asarray(args, kwargs) (line 703)
    asarray_call_result_172229 = invoke(stypy.reporting.localization.Localization(__file__, 703, 16), asarray_172226, *[ydata_172227], **kwargs_172228)
    
    # Assigning a type to the variable 'ydata' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'ydata', asarray_call_result_172229)
    # SSA join for if statement (line 700)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 705)
    # Processing the call arguments (line 705)
    # Getting the type of 'xdata' (line 705)
    xdata_172231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 18), 'xdata', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 705)
    tuple_172232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 705)
    # Adding element type (line 705)
    # Getting the type of 'list' (line 705)
    list_172233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 26), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 26), tuple_172232, list_172233)
    # Adding element type (line 705)
    # Getting the type of 'tuple' (line 705)
    tuple_172234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 32), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 26), tuple_172232, tuple_172234)
    # Adding element type (line 705)
    # Getting the type of 'np' (line 705)
    np_172235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 39), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 705)
    ndarray_172236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 39), np_172235, 'ndarray')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 26), tuple_172232, ndarray_172236)
    
    # Processing the call keyword arguments (line 705)
    kwargs_172237 = {}
    # Getting the type of 'isinstance' (line 705)
    isinstance_172230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 705)
    isinstance_call_result_172238 = invoke(stypy.reporting.localization.Localization(__file__, 705, 7), isinstance_172230, *[xdata_172231, tuple_172232], **kwargs_172237)
    
    # Testing the type of an if condition (line 705)
    if_condition_172239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 705, 4), isinstance_call_result_172238)
    # Assigning a type to the variable 'if_condition_172239' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'if_condition_172239', if_condition_172239)
    # SSA begins for if statement (line 705)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'check_finite' (line 708)
    check_finite_172240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 11), 'check_finite')
    # Testing the type of an if condition (line 708)
    if_condition_172241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 708, 8), check_finite_172240)
    # Assigning a type to the variable 'if_condition_172241' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'if_condition_172241', if_condition_172241)
    # SSA begins for if statement (line 708)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 709):
    
    # Assigning a Call to a Name (line 709):
    
    # Call to asarray_chkfinite(...): (line 709)
    # Processing the call arguments (line 709)
    # Getting the type of 'xdata' (line 709)
    xdata_172244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 41), 'xdata', False)
    # Processing the call keyword arguments (line 709)
    kwargs_172245 = {}
    # Getting the type of 'np' (line 709)
    np_172242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 20), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 709)
    asarray_chkfinite_172243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 20), np_172242, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 709)
    asarray_chkfinite_call_result_172246 = invoke(stypy.reporting.localization.Localization(__file__, 709, 20), asarray_chkfinite_172243, *[xdata_172244], **kwargs_172245)
    
    # Assigning a type to the variable 'xdata' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 12), 'xdata', asarray_chkfinite_call_result_172246)
    # SSA branch for the else part of an if statement (line 708)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 711):
    
    # Assigning a Call to a Name (line 711):
    
    # Call to asarray(...): (line 711)
    # Processing the call arguments (line 711)
    # Getting the type of 'xdata' (line 711)
    xdata_172249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 31), 'xdata', False)
    # Processing the call keyword arguments (line 711)
    kwargs_172250 = {}
    # Getting the type of 'np' (line 711)
    np_172247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 20), 'np', False)
    # Obtaining the member 'asarray' of a type (line 711)
    asarray_172248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 20), np_172247, 'asarray')
    # Calling asarray(args, kwargs) (line 711)
    asarray_call_result_172251 = invoke(stypy.reporting.localization.Localization(__file__, 711, 20), asarray_172248, *[xdata_172249], **kwargs_172250)
    
    # Assigning a type to the variable 'xdata' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), 'xdata', asarray_call_result_172251)
    # SSA join for if statement (line 708)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 705)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 714)
    # Getting the type of 'sigma' (line 714)
    sigma_172252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'sigma')
    # Getting the type of 'None' (line 714)
    None_172253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 20), 'None')
    
    (may_be_172254, more_types_in_union_172255) = may_not_be_none(sigma_172252, None_172253)

    if may_be_172254:

        if more_types_in_union_172255:
            # Runtime conditional SSA (line 714)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 715):
        
        # Assigning a Call to a Name (line 715):
        
        # Call to asarray(...): (line 715)
        # Processing the call arguments (line 715)
        # Getting the type of 'sigma' (line 715)
        sigma_172258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 27), 'sigma', False)
        # Processing the call keyword arguments (line 715)
        kwargs_172259 = {}
        # Getting the type of 'np' (line 715)
        np_172256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 715)
        asarray_172257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 16), np_172256, 'asarray')
        # Calling asarray(args, kwargs) (line 715)
        asarray_call_result_172260 = invoke(stypy.reporting.localization.Localization(__file__, 715, 16), asarray_172257, *[sigma_172258], **kwargs_172259)
        
        # Assigning a type to the variable 'sigma' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'sigma', asarray_call_result_172260)
        
        
        # Getting the type of 'sigma' (line 718)
        sigma_172261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 11), 'sigma')
        # Obtaining the member 'shape' of a type (line 718)
        shape_172262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 11), sigma_172261, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 718)
        tuple_172263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 718)
        # Adding element type (line 718)
        # Getting the type of 'ydata' (line 718)
        ydata_172264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 27), 'ydata')
        # Obtaining the member 'size' of a type (line 718)
        size_172265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 27), ydata_172264, 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 27), tuple_172263, size_172265)
        
        # Applying the binary operator '==' (line 718)
        result_eq_172266 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 11), '==', shape_172262, tuple_172263)
        
        # Testing the type of an if condition (line 718)
        if_condition_172267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 8), result_eq_172266)
        # Assigning a type to the variable 'if_condition_172267' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'if_condition_172267', if_condition_172267)
        # SSA begins for if statement (line 718)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 719):
        
        # Assigning a BinOp to a Name (line 719):
        float_172268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 24), 'float')
        # Getting the type of 'sigma' (line 719)
        sigma_172269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 30), 'sigma')
        # Applying the binary operator 'div' (line 719)
        result_div_172270 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 24), 'div', float_172268, sigma_172269)
        
        # Assigning a type to the variable 'transform' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'transform', result_div_172270)
        # SSA branch for the else part of an if statement (line 718)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sigma' (line 722)
        sigma_172271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 13), 'sigma')
        # Obtaining the member 'shape' of a type (line 722)
        shape_172272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 13), sigma_172271, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 722)
        tuple_172273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 722)
        # Adding element type (line 722)
        # Getting the type of 'ydata' (line 722)
        ydata_172274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 29), 'ydata')
        # Obtaining the member 'size' of a type (line 722)
        size_172275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 29), ydata_172274, 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 29), tuple_172273, size_172275)
        # Adding element type (line 722)
        # Getting the type of 'ydata' (line 722)
        ydata_172276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 41), 'ydata')
        # Obtaining the member 'size' of a type (line 722)
        size_172277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 41), ydata_172276, 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 29), tuple_172273, size_172277)
        
        # Applying the binary operator '==' (line 722)
        result_eq_172278 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 13), '==', shape_172272, tuple_172273)
        
        # Testing the type of an if condition (line 722)
        if_condition_172279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 722, 13), result_eq_172278)
        # Assigning a type to the variable 'if_condition_172279' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 13), 'if_condition_172279', if_condition_172279)
        # SSA begins for if statement (line 722)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 723)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 725):
        
        # Assigning a Call to a Name (line 725):
        
        # Call to cholesky(...): (line 725)
        # Processing the call arguments (line 725)
        # Getting the type of 'sigma' (line 725)
        sigma_172281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 37), 'sigma', False)
        # Processing the call keyword arguments (line 725)
        # Getting the type of 'True' (line 725)
        True_172282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 50), 'True', False)
        keyword_172283 = True_172282
        kwargs_172284 = {'lower': keyword_172283}
        # Getting the type of 'cholesky' (line 725)
        cholesky_172280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 28), 'cholesky', False)
        # Calling cholesky(args, kwargs) (line 725)
        cholesky_call_result_172285 = invoke(stypy.reporting.localization.Localization(__file__, 725, 28), cholesky_172280, *[sigma_172281], **kwargs_172284)
        
        # Assigning a type to the variable 'transform' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 16), 'transform', cholesky_call_result_172285)
        # SSA branch for the except part of a try statement (line 723)
        # SSA branch for the except 'LinAlgError' branch of a try statement (line 723)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 727)
        # Processing the call arguments (line 727)
        str_172287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 33), 'str', '`sigma` must be positive definite.')
        # Processing the call keyword arguments (line 727)
        kwargs_172288 = {}
        # Getting the type of 'ValueError' (line 727)
        ValueError_172286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 727)
        ValueError_call_result_172289 = invoke(stypy.reporting.localization.Localization(__file__, 727, 22), ValueError_172286, *[str_172287], **kwargs_172288)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 727, 16), ValueError_call_result_172289, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 723)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 722)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 729)
        # Processing the call arguments (line 729)
        str_172291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 29), 'str', '`sigma` has incorrect shape.')
        # Processing the call keyword arguments (line 729)
        kwargs_172292 = {}
        # Getting the type of 'ValueError' (line 729)
        ValueError_172290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 729)
        ValueError_call_result_172293 = invoke(stypy.reporting.localization.Localization(__file__, 729, 18), ValueError_172290, *[str_172291], **kwargs_172292)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 729, 12), ValueError_call_result_172293, 'raise parameter', BaseException)
        # SSA join for if statement (line 722)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 718)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_172255:
            # Runtime conditional SSA for else branch (line 714)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_172254) or more_types_in_union_172255):
        
        # Assigning a Name to a Name (line 731):
        
        # Assigning a Name to a Name (line 731):
        # Getting the type of 'None' (line 731)
        None_172294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 20), 'None')
        # Assigning a type to the variable 'transform' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'transform', None_172294)

        if (may_be_172254 and more_types_in_union_172255):
            # SSA join for if statement (line 714)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 733):
    
    # Assigning a Call to a Name (line 733):
    
    # Call to _wrap_func(...): (line 733)
    # Processing the call arguments (line 733)
    # Getting the type of 'f' (line 733)
    f_172296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 22), 'f', False)
    # Getting the type of 'xdata' (line 733)
    xdata_172297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 25), 'xdata', False)
    # Getting the type of 'ydata' (line 733)
    ydata_172298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 32), 'ydata', False)
    # Getting the type of 'transform' (line 733)
    transform_172299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 39), 'transform', False)
    # Processing the call keyword arguments (line 733)
    kwargs_172300 = {}
    # Getting the type of '_wrap_func' (line 733)
    _wrap_func_172295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 11), '_wrap_func', False)
    # Calling _wrap_func(args, kwargs) (line 733)
    _wrap_func_call_result_172301 = invoke(stypy.reporting.localization.Localization(__file__, 733, 11), _wrap_func_172295, *[f_172296, xdata_172297, ydata_172298, transform_172299], **kwargs_172300)
    
    # Assigning a type to the variable 'func' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'func', _wrap_func_call_result_172301)
    
    
    # Call to callable(...): (line 734)
    # Processing the call arguments (line 734)
    # Getting the type of 'jac' (line 734)
    jac_172303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 16), 'jac', False)
    # Processing the call keyword arguments (line 734)
    kwargs_172304 = {}
    # Getting the type of 'callable' (line 734)
    callable_172302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 7), 'callable', False)
    # Calling callable(args, kwargs) (line 734)
    callable_call_result_172305 = invoke(stypy.reporting.localization.Localization(__file__, 734, 7), callable_172302, *[jac_172303], **kwargs_172304)
    
    # Testing the type of an if condition (line 734)
    if_condition_172306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 4), callable_call_result_172305)
    # Assigning a type to the variable 'if_condition_172306' (line 734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 4), 'if_condition_172306', if_condition_172306)
    # SSA begins for if statement (line 734)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 735):
    
    # Assigning a Call to a Name (line 735):
    
    # Call to _wrap_jac(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'jac' (line 735)
    jac_172308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 24), 'jac', False)
    # Getting the type of 'xdata' (line 735)
    xdata_172309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 29), 'xdata', False)
    # Getting the type of 'transform' (line 735)
    transform_172310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 36), 'transform', False)
    # Processing the call keyword arguments (line 735)
    kwargs_172311 = {}
    # Getting the type of '_wrap_jac' (line 735)
    _wrap_jac_172307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 14), '_wrap_jac', False)
    # Calling _wrap_jac(args, kwargs) (line 735)
    _wrap_jac_call_result_172312 = invoke(stypy.reporting.localization.Localization(__file__, 735, 14), _wrap_jac_172307, *[jac_172308, xdata_172309, transform_172310], **kwargs_172311)
    
    # Assigning a type to the variable 'jac' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 8), 'jac', _wrap_jac_call_result_172312)
    # SSA branch for the else part of an if statement (line 734)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'jac' (line 736)
    jac_172313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 9), 'jac')
    # Getting the type of 'None' (line 736)
    None_172314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 16), 'None')
    # Applying the binary operator 'is' (line 736)
    result_is__172315 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 9), 'is', jac_172313, None_172314)
    
    
    # Getting the type of 'method' (line 736)
    method_172316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 25), 'method')
    str_172317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 35), 'str', 'lm')
    # Applying the binary operator '!=' (line 736)
    result_ne_172318 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 25), '!=', method_172316, str_172317)
    
    # Applying the binary operator 'and' (line 736)
    result_and_keyword_172319 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 9), 'and', result_is__172315, result_ne_172318)
    
    # Testing the type of an if condition (line 736)
    if_condition_172320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 9), result_and_keyword_172319)
    # Assigning a type to the variable 'if_condition_172320' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 9), 'if_condition_172320', if_condition_172320)
    # SSA begins for if statement (line 736)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 737):
    
    # Assigning a Str to a Name (line 737):
    str_172321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 14), 'str', '2-point')
    # Assigning a type to the variable 'jac' (line 737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'jac', str_172321)
    # SSA join for if statement (line 736)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 734)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'method' (line 739)
    method_172322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 7), 'method')
    str_172323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 17), 'str', 'lm')
    # Applying the binary operator '==' (line 739)
    result_eq_172324 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 7), '==', method_172322, str_172323)
    
    # Testing the type of an if condition (line 739)
    if_condition_172325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 739, 4), result_eq_172324)
    # Assigning a type to the variable 'if_condition_172325' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'if_condition_172325', if_condition_172325)
    # SSA begins for if statement (line 739)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 741):
    
    # Assigning a Call to a Name (line 741):
    
    # Call to pop(...): (line 741)
    # Processing the call arguments (line 741)
    str_172328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 33), 'str', 'full_output')
    # Getting the type of 'False' (line 741)
    False_172329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 48), 'False', False)
    # Processing the call keyword arguments (line 741)
    kwargs_172330 = {}
    # Getting the type of 'kwargs' (line 741)
    kwargs_172326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 22), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 741)
    pop_172327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 22), kwargs_172326, 'pop')
    # Calling pop(args, kwargs) (line 741)
    pop_call_result_172331 = invoke(stypy.reporting.localization.Localization(__file__, 741, 22), pop_172327, *[str_172328, False_172329], **kwargs_172330)
    
    # Assigning a type to the variable 'return_full' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'return_full', pop_call_result_172331)
    
    # Assigning a Call to a Name (line 742):
    
    # Assigning a Call to a Name (line 742):
    
    # Call to leastsq(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'func' (line 742)
    func_172333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 22), 'func', False)
    # Getting the type of 'p0' (line 742)
    p0_172334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 28), 'p0', False)
    # Processing the call keyword arguments (line 742)
    # Getting the type of 'jac' (line 742)
    jac_172335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 37), 'jac', False)
    keyword_172336 = jac_172335
    int_172337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 54), 'int')
    keyword_172338 = int_172337
    # Getting the type of 'kwargs' (line 742)
    kwargs_172339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 59), 'kwargs', False)
    kwargs_172340 = {'kwargs_172339': kwargs_172339, 'Dfun': keyword_172336, 'full_output': keyword_172338}
    # Getting the type of 'leastsq' (line 742)
    leastsq_172332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 14), 'leastsq', False)
    # Calling leastsq(args, kwargs) (line 742)
    leastsq_call_result_172341 = invoke(stypy.reporting.localization.Localization(__file__, 742, 14), leastsq_172332, *[func_172333, p0_172334], **kwargs_172340)
    
    # Assigning a type to the variable 'res' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'res', leastsq_call_result_172341)
    
    # Assigning a Name to a Tuple (line 743):
    
    # Assigning a Subscript to a Name (line 743):
    
    # Obtaining the type of the subscript
    int_172342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 8), 'int')
    # Getting the type of 'res' (line 743)
    res_172343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'res')
    # Obtaining the member '__getitem__' of a type (line 743)
    getitem___172344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), res_172343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 743)
    subscript_call_result_172345 = invoke(stypy.reporting.localization.Localization(__file__, 743, 8), getitem___172344, int_172342)
    
    # Assigning a type to the variable 'tuple_var_assignment_171062' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171062', subscript_call_result_172345)
    
    # Assigning a Subscript to a Name (line 743):
    
    # Obtaining the type of the subscript
    int_172346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 8), 'int')
    # Getting the type of 'res' (line 743)
    res_172347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'res')
    # Obtaining the member '__getitem__' of a type (line 743)
    getitem___172348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), res_172347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 743)
    subscript_call_result_172349 = invoke(stypy.reporting.localization.Localization(__file__, 743, 8), getitem___172348, int_172346)
    
    # Assigning a type to the variable 'tuple_var_assignment_171063' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171063', subscript_call_result_172349)
    
    # Assigning a Subscript to a Name (line 743):
    
    # Obtaining the type of the subscript
    int_172350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 8), 'int')
    # Getting the type of 'res' (line 743)
    res_172351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'res')
    # Obtaining the member '__getitem__' of a type (line 743)
    getitem___172352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), res_172351, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 743)
    subscript_call_result_172353 = invoke(stypy.reporting.localization.Localization(__file__, 743, 8), getitem___172352, int_172350)
    
    # Assigning a type to the variable 'tuple_var_assignment_171064' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171064', subscript_call_result_172353)
    
    # Assigning a Subscript to a Name (line 743):
    
    # Obtaining the type of the subscript
    int_172354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 8), 'int')
    # Getting the type of 'res' (line 743)
    res_172355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'res')
    # Obtaining the member '__getitem__' of a type (line 743)
    getitem___172356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), res_172355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 743)
    subscript_call_result_172357 = invoke(stypy.reporting.localization.Localization(__file__, 743, 8), getitem___172356, int_172354)
    
    # Assigning a type to the variable 'tuple_var_assignment_171065' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171065', subscript_call_result_172357)
    
    # Assigning a Subscript to a Name (line 743):
    
    # Obtaining the type of the subscript
    int_172358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 8), 'int')
    # Getting the type of 'res' (line 743)
    res_172359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'res')
    # Obtaining the member '__getitem__' of a type (line 743)
    getitem___172360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), res_172359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 743)
    subscript_call_result_172361 = invoke(stypy.reporting.localization.Localization(__file__, 743, 8), getitem___172360, int_172358)
    
    # Assigning a type to the variable 'tuple_var_assignment_171066' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171066', subscript_call_result_172361)
    
    # Assigning a Name to a Name (line 743):
    # Getting the type of 'tuple_var_assignment_171062' (line 743)
    tuple_var_assignment_171062_172362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171062')
    # Assigning a type to the variable 'popt' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'popt', tuple_var_assignment_171062_172362)
    
    # Assigning a Name to a Name (line 743):
    # Getting the type of 'tuple_var_assignment_171063' (line 743)
    tuple_var_assignment_171063_172363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171063')
    # Assigning a type to the variable 'pcov' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 14), 'pcov', tuple_var_assignment_171063_172363)
    
    # Assigning a Name to a Name (line 743):
    # Getting the type of 'tuple_var_assignment_171064' (line 743)
    tuple_var_assignment_171064_172364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171064')
    # Assigning a type to the variable 'infodict' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 20), 'infodict', tuple_var_assignment_171064_172364)
    
    # Assigning a Name to a Name (line 743):
    # Getting the type of 'tuple_var_assignment_171065' (line 743)
    tuple_var_assignment_171065_172365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171065')
    # Assigning a type to the variable 'errmsg' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 30), 'errmsg', tuple_var_assignment_171065_172365)
    
    # Assigning a Name to a Name (line 743):
    # Getting the type of 'tuple_var_assignment_171066' (line 743)
    tuple_var_assignment_171066_172366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'tuple_var_assignment_171066')
    # Assigning a type to the variable 'ier' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 38), 'ier', tuple_var_assignment_171066_172366)
    
    # Assigning a Call to a Name (line 744):
    
    # Assigning a Call to a Name (line 744):
    
    # Call to sum(...): (line 744)
    # Processing the call arguments (line 744)
    
    # Obtaining the type of the subscript
    str_172369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 31), 'str', 'fvec')
    # Getting the type of 'infodict' (line 744)
    infodict_172370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 22), 'infodict', False)
    # Obtaining the member '__getitem__' of a type (line 744)
    getitem___172371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 22), infodict_172370, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 744)
    subscript_call_result_172372 = invoke(stypy.reporting.localization.Localization(__file__, 744, 22), getitem___172371, str_172369)
    
    int_172373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 42), 'int')
    # Applying the binary operator '**' (line 744)
    result_pow_172374 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 22), '**', subscript_call_result_172372, int_172373)
    
    # Processing the call keyword arguments (line 744)
    kwargs_172375 = {}
    # Getting the type of 'np' (line 744)
    np_172367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 15), 'np', False)
    # Obtaining the member 'sum' of a type (line 744)
    sum_172368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 15), np_172367, 'sum')
    # Calling sum(args, kwargs) (line 744)
    sum_call_result_172376 = invoke(stypy.reporting.localization.Localization(__file__, 744, 15), sum_172368, *[result_pow_172374], **kwargs_172375)
    
    # Assigning a type to the variable 'cost' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'cost', sum_call_result_172376)
    
    
    # Getting the type of 'ier' (line 745)
    ier_172377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 11), 'ier')
    
    # Obtaining an instance of the builtin type 'list' (line 745)
    list_172378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 745)
    # Adding element type (line 745)
    int_172379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 22), list_172378, int_172379)
    # Adding element type (line 745)
    int_172380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 22), list_172378, int_172380)
    # Adding element type (line 745)
    int_172381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 22), list_172378, int_172381)
    # Adding element type (line 745)
    int_172382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 22), list_172378, int_172382)
    
    # Applying the binary operator 'notin' (line 745)
    result_contains_172383 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 11), 'notin', ier_172377, list_172378)
    
    # Testing the type of an if condition (line 745)
    if_condition_172384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 745, 8), result_contains_172383)
    # Assigning a type to the variable 'if_condition_172384' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'if_condition_172384', if_condition_172384)
    # SSA begins for if statement (line 745)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 746)
    # Processing the call arguments (line 746)
    str_172386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 31), 'str', 'Optimal parameters not found: ')
    # Getting the type of 'errmsg' (line 746)
    errmsg_172387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 66), 'errmsg', False)
    # Applying the binary operator '+' (line 746)
    result_add_172388 = python_operator(stypy.reporting.localization.Localization(__file__, 746, 31), '+', str_172386, errmsg_172387)
    
    # Processing the call keyword arguments (line 746)
    kwargs_172389 = {}
    # Getting the type of 'RuntimeError' (line 746)
    RuntimeError_172385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 746)
    RuntimeError_call_result_172390 = invoke(stypy.reporting.localization.Localization(__file__, 746, 18), RuntimeError_172385, *[result_add_172388], **kwargs_172389)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 746, 12), RuntimeError_call_result_172390, 'raise parameter', BaseException)
    # SSA join for if statement (line 745)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 739)
    module_type_store.open_ssa_branch('else')
    
    
    str_172391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 11), 'str', 'max_nfev')
    # Getting the type of 'kwargs' (line 749)
    kwargs_172392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 29), 'kwargs')
    # Applying the binary operator 'notin' (line 749)
    result_contains_172393 = python_operator(stypy.reporting.localization.Localization(__file__, 749, 11), 'notin', str_172391, kwargs_172392)
    
    # Testing the type of an if condition (line 749)
    if_condition_172394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 749, 8), result_contains_172393)
    # Assigning a type to the variable 'if_condition_172394' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'if_condition_172394', if_condition_172394)
    # SSA begins for if statement (line 749)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 750):
    
    # Assigning a Call to a Subscript (line 750):
    
    # Call to pop(...): (line 750)
    # Processing the call arguments (line 750)
    str_172397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 44), 'str', 'maxfev')
    # Getting the type of 'None' (line 750)
    None_172398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 54), 'None', False)
    # Processing the call keyword arguments (line 750)
    kwargs_172399 = {}
    # Getting the type of 'kwargs' (line 750)
    kwargs_172395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 33), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 750)
    pop_172396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 33), kwargs_172395, 'pop')
    # Calling pop(args, kwargs) (line 750)
    pop_call_result_172400 = invoke(stypy.reporting.localization.Localization(__file__, 750, 33), pop_172396, *[str_172397, None_172398], **kwargs_172399)
    
    # Getting the type of 'kwargs' (line 750)
    kwargs_172401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'kwargs')
    str_172402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 19), 'str', 'max_nfev')
    # Storing an element on a container (line 750)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 12), kwargs_172401, (str_172402, pop_call_result_172400))
    # SSA join for if statement (line 749)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 752):
    
    # Assigning a Call to a Name (line 752):
    
    # Call to least_squares(...): (line 752)
    # Processing the call arguments (line 752)
    # Getting the type of 'func' (line 752)
    func_172404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 28), 'func', False)
    # Getting the type of 'p0' (line 752)
    p0_172405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 34), 'p0', False)
    # Processing the call keyword arguments (line 752)
    # Getting the type of 'jac' (line 752)
    jac_172406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 42), 'jac', False)
    keyword_172407 = jac_172406
    # Getting the type of 'bounds' (line 752)
    bounds_172408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 54), 'bounds', False)
    keyword_172409 = bounds_172408
    # Getting the type of 'method' (line 752)
    method_172410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 69), 'method', False)
    keyword_172411 = method_172410
    # Getting the type of 'kwargs' (line 753)
    kwargs_172412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 30), 'kwargs', False)
    kwargs_172413 = {'method': keyword_172411, 'kwargs_172412': kwargs_172412, 'jac': keyword_172407, 'bounds': keyword_172409}
    # Getting the type of 'least_squares' (line 752)
    least_squares_172403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 14), 'least_squares', False)
    # Calling least_squares(args, kwargs) (line 752)
    least_squares_call_result_172414 = invoke(stypy.reporting.localization.Localization(__file__, 752, 14), least_squares_172403, *[func_172404, p0_172405], **kwargs_172413)
    
    # Assigning a type to the variable 'res' (line 752)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'res', least_squares_call_result_172414)
    
    
    # Getting the type of 'res' (line 755)
    res_172415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 15), 'res')
    # Obtaining the member 'success' of a type (line 755)
    success_172416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 15), res_172415, 'success')
    # Applying the 'not' unary operator (line 755)
    result_not__172417 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 11), 'not', success_172416)
    
    # Testing the type of an if condition (line 755)
    if_condition_172418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 755, 8), result_not__172417)
    # Assigning a type to the variable 'if_condition_172418' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'if_condition_172418', if_condition_172418)
    # SSA begins for if statement (line 755)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 756)
    # Processing the call arguments (line 756)
    str_172420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 31), 'str', 'Optimal parameters not found: ')
    # Getting the type of 'res' (line 756)
    res_172421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 66), 'res', False)
    # Obtaining the member 'message' of a type (line 756)
    message_172422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 66), res_172421, 'message')
    # Applying the binary operator '+' (line 756)
    result_add_172423 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 31), '+', str_172420, message_172422)
    
    # Processing the call keyword arguments (line 756)
    kwargs_172424 = {}
    # Getting the type of 'RuntimeError' (line 756)
    RuntimeError_172419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 756)
    RuntimeError_call_result_172425 = invoke(stypy.reporting.localization.Localization(__file__, 756, 18), RuntimeError_172419, *[result_add_172423], **kwargs_172424)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 756, 12), RuntimeError_call_result_172425, 'raise parameter', BaseException)
    # SSA join for if statement (line 755)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 758):
    
    # Assigning a BinOp to a Name (line 758):
    int_172426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 15), 'int')
    # Getting the type of 'res' (line 758)
    res_172427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 19), 'res')
    # Obtaining the member 'cost' of a type (line 758)
    cost_172428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 19), res_172427, 'cost')
    # Applying the binary operator '*' (line 758)
    result_mul_172429 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 15), '*', int_172426, cost_172428)
    
    # Assigning a type to the variable 'cost' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'cost', result_mul_172429)
    
    # Assigning a Attribute to a Name (line 759):
    
    # Assigning a Attribute to a Name (line 759):
    # Getting the type of 'res' (line 759)
    res_172430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 15), 'res')
    # Obtaining the member 'x' of a type (line 759)
    x_172431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 15), res_172430, 'x')
    # Assigning a type to the variable 'popt' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'popt', x_172431)
    
    # Assigning a Call to a Tuple (line 762):
    
    # Assigning a Subscript to a Name (line 762):
    
    # Obtaining the type of the subscript
    int_172432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 8), 'int')
    
    # Call to svd(...): (line 762)
    # Processing the call arguments (line 762)
    # Getting the type of 'res' (line 762)
    res_172434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 23), 'res', False)
    # Obtaining the member 'jac' of a type (line 762)
    jac_172435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 23), res_172434, 'jac')
    # Processing the call keyword arguments (line 762)
    # Getting the type of 'False' (line 762)
    False_172436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 46), 'False', False)
    keyword_172437 = False_172436
    kwargs_172438 = {'full_matrices': keyword_172437}
    # Getting the type of 'svd' (line 762)
    svd_172433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 19), 'svd', False)
    # Calling svd(args, kwargs) (line 762)
    svd_call_result_172439 = invoke(stypy.reporting.localization.Localization(__file__, 762, 19), svd_172433, *[jac_172435], **kwargs_172438)
    
    # Obtaining the member '__getitem__' of a type (line 762)
    getitem___172440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 8), svd_call_result_172439, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 762)
    subscript_call_result_172441 = invoke(stypy.reporting.localization.Localization(__file__, 762, 8), getitem___172440, int_172432)
    
    # Assigning a type to the variable 'tuple_var_assignment_171067' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'tuple_var_assignment_171067', subscript_call_result_172441)
    
    # Assigning a Subscript to a Name (line 762):
    
    # Obtaining the type of the subscript
    int_172442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 8), 'int')
    
    # Call to svd(...): (line 762)
    # Processing the call arguments (line 762)
    # Getting the type of 'res' (line 762)
    res_172444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 23), 'res', False)
    # Obtaining the member 'jac' of a type (line 762)
    jac_172445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 23), res_172444, 'jac')
    # Processing the call keyword arguments (line 762)
    # Getting the type of 'False' (line 762)
    False_172446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 46), 'False', False)
    keyword_172447 = False_172446
    kwargs_172448 = {'full_matrices': keyword_172447}
    # Getting the type of 'svd' (line 762)
    svd_172443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 19), 'svd', False)
    # Calling svd(args, kwargs) (line 762)
    svd_call_result_172449 = invoke(stypy.reporting.localization.Localization(__file__, 762, 19), svd_172443, *[jac_172445], **kwargs_172448)
    
    # Obtaining the member '__getitem__' of a type (line 762)
    getitem___172450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 8), svd_call_result_172449, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 762)
    subscript_call_result_172451 = invoke(stypy.reporting.localization.Localization(__file__, 762, 8), getitem___172450, int_172442)
    
    # Assigning a type to the variable 'tuple_var_assignment_171068' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'tuple_var_assignment_171068', subscript_call_result_172451)
    
    # Assigning a Subscript to a Name (line 762):
    
    # Obtaining the type of the subscript
    int_172452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 8), 'int')
    
    # Call to svd(...): (line 762)
    # Processing the call arguments (line 762)
    # Getting the type of 'res' (line 762)
    res_172454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 23), 'res', False)
    # Obtaining the member 'jac' of a type (line 762)
    jac_172455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 23), res_172454, 'jac')
    # Processing the call keyword arguments (line 762)
    # Getting the type of 'False' (line 762)
    False_172456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 46), 'False', False)
    keyword_172457 = False_172456
    kwargs_172458 = {'full_matrices': keyword_172457}
    # Getting the type of 'svd' (line 762)
    svd_172453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 19), 'svd', False)
    # Calling svd(args, kwargs) (line 762)
    svd_call_result_172459 = invoke(stypy.reporting.localization.Localization(__file__, 762, 19), svd_172453, *[jac_172455], **kwargs_172458)
    
    # Obtaining the member '__getitem__' of a type (line 762)
    getitem___172460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 8), svd_call_result_172459, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 762)
    subscript_call_result_172461 = invoke(stypy.reporting.localization.Localization(__file__, 762, 8), getitem___172460, int_172452)
    
    # Assigning a type to the variable 'tuple_var_assignment_171069' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'tuple_var_assignment_171069', subscript_call_result_172461)
    
    # Assigning a Name to a Name (line 762):
    # Getting the type of 'tuple_var_assignment_171067' (line 762)
    tuple_var_assignment_171067_172462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'tuple_var_assignment_171067')
    # Assigning a type to the variable '_' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), '_', tuple_var_assignment_171067_172462)
    
    # Assigning a Name to a Name (line 762):
    # Getting the type of 'tuple_var_assignment_171068' (line 762)
    tuple_var_assignment_171068_172463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'tuple_var_assignment_171068')
    # Assigning a type to the variable 's' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 11), 's', tuple_var_assignment_171068_172463)
    
    # Assigning a Name to a Name (line 762):
    # Getting the type of 'tuple_var_assignment_171069' (line 762)
    tuple_var_assignment_171069_172464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'tuple_var_assignment_171069')
    # Assigning a type to the variable 'VT' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 14), 'VT', tuple_var_assignment_171069_172464)
    
    # Assigning a BinOp to a Name (line 763):
    
    # Assigning a BinOp to a Name (line 763):
    
    # Call to finfo(...): (line 763)
    # Processing the call arguments (line 763)
    # Getting the type of 'float' (line 763)
    float_172467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 29), 'float', False)
    # Processing the call keyword arguments (line 763)
    kwargs_172468 = {}
    # Getting the type of 'np' (line 763)
    np_172465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 20), 'np', False)
    # Obtaining the member 'finfo' of a type (line 763)
    finfo_172466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 20), np_172465, 'finfo')
    # Calling finfo(args, kwargs) (line 763)
    finfo_call_result_172469 = invoke(stypy.reporting.localization.Localization(__file__, 763, 20), finfo_172466, *[float_172467], **kwargs_172468)
    
    # Obtaining the member 'eps' of a type (line 763)
    eps_172470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 20), finfo_call_result_172469, 'eps')
    
    # Call to max(...): (line 763)
    # Processing the call arguments (line 763)
    # Getting the type of 'res' (line 763)
    res_172472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 46), 'res', False)
    # Obtaining the member 'jac' of a type (line 763)
    jac_172473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 46), res_172472, 'jac')
    # Obtaining the member 'shape' of a type (line 763)
    shape_172474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 46), jac_172473, 'shape')
    # Processing the call keyword arguments (line 763)
    kwargs_172475 = {}
    # Getting the type of 'max' (line 763)
    max_172471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 42), 'max', False)
    # Calling max(args, kwargs) (line 763)
    max_call_result_172476 = invoke(stypy.reporting.localization.Localization(__file__, 763, 42), max_172471, *[shape_172474], **kwargs_172475)
    
    # Applying the binary operator '*' (line 763)
    result_mul_172477 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 20), '*', eps_172470, max_call_result_172476)
    
    
    # Obtaining the type of the subscript
    int_172478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 65), 'int')
    # Getting the type of 's' (line 763)
    s_172479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 63), 's')
    # Obtaining the member '__getitem__' of a type (line 763)
    getitem___172480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 63), s_172479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 763)
    subscript_call_result_172481 = invoke(stypy.reporting.localization.Localization(__file__, 763, 63), getitem___172480, int_172478)
    
    # Applying the binary operator '*' (line 763)
    result_mul_172482 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 61), '*', result_mul_172477, subscript_call_result_172481)
    
    # Assigning a type to the variable 'threshold' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 8), 'threshold', result_mul_172482)
    
    # Assigning a Subscript to a Name (line 764):
    
    # Assigning a Subscript to a Name (line 764):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 's' (line 764)
    s_172483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 14), 's')
    # Getting the type of 'threshold' (line 764)
    threshold_172484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 18), 'threshold')
    # Applying the binary operator '>' (line 764)
    result_gt_172485 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 14), '>', s_172483, threshold_172484)
    
    # Getting the type of 's' (line 764)
    s_172486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 's')
    # Obtaining the member '__getitem__' of a type (line 764)
    getitem___172487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), s_172486, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 764)
    subscript_call_result_172488 = invoke(stypy.reporting.localization.Localization(__file__, 764, 12), getitem___172487, result_gt_172485)
    
    # Assigning a type to the variable 's' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 's', subscript_call_result_172488)
    
    # Assigning a Subscript to a Name (line 765):
    
    # Assigning a Subscript to a Name (line 765):
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 765)
    s_172489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 17), 's')
    # Obtaining the member 'size' of a type (line 765)
    size_172490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 17), s_172489, 'size')
    slice_172491 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 765, 13), None, size_172490, None)
    # Getting the type of 'VT' (line 765)
    VT_172492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 13), 'VT')
    # Obtaining the member '__getitem__' of a type (line 765)
    getitem___172493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 13), VT_172492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 765)
    subscript_call_result_172494 = invoke(stypy.reporting.localization.Localization(__file__, 765, 13), getitem___172493, slice_172491)
    
    # Assigning a type to the variable 'VT' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 8), 'VT', subscript_call_result_172494)
    
    # Assigning a Call to a Name (line 766):
    
    # Assigning a Call to a Name (line 766):
    
    # Call to dot(...): (line 766)
    # Processing the call arguments (line 766)
    # Getting the type of 'VT' (line 766)
    VT_172497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 22), 'VT', False)
    # Obtaining the member 'T' of a type (line 766)
    T_172498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 22), VT_172497, 'T')
    # Getting the type of 's' (line 766)
    s_172499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 29), 's', False)
    int_172500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 32), 'int')
    # Applying the binary operator '**' (line 766)
    result_pow_172501 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 29), '**', s_172499, int_172500)
    
    # Applying the binary operator 'div' (line 766)
    result_div_172502 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 22), 'div', T_172498, result_pow_172501)
    
    # Getting the type of 'VT' (line 766)
    VT_172503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 35), 'VT', False)
    # Processing the call keyword arguments (line 766)
    kwargs_172504 = {}
    # Getting the type of 'np' (line 766)
    np_172495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 15), 'np', False)
    # Obtaining the member 'dot' of a type (line 766)
    dot_172496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 15), np_172495, 'dot')
    # Calling dot(args, kwargs) (line 766)
    dot_call_result_172505 = invoke(stypy.reporting.localization.Localization(__file__, 766, 15), dot_172496, *[result_div_172502, VT_172503], **kwargs_172504)
    
    # Assigning a type to the variable 'pcov' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'pcov', dot_call_result_172505)
    
    # Assigning a Name to a Name (line 767):
    
    # Assigning a Name to a Name (line 767):
    # Getting the type of 'False' (line 767)
    False_172506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 22), 'False')
    # Assigning a type to the variable 'return_full' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'return_full', False_172506)
    # SSA join for if statement (line 739)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 769):
    
    # Assigning a Name to a Name (line 769):
    # Getting the type of 'False' (line 769)
    False_172507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 15), 'False')
    # Assigning a type to the variable 'warn_cov' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'warn_cov', False_172507)
    
    # Type idiom detected: calculating its left and rigth part (line 770)
    # Getting the type of 'pcov' (line 770)
    pcov_172508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 7), 'pcov')
    # Getting the type of 'None' (line 770)
    None_172509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 15), 'None')
    
    (may_be_172510, more_types_in_union_172511) = may_be_none(pcov_172508, None_172509)

    if may_be_172510:

        if more_types_in_union_172511:
            # Runtime conditional SSA (line 770)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 772):
        
        # Assigning a Call to a Name (line 772):
        
        # Call to zeros(...): (line 772)
        # Processing the call arguments (line 772)
        
        # Obtaining an instance of the builtin type 'tuple' (line 772)
        tuple_172513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 772)
        # Adding element type (line 772)
        
        # Call to len(...): (line 772)
        # Processing the call arguments (line 772)
        # Getting the type of 'popt' (line 772)
        popt_172515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 26), 'popt', False)
        # Processing the call keyword arguments (line 772)
        kwargs_172516 = {}
        # Getting the type of 'len' (line 772)
        len_172514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 22), 'len', False)
        # Calling len(args, kwargs) (line 772)
        len_call_result_172517 = invoke(stypy.reporting.localization.Localization(__file__, 772, 22), len_172514, *[popt_172515], **kwargs_172516)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 22), tuple_172513, len_call_result_172517)
        # Adding element type (line 772)
        
        # Call to len(...): (line 772)
        # Processing the call arguments (line 772)
        # Getting the type of 'popt' (line 772)
        popt_172519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 37), 'popt', False)
        # Processing the call keyword arguments (line 772)
        kwargs_172520 = {}
        # Getting the type of 'len' (line 772)
        len_172518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 33), 'len', False)
        # Calling len(args, kwargs) (line 772)
        len_call_result_172521 = invoke(stypy.reporting.localization.Localization(__file__, 772, 33), len_172518, *[popt_172519], **kwargs_172520)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 22), tuple_172513, len_call_result_172521)
        
        # Processing the call keyword arguments (line 772)
        # Getting the type of 'float' (line 772)
        float_172522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 51), 'float', False)
        keyword_172523 = float_172522
        kwargs_172524 = {'dtype': keyword_172523}
        # Getting the type of 'zeros' (line 772)
        zeros_172512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 15), 'zeros', False)
        # Calling zeros(args, kwargs) (line 772)
        zeros_call_result_172525 = invoke(stypy.reporting.localization.Localization(__file__, 772, 15), zeros_172512, *[tuple_172513], **kwargs_172524)
        
        # Assigning a type to the variable 'pcov' (line 772)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 8), 'pcov', zeros_call_result_172525)
        
        # Call to fill(...): (line 773)
        # Processing the call arguments (line 773)
        # Getting the type of 'inf' (line 773)
        inf_172528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 18), 'inf', False)
        # Processing the call keyword arguments (line 773)
        kwargs_172529 = {}
        # Getting the type of 'pcov' (line 773)
        pcov_172526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'pcov', False)
        # Obtaining the member 'fill' of a type (line 773)
        fill_172527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 8), pcov_172526, 'fill')
        # Calling fill(args, kwargs) (line 773)
        fill_call_result_172530 = invoke(stypy.reporting.localization.Localization(__file__, 773, 8), fill_172527, *[inf_172528], **kwargs_172529)
        
        
        # Assigning a Name to a Name (line 774):
        
        # Assigning a Name to a Name (line 774):
        # Getting the type of 'True' (line 774)
        True_172531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 19), 'True')
        # Assigning a type to the variable 'warn_cov' (line 774)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'warn_cov', True_172531)

        if more_types_in_union_172511:
            # Runtime conditional SSA for else branch (line 770)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_172510) or more_types_in_union_172511):
        
        
        # Getting the type of 'absolute_sigma' (line 775)
        absolute_sigma_172532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 13), 'absolute_sigma')
        # Applying the 'not' unary operator (line 775)
        result_not__172533 = python_operator(stypy.reporting.localization.Localization(__file__, 775, 9), 'not', absolute_sigma_172532)
        
        # Testing the type of an if condition (line 775)
        if_condition_172534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 775, 9), result_not__172533)
        # Assigning a type to the variable 'if_condition_172534' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 9), 'if_condition_172534', if_condition_172534)
        # SSA begins for if statement (line 775)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'ydata' (line 776)
        ydata_172535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 11), 'ydata')
        # Obtaining the member 'size' of a type (line 776)
        size_172536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 11), ydata_172535, 'size')
        # Getting the type of 'p0' (line 776)
        p0_172537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 24), 'p0')
        # Obtaining the member 'size' of a type (line 776)
        size_172538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 24), p0_172537, 'size')
        # Applying the binary operator '>' (line 776)
        result_gt_172539 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 11), '>', size_172536, size_172538)
        
        # Testing the type of an if condition (line 776)
        if_condition_172540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 776, 8), result_gt_172539)
        # Assigning a type to the variable 'if_condition_172540' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'if_condition_172540', if_condition_172540)
        # SSA begins for if statement (line 776)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 777):
        
        # Assigning a BinOp to a Name (line 777):
        # Getting the type of 'cost' (line 777)
        cost_172541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 19), 'cost')
        # Getting the type of 'ydata' (line 777)
        ydata_172542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 27), 'ydata')
        # Obtaining the member 'size' of a type (line 777)
        size_172543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 27), ydata_172542, 'size')
        # Getting the type of 'p0' (line 777)
        p0_172544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 40), 'p0')
        # Obtaining the member 'size' of a type (line 777)
        size_172545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 40), p0_172544, 'size')
        # Applying the binary operator '-' (line 777)
        result_sub_172546 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 27), '-', size_172543, size_172545)
        
        # Applying the binary operator 'div' (line 777)
        result_div_172547 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 19), 'div', cost_172541, result_sub_172546)
        
        # Assigning a type to the variable 's_sq' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 12), 's_sq', result_div_172547)
        
        # Assigning a BinOp to a Name (line 778):
        
        # Assigning a BinOp to a Name (line 778):
        # Getting the type of 'pcov' (line 778)
        pcov_172548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 19), 'pcov')
        # Getting the type of 's_sq' (line 778)
        s_sq_172549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 26), 's_sq')
        # Applying the binary operator '*' (line 778)
        result_mul_172550 = python_operator(stypy.reporting.localization.Localization(__file__, 778, 19), '*', pcov_172548, s_sq_172549)
        
        # Assigning a type to the variable 'pcov' (line 778)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 12), 'pcov', result_mul_172550)
        # SSA branch for the else part of an if statement (line 776)
        module_type_store.open_ssa_branch('else')
        
        # Call to fill(...): (line 780)
        # Processing the call arguments (line 780)
        # Getting the type of 'inf' (line 780)
        inf_172553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 22), 'inf', False)
        # Processing the call keyword arguments (line 780)
        kwargs_172554 = {}
        # Getting the type of 'pcov' (line 780)
        pcov_172551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 12), 'pcov', False)
        # Obtaining the member 'fill' of a type (line 780)
        fill_172552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 12), pcov_172551, 'fill')
        # Calling fill(args, kwargs) (line 780)
        fill_call_result_172555 = invoke(stypy.reporting.localization.Localization(__file__, 780, 12), fill_172552, *[inf_172553], **kwargs_172554)
        
        
        # Assigning a Name to a Name (line 781):
        
        # Assigning a Name to a Name (line 781):
        # Getting the type of 'True' (line 781)
        True_172556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 23), 'True')
        # Assigning a type to the variable 'warn_cov' (line 781)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'warn_cov', True_172556)
        # SSA join for if statement (line 776)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 775)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_172510 and more_types_in_union_172511):
            # SSA join for if statement (line 770)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'warn_cov' (line 783)
    warn_cov_172557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 7), 'warn_cov')
    # Testing the type of an if condition (line 783)
    if_condition_172558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 783, 4), warn_cov_172557)
    # Assigning a type to the variable 'if_condition_172558' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'if_condition_172558', if_condition_172558)
    # SSA begins for if statement (line 783)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 784)
    # Processing the call arguments (line 784)
    str_172561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 22), 'str', 'Covariance of the parameters could not be estimated')
    # Processing the call keyword arguments (line 784)
    # Getting the type of 'OptimizeWarning' (line 785)
    OptimizeWarning_172562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 31), 'OptimizeWarning', False)
    keyword_172563 = OptimizeWarning_172562
    kwargs_172564 = {'category': keyword_172563}
    # Getting the type of 'warnings' (line 784)
    warnings_172559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 784)
    warn_172560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 8), warnings_172559, 'warn')
    # Calling warn(args, kwargs) (line 784)
    warn_call_result_172565 = invoke(stypy.reporting.localization.Localization(__file__, 784, 8), warn_172560, *[str_172561], **kwargs_172564)
    
    # SSA join for if statement (line 783)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_full' (line 787)
    return_full_172566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 7), 'return_full')
    # Testing the type of an if condition (line 787)
    if_condition_172567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 787, 4), return_full_172566)
    # Assigning a type to the variable 'if_condition_172567' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'if_condition_172567', if_condition_172567)
    # SSA begins for if statement (line 787)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 788)
    tuple_172568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 788)
    # Adding element type (line 788)
    # Getting the type of 'popt' (line 788)
    popt_172569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 15), 'popt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 15), tuple_172568, popt_172569)
    # Adding element type (line 788)
    # Getting the type of 'pcov' (line 788)
    pcov_172570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 21), 'pcov')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 15), tuple_172568, pcov_172570)
    # Adding element type (line 788)
    # Getting the type of 'infodict' (line 788)
    infodict_172571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 27), 'infodict')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 15), tuple_172568, infodict_172571)
    # Adding element type (line 788)
    # Getting the type of 'errmsg' (line 788)
    errmsg_172572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 37), 'errmsg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 15), tuple_172568, errmsg_172572)
    # Adding element type (line 788)
    # Getting the type of 'ier' (line 788)
    ier_172573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 45), 'ier')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 15), tuple_172568, ier_172573)
    
    # Assigning a type to the variable 'stypy_return_type' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'stypy_return_type', tuple_172568)
    # SSA branch for the else part of an if statement (line 787)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 790)
    tuple_172574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 790)
    # Adding element type (line 790)
    # Getting the type of 'popt' (line 790)
    popt_172575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 15), 'popt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 15), tuple_172574, popt_172575)
    # Adding element type (line 790)
    # Getting the type of 'pcov' (line 790)
    pcov_172576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 21), 'pcov')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 15), tuple_172574, pcov_172576)
    
    # Assigning a type to the variable 'stypy_return_type' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'stypy_return_type', tuple_172574)
    # SSA join for if statement (line 787)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'curve_fit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'curve_fit' in the type store
    # Getting the type of 'stypy_return_type' (line 502)
    stypy_return_type_172577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172577)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'curve_fit'
    return stypy_return_type_172577

# Assigning a type to the variable 'curve_fit' (line 502)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 0), 'curve_fit', curve_fit)

@norecursion
def check_gradient(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 793)
    tuple_172578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 793)
    
    int_172579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 53), 'int')
    defaults = [tuple_172578, int_172579]
    # Create a new context for function 'check_gradient'
    module_type_store = module_type_store.open_function_context('check_gradient', 793, 0, False)
    
    # Passed parameters checking function
    check_gradient.stypy_localization = localization
    check_gradient.stypy_type_of_self = None
    check_gradient.stypy_type_store = module_type_store
    check_gradient.stypy_function_name = 'check_gradient'
    check_gradient.stypy_param_names_list = ['fcn', 'Dfcn', 'x0', 'args', 'col_deriv']
    check_gradient.stypy_varargs_param_name = None
    check_gradient.stypy_kwargs_param_name = None
    check_gradient.stypy_call_defaults = defaults
    check_gradient.stypy_call_varargs = varargs
    check_gradient.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_gradient', ['fcn', 'Dfcn', 'x0', 'args', 'col_deriv'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_gradient', localization, ['fcn', 'Dfcn', 'x0', 'args', 'col_deriv'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_gradient(...)' code ##################

    str_172580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, (-1)), 'str', 'Perform a simple check on the gradient for correctness.\n\n    ')
    
    # Assigning a Call to a Name (line 798):
    
    # Assigning a Call to a Name (line 798):
    
    # Call to atleast_1d(...): (line 798)
    # Processing the call arguments (line 798)
    # Getting the type of 'x0' (line 798)
    x0_172582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 19), 'x0', False)
    # Processing the call keyword arguments (line 798)
    kwargs_172583 = {}
    # Getting the type of 'atleast_1d' (line 798)
    atleast_1d_172581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 798)
    atleast_1d_call_result_172584 = invoke(stypy.reporting.localization.Localization(__file__, 798, 8), atleast_1d_172581, *[x0_172582], **kwargs_172583)
    
    # Assigning a type to the variable 'x' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'x', atleast_1d_call_result_172584)
    
    # Assigning a Call to a Name (line 799):
    
    # Assigning a Call to a Name (line 799):
    
    # Call to len(...): (line 799)
    # Processing the call arguments (line 799)
    # Getting the type of 'x' (line 799)
    x_172586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 12), 'x', False)
    # Processing the call keyword arguments (line 799)
    kwargs_172587 = {}
    # Getting the type of 'len' (line 799)
    len_172585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'len', False)
    # Calling len(args, kwargs) (line 799)
    len_call_result_172588 = invoke(stypy.reporting.localization.Localization(__file__, 799, 8), len_172585, *[x_172586], **kwargs_172587)
    
    # Assigning a type to the variable 'n' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'n', len_call_result_172588)
    
    # Assigning a Call to a Name (line 800):
    
    # Assigning a Call to a Name (line 800):
    
    # Call to reshape(...): (line 800)
    # Processing the call arguments (line 800)
    
    # Obtaining an instance of the builtin type 'tuple' (line 800)
    tuple_172591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 800)
    # Adding element type (line 800)
    # Getting the type of 'n' (line 800)
    n_172592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 19), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 800, 19), tuple_172591, n_172592)
    
    # Processing the call keyword arguments (line 800)
    kwargs_172593 = {}
    # Getting the type of 'x' (line 800)
    x_172589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'x', False)
    # Obtaining the member 'reshape' of a type (line 800)
    reshape_172590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 8), x_172589, 'reshape')
    # Calling reshape(args, kwargs) (line 800)
    reshape_call_result_172594 = invoke(stypy.reporting.localization.Localization(__file__, 800, 8), reshape_172590, *[tuple_172591], **kwargs_172593)
    
    # Assigning a type to the variable 'x' (line 800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 4), 'x', reshape_call_result_172594)
    
    # Assigning a Call to a Name (line 801):
    
    # Assigning a Call to a Name (line 801):
    
    # Call to atleast_1d(...): (line 801)
    # Processing the call arguments (line 801)
    
    # Call to fcn(...): (line 801)
    # Processing the call arguments (line 801)
    # Getting the type of 'x' (line 801)
    x_172597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 26), 'x', False)
    # Getting the type of 'args' (line 801)
    args_172598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 30), 'args', False)
    # Processing the call keyword arguments (line 801)
    kwargs_172599 = {}
    # Getting the type of 'fcn' (line 801)
    fcn_172596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 22), 'fcn', False)
    # Calling fcn(args, kwargs) (line 801)
    fcn_call_result_172600 = invoke(stypy.reporting.localization.Localization(__file__, 801, 22), fcn_172596, *[x_172597, args_172598], **kwargs_172599)
    
    # Processing the call keyword arguments (line 801)
    kwargs_172601 = {}
    # Getting the type of 'atleast_1d' (line 801)
    atleast_1d_172595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 11), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 801)
    atleast_1d_call_result_172602 = invoke(stypy.reporting.localization.Localization(__file__, 801, 11), atleast_1d_172595, *[fcn_call_result_172600], **kwargs_172601)
    
    # Assigning a type to the variable 'fvec' (line 801)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 4), 'fvec', atleast_1d_call_result_172602)
    
    # Assigning a Call to a Name (line 802):
    
    # Assigning a Call to a Name (line 802):
    
    # Call to len(...): (line 802)
    # Processing the call arguments (line 802)
    # Getting the type of 'fvec' (line 802)
    fvec_172604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 12), 'fvec', False)
    # Processing the call keyword arguments (line 802)
    kwargs_172605 = {}
    # Getting the type of 'len' (line 802)
    len_172603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'len', False)
    # Calling len(args, kwargs) (line 802)
    len_call_result_172606 = invoke(stypy.reporting.localization.Localization(__file__, 802, 8), len_172603, *[fvec_172604], **kwargs_172605)
    
    # Assigning a type to the variable 'm' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 4), 'm', len_call_result_172606)
    
    # Assigning a Call to a Name (line 803):
    
    # Assigning a Call to a Name (line 803):
    
    # Call to reshape(...): (line 803)
    # Processing the call arguments (line 803)
    
    # Obtaining an instance of the builtin type 'tuple' (line 803)
    tuple_172609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 803)
    # Adding element type (line 803)
    # Getting the type of 'm' (line 803)
    m_172610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 25), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 803, 25), tuple_172609, m_172610)
    
    # Processing the call keyword arguments (line 803)
    kwargs_172611 = {}
    # Getting the type of 'fvec' (line 803)
    fvec_172607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 11), 'fvec', False)
    # Obtaining the member 'reshape' of a type (line 803)
    reshape_172608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 11), fvec_172607, 'reshape')
    # Calling reshape(args, kwargs) (line 803)
    reshape_call_result_172612 = invoke(stypy.reporting.localization.Localization(__file__, 803, 11), reshape_172608, *[tuple_172609], **kwargs_172611)
    
    # Assigning a type to the variable 'fvec' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 4), 'fvec', reshape_call_result_172612)
    
    # Assigning a Name to a Name (line 804):
    
    # Assigning a Name to a Name (line 804):
    # Getting the type of 'm' (line 804)
    m_172613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 13), 'm')
    # Assigning a type to the variable 'ldfjac' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'ldfjac', m_172613)
    
    # Assigning a Call to a Name (line 805):
    
    # Assigning a Call to a Name (line 805):
    
    # Call to atleast_1d(...): (line 805)
    # Processing the call arguments (line 805)
    
    # Call to Dfcn(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'x' (line 805)
    x_172616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 27), 'x', False)
    # Getting the type of 'args' (line 805)
    args_172617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 31), 'args', False)
    # Processing the call keyword arguments (line 805)
    kwargs_172618 = {}
    # Getting the type of 'Dfcn' (line 805)
    Dfcn_172615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 22), 'Dfcn', False)
    # Calling Dfcn(args, kwargs) (line 805)
    Dfcn_call_result_172619 = invoke(stypy.reporting.localization.Localization(__file__, 805, 22), Dfcn_172615, *[x_172616, args_172617], **kwargs_172618)
    
    # Processing the call keyword arguments (line 805)
    kwargs_172620 = {}
    # Getting the type of 'atleast_1d' (line 805)
    atleast_1d_172614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 11), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 805)
    atleast_1d_call_result_172621 = invoke(stypy.reporting.localization.Localization(__file__, 805, 11), atleast_1d_172614, *[Dfcn_call_result_172619], **kwargs_172620)
    
    # Assigning a type to the variable 'fjac' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'fjac', atleast_1d_call_result_172621)
    
    # Assigning a Call to a Name (line 806):
    
    # Assigning a Call to a Name (line 806):
    
    # Call to reshape(...): (line 806)
    # Processing the call arguments (line 806)
    
    # Obtaining an instance of the builtin type 'tuple' (line 806)
    tuple_172624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 806)
    # Adding element type (line 806)
    # Getting the type of 'm' (line 806)
    m_172625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 25), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 25), tuple_172624, m_172625)
    # Adding element type (line 806)
    # Getting the type of 'n' (line 806)
    n_172626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 25), tuple_172624, n_172626)
    
    # Processing the call keyword arguments (line 806)
    kwargs_172627 = {}
    # Getting the type of 'fjac' (line 806)
    fjac_172622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 11), 'fjac', False)
    # Obtaining the member 'reshape' of a type (line 806)
    reshape_172623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 11), fjac_172622, 'reshape')
    # Calling reshape(args, kwargs) (line 806)
    reshape_call_result_172628 = invoke(stypy.reporting.localization.Localization(__file__, 806, 11), reshape_172623, *[tuple_172624], **kwargs_172627)
    
    # Assigning a type to the variable 'fjac' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 4), 'fjac', reshape_call_result_172628)
    
    
    # Getting the type of 'col_deriv' (line 807)
    col_deriv_172629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 7), 'col_deriv')
    int_172630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 20), 'int')
    # Applying the binary operator '==' (line 807)
    result_eq_172631 = python_operator(stypy.reporting.localization.Localization(__file__, 807, 7), '==', col_deriv_172629, int_172630)
    
    # Testing the type of an if condition (line 807)
    if_condition_172632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 807, 4), result_eq_172631)
    # Assigning a type to the variable 'if_condition_172632' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 4), 'if_condition_172632', if_condition_172632)
    # SSA begins for if statement (line 807)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 808):
    
    # Assigning a Call to a Name (line 808):
    
    # Call to transpose(...): (line 808)
    # Processing the call arguments (line 808)
    # Getting the type of 'fjac' (line 808)
    fjac_172634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 25), 'fjac', False)
    # Processing the call keyword arguments (line 808)
    kwargs_172635 = {}
    # Getting the type of 'transpose' (line 808)
    transpose_172633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 15), 'transpose', False)
    # Calling transpose(args, kwargs) (line 808)
    transpose_call_result_172636 = invoke(stypy.reporting.localization.Localization(__file__, 808, 15), transpose_172633, *[fjac_172634], **kwargs_172635)
    
    # Assigning a type to the variable 'fjac' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'fjac', transpose_call_result_172636)
    # SSA join for if statement (line 807)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 810):
    
    # Assigning a Call to a Name (line 810):
    
    # Call to zeros(...): (line 810)
    # Processing the call arguments (line 810)
    
    # Obtaining an instance of the builtin type 'tuple' (line 810)
    tuple_172638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 810)
    # Adding element type (line 810)
    # Getting the type of 'n' (line 810)
    n_172639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 16), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 810, 16), tuple_172638, n_172639)
    
    # Getting the type of 'float' (line 810)
    float_172640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 21), 'float', False)
    # Processing the call keyword arguments (line 810)
    kwargs_172641 = {}
    # Getting the type of 'zeros' (line 810)
    zeros_172637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 9), 'zeros', False)
    # Calling zeros(args, kwargs) (line 810)
    zeros_call_result_172642 = invoke(stypy.reporting.localization.Localization(__file__, 810, 9), zeros_172637, *[tuple_172638, float_172640], **kwargs_172641)
    
    # Assigning a type to the variable 'xp' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'xp', zeros_call_result_172642)
    
    # Assigning a Call to a Name (line 811):
    
    # Assigning a Call to a Name (line 811):
    
    # Call to zeros(...): (line 811)
    # Processing the call arguments (line 811)
    
    # Obtaining an instance of the builtin type 'tuple' (line 811)
    tuple_172644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 811)
    # Adding element type (line 811)
    # Getting the type of 'm' (line 811)
    m_172645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 17), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 17), tuple_172644, m_172645)
    
    # Getting the type of 'float' (line 811)
    float_172646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 22), 'float', False)
    # Processing the call keyword arguments (line 811)
    kwargs_172647 = {}
    # Getting the type of 'zeros' (line 811)
    zeros_172643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 10), 'zeros', False)
    # Calling zeros(args, kwargs) (line 811)
    zeros_call_result_172648 = invoke(stypy.reporting.localization.Localization(__file__, 811, 10), zeros_172643, *[tuple_172644, float_172646], **kwargs_172647)
    
    # Assigning a type to the variable 'err' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 4), 'err', zeros_call_result_172648)
    
    # Assigning a Name to a Name (line 812):
    
    # Assigning a Name to a Name (line 812):
    # Getting the type of 'None' (line 812)
    None_172649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'None')
    # Assigning a type to the variable 'fvecp' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'fvecp', None_172649)
    
    # Call to _chkder(...): (line 813)
    # Processing the call arguments (line 813)
    # Getting the type of 'm' (line 813)
    m_172652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 21), 'm', False)
    # Getting the type of 'n' (line 813)
    n_172653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 24), 'n', False)
    # Getting the type of 'x' (line 813)
    x_172654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 27), 'x', False)
    # Getting the type of 'fvec' (line 813)
    fvec_172655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 30), 'fvec', False)
    # Getting the type of 'fjac' (line 813)
    fjac_172656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 36), 'fjac', False)
    # Getting the type of 'ldfjac' (line 813)
    ldfjac_172657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 42), 'ldfjac', False)
    # Getting the type of 'xp' (line 813)
    xp_172658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 50), 'xp', False)
    # Getting the type of 'fvecp' (line 813)
    fvecp_172659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 54), 'fvecp', False)
    int_172660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 61), 'int')
    # Getting the type of 'err' (line 813)
    err_172661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 64), 'err', False)
    # Processing the call keyword arguments (line 813)
    kwargs_172662 = {}
    # Getting the type of '_minpack' (line 813)
    _minpack_172650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), '_minpack', False)
    # Obtaining the member '_chkder' of a type (line 813)
    _chkder_172651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 4), _minpack_172650, '_chkder')
    # Calling _chkder(args, kwargs) (line 813)
    _chkder_call_result_172663 = invoke(stypy.reporting.localization.Localization(__file__, 813, 4), _chkder_172651, *[m_172652, n_172653, x_172654, fvec_172655, fjac_172656, ldfjac_172657, xp_172658, fvecp_172659, int_172660, err_172661], **kwargs_172662)
    
    
    # Assigning a Call to a Name (line 815):
    
    # Assigning a Call to a Name (line 815):
    
    # Call to atleast_1d(...): (line 815)
    # Processing the call arguments (line 815)
    
    # Call to fcn(...): (line 815)
    # Processing the call arguments (line 815)
    # Getting the type of 'xp' (line 815)
    xp_172666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 27), 'xp', False)
    # Getting the type of 'args' (line 815)
    args_172667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 32), 'args', False)
    # Processing the call keyword arguments (line 815)
    kwargs_172668 = {}
    # Getting the type of 'fcn' (line 815)
    fcn_172665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 23), 'fcn', False)
    # Calling fcn(args, kwargs) (line 815)
    fcn_call_result_172669 = invoke(stypy.reporting.localization.Localization(__file__, 815, 23), fcn_172665, *[xp_172666, args_172667], **kwargs_172668)
    
    # Processing the call keyword arguments (line 815)
    kwargs_172670 = {}
    # Getting the type of 'atleast_1d' (line 815)
    atleast_1d_172664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 12), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 815)
    atleast_1d_call_result_172671 = invoke(stypy.reporting.localization.Localization(__file__, 815, 12), atleast_1d_172664, *[fcn_call_result_172669], **kwargs_172670)
    
    # Assigning a type to the variable 'fvecp' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'fvecp', atleast_1d_call_result_172671)
    
    # Assigning a Call to a Name (line 816):
    
    # Assigning a Call to a Name (line 816):
    
    # Call to reshape(...): (line 816)
    # Processing the call arguments (line 816)
    
    # Obtaining an instance of the builtin type 'tuple' (line 816)
    tuple_172674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 816)
    # Adding element type (line 816)
    # Getting the type of 'm' (line 816)
    m_172675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 27), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 27), tuple_172674, m_172675)
    
    # Processing the call keyword arguments (line 816)
    kwargs_172676 = {}
    # Getting the type of 'fvecp' (line 816)
    fvecp_172672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 12), 'fvecp', False)
    # Obtaining the member 'reshape' of a type (line 816)
    reshape_172673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 12), fvecp_172672, 'reshape')
    # Calling reshape(args, kwargs) (line 816)
    reshape_call_result_172677 = invoke(stypy.reporting.localization.Localization(__file__, 816, 12), reshape_172673, *[tuple_172674], **kwargs_172676)
    
    # Assigning a type to the variable 'fvecp' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'fvecp', reshape_call_result_172677)
    
    # Call to _chkder(...): (line 817)
    # Processing the call arguments (line 817)
    # Getting the type of 'm' (line 817)
    m_172680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 21), 'm', False)
    # Getting the type of 'n' (line 817)
    n_172681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 24), 'n', False)
    # Getting the type of 'x' (line 817)
    x_172682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 27), 'x', False)
    # Getting the type of 'fvec' (line 817)
    fvec_172683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 30), 'fvec', False)
    # Getting the type of 'fjac' (line 817)
    fjac_172684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 36), 'fjac', False)
    # Getting the type of 'ldfjac' (line 817)
    ldfjac_172685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 42), 'ldfjac', False)
    # Getting the type of 'xp' (line 817)
    xp_172686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 50), 'xp', False)
    # Getting the type of 'fvecp' (line 817)
    fvecp_172687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 54), 'fvecp', False)
    int_172688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 61), 'int')
    # Getting the type of 'err' (line 817)
    err_172689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 64), 'err', False)
    # Processing the call keyword arguments (line 817)
    kwargs_172690 = {}
    # Getting the type of '_minpack' (line 817)
    _minpack_172678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), '_minpack', False)
    # Obtaining the member '_chkder' of a type (line 817)
    _chkder_172679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 4), _minpack_172678, '_chkder')
    # Calling _chkder(args, kwargs) (line 817)
    _chkder_call_result_172691 = invoke(stypy.reporting.localization.Localization(__file__, 817, 4), _chkder_172679, *[m_172680, n_172681, x_172682, fvec_172683, fjac_172684, ldfjac_172685, xp_172686, fvecp_172687, int_172688, err_172689], **kwargs_172690)
    
    
    # Assigning a Call to a Name (line 819):
    
    # Assigning a Call to a Name (line 819):
    
    # Call to product(...): (line 819)
    # Processing the call arguments (line 819)
    
    # Call to greater(...): (line 819)
    # Processing the call arguments (line 819)
    # Getting the type of 'err' (line 819)
    err_172694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 28), 'err', False)
    float_172695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 33), 'float')
    # Processing the call keyword arguments (line 819)
    kwargs_172696 = {}
    # Getting the type of 'greater' (line 819)
    greater_172693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 20), 'greater', False)
    # Calling greater(args, kwargs) (line 819)
    greater_call_result_172697 = invoke(stypy.reporting.localization.Localization(__file__, 819, 20), greater_172693, *[err_172694, float_172695], **kwargs_172696)
    
    # Processing the call keyword arguments (line 819)
    int_172698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 44), 'int')
    keyword_172699 = int_172698
    kwargs_172700 = {'axis': keyword_172699}
    # Getting the type of 'product' (line 819)
    product_172692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 12), 'product', False)
    # Calling product(args, kwargs) (line 819)
    product_call_result_172701 = invoke(stypy.reporting.localization.Localization(__file__, 819, 12), product_172692, *[greater_call_result_172697], **kwargs_172700)
    
    # Assigning a type to the variable 'good' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'good', product_call_result_172701)
    
    # Obtaining an instance of the builtin type 'tuple' (line 821)
    tuple_172702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 821)
    # Adding element type (line 821)
    # Getting the type of 'good' (line 821)
    good_172703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 12), 'good')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 12), tuple_172702, good_172703)
    # Adding element type (line 821)
    # Getting the type of 'err' (line 821)
    err_172704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 18), 'err')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 12), tuple_172702, err_172704)
    
    # Assigning a type to the variable 'stypy_return_type' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 4), 'stypy_return_type', tuple_172702)
    
    # ################# End of 'check_gradient(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_gradient' in the type store
    # Getting the type of 'stypy_return_type' (line 793)
    stypy_return_type_172705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172705)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_gradient'
    return stypy_return_type_172705

# Assigning a type to the variable 'check_gradient' (line 793)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 0), 'check_gradient', check_gradient)

@norecursion
def _del2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_del2'
    module_type_store = module_type_store.open_function_context('_del2', 824, 0, False)
    
    # Passed parameters checking function
    _del2.stypy_localization = localization
    _del2.stypy_type_of_self = None
    _del2.stypy_type_store = module_type_store
    _del2.stypy_function_name = '_del2'
    _del2.stypy_param_names_list = ['p0', 'p1', 'd']
    _del2.stypy_varargs_param_name = None
    _del2.stypy_kwargs_param_name = None
    _del2.stypy_call_defaults = defaults
    _del2.stypy_call_varargs = varargs
    _del2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_del2', ['p0', 'p1', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_del2', localization, ['p0', 'p1', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_del2(...)' code ##################

    # Getting the type of 'p0' (line 825)
    p0_172706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 11), 'p0')
    
    # Call to square(...): (line 825)
    # Processing the call arguments (line 825)
    # Getting the type of 'p1' (line 825)
    p1_172709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 26), 'p1', False)
    # Getting the type of 'p0' (line 825)
    p0_172710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 31), 'p0', False)
    # Applying the binary operator '-' (line 825)
    result_sub_172711 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 26), '-', p1_172709, p0_172710)
    
    # Processing the call keyword arguments (line 825)
    kwargs_172712 = {}
    # Getting the type of 'np' (line 825)
    np_172707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 16), 'np', False)
    # Obtaining the member 'square' of a type (line 825)
    square_172708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 16), np_172707, 'square')
    # Calling square(args, kwargs) (line 825)
    square_call_result_172713 = invoke(stypy.reporting.localization.Localization(__file__, 825, 16), square_172708, *[result_sub_172711], **kwargs_172712)
    
    # Getting the type of 'd' (line 825)
    d_172714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 37), 'd')
    # Applying the binary operator 'div' (line 825)
    result_div_172715 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 16), 'div', square_call_result_172713, d_172714)
    
    # Applying the binary operator '-' (line 825)
    result_sub_172716 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), '-', p0_172706, result_div_172715)
    
    # Assigning a type to the variable 'stypy_return_type' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'stypy_return_type', result_sub_172716)
    
    # ################# End of '_del2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_del2' in the type store
    # Getting the type of 'stypy_return_type' (line 824)
    stypy_return_type_172717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172717)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_del2'
    return stypy_return_type_172717

# Assigning a type to the variable '_del2' (line 824)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 0), '_del2', _del2)

@norecursion
def _relerr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_relerr'
    module_type_store = module_type_store.open_function_context('_relerr', 828, 0, False)
    
    # Passed parameters checking function
    _relerr.stypy_localization = localization
    _relerr.stypy_type_of_self = None
    _relerr.stypy_type_store = module_type_store
    _relerr.stypy_function_name = '_relerr'
    _relerr.stypy_param_names_list = ['actual', 'desired']
    _relerr.stypy_varargs_param_name = None
    _relerr.stypy_kwargs_param_name = None
    _relerr.stypy_call_defaults = defaults
    _relerr.stypy_call_varargs = varargs
    _relerr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_relerr', ['actual', 'desired'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_relerr', localization, ['actual', 'desired'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_relerr(...)' code ##################

    # Getting the type of 'actual' (line 829)
    actual_172718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 12), 'actual')
    # Getting the type of 'desired' (line 829)
    desired_172719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 21), 'desired')
    # Applying the binary operator '-' (line 829)
    result_sub_172720 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 12), '-', actual_172718, desired_172719)
    
    # Getting the type of 'desired' (line 829)
    desired_172721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 32), 'desired')
    # Applying the binary operator 'div' (line 829)
    result_div_172722 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 11), 'div', result_sub_172720, desired_172721)
    
    # Assigning a type to the variable 'stypy_return_type' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 4), 'stypy_return_type', result_div_172722)
    
    # ################# End of '_relerr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_relerr' in the type store
    # Getting the type of 'stypy_return_type' (line 828)
    stypy_return_type_172723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172723)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_relerr'
    return stypy_return_type_172723

# Assigning a type to the variable '_relerr' (line 828)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 0), '_relerr', _relerr)

@norecursion
def _fixed_point_helper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fixed_point_helper'
    module_type_store = module_type_store.open_function_context('_fixed_point_helper', 832, 0, False)
    
    # Passed parameters checking function
    _fixed_point_helper.stypy_localization = localization
    _fixed_point_helper.stypy_type_of_self = None
    _fixed_point_helper.stypy_type_store = module_type_store
    _fixed_point_helper.stypy_function_name = '_fixed_point_helper'
    _fixed_point_helper.stypy_param_names_list = ['func', 'x0', 'args', 'xtol', 'maxiter', 'use_accel']
    _fixed_point_helper.stypy_varargs_param_name = None
    _fixed_point_helper.stypy_kwargs_param_name = None
    _fixed_point_helper.stypy_call_defaults = defaults
    _fixed_point_helper.stypy_call_varargs = varargs
    _fixed_point_helper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fixed_point_helper', ['func', 'x0', 'args', 'xtol', 'maxiter', 'use_accel'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fixed_point_helper', localization, ['func', 'x0', 'args', 'xtol', 'maxiter', 'use_accel'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fixed_point_helper(...)' code ##################

    
    # Assigning a Name to a Name (line 833):
    
    # Assigning a Name to a Name (line 833):
    # Getting the type of 'x0' (line 833)
    x0_172724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 9), 'x0')
    # Assigning a type to the variable 'p0' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'p0', x0_172724)
    
    
    # Call to range(...): (line 834)
    # Processing the call arguments (line 834)
    # Getting the type of 'maxiter' (line 834)
    maxiter_172726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 19), 'maxiter', False)
    # Processing the call keyword arguments (line 834)
    kwargs_172727 = {}
    # Getting the type of 'range' (line 834)
    range_172725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 13), 'range', False)
    # Calling range(args, kwargs) (line 834)
    range_call_result_172728 = invoke(stypy.reporting.localization.Localization(__file__, 834, 13), range_172725, *[maxiter_172726], **kwargs_172727)
    
    # Testing the type of a for loop iterable (line 834)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 834, 4), range_call_result_172728)
    # Getting the type of the for loop variable (line 834)
    for_loop_var_172729 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 834, 4), range_call_result_172728)
    # Assigning a type to the variable 'i' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 4), 'i', for_loop_var_172729)
    # SSA begins for a for statement (line 834)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 835):
    
    # Assigning a Call to a Name (line 835):
    
    # Call to func(...): (line 835)
    # Processing the call arguments (line 835)
    # Getting the type of 'p0' (line 835)
    p0_172731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 18), 'p0', False)
    # Getting the type of 'args' (line 835)
    args_172732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 23), 'args', False)
    # Processing the call keyword arguments (line 835)
    kwargs_172733 = {}
    # Getting the type of 'func' (line 835)
    func_172730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 13), 'func', False)
    # Calling func(args, kwargs) (line 835)
    func_call_result_172734 = invoke(stypy.reporting.localization.Localization(__file__, 835, 13), func_172730, *[p0_172731, args_172732], **kwargs_172733)
    
    # Assigning a type to the variable 'p1' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'p1', func_call_result_172734)
    
    # Getting the type of 'use_accel' (line 836)
    use_accel_172735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 11), 'use_accel')
    # Testing the type of an if condition (line 836)
    if_condition_172736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 836, 8), use_accel_172735)
    # Assigning a type to the variable 'if_condition_172736' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'if_condition_172736', if_condition_172736)
    # SSA begins for if statement (line 836)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 837):
    
    # Assigning a Call to a Name (line 837):
    
    # Call to func(...): (line 837)
    # Processing the call arguments (line 837)
    # Getting the type of 'p1' (line 837)
    p1_172738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 22), 'p1', False)
    # Getting the type of 'args' (line 837)
    args_172739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 27), 'args', False)
    # Processing the call keyword arguments (line 837)
    kwargs_172740 = {}
    # Getting the type of 'func' (line 837)
    func_172737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 17), 'func', False)
    # Calling func(args, kwargs) (line 837)
    func_call_result_172741 = invoke(stypy.reporting.localization.Localization(__file__, 837, 17), func_172737, *[p1_172738, args_172739], **kwargs_172740)
    
    # Assigning a type to the variable 'p2' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'p2', func_call_result_172741)
    
    # Assigning a BinOp to a Name (line 838):
    
    # Assigning a BinOp to a Name (line 838):
    # Getting the type of 'p2' (line 838)
    p2_172742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 16), 'p2')
    float_172743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 21), 'float')
    # Getting the type of 'p1' (line 838)
    p1_172744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 27), 'p1')
    # Applying the binary operator '*' (line 838)
    result_mul_172745 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 21), '*', float_172743, p1_172744)
    
    # Applying the binary operator '-' (line 838)
    result_sub_172746 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 16), '-', p2_172742, result_mul_172745)
    
    # Getting the type of 'p0' (line 838)
    p0_172747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 32), 'p0')
    # Applying the binary operator '+' (line 838)
    result_add_172748 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 30), '+', result_sub_172746, p0_172747)
    
    # Assigning a type to the variable 'd' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 12), 'd', result_add_172748)
    
    # Assigning a Call to a Name (line 839):
    
    # Assigning a Call to a Name (line 839):
    
    # Call to _lazywhere(...): (line 839)
    # Processing the call arguments (line 839)
    
    # Getting the type of 'd' (line 839)
    d_172750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 27), 'd', False)
    int_172751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 32), 'int')
    # Applying the binary operator '!=' (line 839)
    result_ne_172752 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 27), '!=', d_172750, int_172751)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 839)
    tuple_172753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 839)
    # Adding element type (line 839)
    # Getting the type of 'p0' (line 839)
    p0_172754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 36), 'p0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 36), tuple_172753, p0_172754)
    # Adding element type (line 839)
    # Getting the type of 'p1' (line 839)
    p1_172755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 40), 'p1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 36), tuple_172753, p1_172755)
    # Adding element type (line 839)
    # Getting the type of 'd' (line 839)
    d_172756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 44), 'd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 36), tuple_172753, d_172756)
    
    # Processing the call keyword arguments (line 839)
    # Getting the type of '_del2' (line 839)
    _del2_172757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 50), '_del2', False)
    keyword_172758 = _del2_172757
    # Getting the type of 'p2' (line 839)
    p2_172759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 67), 'p2', False)
    keyword_172760 = p2_172759
    kwargs_172761 = {'fillvalue': keyword_172760, 'f': keyword_172758}
    # Getting the type of '_lazywhere' (line 839)
    _lazywhere_172749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 16), '_lazywhere', False)
    # Calling _lazywhere(args, kwargs) (line 839)
    _lazywhere_call_result_172762 = invoke(stypy.reporting.localization.Localization(__file__, 839, 16), _lazywhere_172749, *[result_ne_172752, tuple_172753], **kwargs_172761)
    
    # Assigning a type to the variable 'p' (line 839)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'p', _lazywhere_call_result_172762)
    # SSA branch for the else part of an if statement (line 836)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 841):
    
    # Assigning a Name to a Name (line 841):
    # Getting the type of 'p1' (line 841)
    p1_172763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 16), 'p1')
    # Assigning a type to the variable 'p' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 12), 'p', p1_172763)
    # SSA join for if statement (line 836)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 842):
    
    # Assigning a Call to a Name (line 842):
    
    # Call to _lazywhere(...): (line 842)
    # Processing the call arguments (line 842)
    
    # Getting the type of 'p0' (line 842)
    p0_172765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 28), 'p0', False)
    int_172766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 34), 'int')
    # Applying the binary operator '!=' (line 842)
    result_ne_172767 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 28), '!=', p0_172765, int_172766)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 842)
    tuple_172768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 842)
    # Adding element type (line 842)
    # Getting the type of 'p' (line 842)
    p_172769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 38), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 38), tuple_172768, p_172769)
    # Adding element type (line 842)
    # Getting the type of 'p0' (line 842)
    p0_172770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 41), 'p0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 38), tuple_172768, p0_172770)
    
    # Processing the call keyword arguments (line 842)
    # Getting the type of '_relerr' (line 842)
    _relerr_172771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 48), '_relerr', False)
    keyword_172772 = _relerr_172771
    # Getting the type of 'p' (line 842)
    p_172773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 67), 'p', False)
    keyword_172774 = p_172773
    kwargs_172775 = {'fillvalue': keyword_172774, 'f': keyword_172772}
    # Getting the type of '_lazywhere' (line 842)
    _lazywhere_172764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 17), '_lazywhere', False)
    # Calling _lazywhere(args, kwargs) (line 842)
    _lazywhere_call_result_172776 = invoke(stypy.reporting.localization.Localization(__file__, 842, 17), _lazywhere_172764, *[result_ne_172767, tuple_172768], **kwargs_172775)
    
    # Assigning a type to the variable 'relerr' (line 842)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'relerr', _lazywhere_call_result_172776)
    
    
    # Call to all(...): (line 843)
    # Processing the call arguments (line 843)
    
    
    # Call to abs(...): (line 843)
    # Processing the call arguments (line 843)
    # Getting the type of 'relerr' (line 843)
    relerr_172781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 25), 'relerr', False)
    # Processing the call keyword arguments (line 843)
    kwargs_172782 = {}
    # Getting the type of 'np' (line 843)
    np_172779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 843)
    abs_172780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 18), np_172779, 'abs')
    # Calling abs(args, kwargs) (line 843)
    abs_call_result_172783 = invoke(stypy.reporting.localization.Localization(__file__, 843, 18), abs_172780, *[relerr_172781], **kwargs_172782)
    
    # Getting the type of 'xtol' (line 843)
    xtol_172784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 35), 'xtol', False)
    # Applying the binary operator '<' (line 843)
    result_lt_172785 = python_operator(stypy.reporting.localization.Localization(__file__, 843, 18), '<', abs_call_result_172783, xtol_172784)
    
    # Processing the call keyword arguments (line 843)
    kwargs_172786 = {}
    # Getting the type of 'np' (line 843)
    np_172777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 11), 'np', False)
    # Obtaining the member 'all' of a type (line 843)
    all_172778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 11), np_172777, 'all')
    # Calling all(args, kwargs) (line 843)
    all_call_result_172787 = invoke(stypy.reporting.localization.Localization(__file__, 843, 11), all_172778, *[result_lt_172785], **kwargs_172786)
    
    # Testing the type of an if condition (line 843)
    if_condition_172788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 843, 8), all_call_result_172787)
    # Assigning a type to the variable 'if_condition_172788' (line 843)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'if_condition_172788', if_condition_172788)
    # SSA begins for if statement (line 843)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'p' (line 844)
    p_172789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 19), 'p')
    # Assigning a type to the variable 'stypy_return_type' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'stypy_return_type', p_172789)
    # SSA join for if statement (line 843)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 845):
    
    # Assigning a Name to a Name (line 845):
    # Getting the type of 'p' (line 845)
    p_172790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 13), 'p')
    # Assigning a type to the variable 'p0' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 8), 'p0', p_172790)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 846):
    
    # Assigning a BinOp to a Name (line 846):
    str_172791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 10), 'str', 'Failed to converge after %d iterations, value is %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 846)
    tuple_172792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 846)
    # Adding element type (line 846)
    # Getting the type of 'maxiter' (line 846)
    maxiter_172793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 67), 'maxiter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 67), tuple_172792, maxiter_172793)
    # Adding element type (line 846)
    # Getting the type of 'p' (line 846)
    p_172794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 76), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 67), tuple_172792, p_172794)
    
    # Applying the binary operator '%' (line 846)
    result_mod_172795 = python_operator(stypy.reporting.localization.Localization(__file__, 846, 10), '%', str_172791, tuple_172792)
    
    # Assigning a type to the variable 'msg' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 4), 'msg', result_mod_172795)
    
    # Call to RuntimeError(...): (line 847)
    # Processing the call arguments (line 847)
    # Getting the type of 'msg' (line 847)
    msg_172797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 23), 'msg', False)
    # Processing the call keyword arguments (line 847)
    kwargs_172798 = {}
    # Getting the type of 'RuntimeError' (line 847)
    RuntimeError_172796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 10), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 847)
    RuntimeError_call_result_172799 = invoke(stypy.reporting.localization.Localization(__file__, 847, 10), RuntimeError_172796, *[msg_172797], **kwargs_172798)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 847, 4), RuntimeError_call_result_172799, 'raise parameter', BaseException)
    
    # ################# End of '_fixed_point_helper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fixed_point_helper' in the type store
    # Getting the type of 'stypy_return_type' (line 832)
    stypy_return_type_172800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172800)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fixed_point_helper'
    return stypy_return_type_172800

# Assigning a type to the variable '_fixed_point_helper' (line 832)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 0), '_fixed_point_helper', _fixed_point_helper)

@norecursion
def fixed_point(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 850)
    tuple_172801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 850)
    
    float_172802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 40), 'float')
    int_172803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 54), 'int')
    str_172804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 66), 'str', 'del2')
    defaults = [tuple_172801, float_172802, int_172803, str_172804]
    # Create a new context for function 'fixed_point'
    module_type_store = module_type_store.open_function_context('fixed_point', 850, 0, False)
    
    # Passed parameters checking function
    fixed_point.stypy_localization = localization
    fixed_point.stypy_type_of_self = None
    fixed_point.stypy_type_store = module_type_store
    fixed_point.stypy_function_name = 'fixed_point'
    fixed_point.stypy_param_names_list = ['func', 'x0', 'args', 'xtol', 'maxiter', 'method']
    fixed_point.stypy_varargs_param_name = None
    fixed_point.stypy_kwargs_param_name = None
    fixed_point.stypy_call_defaults = defaults
    fixed_point.stypy_call_varargs = varargs
    fixed_point.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fixed_point', ['func', 'x0', 'args', 'xtol', 'maxiter', 'method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fixed_point', localization, ['func', 'x0', 'args', 'xtol', 'maxiter', 'method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fixed_point(...)' code ##################

    str_172805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, (-1)), 'str', '\n    Find a fixed point of the function.\n\n    Given a function of one or more variables and a starting point, find a\n    fixed-point of the function: i.e. where ``func(x0) == x0``.\n\n    Parameters\n    ----------\n    func : function\n        Function to evaluate.\n    x0 : array_like\n        Fixed point of function.\n    args : tuple, optional\n        Extra arguments to `func`.\n    xtol : float, optional\n        Convergence tolerance, defaults to 1e-08.\n    maxiter : int, optional\n        Maximum number of iterations, defaults to 500.\n    method : {"del2", "iteration"}, optional\n        Method of finding the fixed-point, defaults to "del2"\n        which uses Steffensen\'s Method with Aitken\'s ``Del^2``\n        convergence acceleration [1]_. The "iteration" method simply iterates\n        the function until convergence is detected, without attempting to\n        accelerate the convergence.\n\n    References\n    ----------\n    .. [1] Burden, Faires, "Numerical Analysis", 5th edition, pg. 80\n\n    Examples\n    --------\n    >>> from scipy import optimize\n    >>> def func(x, c1, c2):\n    ...    return np.sqrt(c1/(x+c2))\n    >>> c1 = np.array([10,12.])\n    >>> c2 = np.array([3, 5.])\n    >>> optimize.fixed_point(func, [1.2, 1.3], args=(c1,c2))\n    array([ 1.4920333 ,  1.37228132])\n\n    ')
    
    # Assigning a Subscript to a Name (line 891):
    
    # Assigning a Subscript to a Name (line 891):
    
    # Obtaining the type of the subscript
    # Getting the type of 'method' (line 891)
    method_172806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 51), 'method')
    
    # Obtaining an instance of the builtin type 'dict' (line 891)
    dict_172807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 891)
    # Adding element type (key, value) (line 891)
    str_172808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 17), 'str', 'del2')
    # Getting the type of 'True' (line 891)
    True_172809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 25), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 891, 16), dict_172807, (str_172808, True_172809))
    # Adding element type (key, value) (line 891)
    str_172810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 31), 'str', 'iteration')
    # Getting the type of 'False' (line 891)
    False_172811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 44), 'False')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 891, 16), dict_172807, (str_172810, False_172811))
    
    # Obtaining the member '__getitem__' of a type (line 891)
    getitem___172812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 16), dict_172807, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 891)
    subscript_call_result_172813 = invoke(stypy.reporting.localization.Localization(__file__, 891, 16), getitem___172812, method_172806)
    
    # Assigning a type to the variable 'use_accel' (line 891)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 4), 'use_accel', subscript_call_result_172813)
    
    # Assigning a Call to a Name (line 892):
    
    # Assigning a Call to a Name (line 892):
    
    # Call to _asarray_validated(...): (line 892)
    # Processing the call arguments (line 892)
    # Getting the type of 'x0' (line 892)
    x0_172815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 28), 'x0', False)
    # Processing the call keyword arguments (line 892)
    # Getting the type of 'True' (line 892)
    True_172816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 43), 'True', False)
    keyword_172817 = True_172816
    kwargs_172818 = {'as_inexact': keyword_172817}
    # Getting the type of '_asarray_validated' (line 892)
    _asarray_validated_172814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 892)
    _asarray_validated_call_result_172819 = invoke(stypy.reporting.localization.Localization(__file__, 892, 9), _asarray_validated_172814, *[x0_172815], **kwargs_172818)
    
    # Assigning a type to the variable 'x0' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 4), 'x0', _asarray_validated_call_result_172819)
    
    # Call to _fixed_point_helper(...): (line 893)
    # Processing the call arguments (line 893)
    # Getting the type of 'func' (line 893)
    func_172821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 31), 'func', False)
    # Getting the type of 'x0' (line 893)
    x0_172822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 37), 'x0', False)
    # Getting the type of 'args' (line 893)
    args_172823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 41), 'args', False)
    # Getting the type of 'xtol' (line 893)
    xtol_172824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 47), 'xtol', False)
    # Getting the type of 'maxiter' (line 893)
    maxiter_172825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 53), 'maxiter', False)
    # Getting the type of 'use_accel' (line 893)
    use_accel_172826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 62), 'use_accel', False)
    # Processing the call keyword arguments (line 893)
    kwargs_172827 = {}
    # Getting the type of '_fixed_point_helper' (line 893)
    _fixed_point_helper_172820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 11), '_fixed_point_helper', False)
    # Calling _fixed_point_helper(args, kwargs) (line 893)
    _fixed_point_helper_call_result_172828 = invoke(stypy.reporting.localization.Localization(__file__, 893, 11), _fixed_point_helper_172820, *[func_172821, x0_172822, args_172823, xtol_172824, maxiter_172825, use_accel_172826], **kwargs_172827)
    
    # Assigning a type to the variable 'stypy_return_type' (line 893)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 4), 'stypy_return_type', _fixed_point_helper_call_result_172828)
    
    # ################# End of 'fixed_point(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fixed_point' in the type store
    # Getting the type of 'stypy_return_type' (line 850)
    stypy_return_type_172829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172829)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fixed_point'
    return stypy_return_type_172829

# Assigning a type to the variable 'fixed_point' (line 850)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 0), 'fixed_point', fixed_point)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
