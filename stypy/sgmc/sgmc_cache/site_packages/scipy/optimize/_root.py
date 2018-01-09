
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unified interfaces to root finding algorithms.
3: 
4: Functions
5: ---------
6: - root : find a root of a vector function.
7: '''
8: from __future__ import division, print_function, absolute_import
9: 
10: __all__ = ['root']
11: 
12: import numpy as np
13: 
14: from scipy._lib.six import callable
15: 
16: from warnings import warn
17: 
18: from .optimize import MemoizeJac, OptimizeResult, _check_unknown_options
19: from .minpack import _root_hybr, leastsq
20: from ._spectral import _root_df_sane
21: from . import nonlin
22: 
23: 
24: def root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None,
25:          options=None):
26:     '''
27:     Find a root of a vector function.
28: 
29:     Parameters
30:     ----------
31:     fun : callable
32:         A vector function to find a root of.
33:     x0 : ndarray
34:         Initial guess.
35:     args : tuple, optional
36:         Extra arguments passed to the objective function and its Jacobian.
37:     method : str, optional
38:         Type of solver.  Should be one of
39: 
40:             - 'hybr'             :ref:`(see here) <optimize.root-hybr>`
41:             - 'lm'               :ref:`(see here) <optimize.root-lm>`
42:             - 'broyden1'         :ref:`(see here) <optimize.root-broyden1>`
43:             - 'broyden2'         :ref:`(see here) <optimize.root-broyden2>`
44:             - 'anderson'         :ref:`(see here) <optimize.root-anderson>`
45:             - 'linearmixing'     :ref:`(see here) <optimize.root-linearmixing>`
46:             - 'diagbroyden'      :ref:`(see here) <optimize.root-diagbroyden>`
47:             - 'excitingmixing'   :ref:`(see here) <optimize.root-excitingmixing>`
48:             - 'krylov'           :ref:`(see here) <optimize.root-krylov>`
49:             - 'df-sane'          :ref:`(see here) <optimize.root-dfsane>`
50: 
51:     jac : bool or callable, optional
52:         If `jac` is a Boolean and is True, `fun` is assumed to return the
53:         value of Jacobian along with the objective function. If False, the
54:         Jacobian will be estimated numerically.
55:         `jac` can also be a callable returning the Jacobian of `fun`. In
56:         this case, it must accept the same arguments as `fun`.
57:     tol : float, optional
58:         Tolerance for termination. For detailed control, use solver-specific
59:         options.
60:     callback : function, optional
61:         Optional callback function. It is called on every iteration as
62:         ``callback(x, f)`` where `x` is the current solution and `f`
63:         the corresponding residual. For all methods but 'hybr' and 'lm'.
64:     options : dict, optional
65:         A dictionary of solver options. E.g. `xtol` or `maxiter`, see
66:         :obj:`show_options()` for details.
67: 
68:     Returns
69:     -------
70:     sol : OptimizeResult
71:         The solution represented as a ``OptimizeResult`` object.
72:         Important attributes are: ``x`` the solution array, ``success`` a
73:         Boolean flag indicating if the algorithm exited successfully and
74:         ``message`` which describes the cause of the termination. See
75:         `OptimizeResult` for a description of other attributes.
76: 
77:     See also
78:     --------
79:     show_options : Additional options accepted by the solvers
80: 
81:     Notes
82:     -----
83:     This section describes the available solvers that can be selected by the
84:     'method' parameter. The default method is *hybr*.
85: 
86:     Method *hybr* uses a modification of the Powell hybrid method as
87:     implemented in MINPACK [1]_.
88: 
89:     Method *lm* solves the system of nonlinear equations in a least squares
90:     sense using a modification of the Levenberg-Marquardt algorithm as
91:     implemented in MINPACK [1]_.
92: 
93:     Method *df-sane* is a derivative-free spectral method. [3]_
94: 
95:     Methods *broyden1*, *broyden2*, *anderson*, *linearmixing*,
96:     *diagbroyden*, *excitingmixing*, *krylov* are inexact Newton methods,
97:     with backtracking or full line searches [2]_. Each method corresponds
98:     to a particular Jacobian approximations. See `nonlin` for details.
99: 
100:     - Method *broyden1* uses Broyden's first Jacobian approximation, it is
101:       known as Broyden's good method.
102:     - Method *broyden2* uses Broyden's second Jacobian approximation, it
103:       is known as Broyden's bad method.
104:     - Method *anderson* uses (extended) Anderson mixing.
105:     - Method *Krylov* uses Krylov approximation for inverse Jacobian. It
106:       is suitable for large-scale problem.
107:     - Method *diagbroyden* uses diagonal Broyden Jacobian approximation.
108:     - Method *linearmixing* uses a scalar Jacobian approximation.
109:     - Method *excitingmixing* uses a tuned diagonal Jacobian
110:       approximation.
111: 
112:     .. warning::
113: 
114:         The algorithms implemented for methods *diagbroyden*,
115:         *linearmixing* and *excitingmixing* may be useful for specific
116:         problems, but whether they will work may depend strongly on the
117:         problem.
118: 
119:     .. versionadded:: 0.11.0
120: 
121:     References
122:     ----------
123:     .. [1] More, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.
124:        1980. User Guide for MINPACK-1.
125:     .. [2] C. T. Kelley. 1995. Iterative Methods for Linear and Nonlinear
126:         Equations. Society for Industrial and Applied Mathematics.
127:         <http://www.siam.org/books/kelley/fr16/index.php>
128:     .. [3] W. La Cruz, J.M. Martinez, M. Raydan. Math. Comp. 75, 1429 (2006).
129: 
130:     Examples
131:     --------
132:     The following functions define a system of nonlinear equations and its
133:     jacobian.
134: 
135:     >>> def fun(x):
136:     ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
137:     ...             0.5 * (x[1] - x[0])**3 + x[1]]
138: 
139:     >>> def jac(x):
140:     ...     return np.array([[1 + 1.5 * (x[0] - x[1])**2,
141:     ...                       -1.5 * (x[0] - x[1])**2],
142:     ...                      [-1.5 * (x[1] - x[0])**2,
143:     ...                       1 + 1.5 * (x[1] - x[0])**2]])
144: 
145:     A solution can be obtained as follows.
146: 
147:     >>> from scipy import optimize
148:     >>> sol = optimize.root(fun, [0, 0], jac=jac, method='hybr')
149:     >>> sol.x
150:     array([ 0.8411639,  0.1588361])
151:     '''
152:     if not isinstance(args, tuple):
153:         args = (args,)
154: 
155:     meth = method.lower()
156:     if options is None:
157:         options = {}
158: 
159:     if callback is not None and meth in ('hybr', 'lm'):
160:         warn('Method %s does not accept callback.' % method,
161:              RuntimeWarning)
162: 
163:     # fun also returns the jacobian
164:     if not callable(jac) and meth in ('hybr', 'lm'):
165:         if bool(jac):
166:             fun = MemoizeJac(fun)
167:             jac = fun.derivative
168:         else:
169:             jac = None
170: 
171:     # set default tolerances
172:     if tol is not None:
173:         options = dict(options)
174:         if meth in ('hybr', 'lm'):
175:             options.setdefault('xtol', tol)
176:         elif meth in ('df-sane',):
177:             options.setdefault('ftol', tol)
178:         elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
179:                       'diagbroyden', 'excitingmixing', 'krylov'):
180:             options.setdefault('xtol', tol)
181:             options.setdefault('xatol', np.inf)
182:             options.setdefault('ftol', np.inf)
183:             options.setdefault('fatol', np.inf)
184: 
185:     if meth == 'hybr':
186:         sol = _root_hybr(fun, x0, args=args, jac=jac, **options)
187:     elif meth == 'lm':
188:         sol = _root_leastsq(fun, x0, args=args, jac=jac, **options)
189:     elif meth == 'df-sane':
190:         _warn_jac_unused(jac, method)
191:         sol = _root_df_sane(fun, x0, args=args, callback=callback,
192:                             **options)
193:     elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
194:                   'diagbroyden', 'excitingmixing', 'krylov'):
195:         _warn_jac_unused(jac, method)
196:         sol = _root_nonlin_solve(fun, x0, args=args, jac=jac,
197:                                  _method=meth, _callback=callback,
198:                                  **options)
199:     else:
200:         raise ValueError('Unknown solver %s' % method)
201: 
202:     return sol
203: 
204: 
205: def _warn_jac_unused(jac, method):
206:     if jac is not None:
207:         warn('Method %s does not use the jacobian (jac).' % (method,),
208:              RuntimeWarning)
209: 
210: 
211: def _root_leastsq(func, x0, args=(), jac=None,
212:                   col_deriv=0, xtol=1.49012e-08, ftol=1.49012e-08,
213:                   gtol=0.0, maxiter=0, eps=0.0, factor=100, diag=None,
214:                   **unknown_options):
215:     '''
216:     Solve for least squares with Levenberg-Marquardt
217: 
218:     Options
219:     -------
220:     col_deriv : bool
221:         non-zero to specify that the Jacobian function computes derivatives
222:         down the columns (faster, because there is no transpose operation).
223:     ftol : float
224:         Relative error desired in the sum of squares.
225:     xtol : float
226:         Relative error desired in the approximate solution.
227:     gtol : float
228:         Orthogonality desired between the function vector and the columns
229:         of the Jacobian.
230:     maxiter : int
231:         The maximum number of calls to the function. If zero, then
232:         100*(N+1) is the maximum where N is the number of elements in x0.
233:     epsfcn : float
234:         A suitable step length for the forward-difference approximation of
235:         the Jacobian (for Dfun=None). If epsfcn is less than the machine
236:         precision, it is assumed that the relative errors in the functions
237:         are of the order of the machine precision.
238:     factor : float
239:         A parameter determining the initial step bound
240:         (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
241:     diag : sequence
242:         N positive entries that serve as a scale factors for the variables.
243:     '''
244: 
245:     _check_unknown_options(unknown_options)
246:     x, cov_x, info, msg, ier = leastsq(func, x0, args=args, Dfun=jac,
247:                                        full_output=True,
248:                                        col_deriv=col_deriv, xtol=xtol,
249:                                        ftol=ftol, gtol=gtol,
250:                                        maxfev=maxiter, epsfcn=eps,
251:                                        factor=factor, diag=diag)
252:     sol = OptimizeResult(x=x, message=msg, status=ier,
253:                          success=ier in (1, 2, 3, 4), cov_x=cov_x,
254:                          fun=info.pop('fvec'))
255:     sol.update(info)
256:     return sol
257: 
258: 
259: def _root_nonlin_solve(func, x0, args=(), jac=None,
260:                        _callback=None, _method=None,
261:                        nit=None, disp=False, maxiter=None,
262:                        ftol=None, fatol=None, xtol=None, xatol=None,
263:                        tol_norm=None, line_search='armijo', jac_options=None,
264:                        **unknown_options):
265:     _check_unknown_options(unknown_options)
266: 
267:     f_tol = fatol
268:     f_rtol = ftol
269:     x_tol = xatol
270:     x_rtol = xtol
271:     verbose = disp
272:     if jac_options is None:
273:         jac_options = dict()
274: 
275:     jacobian = {'broyden1': nonlin.BroydenFirst,
276:                 'broyden2': nonlin.BroydenSecond,
277:                 'anderson': nonlin.Anderson,
278:                 'linearmixing': nonlin.LinearMixing,
279:                 'diagbroyden': nonlin.DiagBroyden,
280:                 'excitingmixing': nonlin.ExcitingMixing,
281:                 'krylov': nonlin.KrylovJacobian
282:                 }[_method]
283: 
284:     if args:
285:         if jac:
286:             def f(x):
287:                 return func(x, *args)[0]
288:         else:
289:             def f(x):
290:                 return func(x, *args)
291:     else:
292:         f = func
293: 
294:     x, info = nonlin.nonlin_solve(f, x0, jacobian=jacobian(**jac_options),
295:                                   iter=nit, verbose=verbose,
296:                                   maxiter=maxiter, f_tol=f_tol,
297:                                   f_rtol=f_rtol, x_tol=x_tol,
298:                                   x_rtol=x_rtol, tol_norm=tol_norm,
299:                                   line_search=line_search,
300:                                   callback=_callback, full_output=True,
301:                                   raise_exception=False)
302:     sol = OptimizeResult(x=x)
303:     sol.update(info)
304:     return sol
305: 
306: def _root_broyden1_doc():
307:     '''
308:     Options
309:     -------
310:     nit : int, optional
311:         Number of iterations to make. If omitted (default), make as many
312:         as required to meet tolerances.
313:     disp : bool, optional
314:         Print status to stdout on every iteration.
315:     maxiter : int, optional
316:         Maximum number of iterations to make. If more are needed to
317:         meet convergence, `NoConvergence` is raised.
318:     ftol : float, optional
319:         Relative tolerance for the residual. If omitted, not used.
320:     fatol : float, optional
321:         Absolute tolerance (in max-norm) for the residual.
322:         If omitted, default is 6e-6.
323:     xtol : float, optional
324:         Relative minimum step size. If omitted, not used.
325:     xatol : float, optional
326:         Absolute minimum step size, as determined from the Jacobian
327:         approximation. If the step size is smaller than this, optimization
328:         is terminated as successful. If omitted, not used.
329:     tol_norm : function(vector) -> scalar, optional
330:         Norm to use in convergence check. Default is the maximum norm.
331:     line_search : {None, 'armijo' (default), 'wolfe'}, optional
332:         Which type of a line search to use to determine the step size in
333:         the direction given by the Jacobian approximation. Defaults to
334:         'armijo'.
335:     jac_options : dict, optional
336:         Options for the respective Jacobian approximation.
337:             alpha : float, optional
338:                 Initial guess for the Jacobian is (-1/alpha).
339:             reduction_method : str or tuple, optional
340:                 Method used in ensuring that the rank of the Broyden
341:                 matrix stays low. Can either be a string giving the
342:                 name of the method, or a tuple of the form ``(method,
343:                 param1, param2, ...)`` that gives the name of the
344:                 method and values for additional parameters.
345: 
346:                 Methods available:
347:                     - ``restart``: drop all matrix columns. Has no
348:                         extra parameters.
349:                     - ``simple``: drop oldest matrix column. Has no
350:                         extra parameters.
351:                     - ``svd``: keep only the most significant SVD
352:                         components.
353:                       Extra parameters:
354:                           - ``to_retain``: number of SVD components to
355:                               retain when rank reduction is done.
356:                               Default is ``max_rank - 2``.
357:             max_rank : int, optional
358:                 Maximum rank for the Broyden matrix.
359:                 Default is infinity (ie., no rank reduction).
360:     '''
361:     pass
362: 
363: def _root_broyden2_doc():
364:     '''
365:     Options
366:     -------
367:     nit : int, optional
368:         Number of iterations to make. If omitted (default), make as many
369:         as required to meet tolerances.
370:     disp : bool, optional
371:         Print status to stdout on every iteration.
372:     maxiter : int, optional
373:         Maximum number of iterations to make. If more are needed to
374:         meet convergence, `NoConvergence` is raised.
375:     ftol : float, optional
376:         Relative tolerance for the residual. If omitted, not used.
377:     fatol : float, optional
378:         Absolute tolerance (in max-norm) for the residual.
379:         If omitted, default is 6e-6.
380:     xtol : float, optional
381:         Relative minimum step size. If omitted, not used.
382:     xatol : float, optional
383:         Absolute minimum step size, as determined from the Jacobian
384:         approximation. If the step size is smaller than this, optimization
385:         is terminated as successful. If omitted, not used.
386:     tol_norm : function(vector) -> scalar, optional
387:         Norm to use in convergence check. Default is the maximum norm.
388:     line_search : {None, 'armijo' (default), 'wolfe'}, optional
389:         Which type of a line search to use to determine the step size in
390:         the direction given by the Jacobian approximation. Defaults to
391:         'armijo'.
392:     jac_options : dict, optional
393:         Options for the respective Jacobian approximation.
394: 
395:         alpha : float, optional
396:             Initial guess for the Jacobian is (-1/alpha).
397:         reduction_method : str or tuple, optional
398:             Method used in ensuring that the rank of the Broyden
399:             matrix stays low. Can either be a string giving the
400:             name of the method, or a tuple of the form ``(method,
401:             param1, param2, ...)`` that gives the name of the
402:             method and values for additional parameters.
403: 
404:             Methods available:
405:                 - ``restart``: drop all matrix columns. Has no
406:                     extra parameters.
407:                 - ``simple``: drop oldest matrix column. Has no
408:                     extra parameters.
409:                 - ``svd``: keep only the most significant SVD
410:                     components.
411:                   Extra parameters:
412:                       - ``to_retain``: number of SVD components to
413:                           retain when rank reduction is done.
414:                           Default is ``max_rank - 2``.
415:         max_rank : int, optional
416:             Maximum rank for the Broyden matrix.
417:             Default is infinity (ie., no rank reduction).
418:     '''
419:     pass
420: 
421: def _root_anderson_doc():
422:     '''
423:     Options
424:     -------
425:     nit : int, optional
426:         Number of iterations to make. If omitted (default), make as many
427:         as required to meet tolerances.
428:     disp : bool, optional
429:         Print status to stdout on every iteration.
430:     maxiter : int, optional
431:         Maximum number of iterations to make. If more are needed to
432:         meet convergence, `NoConvergence` is raised.
433:     ftol : float, optional
434:         Relative tolerance for the residual. If omitted, not used.
435:     fatol : float, optional
436:         Absolute tolerance (in max-norm) for the residual.
437:         If omitted, default is 6e-6.
438:     xtol : float, optional
439:         Relative minimum step size. If omitted, not used.
440:     xatol : float, optional
441:         Absolute minimum step size, as determined from the Jacobian
442:         approximation. If the step size is smaller than this, optimization
443:         is terminated as successful. If omitted, not used.
444:     tol_norm : function(vector) -> scalar, optional
445:         Norm to use in convergence check. Default is the maximum norm.
446:     line_search : {None, 'armijo' (default), 'wolfe'}, optional
447:         Which type of a line search to use to determine the step size in
448:         the direction given by the Jacobian approximation. Defaults to
449:         'armijo'.
450:     jac_options : dict, optional
451:         Options for the respective Jacobian approximation.
452: 
453:         alpha : float, optional
454:             Initial guess for the Jacobian is (-1/alpha).
455:         M : float, optional
456:             Number of previous vectors to retain. Defaults to 5.
457:         w0 : float, optional
458:             Regularization parameter for numerical stability.
459:             Compared to unity, good values of the order of 0.01.
460:     '''
461:     pass
462: 
463: def _root_linearmixing_doc():
464:     '''
465:     Options
466:     -------
467:     nit : int, optional
468:         Number of iterations to make. If omitted (default), make as many
469:         as required to meet tolerances.
470:     disp : bool, optional
471:         Print status to stdout on every iteration.
472:     maxiter : int, optional
473:         Maximum number of iterations to make. If more are needed to
474:         meet convergence, ``NoConvergence`` is raised.
475:     ftol : float, optional
476:         Relative tolerance for the residual. If omitted, not used.
477:     fatol : float, optional
478:         Absolute tolerance (in max-norm) for the residual.
479:         If omitted, default is 6e-6.
480:     xtol : float, optional
481:         Relative minimum step size. If omitted, not used.
482:     xatol : float, optional
483:         Absolute minimum step size, as determined from the Jacobian
484:         approximation. If the step size is smaller than this, optimization
485:         is terminated as successful. If omitted, not used.
486:     tol_norm : function(vector) -> scalar, optional
487:         Norm to use in convergence check. Default is the maximum norm.
488:     line_search : {None, 'armijo' (default), 'wolfe'}, optional
489:         Which type of a line search to use to determine the step size in
490:         the direction given by the Jacobian approximation. Defaults to
491:         'armijo'.
492:     jac_options : dict, optional
493:         Options for the respective Jacobian approximation.
494: 
495:         alpha : float, optional
496:             initial guess for the jacobian is (-1/alpha).
497:     '''
498:     pass
499: 
500: def _root_diagbroyden_doc():
501:     '''
502:     Options
503:     -------
504:     nit : int, optional
505:         Number of iterations to make. If omitted (default), make as many
506:         as required to meet tolerances.
507:     disp : bool, optional
508:         Print status to stdout on every iteration.
509:     maxiter : int, optional
510:         Maximum number of iterations to make. If more are needed to
511:         meet convergence, `NoConvergence` is raised.
512:     ftol : float, optional
513:         Relative tolerance for the residual. If omitted, not used.
514:     fatol : float, optional
515:         Absolute tolerance (in max-norm) for the residual.
516:         If omitted, default is 6e-6.
517:     xtol : float, optional
518:         Relative minimum step size. If omitted, not used.
519:     xatol : float, optional
520:         Absolute minimum step size, as determined from the Jacobian
521:         approximation. If the step size is smaller than this, optimization
522:         is terminated as successful. If omitted, not used.
523:     tol_norm : function(vector) -> scalar, optional
524:         Norm to use in convergence check. Default is the maximum norm.
525:     line_search : {None, 'armijo' (default), 'wolfe'}, optional
526:         Which type of a line search to use to determine the step size in
527:         the direction given by the Jacobian approximation. Defaults to
528:         'armijo'.
529:     jac_options : dict, optional
530:         Options for the respective Jacobian approximation.
531: 
532:         alpha : float, optional
533:             initial guess for the jacobian is (-1/alpha).
534:     '''
535:     pass
536: 
537: def _root_excitingmixing_doc():
538:     '''
539:     Options
540:     -------
541:     nit : int, optional
542:         Number of iterations to make. If omitted (default), make as many
543:         as required to meet tolerances.
544:     disp : bool, optional
545:         Print status to stdout on every iteration.
546:     maxiter : int, optional
547:         Maximum number of iterations to make. If more are needed to
548:         meet convergence, `NoConvergence` is raised.
549:     ftol : float, optional
550:         Relative tolerance for the residual. If omitted, not used.
551:     fatol : float, optional
552:         Absolute tolerance (in max-norm) for the residual.
553:         If omitted, default is 6e-6.
554:     xtol : float, optional
555:         Relative minimum step size. If omitted, not used.
556:     xatol : float, optional
557:         Absolute minimum step size, as determined from the Jacobian
558:         approximation. If the step size is smaller than this, optimization
559:         is terminated as successful. If omitted, not used.
560:     tol_norm : function(vector) -> scalar, optional
561:         Norm to use in convergence check. Default is the maximum norm.
562:     line_search : {None, 'armijo' (default), 'wolfe'}, optional
563:         Which type of a line search to use to determine the step size in
564:         the direction given by the Jacobian approximation. Defaults to
565:         'armijo'.
566:     jac_options : dict, optional
567:         Options for the respective Jacobian approximation.
568: 
569:         alpha : float, optional
570:             Initial Jacobian approximation is (-1/alpha).
571:         alphamax : float, optional
572:             The entries of the diagonal Jacobian are kept in the range
573:             ``[alpha, alphamax]``.
574:     '''
575:     pass
576: 
577: def _root_krylov_doc():
578:     '''
579:     Options
580:     -------
581:     nit : int, optional
582:         Number of iterations to make. If omitted (default), make as many
583:         as required to meet tolerances.
584:     disp : bool, optional
585:         Print status to stdout on every iteration.
586:     maxiter : int, optional
587:         Maximum number of iterations to make. If more are needed to
588:         meet convergence, `NoConvergence` is raised.
589:     ftol : float, optional
590:         Relative tolerance for the residual. If omitted, not used.
591:     fatol : float, optional
592:         Absolute tolerance (in max-norm) for the residual.
593:         If omitted, default is 6e-6.
594:     xtol : float, optional
595:         Relative minimum step size. If omitted, not used.
596:     xatol : float, optional
597:         Absolute minimum step size, as determined from the Jacobian
598:         approximation. If the step size is smaller than this, optimization
599:         is terminated as successful. If omitted, not used.
600:     tol_norm : function(vector) -> scalar, optional
601:         Norm to use in convergence check. Default is the maximum norm.
602:     line_search : {None, 'armijo' (default), 'wolfe'}, optional
603:         Which type of a line search to use to determine the step size in
604:         the direction given by the Jacobian approximation. Defaults to
605:         'armijo'.
606:     jac_options : dict, optional
607:         Options for the respective Jacobian approximation.
608: 
609:         rdiff : float, optional
610:             Relative step size to use in numerical differentiation.
611:         method : {'lgmres', 'gmres', 'bicgstab', 'cgs', 'minres'} or function
612:             Krylov method to use to approximate the Jacobian.
613:             Can be a string, or a function implementing the same
614:             interface as the iterative solvers in
615:             `scipy.sparse.linalg`.
616: 
617:             The default is `scipy.sparse.linalg.lgmres`.
618:         inner_M : LinearOperator or InverseJacobian
619:             Preconditioner for the inner Krylov iteration.
620:             Note that you can use also inverse Jacobians as (adaptive)
621:             preconditioners. For example,
622: 
623:             >>> jac = BroydenFirst()
624:             >>> kjac = KrylovJacobian(inner_M=jac.inverse).
625: 
626:             If the preconditioner has a method named 'update', it will
627:             be called as ``update(x, f)`` after each nonlinear step,
628:             with ``x`` giving the current point, and ``f`` the current
629:             function value.
630:         inner_tol, inner_maxiter, ...
631:             Parameters to pass on to the "inner" Krylov solver.
632:             See `scipy.sparse.linalg.gmres` for details.
633:         outer_k : int, optional
634:             Size of the subspace kept across LGMRES nonlinear
635:             iterations.
636: 
637:             See `scipy.sparse.linalg.lgmres` for details.
638:     '''
639:     pass
640: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_201145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nUnified interfaces to root finding algorithms.\n\nFunctions\n---------\n- root : find a root of a vector function.\n')

# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):
__all__ = ['root']
module_type_store.set_exportable_members(['root'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_201146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_201147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'root')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_201146, str_201147)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_201146)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import numpy' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201148 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy')

if (type(import_201148) is not StypyTypeError):

    if (import_201148 != 'pyd_module'):
        __import__(import_201148)
        sys_modules_201149 = sys.modules[import_201148]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', sys_modules_201149.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', import_201148)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy._lib.six import callable' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201150 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib.six')

if (type(import_201150) is not StypyTypeError):

    if (import_201150 != 'pyd_module'):
        __import__(import_201150)
        sys_modules_201151 = sys.modules[import_201150]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib.six', sys_modules_201151.module_type_store, module_type_store, ['callable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_201151, sys_modules_201151.module_type_store, module_type_store)
    else:
        from scipy._lib.six import callable

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib.six', None, module_type_store, ['callable'], [callable])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib.six', import_201150)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from warnings import warn' statement (line 16)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.optimize.optimize import MemoizeJac, OptimizeResult, _check_unknown_options' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201152 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize.optimize')

if (type(import_201152) is not StypyTypeError):

    if (import_201152 != 'pyd_module'):
        __import__(import_201152)
        sys_modules_201153 = sys.modules[import_201152]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize.optimize', sys_modules_201153.module_type_store, module_type_store, ['MemoizeJac', 'OptimizeResult', '_check_unknown_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_201153, sys_modules_201153.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import MemoizeJac, OptimizeResult, _check_unknown_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize.optimize', None, module_type_store, ['MemoizeJac', 'OptimizeResult', '_check_unknown_options'], [MemoizeJac, OptimizeResult, _check_unknown_options])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.optimize.optimize', import_201152)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.optimize.minpack import _root_hybr, leastsq' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201154 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.optimize.minpack')

if (type(import_201154) is not StypyTypeError):

    if (import_201154 != 'pyd_module'):
        __import__(import_201154)
        sys_modules_201155 = sys.modules[import_201154]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.optimize.minpack', sys_modules_201155.module_type_store, module_type_store, ['_root_hybr', 'leastsq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_201155, sys_modules_201155.module_type_store, module_type_store)
    else:
        from scipy.optimize.minpack import _root_hybr, leastsq

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.optimize.minpack', None, module_type_store, ['_root_hybr', 'leastsq'], [_root_hybr, leastsq])

else:
    # Assigning a type to the variable 'scipy.optimize.minpack' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.optimize.minpack', import_201154)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.optimize._spectral import _root_df_sane' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201156 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.optimize._spectral')

if (type(import_201156) is not StypyTypeError):

    if (import_201156 != 'pyd_module'):
        __import__(import_201156)
        sys_modules_201157 = sys.modules[import_201156]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.optimize._spectral', sys_modules_201157.module_type_store, module_type_store, ['_root_df_sane'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_201157, sys_modules_201157.module_type_store, module_type_store)
    else:
        from scipy.optimize._spectral import _root_df_sane

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.optimize._spectral', None, module_type_store, ['_root_df_sane'], [_root_df_sane])

else:
    # Assigning a type to the variable 'scipy.optimize._spectral' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.optimize._spectral', import_201156)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.optimize import nonlin' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_201158 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize')

if (type(import_201158) is not StypyTypeError):

    if (import_201158 != 'pyd_module'):
        __import__(import_201158)
        sys_modules_201159 = sys.modules[import_201158]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize', sys_modules_201159.module_type_store, module_type_store, ['nonlin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_201159, sys_modules_201159.module_type_store, module_type_store)
    else:
        from scipy.optimize import nonlin

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize', None, module_type_store, ['nonlin'], [nonlin])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize', import_201158)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


@norecursion
def root(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_201160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    
    str_201161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'str', 'hybr')
    # Getting the type of 'None' (line 24)
    None_201162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 46), 'None')
    # Getting the type of 'None' (line 24)
    None_201163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 56), 'None')
    # Getting the type of 'None' (line 24)
    None_201164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 71), 'None')
    # Getting the type of 'None' (line 25)
    None_201165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'None')
    defaults = [tuple_201160, str_201161, None_201162, None_201163, None_201164, None_201165]
    # Create a new context for function 'root'
    module_type_store = module_type_store.open_function_context('root', 24, 0, False)
    
    # Passed parameters checking function
    root.stypy_localization = localization
    root.stypy_type_of_self = None
    root.stypy_type_store = module_type_store
    root.stypy_function_name = 'root'
    root.stypy_param_names_list = ['fun', 'x0', 'args', 'method', 'jac', 'tol', 'callback', 'options']
    root.stypy_varargs_param_name = None
    root.stypy_kwargs_param_name = None
    root.stypy_call_defaults = defaults
    root.stypy_call_varargs = varargs
    root.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'root', ['fun', 'x0', 'args', 'method', 'jac', 'tol', 'callback', 'options'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'root', localization, ['fun', 'x0', 'args', 'method', 'jac', 'tol', 'callback', 'options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'root(...)' code ##################

    str_201166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'str', "\n    Find a root of a vector function.\n\n    Parameters\n    ----------\n    fun : callable\n        A vector function to find a root of.\n    x0 : ndarray\n        Initial guess.\n    args : tuple, optional\n        Extra arguments passed to the objective function and its Jacobian.\n    method : str, optional\n        Type of solver.  Should be one of\n\n            - 'hybr'             :ref:`(see here) <optimize.root-hybr>`\n            - 'lm'               :ref:`(see here) <optimize.root-lm>`\n            - 'broyden1'         :ref:`(see here) <optimize.root-broyden1>`\n            - 'broyden2'         :ref:`(see here) <optimize.root-broyden2>`\n            - 'anderson'         :ref:`(see here) <optimize.root-anderson>`\n            - 'linearmixing'     :ref:`(see here) <optimize.root-linearmixing>`\n            - 'diagbroyden'      :ref:`(see here) <optimize.root-diagbroyden>`\n            - 'excitingmixing'   :ref:`(see here) <optimize.root-excitingmixing>`\n            - 'krylov'           :ref:`(see here) <optimize.root-krylov>`\n            - 'df-sane'          :ref:`(see here) <optimize.root-dfsane>`\n\n    jac : bool or callable, optional\n        If `jac` is a Boolean and is True, `fun` is assumed to return the\n        value of Jacobian along with the objective function. If False, the\n        Jacobian will be estimated numerically.\n        `jac` can also be a callable returning the Jacobian of `fun`. In\n        this case, it must accept the same arguments as `fun`.\n    tol : float, optional\n        Tolerance for termination. For detailed control, use solver-specific\n        options.\n    callback : function, optional\n        Optional callback function. It is called on every iteration as\n        ``callback(x, f)`` where `x` is the current solution and `f`\n        the corresponding residual. For all methods but 'hybr' and 'lm'.\n    options : dict, optional\n        A dictionary of solver options. E.g. `xtol` or `maxiter`, see\n        :obj:`show_options()` for details.\n\n    Returns\n    -------\n    sol : OptimizeResult\n        The solution represented as a ``OptimizeResult`` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the algorithm exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.\n\n    See also\n    --------\n    show_options : Additional options accepted by the solvers\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    'method' parameter. The default method is *hybr*.\n\n    Method *hybr* uses a modification of the Powell hybrid method as\n    implemented in MINPACK [1]_.\n\n    Method *lm* solves the system of nonlinear equations in a least squares\n    sense using a modification of the Levenberg-Marquardt algorithm as\n    implemented in MINPACK [1]_.\n\n    Method *df-sane* is a derivative-free spectral method. [3]_\n\n    Methods *broyden1*, *broyden2*, *anderson*, *linearmixing*,\n    *diagbroyden*, *excitingmixing*, *krylov* are inexact Newton methods,\n    with backtracking or full line searches [2]_. Each method corresponds\n    to a particular Jacobian approximations. See `nonlin` for details.\n\n    - Method *broyden1* uses Broyden's first Jacobian approximation, it is\n      known as Broyden's good method.\n    - Method *broyden2* uses Broyden's second Jacobian approximation, it\n      is known as Broyden's bad method.\n    - Method *anderson* uses (extended) Anderson mixing.\n    - Method *Krylov* uses Krylov approximation for inverse Jacobian. It\n      is suitable for large-scale problem.\n    - Method *diagbroyden* uses diagonal Broyden Jacobian approximation.\n    - Method *linearmixing* uses a scalar Jacobian approximation.\n    - Method *excitingmixing* uses a tuned diagonal Jacobian\n      approximation.\n\n    .. warning::\n\n        The algorithms implemented for methods *diagbroyden*,\n        *linearmixing* and *excitingmixing* may be useful for specific\n        problems, but whether they will work may depend strongly on the\n        problem.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] More, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.\n       1980. User Guide for MINPACK-1.\n    .. [2] C. T. Kelley. 1995. Iterative Methods for Linear and Nonlinear\n        Equations. Society for Industrial and Applied Mathematics.\n        <http://www.siam.org/books/kelley/fr16/index.php>\n    .. [3] W. La Cruz, J.M. Martinez, M. Raydan. Math. Comp. 75, 1429 (2006).\n\n    Examples\n    --------\n    The following functions define a system of nonlinear equations and its\n    jacobian.\n\n    >>> def fun(x):\n    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,\n    ...             0.5 * (x[1] - x[0])**3 + x[1]]\n\n    >>> def jac(x):\n    ...     return np.array([[1 + 1.5 * (x[0] - x[1])**2,\n    ...                       -1.5 * (x[0] - x[1])**2],\n    ...                      [-1.5 * (x[1] - x[0])**2,\n    ...                       1 + 1.5 * (x[1] - x[0])**2]])\n\n    A solution can be obtained as follows.\n\n    >>> from scipy import optimize\n    >>> sol = optimize.root(fun, [0, 0], jac=jac, method='hybr')\n    >>> sol.x\n    array([ 0.8411639,  0.1588361])\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 152)
    # Getting the type of 'tuple' (line 152)
    tuple_201167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'tuple')
    # Getting the type of 'args' (line 152)
    args_201168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'args')
    
    (may_be_201169, more_types_in_union_201170) = may_not_be_subtype(tuple_201167, args_201168)

    if may_be_201169:

        if more_types_in_union_201170:
            # Runtime conditional SSA (line 152)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'args', remove_subtype_from_union(args_201168, tuple))
        
        # Assigning a Tuple to a Name (line 153):
        
        # Assigning a Tuple to a Name (line 153):
        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_201171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        # Getting the type of 'args' (line 153)
        args_201172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 16), tuple_201171, args_201172)
        
        # Assigning a type to the variable 'args' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'args', tuple_201171)

        if more_types_in_union_201170:
            # SSA join for if statement (line 152)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 155):
    
    # Assigning a Call to a Name (line 155):
    
    # Call to lower(...): (line 155)
    # Processing the call keyword arguments (line 155)
    kwargs_201175 = {}
    # Getting the type of 'method' (line 155)
    method_201173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'method', False)
    # Obtaining the member 'lower' of a type (line 155)
    lower_201174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 11), method_201173, 'lower')
    # Calling lower(args, kwargs) (line 155)
    lower_call_result_201176 = invoke(stypy.reporting.localization.Localization(__file__, 155, 11), lower_201174, *[], **kwargs_201175)
    
    # Assigning a type to the variable 'meth' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'meth', lower_call_result_201176)
    
    # Type idiom detected: calculating its left and rigth part (line 156)
    # Getting the type of 'options' (line 156)
    options_201177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 7), 'options')
    # Getting the type of 'None' (line 156)
    None_201178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 18), 'None')
    
    (may_be_201179, more_types_in_union_201180) = may_be_none(options_201177, None_201178)

    if may_be_201179:

        if more_types_in_union_201180:
            # Runtime conditional SSA (line 156)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 157):
        
        # Assigning a Dict to a Name (line 157):
        
        # Obtaining an instance of the builtin type 'dict' (line 157)
        dict_201181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 157)
        
        # Assigning a type to the variable 'options' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'options', dict_201181)

        if more_types_in_union_201180:
            # SSA join for if statement (line 156)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'callback' (line 159)
    callback_201182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'callback')
    # Getting the type of 'None' (line 159)
    None_201183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'None')
    # Applying the binary operator 'isnot' (line 159)
    result_is_not_201184 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), 'isnot', callback_201182, None_201183)
    
    
    # Getting the type of 'meth' (line 159)
    meth_201185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'meth')
    
    # Obtaining an instance of the builtin type 'tuple' (line 159)
    tuple_201186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 159)
    # Adding element type (line 159)
    str_201187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 41), 'str', 'hybr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 41), tuple_201186, str_201187)
    # Adding element type (line 159)
    str_201188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 49), 'str', 'lm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 41), tuple_201186, str_201188)
    
    # Applying the binary operator 'in' (line 159)
    result_contains_201189 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 32), 'in', meth_201185, tuple_201186)
    
    # Applying the binary operator 'and' (line 159)
    result_and_keyword_201190 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), 'and', result_is_not_201184, result_contains_201189)
    
    # Testing the type of an if condition (line 159)
    if_condition_201191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 4), result_and_keyword_201190)
    # Assigning a type to the variable 'if_condition_201191' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'if_condition_201191', if_condition_201191)
    # SSA begins for if statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 160)
    # Processing the call arguments (line 160)
    str_201193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 13), 'str', 'Method %s does not accept callback.')
    # Getting the type of 'method' (line 160)
    method_201194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 53), 'method', False)
    # Applying the binary operator '%' (line 160)
    result_mod_201195 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 13), '%', str_201193, method_201194)
    
    # Getting the type of 'RuntimeWarning' (line 161)
    RuntimeWarning_201196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 160)
    kwargs_201197 = {}
    # Getting the type of 'warn' (line 160)
    warn_201192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 160)
    warn_call_result_201198 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), warn_201192, *[result_mod_201195, RuntimeWarning_201196], **kwargs_201197)
    
    # SSA join for if statement (line 159)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to callable(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'jac' (line 164)
    jac_201200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'jac', False)
    # Processing the call keyword arguments (line 164)
    kwargs_201201 = {}
    # Getting the type of 'callable' (line 164)
    callable_201199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'callable', False)
    # Calling callable(args, kwargs) (line 164)
    callable_call_result_201202 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), callable_201199, *[jac_201200], **kwargs_201201)
    
    # Applying the 'not' unary operator (line 164)
    result_not__201203 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 7), 'not', callable_call_result_201202)
    
    
    # Getting the type of 'meth' (line 164)
    meth_201204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 29), 'meth')
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_201205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    str_201206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 38), 'str', 'hybr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 38), tuple_201205, str_201206)
    # Adding element type (line 164)
    str_201207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 46), 'str', 'lm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 38), tuple_201205, str_201207)
    
    # Applying the binary operator 'in' (line 164)
    result_contains_201208 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 29), 'in', meth_201204, tuple_201205)
    
    # Applying the binary operator 'and' (line 164)
    result_and_keyword_201209 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 7), 'and', result_not__201203, result_contains_201208)
    
    # Testing the type of an if condition (line 164)
    if_condition_201210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 4), result_and_keyword_201209)
    # Assigning a type to the variable 'if_condition_201210' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'if_condition_201210', if_condition_201210)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to bool(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'jac' (line 165)
    jac_201212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'jac', False)
    # Processing the call keyword arguments (line 165)
    kwargs_201213 = {}
    # Getting the type of 'bool' (line 165)
    bool_201211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'bool', False)
    # Calling bool(args, kwargs) (line 165)
    bool_call_result_201214 = invoke(stypy.reporting.localization.Localization(__file__, 165, 11), bool_201211, *[jac_201212], **kwargs_201213)
    
    # Testing the type of an if condition (line 165)
    if_condition_201215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), bool_call_result_201214)
    # Assigning a type to the variable 'if_condition_201215' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'if_condition_201215', if_condition_201215)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to MemoizeJac(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'fun' (line 166)
    fun_201217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'fun', False)
    # Processing the call keyword arguments (line 166)
    kwargs_201218 = {}
    # Getting the type of 'MemoizeJac' (line 166)
    MemoizeJac_201216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'MemoizeJac', False)
    # Calling MemoizeJac(args, kwargs) (line 166)
    MemoizeJac_call_result_201219 = invoke(stypy.reporting.localization.Localization(__file__, 166, 18), MemoizeJac_201216, *[fun_201217], **kwargs_201218)
    
    # Assigning a type to the variable 'fun' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'fun', MemoizeJac_call_result_201219)
    
    # Assigning a Attribute to a Name (line 167):
    
    # Assigning a Attribute to a Name (line 167):
    # Getting the type of 'fun' (line 167)
    fun_201220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'fun')
    # Obtaining the member 'derivative' of a type (line 167)
    derivative_201221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 18), fun_201220, 'derivative')
    # Assigning a type to the variable 'jac' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'jac', derivative_201221)
    # SSA branch for the else part of an if statement (line 165)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 169):
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'None' (line 169)
    None_201222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 18), 'None')
    # Assigning a type to the variable 'jac' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'jac', None_201222)
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 172)
    # Getting the type of 'tol' (line 172)
    tol_201223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tol')
    # Getting the type of 'None' (line 172)
    None_201224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'None')
    
    (may_be_201225, more_types_in_union_201226) = may_not_be_none(tol_201223, None_201224)

    if may_be_201225:

        if more_types_in_union_201226:
            # Runtime conditional SSA (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to dict(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'options' (line 173)
        options_201228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'options', False)
        # Processing the call keyword arguments (line 173)
        kwargs_201229 = {}
        # Getting the type of 'dict' (line 173)
        dict_201227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 'dict', False)
        # Calling dict(args, kwargs) (line 173)
        dict_call_result_201230 = invoke(stypy.reporting.localization.Localization(__file__, 173, 18), dict_201227, *[options_201228], **kwargs_201229)
        
        # Assigning a type to the variable 'options' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'options', dict_call_result_201230)
        
        
        # Getting the type of 'meth' (line 174)
        meth_201231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'meth')
        
        # Obtaining an instance of the builtin type 'tuple' (line 174)
        tuple_201232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 174)
        # Adding element type (line 174)
        str_201233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 20), 'str', 'hybr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 20), tuple_201232, str_201233)
        # Adding element type (line 174)
        str_201234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 28), 'str', 'lm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 20), tuple_201232, str_201234)
        
        # Applying the binary operator 'in' (line 174)
        result_contains_201235 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 11), 'in', meth_201231, tuple_201232)
        
        # Testing the type of an if condition (line 174)
        if_condition_201236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 8), result_contains_201235)
        # Assigning a type to the variable 'if_condition_201236' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'if_condition_201236', if_condition_201236)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 175)
        # Processing the call arguments (line 175)
        str_201239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 31), 'str', 'xtol')
        # Getting the type of 'tol' (line 175)
        tol_201240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 39), 'tol', False)
        # Processing the call keyword arguments (line 175)
        kwargs_201241 = {}
        # Getting the type of 'options' (line 175)
        options_201237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 175)
        setdefault_201238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), options_201237, 'setdefault')
        # Calling setdefault(args, kwargs) (line 175)
        setdefault_call_result_201242 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), setdefault_201238, *[str_201239, tol_201240], **kwargs_201241)
        
        # SSA branch for the else part of an if statement (line 174)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'meth' (line 176)
        meth_201243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'meth')
        
        # Obtaining an instance of the builtin type 'tuple' (line 176)
        tuple_201244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 176)
        # Adding element type (line 176)
        str_201245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 22), 'str', 'df-sane')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 22), tuple_201244, str_201245)
        
        # Applying the binary operator 'in' (line 176)
        result_contains_201246 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 13), 'in', meth_201243, tuple_201244)
        
        # Testing the type of an if condition (line 176)
        if_condition_201247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 13), result_contains_201246)
        # Assigning a type to the variable 'if_condition_201247' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'if_condition_201247', if_condition_201247)
        # SSA begins for if statement (line 176)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 177)
        # Processing the call arguments (line 177)
        str_201250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'str', 'ftol')
        # Getting the type of 'tol' (line 177)
        tol_201251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 39), 'tol', False)
        # Processing the call keyword arguments (line 177)
        kwargs_201252 = {}
        # Getting the type of 'options' (line 177)
        options_201248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 177)
        setdefault_201249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), options_201248, 'setdefault')
        # Calling setdefault(args, kwargs) (line 177)
        setdefault_call_result_201253 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), setdefault_201249, *[str_201250, tol_201251], **kwargs_201252)
        
        # SSA branch for the else part of an if statement (line 176)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'meth' (line 178)
        meth_201254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'meth')
        
        # Obtaining an instance of the builtin type 'tuple' (line 178)
        tuple_201255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 178)
        # Adding element type (line 178)
        str_201256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 22), 'str', 'broyden1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), tuple_201255, str_201256)
        # Adding element type (line 178)
        str_201257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'str', 'broyden2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), tuple_201255, str_201257)
        # Adding element type (line 178)
        str_201258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 46), 'str', 'anderson')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), tuple_201255, str_201258)
        # Adding element type (line 178)
        str_201259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 58), 'str', 'linearmixing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), tuple_201255, str_201259)
        # Adding element type (line 178)
        str_201260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 22), 'str', 'diagbroyden')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), tuple_201255, str_201260)
        # Adding element type (line 178)
        str_201261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 37), 'str', 'excitingmixing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), tuple_201255, str_201261)
        # Adding element type (line 178)
        str_201262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 55), 'str', 'krylov')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 22), tuple_201255, str_201262)
        
        # Applying the binary operator 'in' (line 178)
        result_contains_201263 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 13), 'in', meth_201254, tuple_201255)
        
        # Testing the type of an if condition (line 178)
        if_condition_201264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 13), result_contains_201263)
        # Assigning a type to the variable 'if_condition_201264' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'if_condition_201264', if_condition_201264)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 180)
        # Processing the call arguments (line 180)
        str_201267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'str', 'xtol')
        # Getting the type of 'tol' (line 180)
        tol_201268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'tol', False)
        # Processing the call keyword arguments (line 180)
        kwargs_201269 = {}
        # Getting the type of 'options' (line 180)
        options_201265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 180)
        setdefault_201266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), options_201265, 'setdefault')
        # Calling setdefault(args, kwargs) (line 180)
        setdefault_call_result_201270 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), setdefault_201266, *[str_201267, tol_201268], **kwargs_201269)
        
        
        # Call to setdefault(...): (line 181)
        # Processing the call arguments (line 181)
        str_201273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 31), 'str', 'xatol')
        # Getting the type of 'np' (line 181)
        np_201274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 40), 'np', False)
        # Obtaining the member 'inf' of a type (line 181)
        inf_201275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 40), np_201274, 'inf')
        # Processing the call keyword arguments (line 181)
        kwargs_201276 = {}
        # Getting the type of 'options' (line 181)
        options_201271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 181)
        setdefault_201272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), options_201271, 'setdefault')
        # Calling setdefault(args, kwargs) (line 181)
        setdefault_call_result_201277 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), setdefault_201272, *[str_201273, inf_201275], **kwargs_201276)
        
        
        # Call to setdefault(...): (line 182)
        # Processing the call arguments (line 182)
        str_201280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 31), 'str', 'ftol')
        # Getting the type of 'np' (line 182)
        np_201281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 39), 'np', False)
        # Obtaining the member 'inf' of a type (line 182)
        inf_201282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 39), np_201281, 'inf')
        # Processing the call keyword arguments (line 182)
        kwargs_201283 = {}
        # Getting the type of 'options' (line 182)
        options_201278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 182)
        setdefault_201279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), options_201278, 'setdefault')
        # Calling setdefault(args, kwargs) (line 182)
        setdefault_call_result_201284 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), setdefault_201279, *[str_201280, inf_201282], **kwargs_201283)
        
        
        # Call to setdefault(...): (line 183)
        # Processing the call arguments (line 183)
        str_201287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 31), 'str', 'fatol')
        # Getting the type of 'np' (line 183)
        np_201288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 40), 'np', False)
        # Obtaining the member 'inf' of a type (line 183)
        inf_201289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 40), np_201288, 'inf')
        # Processing the call keyword arguments (line 183)
        kwargs_201290 = {}
        # Getting the type of 'options' (line 183)
        options_201285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 183)
        setdefault_201286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), options_201285, 'setdefault')
        # Calling setdefault(args, kwargs) (line 183)
        setdefault_call_result_201291 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), setdefault_201286, *[str_201287, inf_201289], **kwargs_201290)
        
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 176)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_201226:
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'meth' (line 185)
    meth_201292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 7), 'meth')
    str_201293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 15), 'str', 'hybr')
    # Applying the binary operator '==' (line 185)
    result_eq_201294 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 7), '==', meth_201292, str_201293)
    
    # Testing the type of an if condition (line 185)
    if_condition_201295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 4), result_eq_201294)
    # Assigning a type to the variable 'if_condition_201295' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'if_condition_201295', if_condition_201295)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to _root_hybr(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'fun' (line 186)
    fun_201297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'fun', False)
    # Getting the type of 'x0' (line 186)
    x0_201298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 30), 'x0', False)
    # Processing the call keyword arguments (line 186)
    # Getting the type of 'args' (line 186)
    args_201299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 39), 'args', False)
    keyword_201300 = args_201299
    # Getting the type of 'jac' (line 186)
    jac_201301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 49), 'jac', False)
    keyword_201302 = jac_201301
    # Getting the type of 'options' (line 186)
    options_201303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 56), 'options', False)
    kwargs_201304 = {'args': keyword_201300, 'jac': keyword_201302, 'options_201303': options_201303}
    # Getting the type of '_root_hybr' (line 186)
    _root_hybr_201296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 14), '_root_hybr', False)
    # Calling _root_hybr(args, kwargs) (line 186)
    _root_hybr_call_result_201305 = invoke(stypy.reporting.localization.Localization(__file__, 186, 14), _root_hybr_201296, *[fun_201297, x0_201298], **kwargs_201304)
    
    # Assigning a type to the variable 'sol' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'sol', _root_hybr_call_result_201305)
    # SSA branch for the else part of an if statement (line 185)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 187)
    meth_201306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 9), 'meth')
    str_201307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 17), 'str', 'lm')
    # Applying the binary operator '==' (line 187)
    result_eq_201308 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 9), '==', meth_201306, str_201307)
    
    # Testing the type of an if condition (line 187)
    if_condition_201309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 9), result_eq_201308)
    # Assigning a type to the variable 'if_condition_201309' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 9), 'if_condition_201309', if_condition_201309)
    # SSA begins for if statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to _root_leastsq(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'fun' (line 188)
    fun_201311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'fun', False)
    # Getting the type of 'x0' (line 188)
    x0_201312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'x0', False)
    # Processing the call keyword arguments (line 188)
    # Getting the type of 'args' (line 188)
    args_201313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 42), 'args', False)
    keyword_201314 = args_201313
    # Getting the type of 'jac' (line 188)
    jac_201315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 52), 'jac', False)
    keyword_201316 = jac_201315
    # Getting the type of 'options' (line 188)
    options_201317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 59), 'options', False)
    kwargs_201318 = {'options_201317': options_201317, 'args': keyword_201314, 'jac': keyword_201316}
    # Getting the type of '_root_leastsq' (line 188)
    _root_leastsq_201310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 14), '_root_leastsq', False)
    # Calling _root_leastsq(args, kwargs) (line 188)
    _root_leastsq_call_result_201319 = invoke(stypy.reporting.localization.Localization(__file__, 188, 14), _root_leastsq_201310, *[fun_201311, x0_201312], **kwargs_201318)
    
    # Assigning a type to the variable 'sol' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'sol', _root_leastsq_call_result_201319)
    # SSA branch for the else part of an if statement (line 187)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 189)
    meth_201320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 9), 'meth')
    str_201321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 17), 'str', 'df-sane')
    # Applying the binary operator '==' (line 189)
    result_eq_201322 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 9), '==', meth_201320, str_201321)
    
    # Testing the type of an if condition (line 189)
    if_condition_201323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 9), result_eq_201322)
    # Assigning a type to the variable 'if_condition_201323' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 9), 'if_condition_201323', if_condition_201323)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _warn_jac_unused(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'jac' (line 190)
    jac_201325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'jac', False)
    # Getting the type of 'method' (line 190)
    method_201326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 30), 'method', False)
    # Processing the call keyword arguments (line 190)
    kwargs_201327 = {}
    # Getting the type of '_warn_jac_unused' (line 190)
    _warn_jac_unused_201324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), '_warn_jac_unused', False)
    # Calling _warn_jac_unused(args, kwargs) (line 190)
    _warn_jac_unused_call_result_201328 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), _warn_jac_unused_201324, *[jac_201325, method_201326], **kwargs_201327)
    
    
    # Assigning a Call to a Name (line 191):
    
    # Assigning a Call to a Name (line 191):
    
    # Call to _root_df_sane(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'fun' (line 191)
    fun_201330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'fun', False)
    # Getting the type of 'x0' (line 191)
    x0_201331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 33), 'x0', False)
    # Processing the call keyword arguments (line 191)
    # Getting the type of 'args' (line 191)
    args_201332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 42), 'args', False)
    keyword_201333 = args_201332
    # Getting the type of 'callback' (line 191)
    callback_201334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 57), 'callback', False)
    keyword_201335 = callback_201334
    # Getting the type of 'options' (line 192)
    options_201336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'options', False)
    kwargs_201337 = {'callback': keyword_201335, 'args': keyword_201333, 'options_201336': options_201336}
    # Getting the type of '_root_df_sane' (line 191)
    _root_df_sane_201329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), '_root_df_sane', False)
    # Calling _root_df_sane(args, kwargs) (line 191)
    _root_df_sane_call_result_201338 = invoke(stypy.reporting.localization.Localization(__file__, 191, 14), _root_df_sane_201329, *[fun_201330, x0_201331], **kwargs_201337)
    
    # Assigning a type to the variable 'sol' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'sol', _root_df_sane_call_result_201338)
    # SSA branch for the else part of an if statement (line 189)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 193)
    meth_201339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 9), 'meth')
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_201340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    str_201341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 18), 'str', 'broyden1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 18), tuple_201340, str_201341)
    # Adding element type (line 193)
    str_201342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 30), 'str', 'broyden2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 18), tuple_201340, str_201342)
    # Adding element type (line 193)
    str_201343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 42), 'str', 'anderson')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 18), tuple_201340, str_201343)
    # Adding element type (line 193)
    str_201344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 54), 'str', 'linearmixing')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 18), tuple_201340, str_201344)
    # Adding element type (line 193)
    str_201345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 18), 'str', 'diagbroyden')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 18), tuple_201340, str_201345)
    # Adding element type (line 193)
    str_201346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 33), 'str', 'excitingmixing')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 18), tuple_201340, str_201346)
    # Adding element type (line 193)
    str_201347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 51), 'str', 'krylov')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 18), tuple_201340, str_201347)
    
    # Applying the binary operator 'in' (line 193)
    result_contains_201348 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 9), 'in', meth_201339, tuple_201340)
    
    # Testing the type of an if condition (line 193)
    if_condition_201349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 9), result_contains_201348)
    # Assigning a type to the variable 'if_condition_201349' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 9), 'if_condition_201349', if_condition_201349)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _warn_jac_unused(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'jac' (line 195)
    jac_201351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 25), 'jac', False)
    # Getting the type of 'method' (line 195)
    method_201352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 30), 'method', False)
    # Processing the call keyword arguments (line 195)
    kwargs_201353 = {}
    # Getting the type of '_warn_jac_unused' (line 195)
    _warn_jac_unused_201350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), '_warn_jac_unused', False)
    # Calling _warn_jac_unused(args, kwargs) (line 195)
    _warn_jac_unused_call_result_201354 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), _warn_jac_unused_201350, *[jac_201351, method_201352], **kwargs_201353)
    
    
    # Assigning a Call to a Name (line 196):
    
    # Assigning a Call to a Name (line 196):
    
    # Call to _root_nonlin_solve(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'fun' (line 196)
    fun_201356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 33), 'fun', False)
    # Getting the type of 'x0' (line 196)
    x0_201357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 38), 'x0', False)
    # Processing the call keyword arguments (line 196)
    # Getting the type of 'args' (line 196)
    args_201358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 47), 'args', False)
    keyword_201359 = args_201358
    # Getting the type of 'jac' (line 196)
    jac_201360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 57), 'jac', False)
    keyword_201361 = jac_201360
    # Getting the type of 'meth' (line 197)
    meth_201362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 41), 'meth', False)
    keyword_201363 = meth_201362
    # Getting the type of 'callback' (line 197)
    callback_201364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 57), 'callback', False)
    keyword_201365 = callback_201364
    # Getting the type of 'options' (line 198)
    options_201366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 35), 'options', False)
    kwargs_201367 = {'args': keyword_201359, '_callback': keyword_201365, 'jac': keyword_201361, 'options_201366': options_201366, '_method': keyword_201363}
    # Getting the type of '_root_nonlin_solve' (line 196)
    _root_nonlin_solve_201355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 14), '_root_nonlin_solve', False)
    # Calling _root_nonlin_solve(args, kwargs) (line 196)
    _root_nonlin_solve_call_result_201368 = invoke(stypy.reporting.localization.Localization(__file__, 196, 14), _root_nonlin_solve_201355, *[fun_201356, x0_201357], **kwargs_201367)
    
    # Assigning a type to the variable 'sol' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'sol', _root_nonlin_solve_call_result_201368)
    # SSA branch for the else part of an if statement (line 193)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 200)
    # Processing the call arguments (line 200)
    str_201370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 25), 'str', 'Unknown solver %s')
    # Getting the type of 'method' (line 200)
    method_201371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 47), 'method', False)
    # Applying the binary operator '%' (line 200)
    result_mod_201372 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 25), '%', str_201370, method_201371)
    
    # Processing the call keyword arguments (line 200)
    kwargs_201373 = {}
    # Getting the type of 'ValueError' (line 200)
    ValueError_201369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 200)
    ValueError_call_result_201374 = invoke(stypy.reporting.localization.Localization(__file__, 200, 14), ValueError_201369, *[result_mod_201372], **kwargs_201373)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 200, 8), ValueError_call_result_201374, 'raise parameter', BaseException)
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'sol' (line 202)
    sol_201375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'sol')
    # Assigning a type to the variable 'stypy_return_type' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type', sol_201375)
    
    # ################# End of 'root(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'root' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_201376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201376)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'root'
    return stypy_return_type_201376

# Assigning a type to the variable 'root' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'root', root)

@norecursion
def _warn_jac_unused(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_warn_jac_unused'
    module_type_store = module_type_store.open_function_context('_warn_jac_unused', 205, 0, False)
    
    # Passed parameters checking function
    _warn_jac_unused.stypy_localization = localization
    _warn_jac_unused.stypy_type_of_self = None
    _warn_jac_unused.stypy_type_store = module_type_store
    _warn_jac_unused.stypy_function_name = '_warn_jac_unused'
    _warn_jac_unused.stypy_param_names_list = ['jac', 'method']
    _warn_jac_unused.stypy_varargs_param_name = None
    _warn_jac_unused.stypy_kwargs_param_name = None
    _warn_jac_unused.stypy_call_defaults = defaults
    _warn_jac_unused.stypy_call_varargs = varargs
    _warn_jac_unused.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_warn_jac_unused', ['jac', 'method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_warn_jac_unused', localization, ['jac', 'method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_warn_jac_unused(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 206)
    # Getting the type of 'jac' (line 206)
    jac_201377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'jac')
    # Getting the type of 'None' (line 206)
    None_201378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'None')
    
    (may_be_201379, more_types_in_union_201380) = may_not_be_none(jac_201377, None_201378)

    if may_be_201379:

        if more_types_in_union_201380:
            # Runtime conditional SSA (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to warn(...): (line 207)
        # Processing the call arguments (line 207)
        str_201382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 13), 'str', 'Method %s does not use the jacobian (jac).')
        
        # Obtaining an instance of the builtin type 'tuple' (line 207)
        tuple_201383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 207)
        # Adding element type (line 207)
        # Getting the type of 'method' (line 207)
        method_201384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 61), 'method', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 61), tuple_201383, method_201384)
        
        # Applying the binary operator '%' (line 207)
        result_mod_201385 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 13), '%', str_201382, tuple_201383)
        
        # Getting the type of 'RuntimeWarning' (line 208)
        RuntimeWarning_201386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 13), 'RuntimeWarning', False)
        # Processing the call keyword arguments (line 207)
        kwargs_201387 = {}
        # Getting the type of 'warn' (line 207)
        warn_201381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'warn', False)
        # Calling warn(args, kwargs) (line 207)
        warn_call_result_201388 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), warn_201381, *[result_mod_201385, RuntimeWarning_201386], **kwargs_201387)
        

        if more_types_in_union_201380:
            # SSA join for if statement (line 206)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_warn_jac_unused(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_warn_jac_unused' in the type store
    # Getting the type of 'stypy_return_type' (line 205)
    stypy_return_type_201389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201389)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_warn_jac_unused'
    return stypy_return_type_201389

# Assigning a type to the variable '_warn_jac_unused' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), '_warn_jac_unused', _warn_jac_unused)

@norecursion
def _root_leastsq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 211)
    tuple_201390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 211)
    
    # Getting the type of 'None' (line 211)
    None_201391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'None')
    int_201392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 28), 'int')
    float_201393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 36), 'float')
    float_201394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 54), 'float')
    float_201395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 23), 'float')
    int_201396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 36), 'int')
    float_201397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 43), 'float')
    int_201398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 55), 'int')
    # Getting the type of 'None' (line 213)
    None_201399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 65), 'None')
    defaults = [tuple_201390, None_201391, int_201392, float_201393, float_201394, float_201395, int_201396, float_201397, int_201398, None_201399]
    # Create a new context for function '_root_leastsq'
    module_type_store = module_type_store.open_function_context('_root_leastsq', 211, 0, False)
    
    # Passed parameters checking function
    _root_leastsq.stypy_localization = localization
    _root_leastsq.stypy_type_of_self = None
    _root_leastsq.stypy_type_store = module_type_store
    _root_leastsq.stypy_function_name = '_root_leastsq'
    _root_leastsq.stypy_param_names_list = ['func', 'x0', 'args', 'jac', 'col_deriv', 'xtol', 'ftol', 'gtol', 'maxiter', 'eps', 'factor', 'diag']
    _root_leastsq.stypy_varargs_param_name = None
    _root_leastsq.stypy_kwargs_param_name = 'unknown_options'
    _root_leastsq.stypy_call_defaults = defaults
    _root_leastsq.stypy_call_varargs = varargs
    _root_leastsq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_leastsq', ['func', 'x0', 'args', 'jac', 'col_deriv', 'xtol', 'ftol', 'gtol', 'maxiter', 'eps', 'factor', 'diag'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_leastsq', localization, ['func', 'x0', 'args', 'jac', 'col_deriv', 'xtol', 'ftol', 'gtol', 'maxiter', 'eps', 'factor', 'diag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_leastsq(...)' code ##################

    str_201400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, (-1)), 'str', '\n    Solve for least squares with Levenberg-Marquardt\n\n    Options\n    -------\n    col_deriv : bool\n        non-zero to specify that the Jacobian function computes derivatives\n        down the columns (faster, because there is no transpose operation).\n    ftol : float\n        Relative error desired in the sum of squares.\n    xtol : float\n        Relative error desired in the approximate solution.\n    gtol : float\n        Orthogonality desired between the function vector and the columns\n        of the Jacobian.\n    maxiter : int\n        The maximum number of calls to the function. If zero, then\n        100*(N+1) is the maximum where N is the number of elements in x0.\n    epsfcn : float\n        A suitable step length for the forward-difference approximation of\n        the Jacobian (for Dfun=None). If epsfcn is less than the machine\n        precision, it is assumed that the relative errors in the functions\n        are of the order of the machine precision.\n    factor : float\n        A parameter determining the initial step bound\n        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.\n    diag : sequence\n        N positive entries that serve as a scale factors for the variables.\n    ')
    
    # Call to _check_unknown_options(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'unknown_options' (line 245)
    unknown_options_201402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 245)
    kwargs_201403 = {}
    # Getting the type of '_check_unknown_options' (line 245)
    _check_unknown_options_201401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 245)
    _check_unknown_options_call_result_201404 = invoke(stypy.reporting.localization.Localization(__file__, 245, 4), _check_unknown_options_201401, *[unknown_options_201402], **kwargs_201403)
    
    
    # Assigning a Call to a Tuple (line 246):
    
    # Assigning a Subscript to a Name (line 246):
    
    # Obtaining the type of the subscript
    int_201405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 4), 'int')
    
    # Call to leastsq(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'func' (line 246)
    func_201407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 39), 'func', False)
    # Getting the type of 'x0' (line 246)
    x0_201408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'x0', False)
    # Processing the call keyword arguments (line 246)
    # Getting the type of 'args' (line 246)
    args_201409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 54), 'args', False)
    keyword_201410 = args_201409
    # Getting the type of 'jac' (line 246)
    jac_201411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 65), 'jac', False)
    keyword_201412 = jac_201411
    # Getting the type of 'True' (line 247)
    True_201413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 51), 'True', False)
    keyword_201414 = True_201413
    # Getting the type of 'col_deriv' (line 248)
    col_deriv_201415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 49), 'col_deriv', False)
    keyword_201416 = col_deriv_201415
    # Getting the type of 'xtol' (line 248)
    xtol_201417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 65), 'xtol', False)
    keyword_201418 = xtol_201417
    # Getting the type of 'ftol' (line 249)
    ftol_201419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 44), 'ftol', False)
    keyword_201420 = ftol_201419
    # Getting the type of 'gtol' (line 249)
    gtol_201421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 55), 'gtol', False)
    keyword_201422 = gtol_201421
    # Getting the type of 'maxiter' (line 250)
    maxiter_201423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 46), 'maxiter', False)
    keyword_201424 = maxiter_201423
    # Getting the type of 'eps' (line 250)
    eps_201425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 62), 'eps', False)
    keyword_201426 = eps_201425
    # Getting the type of 'factor' (line 251)
    factor_201427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 46), 'factor', False)
    keyword_201428 = factor_201427
    # Getting the type of 'diag' (line 251)
    diag_201429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 59), 'diag', False)
    keyword_201430 = diag_201429
    kwargs_201431 = {'factor': keyword_201428, 'Dfun': keyword_201412, 'full_output': keyword_201414, 'col_deriv': keyword_201416, 'diag': keyword_201430, 'args': keyword_201410, 'gtol': keyword_201422, 'maxfev': keyword_201424, 'xtol': keyword_201418, 'epsfcn': keyword_201426, 'ftol': keyword_201420}
    # Getting the type of 'leastsq' (line 246)
    leastsq_201406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'leastsq', False)
    # Calling leastsq(args, kwargs) (line 246)
    leastsq_call_result_201432 = invoke(stypy.reporting.localization.Localization(__file__, 246, 31), leastsq_201406, *[func_201407, x0_201408], **kwargs_201431)
    
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___201433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 4), leastsq_call_result_201432, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_201434 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), getitem___201433, int_201405)
    
    # Assigning a type to the variable 'tuple_var_assignment_201138' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201138', subscript_call_result_201434)
    
    # Assigning a Subscript to a Name (line 246):
    
    # Obtaining the type of the subscript
    int_201435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 4), 'int')
    
    # Call to leastsq(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'func' (line 246)
    func_201437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 39), 'func', False)
    # Getting the type of 'x0' (line 246)
    x0_201438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'x0', False)
    # Processing the call keyword arguments (line 246)
    # Getting the type of 'args' (line 246)
    args_201439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 54), 'args', False)
    keyword_201440 = args_201439
    # Getting the type of 'jac' (line 246)
    jac_201441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 65), 'jac', False)
    keyword_201442 = jac_201441
    # Getting the type of 'True' (line 247)
    True_201443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 51), 'True', False)
    keyword_201444 = True_201443
    # Getting the type of 'col_deriv' (line 248)
    col_deriv_201445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 49), 'col_deriv', False)
    keyword_201446 = col_deriv_201445
    # Getting the type of 'xtol' (line 248)
    xtol_201447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 65), 'xtol', False)
    keyword_201448 = xtol_201447
    # Getting the type of 'ftol' (line 249)
    ftol_201449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 44), 'ftol', False)
    keyword_201450 = ftol_201449
    # Getting the type of 'gtol' (line 249)
    gtol_201451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 55), 'gtol', False)
    keyword_201452 = gtol_201451
    # Getting the type of 'maxiter' (line 250)
    maxiter_201453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 46), 'maxiter', False)
    keyword_201454 = maxiter_201453
    # Getting the type of 'eps' (line 250)
    eps_201455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 62), 'eps', False)
    keyword_201456 = eps_201455
    # Getting the type of 'factor' (line 251)
    factor_201457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 46), 'factor', False)
    keyword_201458 = factor_201457
    # Getting the type of 'diag' (line 251)
    diag_201459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 59), 'diag', False)
    keyword_201460 = diag_201459
    kwargs_201461 = {'factor': keyword_201458, 'Dfun': keyword_201442, 'full_output': keyword_201444, 'col_deriv': keyword_201446, 'diag': keyword_201460, 'args': keyword_201440, 'gtol': keyword_201452, 'maxfev': keyword_201454, 'xtol': keyword_201448, 'epsfcn': keyword_201456, 'ftol': keyword_201450}
    # Getting the type of 'leastsq' (line 246)
    leastsq_201436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'leastsq', False)
    # Calling leastsq(args, kwargs) (line 246)
    leastsq_call_result_201462 = invoke(stypy.reporting.localization.Localization(__file__, 246, 31), leastsq_201436, *[func_201437, x0_201438], **kwargs_201461)
    
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___201463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 4), leastsq_call_result_201462, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_201464 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), getitem___201463, int_201435)
    
    # Assigning a type to the variable 'tuple_var_assignment_201139' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201139', subscript_call_result_201464)
    
    # Assigning a Subscript to a Name (line 246):
    
    # Obtaining the type of the subscript
    int_201465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 4), 'int')
    
    # Call to leastsq(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'func' (line 246)
    func_201467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 39), 'func', False)
    # Getting the type of 'x0' (line 246)
    x0_201468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'x0', False)
    # Processing the call keyword arguments (line 246)
    # Getting the type of 'args' (line 246)
    args_201469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 54), 'args', False)
    keyword_201470 = args_201469
    # Getting the type of 'jac' (line 246)
    jac_201471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 65), 'jac', False)
    keyword_201472 = jac_201471
    # Getting the type of 'True' (line 247)
    True_201473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 51), 'True', False)
    keyword_201474 = True_201473
    # Getting the type of 'col_deriv' (line 248)
    col_deriv_201475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 49), 'col_deriv', False)
    keyword_201476 = col_deriv_201475
    # Getting the type of 'xtol' (line 248)
    xtol_201477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 65), 'xtol', False)
    keyword_201478 = xtol_201477
    # Getting the type of 'ftol' (line 249)
    ftol_201479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 44), 'ftol', False)
    keyword_201480 = ftol_201479
    # Getting the type of 'gtol' (line 249)
    gtol_201481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 55), 'gtol', False)
    keyword_201482 = gtol_201481
    # Getting the type of 'maxiter' (line 250)
    maxiter_201483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 46), 'maxiter', False)
    keyword_201484 = maxiter_201483
    # Getting the type of 'eps' (line 250)
    eps_201485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 62), 'eps', False)
    keyword_201486 = eps_201485
    # Getting the type of 'factor' (line 251)
    factor_201487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 46), 'factor', False)
    keyword_201488 = factor_201487
    # Getting the type of 'diag' (line 251)
    diag_201489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 59), 'diag', False)
    keyword_201490 = diag_201489
    kwargs_201491 = {'factor': keyword_201488, 'Dfun': keyword_201472, 'full_output': keyword_201474, 'col_deriv': keyword_201476, 'diag': keyword_201490, 'args': keyword_201470, 'gtol': keyword_201482, 'maxfev': keyword_201484, 'xtol': keyword_201478, 'epsfcn': keyword_201486, 'ftol': keyword_201480}
    # Getting the type of 'leastsq' (line 246)
    leastsq_201466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'leastsq', False)
    # Calling leastsq(args, kwargs) (line 246)
    leastsq_call_result_201492 = invoke(stypy.reporting.localization.Localization(__file__, 246, 31), leastsq_201466, *[func_201467, x0_201468], **kwargs_201491)
    
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___201493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 4), leastsq_call_result_201492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_201494 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), getitem___201493, int_201465)
    
    # Assigning a type to the variable 'tuple_var_assignment_201140' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201140', subscript_call_result_201494)
    
    # Assigning a Subscript to a Name (line 246):
    
    # Obtaining the type of the subscript
    int_201495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 4), 'int')
    
    # Call to leastsq(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'func' (line 246)
    func_201497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 39), 'func', False)
    # Getting the type of 'x0' (line 246)
    x0_201498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'x0', False)
    # Processing the call keyword arguments (line 246)
    # Getting the type of 'args' (line 246)
    args_201499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 54), 'args', False)
    keyword_201500 = args_201499
    # Getting the type of 'jac' (line 246)
    jac_201501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 65), 'jac', False)
    keyword_201502 = jac_201501
    # Getting the type of 'True' (line 247)
    True_201503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 51), 'True', False)
    keyword_201504 = True_201503
    # Getting the type of 'col_deriv' (line 248)
    col_deriv_201505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 49), 'col_deriv', False)
    keyword_201506 = col_deriv_201505
    # Getting the type of 'xtol' (line 248)
    xtol_201507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 65), 'xtol', False)
    keyword_201508 = xtol_201507
    # Getting the type of 'ftol' (line 249)
    ftol_201509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 44), 'ftol', False)
    keyword_201510 = ftol_201509
    # Getting the type of 'gtol' (line 249)
    gtol_201511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 55), 'gtol', False)
    keyword_201512 = gtol_201511
    # Getting the type of 'maxiter' (line 250)
    maxiter_201513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 46), 'maxiter', False)
    keyword_201514 = maxiter_201513
    # Getting the type of 'eps' (line 250)
    eps_201515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 62), 'eps', False)
    keyword_201516 = eps_201515
    # Getting the type of 'factor' (line 251)
    factor_201517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 46), 'factor', False)
    keyword_201518 = factor_201517
    # Getting the type of 'diag' (line 251)
    diag_201519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 59), 'diag', False)
    keyword_201520 = diag_201519
    kwargs_201521 = {'factor': keyword_201518, 'Dfun': keyword_201502, 'full_output': keyword_201504, 'col_deriv': keyword_201506, 'diag': keyword_201520, 'args': keyword_201500, 'gtol': keyword_201512, 'maxfev': keyword_201514, 'xtol': keyword_201508, 'epsfcn': keyword_201516, 'ftol': keyword_201510}
    # Getting the type of 'leastsq' (line 246)
    leastsq_201496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'leastsq', False)
    # Calling leastsq(args, kwargs) (line 246)
    leastsq_call_result_201522 = invoke(stypy.reporting.localization.Localization(__file__, 246, 31), leastsq_201496, *[func_201497, x0_201498], **kwargs_201521)
    
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___201523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 4), leastsq_call_result_201522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_201524 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), getitem___201523, int_201495)
    
    # Assigning a type to the variable 'tuple_var_assignment_201141' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201141', subscript_call_result_201524)
    
    # Assigning a Subscript to a Name (line 246):
    
    # Obtaining the type of the subscript
    int_201525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 4), 'int')
    
    # Call to leastsq(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'func' (line 246)
    func_201527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 39), 'func', False)
    # Getting the type of 'x0' (line 246)
    x0_201528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'x0', False)
    # Processing the call keyword arguments (line 246)
    # Getting the type of 'args' (line 246)
    args_201529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 54), 'args', False)
    keyword_201530 = args_201529
    # Getting the type of 'jac' (line 246)
    jac_201531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 65), 'jac', False)
    keyword_201532 = jac_201531
    # Getting the type of 'True' (line 247)
    True_201533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 51), 'True', False)
    keyword_201534 = True_201533
    # Getting the type of 'col_deriv' (line 248)
    col_deriv_201535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 49), 'col_deriv', False)
    keyword_201536 = col_deriv_201535
    # Getting the type of 'xtol' (line 248)
    xtol_201537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 65), 'xtol', False)
    keyword_201538 = xtol_201537
    # Getting the type of 'ftol' (line 249)
    ftol_201539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 44), 'ftol', False)
    keyword_201540 = ftol_201539
    # Getting the type of 'gtol' (line 249)
    gtol_201541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 55), 'gtol', False)
    keyword_201542 = gtol_201541
    # Getting the type of 'maxiter' (line 250)
    maxiter_201543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 46), 'maxiter', False)
    keyword_201544 = maxiter_201543
    # Getting the type of 'eps' (line 250)
    eps_201545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 62), 'eps', False)
    keyword_201546 = eps_201545
    # Getting the type of 'factor' (line 251)
    factor_201547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 46), 'factor', False)
    keyword_201548 = factor_201547
    # Getting the type of 'diag' (line 251)
    diag_201549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 59), 'diag', False)
    keyword_201550 = diag_201549
    kwargs_201551 = {'factor': keyword_201548, 'Dfun': keyword_201532, 'full_output': keyword_201534, 'col_deriv': keyword_201536, 'diag': keyword_201550, 'args': keyword_201530, 'gtol': keyword_201542, 'maxfev': keyword_201544, 'xtol': keyword_201538, 'epsfcn': keyword_201546, 'ftol': keyword_201540}
    # Getting the type of 'leastsq' (line 246)
    leastsq_201526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'leastsq', False)
    # Calling leastsq(args, kwargs) (line 246)
    leastsq_call_result_201552 = invoke(stypy.reporting.localization.Localization(__file__, 246, 31), leastsq_201526, *[func_201527, x0_201528], **kwargs_201551)
    
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___201553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 4), leastsq_call_result_201552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_201554 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), getitem___201553, int_201525)
    
    # Assigning a type to the variable 'tuple_var_assignment_201142' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201142', subscript_call_result_201554)
    
    # Assigning a Name to a Name (line 246):
    # Getting the type of 'tuple_var_assignment_201138' (line 246)
    tuple_var_assignment_201138_201555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201138')
    # Assigning a type to the variable 'x' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'x', tuple_var_assignment_201138_201555)
    
    # Assigning a Name to a Name (line 246):
    # Getting the type of 'tuple_var_assignment_201139' (line 246)
    tuple_var_assignment_201139_201556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201139')
    # Assigning a type to the variable 'cov_x' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 7), 'cov_x', tuple_var_assignment_201139_201556)
    
    # Assigning a Name to a Name (line 246):
    # Getting the type of 'tuple_var_assignment_201140' (line 246)
    tuple_var_assignment_201140_201557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201140')
    # Assigning a type to the variable 'info' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 14), 'info', tuple_var_assignment_201140_201557)
    
    # Assigning a Name to a Name (line 246):
    # Getting the type of 'tuple_var_assignment_201141' (line 246)
    tuple_var_assignment_201141_201558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201141')
    # Assigning a type to the variable 'msg' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'msg', tuple_var_assignment_201141_201558)
    
    # Assigning a Name to a Name (line 246):
    # Getting the type of 'tuple_var_assignment_201142' (line 246)
    tuple_var_assignment_201142_201559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'tuple_var_assignment_201142')
    # Assigning a type to the variable 'ier' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'ier', tuple_var_assignment_201142_201559)
    
    # Assigning a Call to a Name (line 252):
    
    # Assigning a Call to a Name (line 252):
    
    # Call to OptimizeResult(...): (line 252)
    # Processing the call keyword arguments (line 252)
    # Getting the type of 'x' (line 252)
    x_201561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 27), 'x', False)
    keyword_201562 = x_201561
    # Getting the type of 'msg' (line 252)
    msg_201563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 38), 'msg', False)
    keyword_201564 = msg_201563
    # Getting the type of 'ier' (line 252)
    ier_201565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 50), 'ier', False)
    keyword_201566 = ier_201565
    
    # Getting the type of 'ier' (line 253)
    ier_201567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'ier', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 253)
    tuple_201568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 253)
    # Adding element type (line 253)
    int_201569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 41), tuple_201568, int_201569)
    # Adding element type (line 253)
    int_201570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 41), tuple_201568, int_201570)
    # Adding element type (line 253)
    int_201571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 41), tuple_201568, int_201571)
    # Adding element type (line 253)
    int_201572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 41), tuple_201568, int_201572)
    
    # Applying the binary operator 'in' (line 253)
    result_contains_201573 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 33), 'in', ier_201567, tuple_201568)
    
    keyword_201574 = result_contains_201573
    # Getting the type of 'cov_x' (line 253)
    cov_x_201575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 60), 'cov_x', False)
    keyword_201576 = cov_x_201575
    
    # Call to pop(...): (line 254)
    # Processing the call arguments (line 254)
    str_201579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 38), 'str', 'fvec')
    # Processing the call keyword arguments (line 254)
    kwargs_201580 = {}
    # Getting the type of 'info' (line 254)
    info_201577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 29), 'info', False)
    # Obtaining the member 'pop' of a type (line 254)
    pop_201578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 29), info_201577, 'pop')
    # Calling pop(args, kwargs) (line 254)
    pop_call_result_201581 = invoke(stypy.reporting.localization.Localization(__file__, 254, 29), pop_201578, *[str_201579], **kwargs_201580)
    
    keyword_201582 = pop_call_result_201581
    kwargs_201583 = {'status': keyword_201566, 'cov_x': keyword_201576, 'success': keyword_201574, 'fun': keyword_201582, 'x': keyword_201562, 'message': keyword_201564}
    # Getting the type of 'OptimizeResult' (line 252)
    OptimizeResult_201560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 10), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 252)
    OptimizeResult_call_result_201584 = invoke(stypy.reporting.localization.Localization(__file__, 252, 10), OptimizeResult_201560, *[], **kwargs_201583)
    
    # Assigning a type to the variable 'sol' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'sol', OptimizeResult_call_result_201584)
    
    # Call to update(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'info' (line 255)
    info_201587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'info', False)
    # Processing the call keyword arguments (line 255)
    kwargs_201588 = {}
    # Getting the type of 'sol' (line 255)
    sol_201585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'sol', False)
    # Obtaining the member 'update' of a type (line 255)
    update_201586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 4), sol_201585, 'update')
    # Calling update(args, kwargs) (line 255)
    update_call_result_201589 = invoke(stypy.reporting.localization.Localization(__file__, 255, 4), update_201586, *[info_201587], **kwargs_201588)
    
    # Getting the type of 'sol' (line 256)
    sol_201590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'sol')
    # Assigning a type to the variable 'stypy_return_type' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type', sol_201590)
    
    # ################# End of '_root_leastsq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_leastsq' in the type store
    # Getting the type of 'stypy_return_type' (line 211)
    stypy_return_type_201591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201591)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_leastsq'
    return stypy_return_type_201591

# Assigning a type to the variable '_root_leastsq' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), '_root_leastsq', _root_leastsq)

@norecursion
def _root_nonlin_solve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 259)
    tuple_201592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 259)
    
    # Getting the type of 'None' (line 259)
    None_201593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 46), 'None')
    # Getting the type of 'None' (line 260)
    None_201594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 33), 'None')
    # Getting the type of 'None' (line 260)
    None_201595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 47), 'None')
    # Getting the type of 'None' (line 261)
    None_201596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 27), 'None')
    # Getting the type of 'False' (line 261)
    False_201597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 38), 'False')
    # Getting the type of 'None' (line 261)
    None_201598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 53), 'None')
    # Getting the type of 'None' (line 262)
    None_201599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'None')
    # Getting the type of 'None' (line 262)
    None_201600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 40), 'None')
    # Getting the type of 'None' (line 262)
    None_201601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 51), 'None')
    # Getting the type of 'None' (line 262)
    None_201602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 63), 'None')
    # Getting the type of 'None' (line 263)
    None_201603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 'None')
    str_201604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 50), 'str', 'armijo')
    # Getting the type of 'None' (line 263)
    None_201605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 72), 'None')
    defaults = [tuple_201592, None_201593, None_201594, None_201595, None_201596, False_201597, None_201598, None_201599, None_201600, None_201601, None_201602, None_201603, str_201604, None_201605]
    # Create a new context for function '_root_nonlin_solve'
    module_type_store = module_type_store.open_function_context('_root_nonlin_solve', 259, 0, False)
    
    # Passed parameters checking function
    _root_nonlin_solve.stypy_localization = localization
    _root_nonlin_solve.stypy_type_of_self = None
    _root_nonlin_solve.stypy_type_store = module_type_store
    _root_nonlin_solve.stypy_function_name = '_root_nonlin_solve'
    _root_nonlin_solve.stypy_param_names_list = ['func', 'x0', 'args', 'jac', '_callback', '_method', 'nit', 'disp', 'maxiter', 'ftol', 'fatol', 'xtol', 'xatol', 'tol_norm', 'line_search', 'jac_options']
    _root_nonlin_solve.stypy_varargs_param_name = None
    _root_nonlin_solve.stypy_kwargs_param_name = 'unknown_options'
    _root_nonlin_solve.stypy_call_defaults = defaults
    _root_nonlin_solve.stypy_call_varargs = varargs
    _root_nonlin_solve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_nonlin_solve', ['func', 'x0', 'args', 'jac', '_callback', '_method', 'nit', 'disp', 'maxiter', 'ftol', 'fatol', 'xtol', 'xatol', 'tol_norm', 'line_search', 'jac_options'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_nonlin_solve', localization, ['func', 'x0', 'args', 'jac', '_callback', '_method', 'nit', 'disp', 'maxiter', 'ftol', 'fatol', 'xtol', 'xatol', 'tol_norm', 'line_search', 'jac_options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_nonlin_solve(...)' code ##################

    
    # Call to _check_unknown_options(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'unknown_options' (line 265)
    unknown_options_201607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 265)
    kwargs_201608 = {}
    # Getting the type of '_check_unknown_options' (line 265)
    _check_unknown_options_201606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 265)
    _check_unknown_options_call_result_201609 = invoke(stypy.reporting.localization.Localization(__file__, 265, 4), _check_unknown_options_201606, *[unknown_options_201607], **kwargs_201608)
    
    
    # Assigning a Name to a Name (line 267):
    
    # Assigning a Name to a Name (line 267):
    # Getting the type of 'fatol' (line 267)
    fatol_201610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'fatol')
    # Assigning a type to the variable 'f_tol' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'f_tol', fatol_201610)
    
    # Assigning a Name to a Name (line 268):
    
    # Assigning a Name to a Name (line 268):
    # Getting the type of 'ftol' (line 268)
    ftol_201611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 13), 'ftol')
    # Assigning a type to the variable 'f_rtol' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'f_rtol', ftol_201611)
    
    # Assigning a Name to a Name (line 269):
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'xatol' (line 269)
    xatol_201612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'xatol')
    # Assigning a type to the variable 'x_tol' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'x_tol', xatol_201612)
    
    # Assigning a Name to a Name (line 270):
    
    # Assigning a Name to a Name (line 270):
    # Getting the type of 'xtol' (line 270)
    xtol_201613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 13), 'xtol')
    # Assigning a type to the variable 'x_rtol' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'x_rtol', xtol_201613)
    
    # Assigning a Name to a Name (line 271):
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'disp' (line 271)
    disp_201614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'disp')
    # Assigning a type to the variable 'verbose' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'verbose', disp_201614)
    
    # Type idiom detected: calculating its left and rigth part (line 272)
    # Getting the type of 'jac_options' (line 272)
    jac_options_201615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 7), 'jac_options')
    # Getting the type of 'None' (line 272)
    None_201616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'None')
    
    (may_be_201617, more_types_in_union_201618) = may_be_none(jac_options_201615, None_201616)

    if may_be_201617:

        if more_types_in_union_201618:
            # Runtime conditional SSA (line 272)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to dict(...): (line 273)
        # Processing the call keyword arguments (line 273)
        kwargs_201620 = {}
        # Getting the type of 'dict' (line 273)
        dict_201619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'dict', False)
        # Calling dict(args, kwargs) (line 273)
        dict_call_result_201621 = invoke(stypy.reporting.localization.Localization(__file__, 273, 22), dict_201619, *[], **kwargs_201620)
        
        # Assigning a type to the variable 'jac_options' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'jac_options', dict_call_result_201621)

        if more_types_in_union_201618:
            # SSA join for if statement (line 272)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 275):
    
    # Assigning a Subscript to a Name (line 275):
    
    # Obtaining the type of the subscript
    # Getting the type of '_method' (line 282)
    _method_201622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), '_method')
    
    # Obtaining an instance of the builtin type 'dict' (line 275)
    dict_201623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 275)
    # Adding element type (key, value) (line 275)
    str_201624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 16), 'str', 'broyden1')
    # Getting the type of 'nonlin' (line 275)
    nonlin_201625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 28), 'nonlin')
    # Obtaining the member 'BroydenFirst' of a type (line 275)
    BroydenFirst_201626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 28), nonlin_201625, 'BroydenFirst')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), dict_201623, (str_201624, BroydenFirst_201626))
    # Adding element type (key, value) (line 275)
    str_201627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'str', 'broyden2')
    # Getting the type of 'nonlin' (line 276)
    nonlin_201628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'nonlin')
    # Obtaining the member 'BroydenSecond' of a type (line 276)
    BroydenSecond_201629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 28), nonlin_201628, 'BroydenSecond')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), dict_201623, (str_201627, BroydenSecond_201629))
    # Adding element type (key, value) (line 275)
    str_201630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 16), 'str', 'anderson')
    # Getting the type of 'nonlin' (line 277)
    nonlin_201631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 28), 'nonlin')
    # Obtaining the member 'Anderson' of a type (line 277)
    Anderson_201632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 28), nonlin_201631, 'Anderson')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), dict_201623, (str_201630, Anderson_201632))
    # Adding element type (key, value) (line 275)
    str_201633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 16), 'str', 'linearmixing')
    # Getting the type of 'nonlin' (line 278)
    nonlin_201634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 32), 'nonlin')
    # Obtaining the member 'LinearMixing' of a type (line 278)
    LinearMixing_201635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 32), nonlin_201634, 'LinearMixing')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), dict_201623, (str_201633, LinearMixing_201635))
    # Adding element type (key, value) (line 275)
    str_201636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 16), 'str', 'diagbroyden')
    # Getting the type of 'nonlin' (line 279)
    nonlin_201637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'nonlin')
    # Obtaining the member 'DiagBroyden' of a type (line 279)
    DiagBroyden_201638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 31), nonlin_201637, 'DiagBroyden')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), dict_201623, (str_201636, DiagBroyden_201638))
    # Adding element type (key, value) (line 275)
    str_201639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'str', 'excitingmixing')
    # Getting the type of 'nonlin' (line 280)
    nonlin_201640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 34), 'nonlin')
    # Obtaining the member 'ExcitingMixing' of a type (line 280)
    ExcitingMixing_201641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 34), nonlin_201640, 'ExcitingMixing')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), dict_201623, (str_201639, ExcitingMixing_201641))
    # Adding element type (key, value) (line 275)
    str_201642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 16), 'str', 'krylov')
    # Getting the type of 'nonlin' (line 281)
    nonlin_201643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'nonlin')
    # Obtaining the member 'KrylovJacobian' of a type (line 281)
    KrylovJacobian_201644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 26), nonlin_201643, 'KrylovJacobian')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), dict_201623, (str_201642, KrylovJacobian_201644))
    
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___201645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 15), dict_201623, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_201646 = invoke(stypy.reporting.localization.Localization(__file__, 275, 15), getitem___201645, _method_201622)
    
    # Assigning a type to the variable 'jacobian' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'jacobian', subscript_call_result_201646)
    
    # Getting the type of 'args' (line 284)
    args_201647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 7), 'args')
    # Testing the type of an if condition (line 284)
    if_condition_201648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 4), args_201647)
    # Assigning a type to the variable 'if_condition_201648' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'if_condition_201648', if_condition_201648)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'jac' (line 285)
    jac_201649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'jac')
    # Testing the type of an if condition (line 285)
    if_condition_201650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), jac_201649)
    # Assigning a type to the variable 'if_condition_201650' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_201650', if_condition_201650)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 286, 12, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['x']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Obtaining the type of the subscript
        int_201651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 38), 'int')
        
        # Call to func(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'x' (line 287)
        x_201653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 28), 'x', False)
        # Getting the type of 'args' (line 287)
        args_201654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'args', False)
        # Processing the call keyword arguments (line 287)
        kwargs_201655 = {}
        # Getting the type of 'func' (line 287)
        func_201652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'func', False)
        # Calling func(args, kwargs) (line 287)
        func_call_result_201656 = invoke(stypy.reporting.localization.Localization(__file__, 287, 23), func_201652, *[x_201653, args_201654], **kwargs_201655)
        
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___201657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 23), func_call_result_201656, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_201658 = invoke(stypy.reporting.localization.Localization(__file__, 287, 23), getitem___201657, int_201651)
        
        # Assigning a type to the variable 'stypy_return_type' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'stypy_return_type', subscript_call_result_201658)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_201659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_201659)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_201659

    # Assigning a type to the variable 'f' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'f', f)
    # SSA branch for the else part of an if statement (line 285)
    module_type_store.open_ssa_branch('else')

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 289, 12, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['x']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to func(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'x' (line 290)
        x_201661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 28), 'x', False)
        # Getting the type of 'args' (line 290)
        args_201662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'args', False)
        # Processing the call keyword arguments (line 290)
        kwargs_201663 = {}
        # Getting the type of 'func' (line 290)
        func_201660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 23), 'func', False)
        # Calling func(args, kwargs) (line 290)
        func_call_result_201664 = invoke(stypy.reporting.localization.Localization(__file__, 290, 23), func_201660, *[x_201661, args_201662], **kwargs_201663)
        
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'stypy_return_type', func_call_result_201664)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 289)
        stypy_return_type_201665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_201665)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_201665

    # Assigning a type to the variable 'f' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'f', f)
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 284)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 292):
    
    # Assigning a Name to a Name (line 292):
    # Getting the type of 'func' (line 292)
    func_201666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'func')
    # Assigning a type to the variable 'f' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'f', func_201666)
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 294):
    
    # Assigning a Subscript to a Name (line 294):
    
    # Obtaining the type of the subscript
    int_201667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 4), 'int')
    
    # Call to nonlin_solve(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'f' (line 294)
    f_201670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 34), 'f', False)
    # Getting the type of 'x0' (line 294)
    x0_201671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 37), 'x0', False)
    # Processing the call keyword arguments (line 294)
    
    # Call to jacobian(...): (line 294)
    # Processing the call keyword arguments (line 294)
    # Getting the type of 'jac_options' (line 294)
    jac_options_201673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 61), 'jac_options', False)
    kwargs_201674 = {'jac_options_201673': jac_options_201673}
    # Getting the type of 'jacobian' (line 294)
    jacobian_201672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 50), 'jacobian', False)
    # Calling jacobian(args, kwargs) (line 294)
    jacobian_call_result_201675 = invoke(stypy.reporting.localization.Localization(__file__, 294, 50), jacobian_201672, *[], **kwargs_201674)
    
    keyword_201676 = jacobian_call_result_201675
    # Getting the type of 'nit' (line 295)
    nit_201677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'nit', False)
    keyword_201678 = nit_201677
    # Getting the type of 'verbose' (line 295)
    verbose_201679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'verbose', False)
    keyword_201680 = verbose_201679
    # Getting the type of 'maxiter' (line 296)
    maxiter_201681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 42), 'maxiter', False)
    keyword_201682 = maxiter_201681
    # Getting the type of 'f_tol' (line 296)
    f_tol_201683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 57), 'f_tol', False)
    keyword_201684 = f_tol_201683
    # Getting the type of 'f_rtol' (line 297)
    f_rtol_201685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 41), 'f_rtol', False)
    keyword_201686 = f_rtol_201685
    # Getting the type of 'x_tol' (line 297)
    x_tol_201687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 55), 'x_tol', False)
    keyword_201688 = x_tol_201687
    # Getting the type of 'x_rtol' (line 298)
    x_rtol_201689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'x_rtol', False)
    keyword_201690 = x_rtol_201689
    # Getting the type of 'tol_norm' (line 298)
    tol_norm_201691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 58), 'tol_norm', False)
    keyword_201692 = tol_norm_201691
    # Getting the type of 'line_search' (line 299)
    line_search_201693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'line_search', False)
    keyword_201694 = line_search_201693
    # Getting the type of '_callback' (line 300)
    _callback_201695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 43), '_callback', False)
    keyword_201696 = _callback_201695
    # Getting the type of 'True' (line 300)
    True_201697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 66), 'True', False)
    keyword_201698 = True_201697
    # Getting the type of 'False' (line 301)
    False_201699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 50), 'False', False)
    keyword_201700 = False_201699
    kwargs_201701 = {'f_rtol': keyword_201686, 'verbose': keyword_201680, 'full_output': keyword_201698, 'raise_exception': keyword_201700, 'f_tol': keyword_201684, 'iter': keyword_201678, 'callback': keyword_201696, 'tol_norm': keyword_201692, 'jacobian': keyword_201676, 'maxiter': keyword_201682, 'x_tol': keyword_201688, 'x_rtol': keyword_201690, 'line_search': keyword_201694}
    # Getting the type of 'nonlin' (line 294)
    nonlin_201668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 14), 'nonlin', False)
    # Obtaining the member 'nonlin_solve' of a type (line 294)
    nonlin_solve_201669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 14), nonlin_201668, 'nonlin_solve')
    # Calling nonlin_solve(args, kwargs) (line 294)
    nonlin_solve_call_result_201702 = invoke(stypy.reporting.localization.Localization(__file__, 294, 14), nonlin_solve_201669, *[f_201670, x0_201671], **kwargs_201701)
    
    # Obtaining the member '__getitem__' of a type (line 294)
    getitem___201703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 4), nonlin_solve_call_result_201702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 294)
    subscript_call_result_201704 = invoke(stypy.reporting.localization.Localization(__file__, 294, 4), getitem___201703, int_201667)
    
    # Assigning a type to the variable 'tuple_var_assignment_201143' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'tuple_var_assignment_201143', subscript_call_result_201704)
    
    # Assigning a Subscript to a Name (line 294):
    
    # Obtaining the type of the subscript
    int_201705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 4), 'int')
    
    # Call to nonlin_solve(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'f' (line 294)
    f_201708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 34), 'f', False)
    # Getting the type of 'x0' (line 294)
    x0_201709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 37), 'x0', False)
    # Processing the call keyword arguments (line 294)
    
    # Call to jacobian(...): (line 294)
    # Processing the call keyword arguments (line 294)
    # Getting the type of 'jac_options' (line 294)
    jac_options_201711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 61), 'jac_options', False)
    kwargs_201712 = {'jac_options_201711': jac_options_201711}
    # Getting the type of 'jacobian' (line 294)
    jacobian_201710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 50), 'jacobian', False)
    # Calling jacobian(args, kwargs) (line 294)
    jacobian_call_result_201713 = invoke(stypy.reporting.localization.Localization(__file__, 294, 50), jacobian_201710, *[], **kwargs_201712)
    
    keyword_201714 = jacobian_call_result_201713
    # Getting the type of 'nit' (line 295)
    nit_201715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'nit', False)
    keyword_201716 = nit_201715
    # Getting the type of 'verbose' (line 295)
    verbose_201717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'verbose', False)
    keyword_201718 = verbose_201717
    # Getting the type of 'maxiter' (line 296)
    maxiter_201719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 42), 'maxiter', False)
    keyword_201720 = maxiter_201719
    # Getting the type of 'f_tol' (line 296)
    f_tol_201721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 57), 'f_tol', False)
    keyword_201722 = f_tol_201721
    # Getting the type of 'f_rtol' (line 297)
    f_rtol_201723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 41), 'f_rtol', False)
    keyword_201724 = f_rtol_201723
    # Getting the type of 'x_tol' (line 297)
    x_tol_201725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 55), 'x_tol', False)
    keyword_201726 = x_tol_201725
    # Getting the type of 'x_rtol' (line 298)
    x_rtol_201727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'x_rtol', False)
    keyword_201728 = x_rtol_201727
    # Getting the type of 'tol_norm' (line 298)
    tol_norm_201729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 58), 'tol_norm', False)
    keyword_201730 = tol_norm_201729
    # Getting the type of 'line_search' (line 299)
    line_search_201731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'line_search', False)
    keyword_201732 = line_search_201731
    # Getting the type of '_callback' (line 300)
    _callback_201733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 43), '_callback', False)
    keyword_201734 = _callback_201733
    # Getting the type of 'True' (line 300)
    True_201735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 66), 'True', False)
    keyword_201736 = True_201735
    # Getting the type of 'False' (line 301)
    False_201737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 50), 'False', False)
    keyword_201738 = False_201737
    kwargs_201739 = {'f_rtol': keyword_201724, 'verbose': keyword_201718, 'full_output': keyword_201736, 'raise_exception': keyword_201738, 'f_tol': keyword_201722, 'iter': keyword_201716, 'callback': keyword_201734, 'tol_norm': keyword_201730, 'jacobian': keyword_201714, 'maxiter': keyword_201720, 'x_tol': keyword_201726, 'x_rtol': keyword_201728, 'line_search': keyword_201732}
    # Getting the type of 'nonlin' (line 294)
    nonlin_201706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 14), 'nonlin', False)
    # Obtaining the member 'nonlin_solve' of a type (line 294)
    nonlin_solve_201707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 14), nonlin_201706, 'nonlin_solve')
    # Calling nonlin_solve(args, kwargs) (line 294)
    nonlin_solve_call_result_201740 = invoke(stypy.reporting.localization.Localization(__file__, 294, 14), nonlin_solve_201707, *[f_201708, x0_201709], **kwargs_201739)
    
    # Obtaining the member '__getitem__' of a type (line 294)
    getitem___201741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 4), nonlin_solve_call_result_201740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 294)
    subscript_call_result_201742 = invoke(stypy.reporting.localization.Localization(__file__, 294, 4), getitem___201741, int_201705)
    
    # Assigning a type to the variable 'tuple_var_assignment_201144' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'tuple_var_assignment_201144', subscript_call_result_201742)
    
    # Assigning a Name to a Name (line 294):
    # Getting the type of 'tuple_var_assignment_201143' (line 294)
    tuple_var_assignment_201143_201743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'tuple_var_assignment_201143')
    # Assigning a type to the variable 'x' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'x', tuple_var_assignment_201143_201743)
    
    # Assigning a Name to a Name (line 294):
    # Getting the type of 'tuple_var_assignment_201144' (line 294)
    tuple_var_assignment_201144_201744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'tuple_var_assignment_201144')
    # Assigning a type to the variable 'info' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 7), 'info', tuple_var_assignment_201144_201744)
    
    # Assigning a Call to a Name (line 302):
    
    # Assigning a Call to a Name (line 302):
    
    # Call to OptimizeResult(...): (line 302)
    # Processing the call keyword arguments (line 302)
    # Getting the type of 'x' (line 302)
    x_201746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 27), 'x', False)
    keyword_201747 = x_201746
    kwargs_201748 = {'x': keyword_201747}
    # Getting the type of 'OptimizeResult' (line 302)
    OptimizeResult_201745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 10), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 302)
    OptimizeResult_call_result_201749 = invoke(stypy.reporting.localization.Localization(__file__, 302, 10), OptimizeResult_201745, *[], **kwargs_201748)
    
    # Assigning a type to the variable 'sol' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'sol', OptimizeResult_call_result_201749)
    
    # Call to update(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'info' (line 303)
    info_201752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'info', False)
    # Processing the call keyword arguments (line 303)
    kwargs_201753 = {}
    # Getting the type of 'sol' (line 303)
    sol_201750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'sol', False)
    # Obtaining the member 'update' of a type (line 303)
    update_201751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 4), sol_201750, 'update')
    # Calling update(args, kwargs) (line 303)
    update_call_result_201754 = invoke(stypy.reporting.localization.Localization(__file__, 303, 4), update_201751, *[info_201752], **kwargs_201753)
    
    # Getting the type of 'sol' (line 304)
    sol_201755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 11), 'sol')
    # Assigning a type to the variable 'stypy_return_type' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type', sol_201755)
    
    # ################# End of '_root_nonlin_solve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_nonlin_solve' in the type store
    # Getting the type of 'stypy_return_type' (line 259)
    stypy_return_type_201756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201756)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_nonlin_solve'
    return stypy_return_type_201756

# Assigning a type to the variable '_root_nonlin_solve' (line 259)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), '_root_nonlin_solve', _root_nonlin_solve)

@norecursion
def _root_broyden1_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_root_broyden1_doc'
    module_type_store = module_type_store.open_function_context('_root_broyden1_doc', 306, 0, False)
    
    # Passed parameters checking function
    _root_broyden1_doc.stypy_localization = localization
    _root_broyden1_doc.stypy_type_of_self = None
    _root_broyden1_doc.stypy_type_store = module_type_store
    _root_broyden1_doc.stypy_function_name = '_root_broyden1_doc'
    _root_broyden1_doc.stypy_param_names_list = []
    _root_broyden1_doc.stypy_varargs_param_name = None
    _root_broyden1_doc.stypy_kwargs_param_name = None
    _root_broyden1_doc.stypy_call_defaults = defaults
    _root_broyden1_doc.stypy_call_varargs = varargs
    _root_broyden1_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_broyden1_doc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_broyden1_doc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_broyden1_doc(...)' code ##################

    str_201757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, (-1)), 'str', "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n            alpha : float, optional\n                Initial guess for the Jacobian is (-1/alpha).\n            reduction_method : str or tuple, optional\n                Method used in ensuring that the rank of the Broyden\n                matrix stays low. Can either be a string giving the\n                name of the method, or a tuple of the form ``(method,\n                param1, param2, ...)`` that gives the name of the\n                method and values for additional parameters.\n\n                Methods available:\n                    - ``restart``: drop all matrix columns. Has no\n                        extra parameters.\n                    - ``simple``: drop oldest matrix column. Has no\n                        extra parameters.\n                    - ``svd``: keep only the most significant SVD\n                        components.\n                      Extra parameters:\n                          - ``to_retain``: number of SVD components to\n                              retain when rank reduction is done.\n                              Default is ``max_rank - 2``.\n            max_rank : int, optional\n                Maximum rank for the Broyden matrix.\n                Default is infinity (ie., no rank reduction).\n    ")
    pass
    
    # ################# End of '_root_broyden1_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_broyden1_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 306)
    stypy_return_type_201758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201758)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_broyden1_doc'
    return stypy_return_type_201758

# Assigning a type to the variable '_root_broyden1_doc' (line 306)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), '_root_broyden1_doc', _root_broyden1_doc)

@norecursion
def _root_broyden2_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_root_broyden2_doc'
    module_type_store = module_type_store.open_function_context('_root_broyden2_doc', 363, 0, False)
    
    # Passed parameters checking function
    _root_broyden2_doc.stypy_localization = localization
    _root_broyden2_doc.stypy_type_of_self = None
    _root_broyden2_doc.stypy_type_store = module_type_store
    _root_broyden2_doc.stypy_function_name = '_root_broyden2_doc'
    _root_broyden2_doc.stypy_param_names_list = []
    _root_broyden2_doc.stypy_varargs_param_name = None
    _root_broyden2_doc.stypy_kwargs_param_name = None
    _root_broyden2_doc.stypy_call_defaults = defaults
    _root_broyden2_doc.stypy_call_varargs = varargs
    _root_broyden2_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_broyden2_doc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_broyden2_doc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_broyden2_doc(...)' code ##################

    str_201759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, (-1)), 'str', "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            Initial guess for the Jacobian is (-1/alpha).\n        reduction_method : str or tuple, optional\n            Method used in ensuring that the rank of the Broyden\n            matrix stays low. Can either be a string giving the\n            name of the method, or a tuple of the form ``(method,\n            param1, param2, ...)`` that gives the name of the\n            method and values for additional parameters.\n\n            Methods available:\n                - ``restart``: drop all matrix columns. Has no\n                    extra parameters.\n                - ``simple``: drop oldest matrix column. Has no\n                    extra parameters.\n                - ``svd``: keep only the most significant SVD\n                    components.\n                  Extra parameters:\n                      - ``to_retain``: number of SVD components to\n                          retain when rank reduction is done.\n                          Default is ``max_rank - 2``.\n        max_rank : int, optional\n            Maximum rank for the Broyden matrix.\n            Default is infinity (ie., no rank reduction).\n    ")
    pass
    
    # ################# End of '_root_broyden2_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_broyden2_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 363)
    stypy_return_type_201760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201760)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_broyden2_doc'
    return stypy_return_type_201760

# Assigning a type to the variable '_root_broyden2_doc' (line 363)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 0), '_root_broyden2_doc', _root_broyden2_doc)

@norecursion
def _root_anderson_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_root_anderson_doc'
    module_type_store = module_type_store.open_function_context('_root_anderson_doc', 421, 0, False)
    
    # Passed parameters checking function
    _root_anderson_doc.stypy_localization = localization
    _root_anderson_doc.stypy_type_of_self = None
    _root_anderson_doc.stypy_type_store = module_type_store
    _root_anderson_doc.stypy_function_name = '_root_anderson_doc'
    _root_anderson_doc.stypy_param_names_list = []
    _root_anderson_doc.stypy_varargs_param_name = None
    _root_anderson_doc.stypy_kwargs_param_name = None
    _root_anderson_doc.stypy_call_defaults = defaults
    _root_anderson_doc.stypy_call_varargs = varargs
    _root_anderson_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_anderson_doc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_anderson_doc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_anderson_doc(...)' code ##################

    str_201761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, (-1)), 'str', "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            Initial guess for the Jacobian is (-1/alpha).\n        M : float, optional\n            Number of previous vectors to retain. Defaults to 5.\n        w0 : float, optional\n            Regularization parameter for numerical stability.\n            Compared to unity, good values of the order of 0.01.\n    ")
    pass
    
    # ################# End of '_root_anderson_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_anderson_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 421)
    stypy_return_type_201762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201762)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_anderson_doc'
    return stypy_return_type_201762

# Assigning a type to the variable '_root_anderson_doc' (line 421)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 0), '_root_anderson_doc', _root_anderson_doc)

@norecursion
def _root_linearmixing_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_root_linearmixing_doc'
    module_type_store = module_type_store.open_function_context('_root_linearmixing_doc', 463, 0, False)
    
    # Passed parameters checking function
    _root_linearmixing_doc.stypy_localization = localization
    _root_linearmixing_doc.stypy_type_of_self = None
    _root_linearmixing_doc.stypy_type_store = module_type_store
    _root_linearmixing_doc.stypy_function_name = '_root_linearmixing_doc'
    _root_linearmixing_doc.stypy_param_names_list = []
    _root_linearmixing_doc.stypy_varargs_param_name = None
    _root_linearmixing_doc.stypy_kwargs_param_name = None
    _root_linearmixing_doc.stypy_call_defaults = defaults
    _root_linearmixing_doc.stypy_call_varargs = varargs
    _root_linearmixing_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_linearmixing_doc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_linearmixing_doc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_linearmixing_doc(...)' code ##################

    str_201763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, (-1)), 'str', "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, ``NoConvergence`` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            initial guess for the jacobian is (-1/alpha).\n    ")
    pass
    
    # ################# End of '_root_linearmixing_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_linearmixing_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 463)
    stypy_return_type_201764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201764)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_linearmixing_doc'
    return stypy_return_type_201764

# Assigning a type to the variable '_root_linearmixing_doc' (line 463)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 0), '_root_linearmixing_doc', _root_linearmixing_doc)

@norecursion
def _root_diagbroyden_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_root_diagbroyden_doc'
    module_type_store = module_type_store.open_function_context('_root_diagbroyden_doc', 500, 0, False)
    
    # Passed parameters checking function
    _root_diagbroyden_doc.stypy_localization = localization
    _root_diagbroyden_doc.stypy_type_of_self = None
    _root_diagbroyden_doc.stypy_type_store = module_type_store
    _root_diagbroyden_doc.stypy_function_name = '_root_diagbroyden_doc'
    _root_diagbroyden_doc.stypy_param_names_list = []
    _root_diagbroyden_doc.stypy_varargs_param_name = None
    _root_diagbroyden_doc.stypy_kwargs_param_name = None
    _root_diagbroyden_doc.stypy_call_defaults = defaults
    _root_diagbroyden_doc.stypy_call_varargs = varargs
    _root_diagbroyden_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_diagbroyden_doc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_diagbroyden_doc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_diagbroyden_doc(...)' code ##################

    str_201765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, (-1)), 'str', "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            initial guess for the jacobian is (-1/alpha).\n    ")
    pass
    
    # ################# End of '_root_diagbroyden_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_diagbroyden_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 500)
    stypy_return_type_201766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201766)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_diagbroyden_doc'
    return stypy_return_type_201766

# Assigning a type to the variable '_root_diagbroyden_doc' (line 500)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 0), '_root_diagbroyden_doc', _root_diagbroyden_doc)

@norecursion
def _root_excitingmixing_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_root_excitingmixing_doc'
    module_type_store = module_type_store.open_function_context('_root_excitingmixing_doc', 537, 0, False)
    
    # Passed parameters checking function
    _root_excitingmixing_doc.stypy_localization = localization
    _root_excitingmixing_doc.stypy_type_of_self = None
    _root_excitingmixing_doc.stypy_type_store = module_type_store
    _root_excitingmixing_doc.stypy_function_name = '_root_excitingmixing_doc'
    _root_excitingmixing_doc.stypy_param_names_list = []
    _root_excitingmixing_doc.stypy_varargs_param_name = None
    _root_excitingmixing_doc.stypy_kwargs_param_name = None
    _root_excitingmixing_doc.stypy_call_defaults = defaults
    _root_excitingmixing_doc.stypy_call_varargs = varargs
    _root_excitingmixing_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_excitingmixing_doc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_excitingmixing_doc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_excitingmixing_doc(...)' code ##################

    str_201767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, (-1)), 'str', "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            Initial Jacobian approximation is (-1/alpha).\n        alphamax : float, optional\n            The entries of the diagonal Jacobian are kept in the range\n            ``[alpha, alphamax]``.\n    ")
    pass
    
    # ################# End of '_root_excitingmixing_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_excitingmixing_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 537)
    stypy_return_type_201768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201768)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_excitingmixing_doc'
    return stypy_return_type_201768

# Assigning a type to the variable '_root_excitingmixing_doc' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), '_root_excitingmixing_doc', _root_excitingmixing_doc)

@norecursion
def _root_krylov_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_root_krylov_doc'
    module_type_store = module_type_store.open_function_context('_root_krylov_doc', 577, 0, False)
    
    # Passed parameters checking function
    _root_krylov_doc.stypy_localization = localization
    _root_krylov_doc.stypy_type_of_self = None
    _root_krylov_doc.stypy_type_store = module_type_store
    _root_krylov_doc.stypy_function_name = '_root_krylov_doc'
    _root_krylov_doc.stypy_param_names_list = []
    _root_krylov_doc.stypy_varargs_param_name = None
    _root_krylov_doc.stypy_kwargs_param_name = None
    _root_krylov_doc.stypy_call_defaults = defaults
    _root_krylov_doc.stypy_call_varargs = varargs
    _root_krylov_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_root_krylov_doc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_root_krylov_doc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_root_krylov_doc(...)' code ##################

    str_201769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, (-1)), 'str', '\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, \'armijo\' (default), \'wolfe\'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        \'armijo\'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        rdiff : float, optional\n            Relative step size to use in numerical differentiation.\n        method : {\'lgmres\', \'gmres\', \'bicgstab\', \'cgs\', \'minres\'} or function\n            Krylov method to use to approximate the Jacobian.\n            Can be a string, or a function implementing the same\n            interface as the iterative solvers in\n            `scipy.sparse.linalg`.\n\n            The default is `scipy.sparse.linalg.lgmres`.\n        inner_M : LinearOperator or InverseJacobian\n            Preconditioner for the inner Krylov iteration.\n            Note that you can use also inverse Jacobians as (adaptive)\n            preconditioners. For example,\n\n            >>> jac = BroydenFirst()\n            >>> kjac = KrylovJacobian(inner_M=jac.inverse).\n\n            If the preconditioner has a method named \'update\', it will\n            be called as ``update(x, f)`` after each nonlinear step,\n            with ``x`` giving the current point, and ``f`` the current\n            function value.\n        inner_tol, inner_maxiter, ...\n            Parameters to pass on to the "inner" Krylov solver.\n            See `scipy.sparse.linalg.gmres` for details.\n        outer_k : int, optional\n            Size of the subspace kept across LGMRES nonlinear\n            iterations.\n\n            See `scipy.sparse.linalg.lgmres` for details.\n    ')
    pass
    
    # ################# End of '_root_krylov_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_root_krylov_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 577)
    stypy_return_type_201770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201770)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_root_krylov_doc'
    return stypy_return_type_201770

# Assigning a type to the variable '_root_krylov_doc' (line 577)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 0), '_root_krylov_doc', _root_krylov_doc)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
