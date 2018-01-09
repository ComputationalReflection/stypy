
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unified interfaces to minimization algorithms.
3: 
4: Functions
5: ---------
6: - minimize : minimization of a function of several variables.
7: - minimize_scalar : minimization of a function of one variable.
8: '''
9: from __future__ import division, print_function, absolute_import
10: 
11: 
12: __all__ = ['minimize', 'minimize_scalar']
13: 
14: 
15: from warnings import warn
16: 
17: import numpy as np
18: 
19: from scipy._lib.six import callable
20: 
21: # unconstrained minimization
22: from .optimize import (_minimize_neldermead, _minimize_powell, _minimize_cg,
23:                       _minimize_bfgs, _minimize_newtoncg,
24:                       _minimize_scalar_brent, _minimize_scalar_bounded,
25:                       _minimize_scalar_golden, MemoizeJac)
26: from ._trustregion_dogleg import _minimize_dogleg
27: from ._trustregion_ncg import _minimize_trust_ncg
28: from ._trustregion_krylov import _minimize_trust_krylov
29: from ._trustregion_exact import _minimize_trustregion_exact
30: 
31: # constrained minimization
32: from .lbfgsb import _minimize_lbfgsb
33: from .tnc import _minimize_tnc
34: from .cobyla import _minimize_cobyla
35: from .slsqp import _minimize_slsqp
36: 
37: 
38: def minimize(fun, x0, args=(), method=None, jac=None, hess=None,
39:              hessp=None, bounds=None, constraints=(), tol=None,
40:              callback=None, options=None):
41:     '''Minimization of scalar function of one or more variables.
42: 
43:     In general, the optimization problems are of the form::
44: 
45:         minimize f(x) subject to
46: 
47:         g_i(x) >= 0,  i = 1,...,m
48:         h_j(x)  = 0,  j = 1,...,p
49: 
50:     where x is a vector of one or more variables.
51:     ``g_i(x)`` are the inequality constraints.
52:     ``h_j(x)`` are the equality constrains.
53: 
54:     Optionally, the lower and upper bounds for each element in x can also be
55:     specified using the `bounds` argument.
56: 
57:     Parameters
58:     ----------
59:     fun : callable
60:         The objective function to be minimized. Must be in the form
61:         ``f(x, *args)``. The optimizing argument, ``x``, is a 1-D array
62:         of points, and ``args`` is a tuple of any additional fixed parameters
63:         needed to completely specify the function.
64:     x0 : ndarray
65:         Initial guess. ``len(x0)`` is the dimensionality of the minimization
66:         problem.
67:     args : tuple, optional
68:         Extra arguments passed to the objective function and its
69:         derivatives (Jacobian, Hessian).
70:     method : str or callable, optional
71:         Type of solver.  Should be one of
72: 
73:             - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
74:             - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
75:             - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
76:             - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
77:             - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
78:             - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
79:             - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
80:             - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
81:             - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
82:             - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
83:             - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
84:             - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
85:             - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
86:             - custom - a callable object (added in version 0.14.0),
87:               see below for description.
88: 
89:         If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
90:         depending if the problem has constraints or bounds.
91:     jac : bool or callable, optional
92:         Jacobian (gradient) of objective function. Only for CG, BFGS,
93:         Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,
94:         trust-region-exact.
95:         If `jac` is a Boolean and is True, `fun` is assumed to return the
96:         gradient along with the objective function. If False, the
97:         gradient will be estimated numerically.
98:         `jac` can also be a callable returning the gradient of the
99:         objective. In this case, it must accept the same arguments as `fun`.
100:     hess, hessp : callable, optional
101:         Hessian (matrix of second-order derivatives) of objective function or
102:         Hessian of objective function times an arbitrary vector p.  Only for
103:         Newton-CG, dogleg, trust-ncg, trust-krylov, trust-region-exact.
104:         Only one of `hessp` or `hess` needs to be given.  If `hess` is
105:         provided, then `hessp` will be ignored.  If neither `hess` nor
106:         `hessp` is provided, then the Hessian product will be approximated
107:         using finite differences on `jac`. `hessp` must compute the Hessian
108:         times an arbitrary vector.
109:     bounds : sequence, optional
110:         Bounds for variables (only for L-BFGS-B, TNC and SLSQP).
111:         ``(min, max)`` pairs for each element in ``x``, defining
112:         the bounds on that parameter. Use None for one of ``min`` or
113:         ``max`` when there is no bound in that direction.
114:     constraints : dict or sequence of dict, optional
115:         Constraints definition (only for COBYLA and SLSQP).
116:         Each constraint is defined in a dictionary with fields:
117: 
118:             type : str
119:                 Constraint type: 'eq' for equality, 'ineq' for inequality.
120:             fun : callable
121:                 The function defining the constraint.
122:             jac : callable, optional
123:                 The Jacobian of `fun` (only for SLSQP).
124:             args : sequence, optional
125:                 Extra arguments to be passed to the function and Jacobian.
126: 
127:         Equality constraint means that the constraint function result is to
128:         be zero whereas inequality means that it is to be non-negative.
129:         Note that COBYLA only supports inequality constraints.
130:     tol : float, optional
131:         Tolerance for termination. For detailed control, use solver-specific
132:         options.
133:     options : dict, optional
134:         A dictionary of solver options. All methods accept the following
135:         generic options:
136: 
137:             maxiter : int
138:                 Maximum number of iterations to perform.
139:             disp : bool
140:                 Set to True to print convergence messages.
141: 
142:         For method-specific options, see :func:`show_options()`.
143:     callback : callable, optional
144:         Called after each iteration, as ``callback(xk)``, where ``xk`` is the
145:         current parameter vector.
146: 
147:     Returns
148:     -------
149:     res : OptimizeResult
150:         The optimization result represented as a ``OptimizeResult`` object.
151:         Important attributes are: ``x`` the solution array, ``success`` a
152:         Boolean flag indicating if the optimizer exited successfully and
153:         ``message`` which describes the cause of the termination. See
154:         `OptimizeResult` for a description of other attributes.
155: 
156: 
157:     See also
158:     --------
159:     minimize_scalar : Interface to minimization algorithms for scalar
160:         univariate functions
161:     show_options : Additional options accepted by the solvers
162: 
163:     Notes
164:     -----
165:     This section describes the available solvers that can be selected by the
166:     'method' parameter. The default method is *BFGS*.
167: 
168:     **Unconstrained minimization**
169: 
170:     Method :ref:`Nelder-Mead <optimize.minimize-neldermead>` uses the
171:     Simplex algorithm [1]_, [2]_. This algorithm is robust in many
172:     applications. However, if numerical computation of derivative can be
173:     trusted, other algorithms using the first and/or second derivatives
174:     information might be preferred for their better performance in
175:     general.
176: 
177:     Method :ref:`Powell <optimize.minimize-powell>` is a modification
178:     of Powell's method [3]_, [4]_ which is a conjugate direction
179:     method. It performs sequential one-dimensional minimizations along
180:     each vector of the directions set (`direc` field in `options` and
181:     `info`), which is updated at each iteration of the main
182:     minimization loop. The function need not be differentiable, and no
183:     derivatives are taken.
184: 
185:     Method :ref:`CG <optimize.minimize-cg>` uses a nonlinear conjugate
186:     gradient algorithm by Polak and Ribiere, a variant of the
187:     Fletcher-Reeves method described in [5]_ pp.  120-122. Only the
188:     first derivatives are used.
189: 
190:     Method :ref:`BFGS <optimize.minimize-bfgs>` uses the quasi-Newton
191:     method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS) [5]_
192:     pp. 136. It uses the first derivatives only. BFGS has proven good
193:     performance even for non-smooth optimizations. This method also
194:     returns an approximation of the Hessian inverse, stored as
195:     `hess_inv` in the OptimizeResult object.
196: 
197:     Method :ref:`Newton-CG <optimize.minimize-newtoncg>` uses a
198:     Newton-CG algorithm [5]_ pp. 168 (also known as the truncated
199:     Newton method). It uses a CG method to the compute the search
200:     direction. See also *TNC* method for a box-constrained
201:     minimization with a similar algorithm. Suitable for large-scale
202:     problems.
203: 
204:     Method :ref:`dogleg <optimize.minimize-dogleg>` uses the dog-leg
205:     trust-region algorithm [5]_ for unconstrained minimization. This
206:     algorithm requires the gradient and Hessian; furthermore the
207:     Hessian is required to be positive definite.
208: 
209:     Method :ref:`trust-ncg <optimize.minimize-trustncg>` uses the
210:     Newton conjugate gradient trust-region algorithm [5]_ for
211:     unconstrained minimization. This algorithm requires the gradient
212:     and either the Hessian or a function that computes the product of
213:     the Hessian with a given vector. Suitable for large-scale problems.
214: 
215:     Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` uses
216:     the Newton GLTR trust-region algorithm [14]_, [15]_ for unconstrained
217:     minimization. This algorithm requires the gradient
218:     and either the Hessian or a function that computes the product of
219:     the Hessian with a given vector. Suitable for large-scale problems.
220:     On indefinite problems it requires usually less iterations than the
221:     `trust-ncg` method and is recommended for medium and large-scale problems.
222: 
223:     Method :ref:`trust-exact <optimize.minimize-trustexact>`
224:     is a trust-region method for unconstrained minimization in which
225:     quadratic subproblems are solved almost exactly [13]_. This
226:     algorithm requires the gradient and the Hessian (which is
227:     *not* required to be positive definite). It is, in many
228:     situations, the Newton method to converge in fewer iteraction
229:     and the most recommended for small and medium-size problems.
230: 
231:     **Constrained minimization**
232: 
233:     Method :ref:`L-BFGS-B <optimize.minimize-lbfgsb>` uses the L-BFGS-B
234:     algorithm [6]_, [7]_ for bound constrained minimization.
235: 
236:     Method :ref:`TNC <optimize.minimize-tnc>` uses a truncated Newton
237:     algorithm [5]_, [8]_ to minimize a function with variables subject
238:     to bounds. This algorithm uses gradient information; it is also
239:     called Newton Conjugate-Gradient. It differs from the *Newton-CG*
240:     method described above as it wraps a C implementation and allows
241:     each variable to be given upper and lower bounds.
242: 
243:     Method :ref:`COBYLA <optimize.minimize-cobyla>` uses the
244:     Constrained Optimization BY Linear Approximation (COBYLA) method
245:     [9]_, [10]_, [11]_. The algorithm is based on linear
246:     approximations to the objective function and each constraint. The
247:     method wraps a FORTRAN implementation of the algorithm. The
248:     constraints functions 'fun' may return either a single number
249:     or an array or list of numbers.
250: 
251:     Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential
252:     Least SQuares Programming to minimize a function of several
253:     variables with any combination of bounds, equality and inequality
254:     constraints. The method wraps the SLSQP Optimization subroutine
255:     originally implemented by Dieter Kraft [12]_. Note that the
256:     wrapper handles infinite values in bounds by converting them into
257:     large floating values.
258: 
259:     **Custom minimizers**
260: 
261:     It may be useful to pass a custom minimization method, for example
262:     when using a frontend to this method such as `scipy.optimize.basinhopping`
263:     or a different library.  You can simply pass a callable as the ``method``
264:     parameter.
265: 
266:     The callable is called as ``method(fun, x0, args, **kwargs, **options)``
267:     where ``kwargs`` corresponds to any other parameters passed to `minimize`
268:     (such as `callback`, `hess`, etc.), except the `options` dict, which has
269:     its contents also passed as `method` parameters pair by pair.  Also, if
270:     `jac` has been passed as a bool type, `jac` and `fun` are mangled so that
271:     `fun` returns just the function values and `jac` is converted to a function
272:     returning the Jacobian.  The method shall return an ``OptimizeResult``
273:     object.
274: 
275:     The provided `method` callable must be able to accept (and possibly ignore)
276:     arbitrary parameters; the set of parameters accepted by `minimize` may
277:     expand in future versions and then these parameters will be passed to
278:     the method.  You can find an example in the scipy.optimize tutorial.
279: 
280:     .. versionadded:: 0.11.0
281: 
282:     References
283:     ----------
284:     .. [1] Nelder, J A, and R Mead. 1965. A Simplex Method for Function
285:         Minimization. The Computer Journal 7: 308-13.
286:     .. [2] Wright M H. 1996. Direct search methods: Once scorned, now
287:         respectable, in Numerical Analysis 1995: Proceedings of the 1995
288:         Dundee Biennial Conference in Numerical Analysis (Eds. D F
289:         Griffiths and G A Watson). Addison Wesley Longman, Harlow, UK.
290:         191-208.
291:     .. [3] Powell, M J D. 1964. An efficient method for finding the minimum of
292:        a function of several variables without calculating derivatives. The
293:        Computer Journal 7: 155-162.
294:     .. [4] Press W, S A Teukolsky, W T Vetterling and B P Flannery.
295:        Numerical Recipes (any edition), Cambridge University Press.
296:     .. [5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.
297:        Springer New York.
298:     .. [6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory
299:        Algorithm for Bound Constrained Optimization. SIAM Journal on
300:        Scientific and Statistical Computing 16 (5): 1190-1208.
301:     .. [7] Zhu, C and R H Byrd and J Nocedal. 1997. L-BFGS-B: Algorithm
302:        778: L-BFGS-B, FORTRAN routines for large scale bound constrained
303:        optimization. ACM Transactions on Mathematical Software 23 (4):
304:        550-560.
305:     .. [8] Nash, S G. Newton-Type Minimization Via the Lanczos Method.
306:        1984. SIAM Journal of Numerical Analysis 21: 770-778.
307:     .. [9] Powell, M J D. A direct search optimization method that models
308:        the objective and constraint functions by linear interpolation.
309:        1994. Advances in Optimization and Numerical Analysis, eds. S. Gomez
310:        and J-P Hennart, Kluwer Academic (Dordrecht), 51-67.
311:     .. [10] Powell M J D. Direct search algorithms for optimization
312:        calculations. 1998. Acta Numerica 7: 287-336.
313:     .. [11] Powell M J D. A view of algorithms for optimization without
314:        derivatives. 2007.Cambridge University Technical Report DAMTP
315:        2007/NA03
316:     .. [12] Kraft, D. A software package for sequential quadratic
317:        programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace
318:        Center -- Institute for Flight Mechanics, Koln, Germany.
319:     .. [13] Conn, A. R., Gould, N. I., and Toint, P. L.
320:        Trust region methods. 2000. Siam. pp. 169-200.
321:     .. [14] F. Lenders, C. Kirches, A. Potschka: "trlib: A vector-free
322:        implementation of the GLTR method for iterative solution of
323:        the trust region problem", https://arxiv.org/abs/1611.04718
324:     .. [15] N. Gould, S. Lucidi, M. Roma, P. Toint: "Solving the
325:        Trust-Region Subproblem using the Lanczos Method",
326:        SIAM J. Optim., 9(2), 504--525, (1999).
327: 
328:     Examples
329:     --------
330:     Let us consider the problem of minimizing the Rosenbrock function. This
331:     function (and its respective derivatives) is implemented in `rosen`
332:     (resp. `rosen_der`, `rosen_hess`) in the `scipy.optimize`.
333: 
334:     >>> from scipy.optimize import minimize, rosen, rosen_der
335: 
336:     A simple application of the *Nelder-Mead* method is:
337: 
338:     >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
339:     >>> res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
340:     >>> res.x
341:     array([ 1.,  1.,  1.,  1.,  1.])
342: 
343:     Now using the *BFGS* algorithm, using the first derivative and a few
344:     options:
345: 
346:     >>> res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
347:     ...                options={'gtol': 1e-6, 'disp': True})
348:     Optimization terminated successfully.
349:              Current function value: 0.000000
350:              Iterations: 26
351:              Function evaluations: 31
352:              Gradient evaluations: 31
353:     >>> res.x
354:     array([ 1.,  1.,  1.,  1.,  1.])
355:     >>> print(res.message)
356:     Optimization terminated successfully.
357:     >>> res.hess_inv
358:     array([[ 0.00749589,  0.01255155,  0.02396251,  0.04750988,  0.09495377],  # may vary
359:            [ 0.01255155,  0.02510441,  0.04794055,  0.09502834,  0.18996269],
360:            [ 0.02396251,  0.04794055,  0.09631614,  0.19092151,  0.38165151],
361:            [ 0.04750988,  0.09502834,  0.19092151,  0.38341252,  0.7664427 ],
362:            [ 0.09495377,  0.18996269,  0.38165151,  0.7664427,   1.53713523]])
363: 
364: 
365:     Next, consider a minimization problem with several constraints (namely
366:     Example 16.4 from [5]_). The objective function is:
367: 
368:     >>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
369: 
370:     There are three constraints defined as:
371: 
372:     >>> cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
373:     ...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
374:     ...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
375: 
376:     And variables must be positive, hence the following bounds:
377: 
378:     >>> bnds = ((0, None), (0, None))
379: 
380:     The optimization problem is solved using the SLSQP method as:
381: 
382:     >>> res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
383:     ...                constraints=cons)
384: 
385:     It should converge to the theoretical solution (1.4 ,1.7).
386: 
387:     '''
388:     x0 = np.asarray(x0)
389:     if x0.dtype.kind in np.typecodes["AllInteger"]:
390:         x0 = np.asarray(x0, dtype=float)
391: 
392:     if not isinstance(args, tuple):
393:         args = (args,)
394: 
395:     if method is None:
396:         # Select automatically
397:         if constraints:
398:             method = 'SLSQP'
399:         elif bounds is not None:
400:             method = 'L-BFGS-B'
401:         else:
402:             method = 'BFGS'
403: 
404:     if callable(method):
405:         meth = "_custom"
406:     else:
407:         meth = method.lower()
408: 
409:     if options is None:
410:         options = {}
411:     # check if optional parameters are supported by the selected method
412:     # - jac
413:     if meth in ['nelder-mead', 'powell', 'cobyla'] and bool(jac):
414:         warn('Method %s does not use gradient information (jac).' % method,
415:              RuntimeWarning)
416:     # - hess
417:     if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov',
418:                     'trust-exact', '_custom') and hess is not None:
419:         warn('Method %s does not use Hessian information (hess).' % method,
420:              RuntimeWarning)
421:     # - hessp
422:     if meth not in ('newton-cg', 'dogleg', 'trust-ncg',
423:                     'trust-krylov', '_custom') and hessp is not None:
424:         warn('Method %s does not use Hessian-vector product '
425:                 'information (hessp).' % method, RuntimeWarning)
426:     # - constraints or bounds
427:     if (meth in ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'dogleg',
428:                  'trust-ncg'] and (bounds is not None or np.any(constraints))):
429:         warn('Method %s cannot handle constraints nor bounds.' % method,
430:              RuntimeWarning)
431:     if meth in ['l-bfgs-b', 'tnc'] and np.any(constraints):
432:         warn('Method %s cannot handle constraints.' % method,
433:              RuntimeWarning)
434:     if meth == 'cobyla' and bounds is not None:
435:         warn('Method %s cannot handle bounds.' % method,
436:              RuntimeWarning)
437:     # - callback
438:     if (meth in ['cobyla'] and callback is not None):
439:         warn('Method %s does not support callback.' % method, RuntimeWarning)
440:     # - return_all
441:     if (meth in ['l-bfgs-b', 'tnc', 'cobyla', 'slsqp'] and
442:             options.get('return_all', False)):
443:         warn('Method %s does not support the return_all option.' % method,
444:              RuntimeWarning)
445: 
446:     # fun also returns the jacobian
447:     if not callable(jac):
448:         if bool(jac):
449:             fun = MemoizeJac(fun)
450:             jac = fun.derivative
451:         else:
452:             jac = None
453: 
454:     # set default tolerances
455:     if tol is not None:
456:         options = dict(options)
457:         if meth == 'nelder-mead':
458:             options.setdefault('xatol', tol)
459:             options.setdefault('fatol', tol)
460:         if meth in ['newton-cg', 'powell', 'tnc']:
461:             options.setdefault('xtol', tol)
462:         if meth in ['powell', 'l-bfgs-b', 'tnc', 'slsqp']:
463:             options.setdefault('ftol', tol)
464:         if meth in ['bfgs', 'cg', 'l-bfgs-b', 'tnc', 'dogleg',
465:                     'trust-ncg', 'trust-exact', 'trust-krylov']:
466:             options.setdefault('gtol', tol)
467:         if meth in ['cobyla', '_custom']:
468:             options.setdefault('tol', tol)
469: 
470:     if meth == '_custom':
471:         return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
472:                       bounds=bounds, constraints=constraints,
473:                       callback=callback, **options)
474:     elif meth == 'nelder-mead':
475:         return _minimize_neldermead(fun, x0, args, callback, **options)
476:     elif meth == 'powell':
477:         return _minimize_powell(fun, x0, args, callback, **options)
478:     elif meth == 'cg':
479:         return _minimize_cg(fun, x0, args, jac, callback, **options)
480:     elif meth == 'bfgs':
481:         return _minimize_bfgs(fun, x0, args, jac, callback, **options)
482:     elif meth == 'newton-cg':
483:         return _minimize_newtoncg(fun, x0, args, jac, hess, hessp, callback,
484:                                   **options)
485:     elif meth == 'l-bfgs-b':
486:         return _minimize_lbfgsb(fun, x0, args, jac, bounds,
487:                                 callback=callback, **options)
488:     elif meth == 'tnc':
489:         return _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,
490:                              **options)
491:     elif meth == 'cobyla':
492:         return _minimize_cobyla(fun, x0, args, constraints, **options)
493:     elif meth == 'slsqp':
494:         return _minimize_slsqp(fun, x0, args, jac, bounds,
495:                                constraints, callback=callback, **options)
496:     elif meth == 'dogleg':
497:         return _minimize_dogleg(fun, x0, args, jac, hess,
498:                                 callback=callback, **options)
499:     elif meth == 'trust-ncg':
500:         return _minimize_trust_ncg(fun, x0, args, jac, hess, hessp,
501:                                    callback=callback, **options)
502:     elif meth == 'trust-krylov':
503:         return _minimize_trust_krylov(fun, x0, args, jac, hess, hessp,
504:                                       callback=callback, **options)
505:     elif meth == 'trust-exact':
506:         return _minimize_trustregion_exact(fun, x0, args, jac, hess,
507:                                            callback=callback, **options)
508:     else:
509:         raise ValueError('Unknown solver %s' % method)
510: 
511: 
512: def minimize_scalar(fun, bracket=None, bounds=None, args=(),
513:                     method='brent', tol=None, options=None):
514:     '''Minimization of scalar function of one variable.
515: 
516:     Parameters
517:     ----------
518:     fun : callable
519:         Objective function.
520:         Scalar function, must return a scalar.
521:     bracket : sequence, optional
522:         For methods 'brent' and 'golden', `bracket` defines the bracketing
523:         interval and can either have three items ``(a, b, c)`` so that
524:         ``a < b < c`` and ``fun(b) < fun(a), fun(c)`` or two items ``a`` and
525:         ``c`` which are assumed to be a starting interval for a downhill
526:         bracket search (see `bracket`); it doesn't always mean that the
527:         obtained solution will satisfy ``a <= x <= c``.
528:     bounds : sequence, optional
529:         For method 'bounded', `bounds` is mandatory and must have two items
530:         corresponding to the optimization bounds.
531:     args : tuple, optional
532:         Extra arguments passed to the objective function.
533:     method : str or callable, optional
534:         Type of solver.  Should be one of:
535: 
536:             - 'Brent'     :ref:`(see here) <optimize.minimize_scalar-brent>`
537:             - 'Bounded'   :ref:`(see here) <optimize.minimize_scalar-bounded>`
538:             - 'Golden'    :ref:`(see here) <optimize.minimize_scalar-golden>`
539:             - custom - a callable object (added in version 0.14.0), see below
540: 
541:     tol : float, optional
542:         Tolerance for termination. For detailed control, use solver-specific
543:         options.
544:     options : dict, optional
545:         A dictionary of solver options.
546: 
547:             maxiter : int
548:                 Maximum number of iterations to perform.
549:             disp : bool
550:                 Set to True to print convergence messages.
551: 
552:         See :func:`show_options()` for solver-specific options.
553: 
554:     Returns
555:     -------
556:     res : OptimizeResult
557:         The optimization result represented as a ``OptimizeResult`` object.
558:         Important attributes are: ``x`` the solution array, ``success`` a
559:         Boolean flag indicating if the optimizer exited successfully and
560:         ``message`` which describes the cause of the termination. See
561:         `OptimizeResult` for a description of other attributes.
562: 
563:     See also
564:     --------
565:     minimize : Interface to minimization algorithms for scalar multivariate
566:         functions
567:     show_options : Additional options accepted by the solvers
568: 
569:     Notes
570:     -----
571:     This section describes the available solvers that can be selected by the
572:     'method' parameter. The default method is *Brent*.
573: 
574:     Method :ref:`Brent <optimize.minimize_scalar-brent>` uses Brent's
575:     algorithm to find a local minimum.  The algorithm uses inverse
576:     parabolic interpolation when possible to speed up convergence of
577:     the golden section method.
578: 
579:     Method :ref:`Golden <optimize.minimize_scalar-golden>` uses the
580:     golden section search technique. It uses analog of the bisection
581:     method to decrease the bracketed interval. It is usually
582:     preferable to use the *Brent* method.
583: 
584:     Method :ref:`Bounded <optimize.minimize_scalar-bounded>` can
585:     perform bounded minimization. It uses the Brent method to find a
586:     local minimum in the interval x1 < xopt < x2.
587: 
588:     **Custom minimizers**
589: 
590:     It may be useful to pass a custom minimization method, for example
591:     when using some library frontend to minimize_scalar.  You can simply
592:     pass a callable as the ``method`` parameter.
593: 
594:     The callable is called as ``method(fun, args, **kwargs, **options)``
595:     where ``kwargs`` corresponds to any other parameters passed to `minimize`
596:     (such as `bracket`, `tol`, etc.), except the `options` dict, which has
597:     its contents also passed as `method` parameters pair by pair.  The method
598:     shall return an ``OptimizeResult`` object.
599: 
600:     The provided `method` callable must be able to accept (and possibly ignore)
601:     arbitrary parameters; the set of parameters accepted by `minimize` may
602:     expand in future versions and then these parameters will be passed to
603:     the method.  You can find an example in the scipy.optimize tutorial.
604: 
605:     .. versionadded:: 0.11.0
606: 
607:     Examples
608:     --------
609:     Consider the problem of minimizing the following function.
610: 
611:     >>> def f(x):
612:     ...     return (x - 2) * x * (x + 2)**2
613: 
614:     Using the *Brent* method, we find the local minimum as:
615: 
616:     >>> from scipy.optimize import minimize_scalar
617:     >>> res = minimize_scalar(f)
618:     >>> res.x
619:     1.28077640403
620: 
621:     Using the *Bounded* method, we find a local minimum with specified
622:     bounds as:
623: 
624:     >>> res = minimize_scalar(f, bounds=(-3, -1), method='bounded')
625:     >>> res.x
626:     -2.0000002026
627: 
628:     '''
629:     if not isinstance(args, tuple):
630:         args = (args,)
631: 
632:     if callable(method):
633:         meth = "_custom"
634:     else:
635:         meth = method.lower()
636:     if options is None:
637:         options = {}
638: 
639:     if tol is not None:
640:         options = dict(options)
641:         if meth == 'bounded' and 'xatol' not in options:
642:             warn("Method 'bounded' does not support relative tolerance in x; "
643:                  "defaulting to absolute tolerance.", RuntimeWarning)
644:             options['xatol'] = tol
645:         elif meth == '_custom':
646:             options.setdefault('tol', tol)
647:         else:
648:             options.setdefault('xtol', tol)
649: 
650:     if meth == '_custom':
651:         return method(fun, args=args, bracket=bracket, bounds=bounds, **options)
652:     elif meth == 'brent':
653:         return _minimize_scalar_brent(fun, bracket, args, **options)
654:     elif meth == 'bounded':
655:         if bounds is None:
656:             raise ValueError('The `bounds` parameter is mandatory for '
657:                              'method `bounded`.')
658:         return _minimize_scalar_bounded(fun, bounds, args, **options)
659:     elif meth == 'golden':
660:         return _minimize_scalar_golden(fun, bracket, args, **options)
661:     else:
662:         raise ValueError('Unknown solver %s' % method)
663: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_197898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nUnified interfaces to minimization algorithms.\n\nFunctions\n---------\n- minimize : minimization of a function of several variables.\n- minimize_scalar : minimization of a function of one variable.\n')

# Assigning a List to a Name (line 12):
__all__ = ['minimize', 'minimize_scalar']
module_type_store.set_exportable_members(['minimize', 'minimize_scalar'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_197899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_197900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'minimize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_197899, str_197900)
# Adding element type (line 12)
str_197901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'str', 'minimize_scalar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_197899, str_197901)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_197899)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from warnings import warn' statement (line 15)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import numpy' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197902 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy')

if (type(import_197902) is not StypyTypeError):

    if (import_197902 != 'pyd_module'):
        __import__(import_197902)
        sys_modules_197903 = sys.modules[import_197902]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'np', sys_modules_197903.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy', import_197902)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy._lib.six import callable' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197904 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib.six')

if (type(import_197904) is not StypyTypeError):

    if (import_197904 != 'pyd_module'):
        __import__(import_197904)
        sys_modules_197905 = sys.modules[import_197904]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib.six', sys_modules_197905.module_type_store, module_type_store, ['callable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_197905, sys_modules_197905.module_type_store, module_type_store)
    else:
        from scipy._lib.six import callable

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib.six', None, module_type_store, ['callable'], [callable])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib.six', import_197904)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from scipy.optimize.optimize import _minimize_neldermead, _minimize_powell, _minimize_cg, _minimize_bfgs, _minimize_newtoncg, _minimize_scalar_brent, _minimize_scalar_bounded, _minimize_scalar_golden, MemoizeJac' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197906 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.optimize.optimize')

if (type(import_197906) is not StypyTypeError):

    if (import_197906 != 'pyd_module'):
        __import__(import_197906)
        sys_modules_197907 = sys.modules[import_197906]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.optimize.optimize', sys_modules_197907.module_type_store, module_type_store, ['_minimize_neldermead', '_minimize_powell', '_minimize_cg', '_minimize_bfgs', '_minimize_newtoncg', '_minimize_scalar_brent', '_minimize_scalar_bounded', '_minimize_scalar_golden', 'MemoizeJac'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_197907, sys_modules_197907.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import _minimize_neldermead, _minimize_powell, _minimize_cg, _minimize_bfgs, _minimize_newtoncg, _minimize_scalar_brent, _minimize_scalar_bounded, _minimize_scalar_golden, MemoizeJac

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.optimize.optimize', None, module_type_store, ['_minimize_neldermead', '_minimize_powell', '_minimize_cg', '_minimize_bfgs', '_minimize_newtoncg', '_minimize_scalar_brent', '_minimize_scalar_bounded', '_minimize_scalar_golden', 'MemoizeJac'], [_minimize_neldermead, _minimize_powell, _minimize_cg, _minimize_bfgs, _minimize_newtoncg, _minimize_scalar_brent, _minimize_scalar_bounded, _minimize_scalar_golden, MemoizeJac])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.optimize.optimize', import_197906)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy.optimize._trustregion_dogleg import _minimize_dogleg' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197908 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.optimize._trustregion_dogleg')

if (type(import_197908) is not StypyTypeError):

    if (import_197908 != 'pyd_module'):
        __import__(import_197908)
        sys_modules_197909 = sys.modules[import_197908]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.optimize._trustregion_dogleg', sys_modules_197909.module_type_store, module_type_store, ['_minimize_dogleg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_197909, sys_modules_197909.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion_dogleg import _minimize_dogleg

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.optimize._trustregion_dogleg', None, module_type_store, ['_minimize_dogleg'], [_minimize_dogleg])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion_dogleg' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.optimize._trustregion_dogleg', import_197908)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from scipy.optimize._trustregion_ncg import _minimize_trust_ncg' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197910 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.optimize._trustregion_ncg')

if (type(import_197910) is not StypyTypeError):

    if (import_197910 != 'pyd_module'):
        __import__(import_197910)
        sys_modules_197911 = sys.modules[import_197910]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.optimize._trustregion_ncg', sys_modules_197911.module_type_store, module_type_store, ['_minimize_trust_ncg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_197911, sys_modules_197911.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion_ncg import _minimize_trust_ncg

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.optimize._trustregion_ncg', None, module_type_store, ['_minimize_trust_ncg'], [_minimize_trust_ncg])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion_ncg' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.optimize._trustregion_ncg', import_197910)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from scipy.optimize._trustregion_krylov import _minimize_trust_krylov' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197912 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.optimize._trustregion_krylov')

if (type(import_197912) is not StypyTypeError):

    if (import_197912 != 'pyd_module'):
        __import__(import_197912)
        sys_modules_197913 = sys.modules[import_197912]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.optimize._trustregion_krylov', sys_modules_197913.module_type_store, module_type_store, ['_minimize_trust_krylov'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_197913, sys_modules_197913.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion_krylov import _minimize_trust_krylov

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.optimize._trustregion_krylov', None, module_type_store, ['_minimize_trust_krylov'], [_minimize_trust_krylov])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion_krylov' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.optimize._trustregion_krylov', import_197912)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from scipy.optimize._trustregion_exact import _minimize_trustregion_exact' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197914 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.optimize._trustregion_exact')

if (type(import_197914) is not StypyTypeError):

    if (import_197914 != 'pyd_module'):
        __import__(import_197914)
        sys_modules_197915 = sys.modules[import_197914]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.optimize._trustregion_exact', sys_modules_197915.module_type_store, module_type_store, ['_minimize_trustregion_exact'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_197915, sys_modules_197915.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion_exact import _minimize_trustregion_exact

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.optimize._trustregion_exact', None, module_type_store, ['_minimize_trustregion_exact'], [_minimize_trustregion_exact])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion_exact' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.optimize._trustregion_exact', import_197914)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from scipy.optimize.lbfgsb import _minimize_lbfgsb' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197916 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.optimize.lbfgsb')

if (type(import_197916) is not StypyTypeError):

    if (import_197916 != 'pyd_module'):
        __import__(import_197916)
        sys_modules_197917 = sys.modules[import_197916]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.optimize.lbfgsb', sys_modules_197917.module_type_store, module_type_store, ['_minimize_lbfgsb'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_197917, sys_modules_197917.module_type_store, module_type_store)
    else:
        from scipy.optimize.lbfgsb import _minimize_lbfgsb

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.optimize.lbfgsb', None, module_type_store, ['_minimize_lbfgsb'], [_minimize_lbfgsb])

else:
    # Assigning a type to the variable 'scipy.optimize.lbfgsb' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.optimize.lbfgsb', import_197916)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from scipy.optimize.tnc import _minimize_tnc' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197918 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'scipy.optimize.tnc')

if (type(import_197918) is not StypyTypeError):

    if (import_197918 != 'pyd_module'):
        __import__(import_197918)
        sys_modules_197919 = sys.modules[import_197918]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'scipy.optimize.tnc', sys_modules_197919.module_type_store, module_type_store, ['_minimize_tnc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_197919, sys_modules_197919.module_type_store, module_type_store)
    else:
        from scipy.optimize.tnc import _minimize_tnc

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'scipy.optimize.tnc', None, module_type_store, ['_minimize_tnc'], [_minimize_tnc])

else:
    # Assigning a type to the variable 'scipy.optimize.tnc' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'scipy.optimize.tnc', import_197918)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from scipy.optimize.cobyla import _minimize_cobyla' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197920 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.optimize.cobyla')

if (type(import_197920) is not StypyTypeError):

    if (import_197920 != 'pyd_module'):
        __import__(import_197920)
        sys_modules_197921 = sys.modules[import_197920]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.optimize.cobyla', sys_modules_197921.module_type_store, module_type_store, ['_minimize_cobyla'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_197921, sys_modules_197921.module_type_store, module_type_store)
    else:
        from scipy.optimize.cobyla import _minimize_cobyla

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.optimize.cobyla', None, module_type_store, ['_minimize_cobyla'], [_minimize_cobyla])

else:
    # Assigning a type to the variable 'scipy.optimize.cobyla' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.optimize.cobyla', import_197920)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from scipy.optimize.slsqp import _minimize_slsqp' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_197922 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.optimize.slsqp')

if (type(import_197922) is not StypyTypeError):

    if (import_197922 != 'pyd_module'):
        __import__(import_197922)
        sys_modules_197923 = sys.modules[import_197922]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.optimize.slsqp', sys_modules_197923.module_type_store, module_type_store, ['_minimize_slsqp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_197923, sys_modules_197923.module_type_store, module_type_store)
    else:
        from scipy.optimize.slsqp import _minimize_slsqp

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.optimize.slsqp', None, module_type_store, ['_minimize_slsqp'], [_minimize_slsqp])

else:
    # Assigning a type to the variable 'scipy.optimize.slsqp' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.optimize.slsqp', import_197922)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


@norecursion
def minimize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_197924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    
    # Getting the type of 'None' (line 38)
    None_197925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 38), 'None')
    # Getting the type of 'None' (line 38)
    None_197926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 48), 'None')
    # Getting the type of 'None' (line 38)
    None_197927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 59), 'None')
    # Getting the type of 'None' (line 39)
    None_197928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'None')
    # Getting the type of 'None' (line 39)
    None_197929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 32), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 39)
    tuple_197930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 39)
    
    # Getting the type of 'None' (line 39)
    None_197931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 58), 'None')
    # Getting the type of 'None' (line 40)
    None_197932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'None')
    # Getting the type of 'None' (line 40)
    None_197933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'None')
    defaults = [tuple_197924, None_197925, None_197926, None_197927, None_197928, None_197929, tuple_197930, None_197931, None_197932, None_197933]
    # Create a new context for function 'minimize'
    module_type_store = module_type_store.open_function_context('minimize', 38, 0, False)
    
    # Passed parameters checking function
    minimize.stypy_localization = localization
    minimize.stypy_type_of_self = None
    minimize.stypy_type_store = module_type_store
    minimize.stypy_function_name = 'minimize'
    minimize.stypy_param_names_list = ['fun', 'x0', 'args', 'method', 'jac', 'hess', 'hessp', 'bounds', 'constraints', 'tol', 'callback', 'options']
    minimize.stypy_varargs_param_name = None
    minimize.stypy_kwargs_param_name = None
    minimize.stypy_call_defaults = defaults
    minimize.stypy_call_varargs = varargs
    minimize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minimize', ['fun', 'x0', 'args', 'method', 'jac', 'hess', 'hessp', 'bounds', 'constraints', 'tol', 'callback', 'options'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minimize', localization, ['fun', 'x0', 'args', 'method', 'jac', 'hess', 'hessp', 'bounds', 'constraints', 'tol', 'callback', 'options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minimize(...)' code ##################

    str_197934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, (-1)), 'str', 'Minimization of scalar function of one or more variables.\n\n    In general, the optimization problems are of the form::\n\n        minimize f(x) subject to\n\n        g_i(x) >= 0,  i = 1,...,m\n        h_j(x)  = 0,  j = 1,...,p\n\n    where x is a vector of one or more variables.\n    ``g_i(x)`` are the inequality constraints.\n    ``h_j(x)`` are the equality constrains.\n\n    Optionally, the lower and upper bounds for each element in x can also be\n    specified using the `bounds` argument.\n\n    Parameters\n    ----------\n    fun : callable\n        The objective function to be minimized. Must be in the form\n        ``f(x, *args)``. The optimizing argument, ``x``, is a 1-D array\n        of points, and ``args`` is a tuple of any additional fixed parameters\n        needed to completely specify the function.\n    x0 : ndarray\n        Initial guess. ``len(x0)`` is the dimensionality of the minimization\n        problem.\n    args : tuple, optional\n        Extra arguments passed to the objective function and its\n        derivatives (Jacobian, Hessian).\n    method : str or callable, optional\n        Type of solver.  Should be one of\n\n            - \'Nelder-Mead\' :ref:`(see here) <optimize.minimize-neldermead>`\n            - \'Powell\'      :ref:`(see here) <optimize.minimize-powell>`\n            - \'CG\'          :ref:`(see here) <optimize.minimize-cg>`\n            - \'BFGS\'        :ref:`(see here) <optimize.minimize-bfgs>`\n            - \'Newton-CG\'   :ref:`(see here) <optimize.minimize-newtoncg>`\n            - \'L-BFGS-B\'    :ref:`(see here) <optimize.minimize-lbfgsb>`\n            - \'TNC\'         :ref:`(see here) <optimize.minimize-tnc>`\n            - \'COBYLA\'      :ref:`(see here) <optimize.minimize-cobyla>`\n            - \'SLSQP\'       :ref:`(see here) <optimize.minimize-slsqp>`\n            - \'dogleg\'      :ref:`(see here) <optimize.minimize-dogleg>`\n            - \'trust-ncg\'   :ref:`(see here) <optimize.minimize-trustncg>`\n            - \'trust-exact\' :ref:`(see here) <optimize.minimize-trustexact>`\n            - \'trust-krylov\' :ref:`(see here) <optimize.minimize-trustkrylov>`\n            - custom - a callable object (added in version 0.14.0),\n              see below for description.\n\n        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,\n        depending if the problem has constraints or bounds.\n    jac : bool or callable, optional\n        Jacobian (gradient) of objective function. Only for CG, BFGS,\n        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,\n        trust-region-exact.\n        If `jac` is a Boolean and is True, `fun` is assumed to return the\n        gradient along with the objective function. If False, the\n        gradient will be estimated numerically.\n        `jac` can also be a callable returning the gradient of the\n        objective. In this case, it must accept the same arguments as `fun`.\n    hess, hessp : callable, optional\n        Hessian (matrix of second-order derivatives) of objective function or\n        Hessian of objective function times an arbitrary vector p.  Only for\n        Newton-CG, dogleg, trust-ncg, trust-krylov, trust-region-exact.\n        Only one of `hessp` or `hess` needs to be given.  If `hess` is\n        provided, then `hessp` will be ignored.  If neither `hess` nor\n        `hessp` is provided, then the Hessian product will be approximated\n        using finite differences on `jac`. `hessp` must compute the Hessian\n        times an arbitrary vector.\n    bounds : sequence, optional\n        Bounds for variables (only for L-BFGS-B, TNC and SLSQP).\n        ``(min, max)`` pairs for each element in ``x``, defining\n        the bounds on that parameter. Use None for one of ``min`` or\n        ``max`` when there is no bound in that direction.\n    constraints : dict or sequence of dict, optional\n        Constraints definition (only for COBYLA and SLSQP).\n        Each constraint is defined in a dictionary with fields:\n\n            type : str\n                Constraint type: \'eq\' for equality, \'ineq\' for inequality.\n            fun : callable\n                The function defining the constraint.\n            jac : callable, optional\n                The Jacobian of `fun` (only for SLSQP).\n            args : sequence, optional\n                Extra arguments to be passed to the function and Jacobian.\n\n        Equality constraint means that the constraint function result is to\n        be zero whereas inequality means that it is to be non-negative.\n        Note that COBYLA only supports inequality constraints.\n    tol : float, optional\n        Tolerance for termination. For detailed control, use solver-specific\n        options.\n    options : dict, optional\n        A dictionary of solver options. All methods accept the following\n        generic options:\n\n            maxiter : int\n                Maximum number of iterations to perform.\n            disp : bool\n                Set to True to print convergence messages.\n\n        For method-specific options, see :func:`show_options()`.\n    callback : callable, optional\n        Called after each iteration, as ``callback(xk)``, where ``xk`` is the\n        current parameter vector.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a ``OptimizeResult`` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.\n\n\n    See also\n    --------\n    minimize_scalar : Interface to minimization algorithms for scalar\n        univariate functions\n    show_options : Additional options accepted by the solvers\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    \'method\' parameter. The default method is *BFGS*.\n\n    **Unconstrained minimization**\n\n    Method :ref:`Nelder-Mead <optimize.minimize-neldermead>` uses the\n    Simplex algorithm [1]_, [2]_. This algorithm is robust in many\n    applications. However, if numerical computation of derivative can be\n    trusted, other algorithms using the first and/or second derivatives\n    information might be preferred for their better performance in\n    general.\n\n    Method :ref:`Powell <optimize.minimize-powell>` is a modification\n    of Powell\'s method [3]_, [4]_ which is a conjugate direction\n    method. It performs sequential one-dimensional minimizations along\n    each vector of the directions set (`direc` field in `options` and\n    `info`), which is updated at each iteration of the main\n    minimization loop. The function need not be differentiable, and no\n    derivatives are taken.\n\n    Method :ref:`CG <optimize.minimize-cg>` uses a nonlinear conjugate\n    gradient algorithm by Polak and Ribiere, a variant of the\n    Fletcher-Reeves method described in [5]_ pp.  120-122. Only the\n    first derivatives are used.\n\n    Method :ref:`BFGS <optimize.minimize-bfgs>` uses the quasi-Newton\n    method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS) [5]_\n    pp. 136. It uses the first derivatives only. BFGS has proven good\n    performance even for non-smooth optimizations. This method also\n    returns an approximation of the Hessian inverse, stored as\n    `hess_inv` in the OptimizeResult object.\n\n    Method :ref:`Newton-CG <optimize.minimize-newtoncg>` uses a\n    Newton-CG algorithm [5]_ pp. 168 (also known as the truncated\n    Newton method). It uses a CG method to the compute the search\n    direction. See also *TNC* method for a box-constrained\n    minimization with a similar algorithm. Suitable for large-scale\n    problems.\n\n    Method :ref:`dogleg <optimize.minimize-dogleg>` uses the dog-leg\n    trust-region algorithm [5]_ for unconstrained minimization. This\n    algorithm requires the gradient and Hessian; furthermore the\n    Hessian is required to be positive definite.\n\n    Method :ref:`trust-ncg <optimize.minimize-trustncg>` uses the\n    Newton conjugate gradient trust-region algorithm [5]_ for\n    unconstrained minimization. This algorithm requires the gradient\n    and either the Hessian or a function that computes the product of\n    the Hessian with a given vector. Suitable for large-scale problems.\n\n    Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` uses\n    the Newton GLTR trust-region algorithm [14]_, [15]_ for unconstrained\n    minimization. This algorithm requires the gradient\n    and either the Hessian or a function that computes the product of\n    the Hessian with a given vector. Suitable for large-scale problems.\n    On indefinite problems it requires usually less iterations than the\n    `trust-ncg` method and is recommended for medium and large-scale problems.\n\n    Method :ref:`trust-exact <optimize.minimize-trustexact>`\n    is a trust-region method for unconstrained minimization in which\n    quadratic subproblems are solved almost exactly [13]_. This\n    algorithm requires the gradient and the Hessian (which is\n    *not* required to be positive definite). It is, in many\n    situations, the Newton method to converge in fewer iteraction\n    and the most recommended for small and medium-size problems.\n\n    **Constrained minimization**\n\n    Method :ref:`L-BFGS-B <optimize.minimize-lbfgsb>` uses the L-BFGS-B\n    algorithm [6]_, [7]_ for bound constrained minimization.\n\n    Method :ref:`TNC <optimize.minimize-tnc>` uses a truncated Newton\n    algorithm [5]_, [8]_ to minimize a function with variables subject\n    to bounds. This algorithm uses gradient information; it is also\n    called Newton Conjugate-Gradient. It differs from the *Newton-CG*\n    method described above as it wraps a C implementation and allows\n    each variable to be given upper and lower bounds.\n\n    Method :ref:`COBYLA <optimize.minimize-cobyla>` uses the\n    Constrained Optimization BY Linear Approximation (COBYLA) method\n    [9]_, [10]_, [11]_. The algorithm is based on linear\n    approximations to the objective function and each constraint. The\n    method wraps a FORTRAN implementation of the algorithm. The\n    constraints functions \'fun\' may return either a single number\n    or an array or list of numbers.\n\n    Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential\n    Least SQuares Programming to minimize a function of several\n    variables with any combination of bounds, equality and inequality\n    constraints. The method wraps the SLSQP Optimization subroutine\n    originally implemented by Dieter Kraft [12]_. Note that the\n    wrapper handles infinite values in bounds by converting them into\n    large floating values.\n\n    **Custom minimizers**\n\n    It may be useful to pass a custom minimization method, for example\n    when using a frontend to this method such as `scipy.optimize.basinhopping`\n    or a different library.  You can simply pass a callable as the ``method``\n    parameter.\n\n    The callable is called as ``method(fun, x0, args, **kwargs, **options)``\n    where ``kwargs`` corresponds to any other parameters passed to `minimize`\n    (such as `callback`, `hess`, etc.), except the `options` dict, which has\n    its contents also passed as `method` parameters pair by pair.  Also, if\n    `jac` has been passed as a bool type, `jac` and `fun` are mangled so that\n    `fun` returns just the function values and `jac` is converted to a function\n    returning the Jacobian.  The method shall return an ``OptimizeResult``\n    object.\n\n    The provided `method` callable must be able to accept (and possibly ignore)\n    arbitrary parameters; the set of parameters accepted by `minimize` may\n    expand in future versions and then these parameters will be passed to\n    the method.  You can find an example in the scipy.optimize tutorial.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] Nelder, J A, and R Mead. 1965. A Simplex Method for Function\n        Minimization. The Computer Journal 7: 308-13.\n    .. [2] Wright M H. 1996. Direct search methods: Once scorned, now\n        respectable, in Numerical Analysis 1995: Proceedings of the 1995\n        Dundee Biennial Conference in Numerical Analysis (Eds. D F\n        Griffiths and G A Watson). Addison Wesley Longman, Harlow, UK.\n        191-208.\n    .. [3] Powell, M J D. 1964. An efficient method for finding the minimum of\n       a function of several variables without calculating derivatives. The\n       Computer Journal 7: 155-162.\n    .. [4] Press W, S A Teukolsky, W T Vetterling and B P Flannery.\n       Numerical Recipes (any edition), Cambridge University Press.\n    .. [5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.\n       Springer New York.\n    .. [6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory\n       Algorithm for Bound Constrained Optimization. SIAM Journal on\n       Scientific and Statistical Computing 16 (5): 1190-1208.\n    .. [7] Zhu, C and R H Byrd and J Nocedal. 1997. L-BFGS-B: Algorithm\n       778: L-BFGS-B, FORTRAN routines for large scale bound constrained\n       optimization. ACM Transactions on Mathematical Software 23 (4):\n       550-560.\n    .. [8] Nash, S G. Newton-Type Minimization Via the Lanczos Method.\n       1984. SIAM Journal of Numerical Analysis 21: 770-778.\n    .. [9] Powell, M J D. A direct search optimization method that models\n       the objective and constraint functions by linear interpolation.\n       1994. Advances in Optimization and Numerical Analysis, eds. S. Gomez\n       and J-P Hennart, Kluwer Academic (Dordrecht), 51-67.\n    .. [10] Powell M J D. Direct search algorithms for optimization\n       calculations. 1998. Acta Numerica 7: 287-336.\n    .. [11] Powell M J D. A view of algorithms for optimization without\n       derivatives. 2007.Cambridge University Technical Report DAMTP\n       2007/NA03\n    .. [12] Kraft, D. A software package for sequential quadratic\n       programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace\n       Center -- Institute for Flight Mechanics, Koln, Germany.\n    .. [13] Conn, A. R., Gould, N. I., and Toint, P. L.\n       Trust region methods. 2000. Siam. pp. 169-200.\n    .. [14] F. Lenders, C. Kirches, A. Potschka: "trlib: A vector-free\n       implementation of the GLTR method for iterative solution of\n       the trust region problem", https://arxiv.org/abs/1611.04718\n    .. [15] N. Gould, S. Lucidi, M. Roma, P. Toint: "Solving the\n       Trust-Region Subproblem using the Lanczos Method",\n       SIAM J. Optim., 9(2), 504--525, (1999).\n\n    Examples\n    --------\n    Let us consider the problem of minimizing the Rosenbrock function. This\n    function (and its respective derivatives) is implemented in `rosen`\n    (resp. `rosen_der`, `rosen_hess`) in the `scipy.optimize`.\n\n    >>> from scipy.optimize import minimize, rosen, rosen_der\n\n    A simple application of the *Nelder-Mead* method is:\n\n    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]\n    >>> res = minimize(rosen, x0, method=\'Nelder-Mead\', tol=1e-6)\n    >>> res.x\n    array([ 1.,  1.,  1.,  1.,  1.])\n\n    Now using the *BFGS* algorithm, using the first derivative and a few\n    options:\n\n    >>> res = minimize(rosen, x0, method=\'BFGS\', jac=rosen_der,\n    ...                options={\'gtol\': 1e-6, \'disp\': True})\n    Optimization terminated successfully.\n             Current function value: 0.000000\n             Iterations: 26\n             Function evaluations: 31\n             Gradient evaluations: 31\n    >>> res.x\n    array([ 1.,  1.,  1.,  1.,  1.])\n    >>> print(res.message)\n    Optimization terminated successfully.\n    >>> res.hess_inv\n    array([[ 0.00749589,  0.01255155,  0.02396251,  0.04750988,  0.09495377],  # may vary\n           [ 0.01255155,  0.02510441,  0.04794055,  0.09502834,  0.18996269],\n           [ 0.02396251,  0.04794055,  0.09631614,  0.19092151,  0.38165151],\n           [ 0.04750988,  0.09502834,  0.19092151,  0.38341252,  0.7664427 ],\n           [ 0.09495377,  0.18996269,  0.38165151,  0.7664427,   1.53713523]])\n\n\n    Next, consider a minimization problem with several constraints (namely\n    Example 16.4 from [5]_). The objective function is:\n\n    >>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2\n\n    There are three constraints defined as:\n\n    >>> cons = ({\'type\': \'ineq\', \'fun\': lambda x:  x[0] - 2 * x[1] + 2},\n    ...         {\'type\': \'ineq\', \'fun\': lambda x: -x[0] - 2 * x[1] + 6},\n    ...         {\'type\': \'ineq\', \'fun\': lambda x: -x[0] + 2 * x[1] + 2})\n\n    And variables must be positive, hence the following bounds:\n\n    >>> bnds = ((0, None), (0, None))\n\n    The optimization problem is solved using the SLSQP method as:\n\n    >>> res = minimize(fun, (2, 0), method=\'SLSQP\', bounds=bnds,\n    ...                constraints=cons)\n\n    It should converge to the theoretical solution (1.4 ,1.7).\n\n    ')
    
    # Assigning a Call to a Name (line 388):
    
    # Call to asarray(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'x0' (line 388)
    x0_197937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'x0', False)
    # Processing the call keyword arguments (line 388)
    kwargs_197938 = {}
    # Getting the type of 'np' (line 388)
    np_197935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 9), 'np', False)
    # Obtaining the member 'asarray' of a type (line 388)
    asarray_197936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 9), np_197935, 'asarray')
    # Calling asarray(args, kwargs) (line 388)
    asarray_call_result_197939 = invoke(stypy.reporting.localization.Localization(__file__, 388, 9), asarray_197936, *[x0_197937], **kwargs_197938)
    
    # Assigning a type to the variable 'x0' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'x0', asarray_call_result_197939)
    
    
    # Getting the type of 'x0' (line 389)
    x0_197940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 7), 'x0')
    # Obtaining the member 'dtype' of a type (line 389)
    dtype_197941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 7), x0_197940, 'dtype')
    # Obtaining the member 'kind' of a type (line 389)
    kind_197942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 7), dtype_197941, 'kind')
    
    # Obtaining the type of the subscript
    str_197943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 37), 'str', 'AllInteger')
    # Getting the type of 'np' (line 389)
    np_197944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 24), 'np')
    # Obtaining the member 'typecodes' of a type (line 389)
    typecodes_197945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 24), np_197944, 'typecodes')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___197946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 24), typecodes_197945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_197947 = invoke(stypy.reporting.localization.Localization(__file__, 389, 24), getitem___197946, str_197943)
    
    # Applying the binary operator 'in' (line 389)
    result_contains_197948 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 7), 'in', kind_197942, subscript_call_result_197947)
    
    # Testing the type of an if condition (line 389)
    if_condition_197949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 4), result_contains_197948)
    # Assigning a type to the variable 'if_condition_197949' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'if_condition_197949', if_condition_197949)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 390):
    
    # Call to asarray(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'x0' (line 390)
    x0_197952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'x0', False)
    # Processing the call keyword arguments (line 390)
    # Getting the type of 'float' (line 390)
    float_197953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 34), 'float', False)
    keyword_197954 = float_197953
    kwargs_197955 = {'dtype': keyword_197954}
    # Getting the type of 'np' (line 390)
    np_197950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 390)
    asarray_197951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 13), np_197950, 'asarray')
    # Calling asarray(args, kwargs) (line 390)
    asarray_call_result_197956 = invoke(stypy.reporting.localization.Localization(__file__, 390, 13), asarray_197951, *[x0_197952], **kwargs_197955)
    
    # Assigning a type to the variable 'x0' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'x0', asarray_call_result_197956)
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 392)
    # Getting the type of 'tuple' (line 392)
    tuple_197957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 28), 'tuple')
    # Getting the type of 'args' (line 392)
    args_197958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'args')
    
    (may_be_197959, more_types_in_union_197960) = may_not_be_subtype(tuple_197957, args_197958)

    if may_be_197959:

        if more_types_in_union_197960:
            # Runtime conditional SSA (line 392)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'args', remove_subtype_from_union(args_197958, tuple))
        
        # Assigning a Tuple to a Name (line 393):
        
        # Obtaining an instance of the builtin type 'tuple' (line 393)
        tuple_197961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 393)
        # Adding element type (line 393)
        # Getting the type of 'args' (line 393)
        args_197962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 16), tuple_197961, args_197962)
        
        # Assigning a type to the variable 'args' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'args', tuple_197961)

        if more_types_in_union_197960:
            # SSA join for if statement (line 392)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 395)
    # Getting the type of 'method' (line 395)
    method_197963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 7), 'method')
    # Getting the type of 'None' (line 395)
    None_197964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 17), 'None')
    
    (may_be_197965, more_types_in_union_197966) = may_be_none(method_197963, None_197964)

    if may_be_197965:

        if more_types_in_union_197966:
            # Runtime conditional SSA (line 395)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'constraints' (line 397)
        constraints_197967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 11), 'constraints')
        # Testing the type of an if condition (line 397)
        if_condition_197968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 8), constraints_197967)
        # Assigning a type to the variable 'if_condition_197968' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'if_condition_197968', if_condition_197968)
        # SSA begins for if statement (line 397)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 398):
        str_197969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 21), 'str', 'SLSQP')
        # Assigning a type to the variable 'method' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'method', str_197969)
        # SSA branch for the else part of an if statement (line 397)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 399)
        # Getting the type of 'bounds' (line 399)
        bounds_197970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 13), 'bounds')
        # Getting the type of 'None' (line 399)
        None_197971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 27), 'None')
        
        (may_be_197972, more_types_in_union_197973) = may_not_be_none(bounds_197970, None_197971)

        if may_be_197972:

            if more_types_in_union_197973:
                # Runtime conditional SSA (line 399)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 400):
            str_197974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 21), 'str', 'L-BFGS-B')
            # Assigning a type to the variable 'method' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'method', str_197974)

            if more_types_in_union_197973:
                # Runtime conditional SSA for else branch (line 399)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_197972) or more_types_in_union_197973):
            
            # Assigning a Str to a Name (line 402):
            str_197975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 21), 'str', 'BFGS')
            # Assigning a type to the variable 'method' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'method', str_197975)

            if (may_be_197972 and more_types_in_union_197973):
                # SSA join for if statement (line 399)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 397)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_197966:
            # SSA join for if statement (line 395)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to callable(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'method' (line 404)
    method_197977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 16), 'method', False)
    # Processing the call keyword arguments (line 404)
    kwargs_197978 = {}
    # Getting the type of 'callable' (line 404)
    callable_197976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 7), 'callable', False)
    # Calling callable(args, kwargs) (line 404)
    callable_call_result_197979 = invoke(stypy.reporting.localization.Localization(__file__, 404, 7), callable_197976, *[method_197977], **kwargs_197978)
    
    # Testing the type of an if condition (line 404)
    if_condition_197980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 4), callable_call_result_197979)
    # Assigning a type to the variable 'if_condition_197980' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'if_condition_197980', if_condition_197980)
    # SSA begins for if statement (line 404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 405):
    str_197981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 15), 'str', '_custom')
    # Assigning a type to the variable 'meth' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'meth', str_197981)
    # SSA branch for the else part of an if statement (line 404)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 407):
    
    # Call to lower(...): (line 407)
    # Processing the call keyword arguments (line 407)
    kwargs_197984 = {}
    # Getting the type of 'method' (line 407)
    method_197982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 15), 'method', False)
    # Obtaining the member 'lower' of a type (line 407)
    lower_197983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 15), method_197982, 'lower')
    # Calling lower(args, kwargs) (line 407)
    lower_call_result_197985 = invoke(stypy.reporting.localization.Localization(__file__, 407, 15), lower_197983, *[], **kwargs_197984)
    
    # Assigning a type to the variable 'meth' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'meth', lower_call_result_197985)
    # SSA join for if statement (line 404)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 409)
    # Getting the type of 'options' (line 409)
    options_197986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 7), 'options')
    # Getting the type of 'None' (line 409)
    None_197987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 18), 'None')
    
    (may_be_197988, more_types_in_union_197989) = may_be_none(options_197986, None_197987)

    if may_be_197988:

        if more_types_in_union_197989:
            # Runtime conditional SSA (line 409)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 410):
        
        # Obtaining an instance of the builtin type 'dict' (line 410)
        dict_197990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 410)
        
        # Assigning a type to the variable 'options' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'options', dict_197990)

        if more_types_in_union_197989:
            # SSA join for if statement (line 409)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'meth' (line 413)
    meth_197991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 7), 'meth')
    
    # Obtaining an instance of the builtin type 'list' (line 413)
    list_197992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 413)
    # Adding element type (line 413)
    str_197993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 16), 'str', 'nelder-mead')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 15), list_197992, str_197993)
    # Adding element type (line 413)
    str_197994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 31), 'str', 'powell')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 15), list_197992, str_197994)
    # Adding element type (line 413)
    str_197995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 41), 'str', 'cobyla')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 15), list_197992, str_197995)
    
    # Applying the binary operator 'in' (line 413)
    result_contains_197996 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 7), 'in', meth_197991, list_197992)
    
    
    # Call to bool(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'jac' (line 413)
    jac_197998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 60), 'jac', False)
    # Processing the call keyword arguments (line 413)
    kwargs_197999 = {}
    # Getting the type of 'bool' (line 413)
    bool_197997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 55), 'bool', False)
    # Calling bool(args, kwargs) (line 413)
    bool_call_result_198000 = invoke(stypy.reporting.localization.Localization(__file__, 413, 55), bool_197997, *[jac_197998], **kwargs_197999)
    
    # Applying the binary operator 'and' (line 413)
    result_and_keyword_198001 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 7), 'and', result_contains_197996, bool_call_result_198000)
    
    # Testing the type of an if condition (line 413)
    if_condition_198002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 4), result_and_keyword_198001)
    # Assigning a type to the variable 'if_condition_198002' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'if_condition_198002', if_condition_198002)
    # SSA begins for if statement (line 413)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 414)
    # Processing the call arguments (line 414)
    str_198004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 13), 'str', 'Method %s does not use gradient information (jac).')
    # Getting the type of 'method' (line 414)
    method_198005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 68), 'method', False)
    # Applying the binary operator '%' (line 414)
    result_mod_198006 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 13), '%', str_198004, method_198005)
    
    # Getting the type of 'RuntimeWarning' (line 415)
    RuntimeWarning_198007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 13), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 414)
    kwargs_198008 = {}
    # Getting the type of 'warn' (line 414)
    warn_198003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 414)
    warn_call_result_198009 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), warn_198003, *[result_mod_198006, RuntimeWarning_198007], **kwargs_198008)
    
    # SSA join for if statement (line 413)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'meth' (line 417)
    meth_198010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 7), 'meth')
    
    # Obtaining an instance of the builtin type 'tuple' (line 417)
    tuple_198011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 417)
    # Adding element type (line 417)
    str_198012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 20), 'str', 'newton-cg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 20), tuple_198011, str_198012)
    # Adding element type (line 417)
    str_198013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 33), 'str', 'dogleg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 20), tuple_198011, str_198013)
    # Adding element type (line 417)
    str_198014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 43), 'str', 'trust-ncg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 20), tuple_198011, str_198014)
    # Adding element type (line 417)
    str_198015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 56), 'str', 'trust-krylov')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 20), tuple_198011, str_198015)
    # Adding element type (line 417)
    str_198016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 20), 'str', 'trust-exact')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 20), tuple_198011, str_198016)
    # Adding element type (line 417)
    str_198017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 35), 'str', '_custom')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 20), tuple_198011, str_198017)
    
    # Applying the binary operator 'notin' (line 417)
    result_contains_198018 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 7), 'notin', meth_198010, tuple_198011)
    
    
    # Getting the type of 'hess' (line 418)
    hess_198019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 50), 'hess')
    # Getting the type of 'None' (line 418)
    None_198020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 62), 'None')
    # Applying the binary operator 'isnot' (line 418)
    result_is_not_198021 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 50), 'isnot', hess_198019, None_198020)
    
    # Applying the binary operator 'and' (line 417)
    result_and_keyword_198022 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 7), 'and', result_contains_198018, result_is_not_198021)
    
    # Testing the type of an if condition (line 417)
    if_condition_198023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 4), result_and_keyword_198022)
    # Assigning a type to the variable 'if_condition_198023' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'if_condition_198023', if_condition_198023)
    # SSA begins for if statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 419)
    # Processing the call arguments (line 419)
    str_198025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 13), 'str', 'Method %s does not use Hessian information (hess).')
    # Getting the type of 'method' (line 419)
    method_198026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 68), 'method', False)
    # Applying the binary operator '%' (line 419)
    result_mod_198027 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 13), '%', str_198025, method_198026)
    
    # Getting the type of 'RuntimeWarning' (line 420)
    RuntimeWarning_198028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 13), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 419)
    kwargs_198029 = {}
    # Getting the type of 'warn' (line 419)
    warn_198024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 419)
    warn_call_result_198030 = invoke(stypy.reporting.localization.Localization(__file__, 419, 8), warn_198024, *[result_mod_198027, RuntimeWarning_198028], **kwargs_198029)
    
    # SSA join for if statement (line 417)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'meth' (line 422)
    meth_198031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 7), 'meth')
    
    # Obtaining an instance of the builtin type 'tuple' (line 422)
    tuple_198032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 422)
    # Adding element type (line 422)
    str_198033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 20), 'str', 'newton-cg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 20), tuple_198032, str_198033)
    # Adding element type (line 422)
    str_198034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 33), 'str', 'dogleg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 20), tuple_198032, str_198034)
    # Adding element type (line 422)
    str_198035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 43), 'str', 'trust-ncg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 20), tuple_198032, str_198035)
    # Adding element type (line 422)
    str_198036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 20), 'str', 'trust-krylov')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 20), tuple_198032, str_198036)
    # Adding element type (line 422)
    str_198037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 36), 'str', '_custom')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 20), tuple_198032, str_198037)
    
    # Applying the binary operator 'notin' (line 422)
    result_contains_198038 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 7), 'notin', meth_198031, tuple_198032)
    
    
    # Getting the type of 'hessp' (line 423)
    hessp_198039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 51), 'hessp')
    # Getting the type of 'None' (line 423)
    None_198040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 64), 'None')
    # Applying the binary operator 'isnot' (line 423)
    result_is_not_198041 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 51), 'isnot', hessp_198039, None_198040)
    
    # Applying the binary operator 'and' (line 422)
    result_and_keyword_198042 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 7), 'and', result_contains_198038, result_is_not_198041)
    
    # Testing the type of an if condition (line 422)
    if_condition_198043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 4), result_and_keyword_198042)
    # Assigning a type to the variable 'if_condition_198043' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'if_condition_198043', if_condition_198043)
    # SSA begins for if statement (line 422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 424)
    # Processing the call arguments (line 424)
    str_198045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 13), 'str', 'Method %s does not use Hessian-vector product information (hessp).')
    # Getting the type of 'method' (line 425)
    method_198046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 41), 'method', False)
    # Applying the binary operator '%' (line 424)
    result_mod_198047 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 13), '%', str_198045, method_198046)
    
    # Getting the type of 'RuntimeWarning' (line 425)
    RuntimeWarning_198048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 49), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 424)
    kwargs_198049 = {}
    # Getting the type of 'warn' (line 424)
    warn_198044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 424)
    warn_call_result_198050 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), warn_198044, *[result_mod_198047, RuntimeWarning_198048], **kwargs_198049)
    
    # SSA join for if statement (line 422)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'meth' (line 427)
    meth_198051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'meth')
    
    # Obtaining an instance of the builtin type 'list' (line 427)
    list_198052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 427)
    # Adding element type (line 427)
    str_198053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 17), 'str', 'nelder-mead')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 16), list_198052, str_198053)
    # Adding element type (line 427)
    str_198054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 32), 'str', 'powell')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 16), list_198052, str_198054)
    # Adding element type (line 427)
    str_198055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 42), 'str', 'cg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 16), list_198052, str_198055)
    # Adding element type (line 427)
    str_198056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 48), 'str', 'bfgs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 16), list_198052, str_198056)
    # Adding element type (line 427)
    str_198057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 56), 'str', 'newton-cg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 16), list_198052, str_198057)
    # Adding element type (line 427)
    str_198058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 69), 'str', 'dogleg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 16), list_198052, str_198058)
    # Adding element type (line 427)
    str_198059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 17), 'str', 'trust-ncg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 16), list_198052, str_198059)
    
    # Applying the binary operator 'in' (line 427)
    result_contains_198060 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 8), 'in', meth_198051, list_198052)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'bounds' (line 428)
    bounds_198061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 35), 'bounds')
    # Getting the type of 'None' (line 428)
    None_198062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 49), 'None')
    # Applying the binary operator 'isnot' (line 428)
    result_is_not_198063 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 35), 'isnot', bounds_198061, None_198062)
    
    
    # Call to any(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'constraints' (line 428)
    constraints_198066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 64), 'constraints', False)
    # Processing the call keyword arguments (line 428)
    kwargs_198067 = {}
    # Getting the type of 'np' (line 428)
    np_198064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 57), 'np', False)
    # Obtaining the member 'any' of a type (line 428)
    any_198065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 57), np_198064, 'any')
    # Calling any(args, kwargs) (line 428)
    any_call_result_198068 = invoke(stypy.reporting.localization.Localization(__file__, 428, 57), any_198065, *[constraints_198066], **kwargs_198067)
    
    # Applying the binary operator 'or' (line 428)
    result_or_keyword_198069 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 35), 'or', result_is_not_198063, any_call_result_198068)
    
    # Applying the binary operator 'and' (line 427)
    result_and_keyword_198070 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 8), 'and', result_contains_198060, result_or_keyword_198069)
    
    # Testing the type of an if condition (line 427)
    if_condition_198071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 4), result_and_keyword_198070)
    # Assigning a type to the variable 'if_condition_198071' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'if_condition_198071', if_condition_198071)
    # SSA begins for if statement (line 427)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 429)
    # Processing the call arguments (line 429)
    str_198073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 13), 'str', 'Method %s cannot handle constraints nor bounds.')
    # Getting the type of 'method' (line 429)
    method_198074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 65), 'method', False)
    # Applying the binary operator '%' (line 429)
    result_mod_198075 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 13), '%', str_198073, method_198074)
    
    # Getting the type of 'RuntimeWarning' (line 430)
    RuntimeWarning_198076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 13), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 429)
    kwargs_198077 = {}
    # Getting the type of 'warn' (line 429)
    warn_198072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 429)
    warn_call_result_198078 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), warn_198072, *[result_mod_198075, RuntimeWarning_198076], **kwargs_198077)
    
    # SSA join for if statement (line 427)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'meth' (line 431)
    meth_198079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 7), 'meth')
    
    # Obtaining an instance of the builtin type 'list' (line 431)
    list_198080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 431)
    # Adding element type (line 431)
    str_198081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 16), 'str', 'l-bfgs-b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 15), list_198080, str_198081)
    # Adding element type (line 431)
    str_198082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 28), 'str', 'tnc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 15), list_198080, str_198082)
    
    # Applying the binary operator 'in' (line 431)
    result_contains_198083 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 7), 'in', meth_198079, list_198080)
    
    
    # Call to any(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'constraints' (line 431)
    constraints_198086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 46), 'constraints', False)
    # Processing the call keyword arguments (line 431)
    kwargs_198087 = {}
    # Getting the type of 'np' (line 431)
    np_198084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 39), 'np', False)
    # Obtaining the member 'any' of a type (line 431)
    any_198085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 39), np_198084, 'any')
    # Calling any(args, kwargs) (line 431)
    any_call_result_198088 = invoke(stypy.reporting.localization.Localization(__file__, 431, 39), any_198085, *[constraints_198086], **kwargs_198087)
    
    # Applying the binary operator 'and' (line 431)
    result_and_keyword_198089 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 7), 'and', result_contains_198083, any_call_result_198088)
    
    # Testing the type of an if condition (line 431)
    if_condition_198090 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 4), result_and_keyword_198089)
    # Assigning a type to the variable 'if_condition_198090' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'if_condition_198090', if_condition_198090)
    # SSA begins for if statement (line 431)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 432)
    # Processing the call arguments (line 432)
    str_198092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 13), 'str', 'Method %s cannot handle constraints.')
    # Getting the type of 'method' (line 432)
    method_198093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 54), 'method', False)
    # Applying the binary operator '%' (line 432)
    result_mod_198094 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 13), '%', str_198092, method_198093)
    
    # Getting the type of 'RuntimeWarning' (line 433)
    RuntimeWarning_198095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 13), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 432)
    kwargs_198096 = {}
    # Getting the type of 'warn' (line 432)
    warn_198091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 432)
    warn_call_result_198097 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), warn_198091, *[result_mod_198094, RuntimeWarning_198095], **kwargs_198096)
    
    # SSA join for if statement (line 431)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'meth' (line 434)
    meth_198098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 7), 'meth')
    str_198099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 15), 'str', 'cobyla')
    # Applying the binary operator '==' (line 434)
    result_eq_198100 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 7), '==', meth_198098, str_198099)
    
    
    # Getting the type of 'bounds' (line 434)
    bounds_198101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 28), 'bounds')
    # Getting the type of 'None' (line 434)
    None_198102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 42), 'None')
    # Applying the binary operator 'isnot' (line 434)
    result_is_not_198103 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 28), 'isnot', bounds_198101, None_198102)
    
    # Applying the binary operator 'and' (line 434)
    result_and_keyword_198104 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 7), 'and', result_eq_198100, result_is_not_198103)
    
    # Testing the type of an if condition (line 434)
    if_condition_198105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 4), result_and_keyword_198104)
    # Assigning a type to the variable 'if_condition_198105' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'if_condition_198105', if_condition_198105)
    # SSA begins for if statement (line 434)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 435)
    # Processing the call arguments (line 435)
    str_198107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 13), 'str', 'Method %s cannot handle bounds.')
    # Getting the type of 'method' (line 435)
    method_198108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 49), 'method', False)
    # Applying the binary operator '%' (line 435)
    result_mod_198109 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 13), '%', str_198107, method_198108)
    
    # Getting the type of 'RuntimeWarning' (line 436)
    RuntimeWarning_198110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 13), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 435)
    kwargs_198111 = {}
    # Getting the type of 'warn' (line 435)
    warn_198106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 435)
    warn_call_result_198112 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), warn_198106, *[result_mod_198109, RuntimeWarning_198110], **kwargs_198111)
    
    # SSA join for if statement (line 434)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'meth' (line 438)
    meth_198113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'meth')
    
    # Obtaining an instance of the builtin type 'list' (line 438)
    list_198114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 438)
    # Adding element type (line 438)
    str_198115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 17), 'str', 'cobyla')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 16), list_198114, str_198115)
    
    # Applying the binary operator 'in' (line 438)
    result_contains_198116 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 8), 'in', meth_198113, list_198114)
    
    
    # Getting the type of 'callback' (line 438)
    callback_198117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 31), 'callback')
    # Getting the type of 'None' (line 438)
    None_198118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 47), 'None')
    # Applying the binary operator 'isnot' (line 438)
    result_is_not_198119 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 31), 'isnot', callback_198117, None_198118)
    
    # Applying the binary operator 'and' (line 438)
    result_and_keyword_198120 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 8), 'and', result_contains_198116, result_is_not_198119)
    
    # Testing the type of an if condition (line 438)
    if_condition_198121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 4), result_and_keyword_198120)
    # Assigning a type to the variable 'if_condition_198121' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'if_condition_198121', if_condition_198121)
    # SSA begins for if statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 439)
    # Processing the call arguments (line 439)
    str_198123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 13), 'str', 'Method %s does not support callback.')
    # Getting the type of 'method' (line 439)
    method_198124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 54), 'method', False)
    # Applying the binary operator '%' (line 439)
    result_mod_198125 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 13), '%', str_198123, method_198124)
    
    # Getting the type of 'RuntimeWarning' (line 439)
    RuntimeWarning_198126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 62), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 439)
    kwargs_198127 = {}
    # Getting the type of 'warn' (line 439)
    warn_198122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 439)
    warn_call_result_198128 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), warn_198122, *[result_mod_198125, RuntimeWarning_198126], **kwargs_198127)
    
    # SSA join for if statement (line 438)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'meth' (line 441)
    meth_198129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'meth')
    
    # Obtaining an instance of the builtin type 'list' (line 441)
    list_198130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 441)
    # Adding element type (line 441)
    str_198131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 17), 'str', 'l-bfgs-b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 16), list_198130, str_198131)
    # Adding element type (line 441)
    str_198132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 29), 'str', 'tnc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 16), list_198130, str_198132)
    # Adding element type (line 441)
    str_198133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 36), 'str', 'cobyla')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 16), list_198130, str_198133)
    # Adding element type (line 441)
    str_198134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 46), 'str', 'slsqp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 16), list_198130, str_198134)
    
    # Applying the binary operator 'in' (line 441)
    result_contains_198135 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 8), 'in', meth_198129, list_198130)
    
    
    # Call to get(...): (line 442)
    # Processing the call arguments (line 442)
    str_198138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 24), 'str', 'return_all')
    # Getting the type of 'False' (line 442)
    False_198139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 38), 'False', False)
    # Processing the call keyword arguments (line 442)
    kwargs_198140 = {}
    # Getting the type of 'options' (line 442)
    options_198136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'options', False)
    # Obtaining the member 'get' of a type (line 442)
    get_198137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), options_198136, 'get')
    # Calling get(args, kwargs) (line 442)
    get_call_result_198141 = invoke(stypy.reporting.localization.Localization(__file__, 442, 12), get_198137, *[str_198138, False_198139], **kwargs_198140)
    
    # Applying the binary operator 'and' (line 441)
    result_and_keyword_198142 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 8), 'and', result_contains_198135, get_call_result_198141)
    
    # Testing the type of an if condition (line 441)
    if_condition_198143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 4), result_and_keyword_198142)
    # Assigning a type to the variable 'if_condition_198143' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'if_condition_198143', if_condition_198143)
    # SSA begins for if statement (line 441)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 443)
    # Processing the call arguments (line 443)
    str_198145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 13), 'str', 'Method %s does not support the return_all option.')
    # Getting the type of 'method' (line 443)
    method_198146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 67), 'method', False)
    # Applying the binary operator '%' (line 443)
    result_mod_198147 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 13), '%', str_198145, method_198146)
    
    # Getting the type of 'RuntimeWarning' (line 444)
    RuntimeWarning_198148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 13), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 443)
    kwargs_198149 = {}
    # Getting the type of 'warn' (line 443)
    warn_198144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 443)
    warn_call_result_198150 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), warn_198144, *[result_mod_198147, RuntimeWarning_198148], **kwargs_198149)
    
    # SSA join for if statement (line 441)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to callable(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'jac' (line 447)
    jac_198152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 20), 'jac', False)
    # Processing the call keyword arguments (line 447)
    kwargs_198153 = {}
    # Getting the type of 'callable' (line 447)
    callable_198151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'callable', False)
    # Calling callable(args, kwargs) (line 447)
    callable_call_result_198154 = invoke(stypy.reporting.localization.Localization(__file__, 447, 11), callable_198151, *[jac_198152], **kwargs_198153)
    
    # Applying the 'not' unary operator (line 447)
    result_not__198155 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 7), 'not', callable_call_result_198154)
    
    # Testing the type of an if condition (line 447)
    if_condition_198156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 4), result_not__198155)
    # Assigning a type to the variable 'if_condition_198156' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'if_condition_198156', if_condition_198156)
    # SSA begins for if statement (line 447)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to bool(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'jac' (line 448)
    jac_198158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'jac', False)
    # Processing the call keyword arguments (line 448)
    kwargs_198159 = {}
    # Getting the type of 'bool' (line 448)
    bool_198157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'bool', False)
    # Calling bool(args, kwargs) (line 448)
    bool_call_result_198160 = invoke(stypy.reporting.localization.Localization(__file__, 448, 11), bool_198157, *[jac_198158], **kwargs_198159)
    
    # Testing the type of an if condition (line 448)
    if_condition_198161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 8), bool_call_result_198160)
    # Assigning a type to the variable 'if_condition_198161' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'if_condition_198161', if_condition_198161)
    # SSA begins for if statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 449):
    
    # Call to MemoizeJac(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'fun' (line 449)
    fun_198163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 29), 'fun', False)
    # Processing the call keyword arguments (line 449)
    kwargs_198164 = {}
    # Getting the type of 'MemoizeJac' (line 449)
    MemoizeJac_198162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 18), 'MemoizeJac', False)
    # Calling MemoizeJac(args, kwargs) (line 449)
    MemoizeJac_call_result_198165 = invoke(stypy.reporting.localization.Localization(__file__, 449, 18), MemoizeJac_198162, *[fun_198163], **kwargs_198164)
    
    # Assigning a type to the variable 'fun' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'fun', MemoizeJac_call_result_198165)
    
    # Assigning a Attribute to a Name (line 450):
    # Getting the type of 'fun' (line 450)
    fun_198166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 18), 'fun')
    # Obtaining the member 'derivative' of a type (line 450)
    derivative_198167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 18), fun_198166, 'derivative')
    # Assigning a type to the variable 'jac' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'jac', derivative_198167)
    # SSA branch for the else part of an if statement (line 448)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 452):
    # Getting the type of 'None' (line 452)
    None_198168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 18), 'None')
    # Assigning a type to the variable 'jac' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'jac', None_198168)
    # SSA join for if statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 447)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 455)
    # Getting the type of 'tol' (line 455)
    tol_198169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tol')
    # Getting the type of 'None' (line 455)
    None_198170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 18), 'None')
    
    (may_be_198171, more_types_in_union_198172) = may_not_be_none(tol_198169, None_198170)

    if may_be_198171:

        if more_types_in_union_198172:
            # Runtime conditional SSA (line 455)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 456):
        
        # Call to dict(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'options' (line 456)
        options_198174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'options', False)
        # Processing the call keyword arguments (line 456)
        kwargs_198175 = {}
        # Getting the type of 'dict' (line 456)
        dict_198173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 18), 'dict', False)
        # Calling dict(args, kwargs) (line 456)
        dict_call_result_198176 = invoke(stypy.reporting.localization.Localization(__file__, 456, 18), dict_198173, *[options_198174], **kwargs_198175)
        
        # Assigning a type to the variable 'options' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'options', dict_call_result_198176)
        
        
        # Getting the type of 'meth' (line 457)
        meth_198177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 11), 'meth')
        str_198178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 19), 'str', 'nelder-mead')
        # Applying the binary operator '==' (line 457)
        result_eq_198179 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 11), '==', meth_198177, str_198178)
        
        # Testing the type of an if condition (line 457)
        if_condition_198180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 8), result_eq_198179)
        # Assigning a type to the variable 'if_condition_198180' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'if_condition_198180', if_condition_198180)
        # SSA begins for if statement (line 457)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 458)
        # Processing the call arguments (line 458)
        str_198183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 31), 'str', 'xatol')
        # Getting the type of 'tol' (line 458)
        tol_198184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 40), 'tol', False)
        # Processing the call keyword arguments (line 458)
        kwargs_198185 = {}
        # Getting the type of 'options' (line 458)
        options_198181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 458)
        setdefault_198182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 12), options_198181, 'setdefault')
        # Calling setdefault(args, kwargs) (line 458)
        setdefault_call_result_198186 = invoke(stypy.reporting.localization.Localization(__file__, 458, 12), setdefault_198182, *[str_198183, tol_198184], **kwargs_198185)
        
        
        # Call to setdefault(...): (line 459)
        # Processing the call arguments (line 459)
        str_198189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 31), 'str', 'fatol')
        # Getting the type of 'tol' (line 459)
        tol_198190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 40), 'tol', False)
        # Processing the call keyword arguments (line 459)
        kwargs_198191 = {}
        # Getting the type of 'options' (line 459)
        options_198187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 459)
        setdefault_198188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 12), options_198187, 'setdefault')
        # Calling setdefault(args, kwargs) (line 459)
        setdefault_call_result_198192 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), setdefault_198188, *[str_198189, tol_198190], **kwargs_198191)
        
        # SSA join for if statement (line 457)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'meth' (line 460)
        meth_198193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'meth')
        
        # Obtaining an instance of the builtin type 'list' (line 460)
        list_198194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 460)
        # Adding element type (line 460)
        str_198195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 20), 'str', 'newton-cg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 19), list_198194, str_198195)
        # Adding element type (line 460)
        str_198196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 33), 'str', 'powell')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 19), list_198194, str_198196)
        # Adding element type (line 460)
        str_198197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 43), 'str', 'tnc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 19), list_198194, str_198197)
        
        # Applying the binary operator 'in' (line 460)
        result_contains_198198 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 11), 'in', meth_198193, list_198194)
        
        # Testing the type of an if condition (line 460)
        if_condition_198199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 8), result_contains_198198)
        # Assigning a type to the variable 'if_condition_198199' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'if_condition_198199', if_condition_198199)
        # SSA begins for if statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 461)
        # Processing the call arguments (line 461)
        str_198202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 31), 'str', 'xtol')
        # Getting the type of 'tol' (line 461)
        tol_198203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 39), 'tol', False)
        # Processing the call keyword arguments (line 461)
        kwargs_198204 = {}
        # Getting the type of 'options' (line 461)
        options_198200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 461)
        setdefault_198201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 12), options_198200, 'setdefault')
        # Calling setdefault(args, kwargs) (line 461)
        setdefault_call_result_198205 = invoke(stypy.reporting.localization.Localization(__file__, 461, 12), setdefault_198201, *[str_198202, tol_198203], **kwargs_198204)
        
        # SSA join for if statement (line 460)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'meth' (line 462)
        meth_198206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 11), 'meth')
        
        # Obtaining an instance of the builtin type 'list' (line 462)
        list_198207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 462)
        # Adding element type (line 462)
        str_198208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 20), 'str', 'powell')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 19), list_198207, str_198208)
        # Adding element type (line 462)
        str_198209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 30), 'str', 'l-bfgs-b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 19), list_198207, str_198209)
        # Adding element type (line 462)
        str_198210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 42), 'str', 'tnc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 19), list_198207, str_198210)
        # Adding element type (line 462)
        str_198211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 49), 'str', 'slsqp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 19), list_198207, str_198211)
        
        # Applying the binary operator 'in' (line 462)
        result_contains_198212 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 11), 'in', meth_198206, list_198207)
        
        # Testing the type of an if condition (line 462)
        if_condition_198213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 8), result_contains_198212)
        # Assigning a type to the variable 'if_condition_198213' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'if_condition_198213', if_condition_198213)
        # SSA begins for if statement (line 462)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 463)
        # Processing the call arguments (line 463)
        str_198216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 31), 'str', 'ftol')
        # Getting the type of 'tol' (line 463)
        tol_198217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 39), 'tol', False)
        # Processing the call keyword arguments (line 463)
        kwargs_198218 = {}
        # Getting the type of 'options' (line 463)
        options_198214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 463)
        setdefault_198215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 12), options_198214, 'setdefault')
        # Calling setdefault(args, kwargs) (line 463)
        setdefault_call_result_198219 = invoke(stypy.reporting.localization.Localization(__file__, 463, 12), setdefault_198215, *[str_198216, tol_198217], **kwargs_198218)
        
        # SSA join for if statement (line 462)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'meth' (line 464)
        meth_198220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 11), 'meth')
        
        # Obtaining an instance of the builtin type 'list' (line 464)
        list_198221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 464)
        # Adding element type (line 464)
        str_198222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 20), 'str', 'bfgs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_198221, str_198222)
        # Adding element type (line 464)
        str_198223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 28), 'str', 'cg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_198221, str_198223)
        # Adding element type (line 464)
        str_198224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 34), 'str', 'l-bfgs-b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_198221, str_198224)
        # Adding element type (line 464)
        str_198225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 46), 'str', 'tnc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_198221, str_198225)
        # Adding element type (line 464)
        str_198226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 53), 'str', 'dogleg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_198221, str_198226)
        # Adding element type (line 464)
        str_198227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 20), 'str', 'trust-ncg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_198221, str_198227)
        # Adding element type (line 464)
        str_198228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 33), 'str', 'trust-exact')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_198221, str_198228)
        # Adding element type (line 464)
        str_198229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 48), 'str', 'trust-krylov')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_198221, str_198229)
        
        # Applying the binary operator 'in' (line 464)
        result_contains_198230 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 11), 'in', meth_198220, list_198221)
        
        # Testing the type of an if condition (line 464)
        if_condition_198231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 8), result_contains_198230)
        # Assigning a type to the variable 'if_condition_198231' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'if_condition_198231', if_condition_198231)
        # SSA begins for if statement (line 464)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 466)
        # Processing the call arguments (line 466)
        str_198234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 31), 'str', 'gtol')
        # Getting the type of 'tol' (line 466)
        tol_198235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 39), 'tol', False)
        # Processing the call keyword arguments (line 466)
        kwargs_198236 = {}
        # Getting the type of 'options' (line 466)
        options_198232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 466)
        setdefault_198233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), options_198232, 'setdefault')
        # Calling setdefault(args, kwargs) (line 466)
        setdefault_call_result_198237 = invoke(stypy.reporting.localization.Localization(__file__, 466, 12), setdefault_198233, *[str_198234, tol_198235], **kwargs_198236)
        
        # SSA join for if statement (line 464)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'meth' (line 467)
        meth_198238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 11), 'meth')
        
        # Obtaining an instance of the builtin type 'list' (line 467)
        list_198239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 467)
        # Adding element type (line 467)
        str_198240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 20), 'str', 'cobyla')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 19), list_198239, str_198240)
        # Adding element type (line 467)
        str_198241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 30), 'str', '_custom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 19), list_198239, str_198241)
        
        # Applying the binary operator 'in' (line 467)
        result_contains_198242 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 11), 'in', meth_198238, list_198239)
        
        # Testing the type of an if condition (line 467)
        if_condition_198243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 8), result_contains_198242)
        # Assigning a type to the variable 'if_condition_198243' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'if_condition_198243', if_condition_198243)
        # SSA begins for if statement (line 467)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 468)
        # Processing the call arguments (line 468)
        str_198246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 31), 'str', 'tol')
        # Getting the type of 'tol' (line 468)
        tol_198247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 38), 'tol', False)
        # Processing the call keyword arguments (line 468)
        kwargs_198248 = {}
        # Getting the type of 'options' (line 468)
        options_198244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 468)
        setdefault_198245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 12), options_198244, 'setdefault')
        # Calling setdefault(args, kwargs) (line 468)
        setdefault_call_result_198249 = invoke(stypy.reporting.localization.Localization(__file__, 468, 12), setdefault_198245, *[str_198246, tol_198247], **kwargs_198248)
        
        # SSA join for if statement (line 467)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_198172:
            # SSA join for if statement (line 455)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'meth' (line 470)
    meth_198250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 7), 'meth')
    str_198251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 15), 'str', '_custom')
    # Applying the binary operator '==' (line 470)
    result_eq_198252 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 7), '==', meth_198250, str_198251)
    
    # Testing the type of an if condition (line 470)
    if_condition_198253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 4), result_eq_198252)
    # Assigning a type to the variable 'if_condition_198253' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'if_condition_198253', if_condition_198253)
    # SSA begins for if statement (line 470)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to method(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'fun' (line 471)
    fun_198255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 22), 'fun', False)
    # Getting the type of 'x0' (line 471)
    x0_198256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 27), 'x0', False)
    # Processing the call keyword arguments (line 471)
    # Getting the type of 'args' (line 471)
    args_198257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 36), 'args', False)
    keyword_198258 = args_198257
    # Getting the type of 'jac' (line 471)
    jac_198259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 46), 'jac', False)
    keyword_198260 = jac_198259
    # Getting the type of 'hess' (line 471)
    hess_198261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 56), 'hess', False)
    keyword_198262 = hess_198261
    # Getting the type of 'hessp' (line 471)
    hessp_198263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 68), 'hessp', False)
    keyword_198264 = hessp_198263
    # Getting the type of 'bounds' (line 472)
    bounds_198265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 29), 'bounds', False)
    keyword_198266 = bounds_198265
    # Getting the type of 'constraints' (line 472)
    constraints_198267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 49), 'constraints', False)
    keyword_198268 = constraints_198267
    # Getting the type of 'callback' (line 473)
    callback_198269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 31), 'callback', False)
    keyword_198270 = callback_198269
    # Getting the type of 'options' (line 473)
    options_198271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 43), 'options', False)
    kwargs_198272 = {'hessp': keyword_198264, 'options_198271': options_198271, 'args': keyword_198258, 'bounds': keyword_198266, 'callback': keyword_198270, 'hess': keyword_198262, 'jac': keyword_198260, 'constraints': keyword_198268}
    # Getting the type of 'method' (line 471)
    method_198254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 15), 'method', False)
    # Calling method(args, kwargs) (line 471)
    method_call_result_198273 = invoke(stypy.reporting.localization.Localization(__file__, 471, 15), method_198254, *[fun_198255, x0_198256], **kwargs_198272)
    
    # Assigning a type to the variable 'stypy_return_type' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'stypy_return_type', method_call_result_198273)
    # SSA branch for the else part of an if statement (line 470)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 474)
    meth_198274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 9), 'meth')
    str_198275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 17), 'str', 'nelder-mead')
    # Applying the binary operator '==' (line 474)
    result_eq_198276 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 9), '==', meth_198274, str_198275)
    
    # Testing the type of an if condition (line 474)
    if_condition_198277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 9), result_eq_198276)
    # Assigning a type to the variable 'if_condition_198277' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 9), 'if_condition_198277', if_condition_198277)
    # SSA begins for if statement (line 474)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_neldermead(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'fun' (line 475)
    fun_198279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 36), 'fun', False)
    # Getting the type of 'x0' (line 475)
    x0_198280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 41), 'x0', False)
    # Getting the type of 'args' (line 475)
    args_198281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'args', False)
    # Getting the type of 'callback' (line 475)
    callback_198282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 51), 'callback', False)
    # Processing the call keyword arguments (line 475)
    # Getting the type of 'options' (line 475)
    options_198283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 63), 'options', False)
    kwargs_198284 = {'options_198283': options_198283}
    # Getting the type of '_minimize_neldermead' (line 475)
    _minimize_neldermead_198278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), '_minimize_neldermead', False)
    # Calling _minimize_neldermead(args, kwargs) (line 475)
    _minimize_neldermead_call_result_198285 = invoke(stypy.reporting.localization.Localization(__file__, 475, 15), _minimize_neldermead_198278, *[fun_198279, x0_198280, args_198281, callback_198282], **kwargs_198284)
    
    # Assigning a type to the variable 'stypy_return_type' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'stypy_return_type', _minimize_neldermead_call_result_198285)
    # SSA branch for the else part of an if statement (line 474)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 476)
    meth_198286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 9), 'meth')
    str_198287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 17), 'str', 'powell')
    # Applying the binary operator '==' (line 476)
    result_eq_198288 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 9), '==', meth_198286, str_198287)
    
    # Testing the type of an if condition (line 476)
    if_condition_198289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 476, 9), result_eq_198288)
    # Assigning a type to the variable 'if_condition_198289' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 9), 'if_condition_198289', if_condition_198289)
    # SSA begins for if statement (line 476)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_powell(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'fun' (line 477)
    fun_198291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 32), 'fun', False)
    # Getting the type of 'x0' (line 477)
    x0_198292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 37), 'x0', False)
    # Getting the type of 'args' (line 477)
    args_198293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 41), 'args', False)
    # Getting the type of 'callback' (line 477)
    callback_198294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 47), 'callback', False)
    # Processing the call keyword arguments (line 477)
    # Getting the type of 'options' (line 477)
    options_198295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 59), 'options', False)
    kwargs_198296 = {'options_198295': options_198295}
    # Getting the type of '_minimize_powell' (line 477)
    _minimize_powell_198290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), '_minimize_powell', False)
    # Calling _minimize_powell(args, kwargs) (line 477)
    _minimize_powell_call_result_198297 = invoke(stypy.reporting.localization.Localization(__file__, 477, 15), _minimize_powell_198290, *[fun_198291, x0_198292, args_198293, callback_198294], **kwargs_198296)
    
    # Assigning a type to the variable 'stypy_return_type' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'stypy_return_type', _minimize_powell_call_result_198297)
    # SSA branch for the else part of an if statement (line 476)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 478)
    meth_198298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 9), 'meth')
    str_198299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 17), 'str', 'cg')
    # Applying the binary operator '==' (line 478)
    result_eq_198300 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 9), '==', meth_198298, str_198299)
    
    # Testing the type of an if condition (line 478)
    if_condition_198301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 9), result_eq_198300)
    # Assigning a type to the variable 'if_condition_198301' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 9), 'if_condition_198301', if_condition_198301)
    # SSA begins for if statement (line 478)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_cg(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'fun' (line 479)
    fun_198303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'fun', False)
    # Getting the type of 'x0' (line 479)
    x0_198304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'x0', False)
    # Getting the type of 'args' (line 479)
    args_198305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 37), 'args', False)
    # Getting the type of 'jac' (line 479)
    jac_198306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 43), 'jac', False)
    # Getting the type of 'callback' (line 479)
    callback_198307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 48), 'callback', False)
    # Processing the call keyword arguments (line 479)
    # Getting the type of 'options' (line 479)
    options_198308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 60), 'options', False)
    kwargs_198309 = {'options_198308': options_198308}
    # Getting the type of '_minimize_cg' (line 479)
    _minimize_cg_198302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), '_minimize_cg', False)
    # Calling _minimize_cg(args, kwargs) (line 479)
    _minimize_cg_call_result_198310 = invoke(stypy.reporting.localization.Localization(__file__, 479, 15), _minimize_cg_198302, *[fun_198303, x0_198304, args_198305, jac_198306, callback_198307], **kwargs_198309)
    
    # Assigning a type to the variable 'stypy_return_type' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'stypy_return_type', _minimize_cg_call_result_198310)
    # SSA branch for the else part of an if statement (line 478)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 480)
    meth_198311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 9), 'meth')
    str_198312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 17), 'str', 'bfgs')
    # Applying the binary operator '==' (line 480)
    result_eq_198313 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 9), '==', meth_198311, str_198312)
    
    # Testing the type of an if condition (line 480)
    if_condition_198314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 9), result_eq_198313)
    # Assigning a type to the variable 'if_condition_198314' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 9), 'if_condition_198314', if_condition_198314)
    # SSA begins for if statement (line 480)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_bfgs(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'fun' (line 481)
    fun_198316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 30), 'fun', False)
    # Getting the type of 'x0' (line 481)
    x0_198317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 35), 'x0', False)
    # Getting the type of 'args' (line 481)
    args_198318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 39), 'args', False)
    # Getting the type of 'jac' (line 481)
    jac_198319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 45), 'jac', False)
    # Getting the type of 'callback' (line 481)
    callback_198320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 50), 'callback', False)
    # Processing the call keyword arguments (line 481)
    # Getting the type of 'options' (line 481)
    options_198321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 62), 'options', False)
    kwargs_198322 = {'options_198321': options_198321}
    # Getting the type of '_minimize_bfgs' (line 481)
    _minimize_bfgs_198315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), '_minimize_bfgs', False)
    # Calling _minimize_bfgs(args, kwargs) (line 481)
    _minimize_bfgs_call_result_198323 = invoke(stypy.reporting.localization.Localization(__file__, 481, 15), _minimize_bfgs_198315, *[fun_198316, x0_198317, args_198318, jac_198319, callback_198320], **kwargs_198322)
    
    # Assigning a type to the variable 'stypy_return_type' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'stypy_return_type', _minimize_bfgs_call_result_198323)
    # SSA branch for the else part of an if statement (line 480)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 482)
    meth_198324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 9), 'meth')
    str_198325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 17), 'str', 'newton-cg')
    # Applying the binary operator '==' (line 482)
    result_eq_198326 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 9), '==', meth_198324, str_198325)
    
    # Testing the type of an if condition (line 482)
    if_condition_198327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 482, 9), result_eq_198326)
    # Assigning a type to the variable 'if_condition_198327' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 9), 'if_condition_198327', if_condition_198327)
    # SSA begins for if statement (line 482)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_newtoncg(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'fun' (line 483)
    fun_198329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 34), 'fun', False)
    # Getting the type of 'x0' (line 483)
    x0_198330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 39), 'x0', False)
    # Getting the type of 'args' (line 483)
    args_198331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 43), 'args', False)
    # Getting the type of 'jac' (line 483)
    jac_198332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 49), 'jac', False)
    # Getting the type of 'hess' (line 483)
    hess_198333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 54), 'hess', False)
    # Getting the type of 'hessp' (line 483)
    hessp_198334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 60), 'hessp', False)
    # Getting the type of 'callback' (line 483)
    callback_198335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 67), 'callback', False)
    # Processing the call keyword arguments (line 483)
    # Getting the type of 'options' (line 484)
    options_198336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 36), 'options', False)
    kwargs_198337 = {'options_198336': options_198336}
    # Getting the type of '_minimize_newtoncg' (line 483)
    _minimize_newtoncg_198328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 15), '_minimize_newtoncg', False)
    # Calling _minimize_newtoncg(args, kwargs) (line 483)
    _minimize_newtoncg_call_result_198338 = invoke(stypy.reporting.localization.Localization(__file__, 483, 15), _minimize_newtoncg_198328, *[fun_198329, x0_198330, args_198331, jac_198332, hess_198333, hessp_198334, callback_198335], **kwargs_198337)
    
    # Assigning a type to the variable 'stypy_return_type' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'stypy_return_type', _minimize_newtoncg_call_result_198338)
    # SSA branch for the else part of an if statement (line 482)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 485)
    meth_198339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 9), 'meth')
    str_198340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 17), 'str', 'l-bfgs-b')
    # Applying the binary operator '==' (line 485)
    result_eq_198341 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 9), '==', meth_198339, str_198340)
    
    # Testing the type of an if condition (line 485)
    if_condition_198342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 9), result_eq_198341)
    # Assigning a type to the variable 'if_condition_198342' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 9), 'if_condition_198342', if_condition_198342)
    # SSA begins for if statement (line 485)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_lbfgsb(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'fun' (line 486)
    fun_198344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 32), 'fun', False)
    # Getting the type of 'x0' (line 486)
    x0_198345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 37), 'x0', False)
    # Getting the type of 'args' (line 486)
    args_198346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 41), 'args', False)
    # Getting the type of 'jac' (line 486)
    jac_198347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 47), 'jac', False)
    # Getting the type of 'bounds' (line 486)
    bounds_198348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 52), 'bounds', False)
    # Processing the call keyword arguments (line 486)
    # Getting the type of 'callback' (line 487)
    callback_198349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 41), 'callback', False)
    keyword_198350 = callback_198349
    # Getting the type of 'options' (line 487)
    options_198351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 53), 'options', False)
    kwargs_198352 = {'callback': keyword_198350, 'options_198351': options_198351}
    # Getting the type of '_minimize_lbfgsb' (line 486)
    _minimize_lbfgsb_198343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 15), '_minimize_lbfgsb', False)
    # Calling _minimize_lbfgsb(args, kwargs) (line 486)
    _minimize_lbfgsb_call_result_198353 = invoke(stypy.reporting.localization.Localization(__file__, 486, 15), _minimize_lbfgsb_198343, *[fun_198344, x0_198345, args_198346, jac_198347, bounds_198348], **kwargs_198352)
    
    # Assigning a type to the variable 'stypy_return_type' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'stypy_return_type', _minimize_lbfgsb_call_result_198353)
    # SSA branch for the else part of an if statement (line 485)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 488)
    meth_198354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 9), 'meth')
    str_198355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 17), 'str', 'tnc')
    # Applying the binary operator '==' (line 488)
    result_eq_198356 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 9), '==', meth_198354, str_198355)
    
    # Testing the type of an if condition (line 488)
    if_condition_198357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 9), result_eq_198356)
    # Assigning a type to the variable 'if_condition_198357' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 9), 'if_condition_198357', if_condition_198357)
    # SSA begins for if statement (line 488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_tnc(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'fun' (line 489)
    fun_198359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 29), 'fun', False)
    # Getting the type of 'x0' (line 489)
    x0_198360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 34), 'x0', False)
    # Getting the type of 'args' (line 489)
    args_198361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 38), 'args', False)
    # Getting the type of 'jac' (line 489)
    jac_198362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 44), 'jac', False)
    # Getting the type of 'bounds' (line 489)
    bounds_198363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 49), 'bounds', False)
    # Processing the call keyword arguments (line 489)
    # Getting the type of 'callback' (line 489)
    callback_198364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 66), 'callback', False)
    keyword_198365 = callback_198364
    # Getting the type of 'options' (line 490)
    options_198366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 31), 'options', False)
    kwargs_198367 = {'callback': keyword_198365, 'options_198366': options_198366}
    # Getting the type of '_minimize_tnc' (line 489)
    _minimize_tnc_198358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), '_minimize_tnc', False)
    # Calling _minimize_tnc(args, kwargs) (line 489)
    _minimize_tnc_call_result_198368 = invoke(stypy.reporting.localization.Localization(__file__, 489, 15), _minimize_tnc_198358, *[fun_198359, x0_198360, args_198361, jac_198362, bounds_198363], **kwargs_198367)
    
    # Assigning a type to the variable 'stypy_return_type' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'stypy_return_type', _minimize_tnc_call_result_198368)
    # SSA branch for the else part of an if statement (line 488)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 491)
    meth_198369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 9), 'meth')
    str_198370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 17), 'str', 'cobyla')
    # Applying the binary operator '==' (line 491)
    result_eq_198371 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 9), '==', meth_198369, str_198370)
    
    # Testing the type of an if condition (line 491)
    if_condition_198372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 491, 9), result_eq_198371)
    # Assigning a type to the variable 'if_condition_198372' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 9), 'if_condition_198372', if_condition_198372)
    # SSA begins for if statement (line 491)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_cobyla(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'fun' (line 492)
    fun_198374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 32), 'fun', False)
    # Getting the type of 'x0' (line 492)
    x0_198375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 37), 'x0', False)
    # Getting the type of 'args' (line 492)
    args_198376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 41), 'args', False)
    # Getting the type of 'constraints' (line 492)
    constraints_198377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 47), 'constraints', False)
    # Processing the call keyword arguments (line 492)
    # Getting the type of 'options' (line 492)
    options_198378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 62), 'options', False)
    kwargs_198379 = {'options_198378': options_198378}
    # Getting the type of '_minimize_cobyla' (line 492)
    _minimize_cobyla_198373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 15), '_minimize_cobyla', False)
    # Calling _minimize_cobyla(args, kwargs) (line 492)
    _minimize_cobyla_call_result_198380 = invoke(stypy.reporting.localization.Localization(__file__, 492, 15), _minimize_cobyla_198373, *[fun_198374, x0_198375, args_198376, constraints_198377], **kwargs_198379)
    
    # Assigning a type to the variable 'stypy_return_type' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'stypy_return_type', _minimize_cobyla_call_result_198380)
    # SSA branch for the else part of an if statement (line 491)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 493)
    meth_198381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 9), 'meth')
    str_198382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 17), 'str', 'slsqp')
    # Applying the binary operator '==' (line 493)
    result_eq_198383 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 9), '==', meth_198381, str_198382)
    
    # Testing the type of an if condition (line 493)
    if_condition_198384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 493, 9), result_eq_198383)
    # Assigning a type to the variable 'if_condition_198384' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 9), 'if_condition_198384', if_condition_198384)
    # SSA begins for if statement (line 493)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_slsqp(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'fun' (line 494)
    fun_198386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 31), 'fun', False)
    # Getting the type of 'x0' (line 494)
    x0_198387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 36), 'x0', False)
    # Getting the type of 'args' (line 494)
    args_198388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 40), 'args', False)
    # Getting the type of 'jac' (line 494)
    jac_198389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 46), 'jac', False)
    # Getting the type of 'bounds' (line 494)
    bounds_198390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 51), 'bounds', False)
    # Getting the type of 'constraints' (line 495)
    constraints_198391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 31), 'constraints', False)
    # Processing the call keyword arguments (line 494)
    # Getting the type of 'callback' (line 495)
    callback_198392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 53), 'callback', False)
    keyword_198393 = callback_198392
    # Getting the type of 'options' (line 495)
    options_198394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 65), 'options', False)
    kwargs_198395 = {'callback': keyword_198393, 'options_198394': options_198394}
    # Getting the type of '_minimize_slsqp' (line 494)
    _minimize_slsqp_198385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), '_minimize_slsqp', False)
    # Calling _minimize_slsqp(args, kwargs) (line 494)
    _minimize_slsqp_call_result_198396 = invoke(stypy.reporting.localization.Localization(__file__, 494, 15), _minimize_slsqp_198385, *[fun_198386, x0_198387, args_198388, jac_198389, bounds_198390, constraints_198391], **kwargs_198395)
    
    # Assigning a type to the variable 'stypy_return_type' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'stypy_return_type', _minimize_slsqp_call_result_198396)
    # SSA branch for the else part of an if statement (line 493)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 496)
    meth_198397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 9), 'meth')
    str_198398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 17), 'str', 'dogleg')
    # Applying the binary operator '==' (line 496)
    result_eq_198399 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 9), '==', meth_198397, str_198398)
    
    # Testing the type of an if condition (line 496)
    if_condition_198400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 9), result_eq_198399)
    # Assigning a type to the variable 'if_condition_198400' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 9), 'if_condition_198400', if_condition_198400)
    # SSA begins for if statement (line 496)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_dogleg(...): (line 497)
    # Processing the call arguments (line 497)
    # Getting the type of 'fun' (line 497)
    fun_198402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 32), 'fun', False)
    # Getting the type of 'x0' (line 497)
    x0_198403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 37), 'x0', False)
    # Getting the type of 'args' (line 497)
    args_198404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 41), 'args', False)
    # Getting the type of 'jac' (line 497)
    jac_198405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 47), 'jac', False)
    # Getting the type of 'hess' (line 497)
    hess_198406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 52), 'hess', False)
    # Processing the call keyword arguments (line 497)
    # Getting the type of 'callback' (line 498)
    callback_198407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 41), 'callback', False)
    keyword_198408 = callback_198407
    # Getting the type of 'options' (line 498)
    options_198409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 53), 'options', False)
    kwargs_198410 = {'callback': keyword_198408, 'options_198409': options_198409}
    # Getting the type of '_minimize_dogleg' (line 497)
    _minimize_dogleg_198401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), '_minimize_dogleg', False)
    # Calling _minimize_dogleg(args, kwargs) (line 497)
    _minimize_dogleg_call_result_198411 = invoke(stypy.reporting.localization.Localization(__file__, 497, 15), _minimize_dogleg_198401, *[fun_198402, x0_198403, args_198404, jac_198405, hess_198406], **kwargs_198410)
    
    # Assigning a type to the variable 'stypy_return_type' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'stypy_return_type', _minimize_dogleg_call_result_198411)
    # SSA branch for the else part of an if statement (line 496)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 499)
    meth_198412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 9), 'meth')
    str_198413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 17), 'str', 'trust-ncg')
    # Applying the binary operator '==' (line 499)
    result_eq_198414 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 9), '==', meth_198412, str_198413)
    
    # Testing the type of an if condition (line 499)
    if_condition_198415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 9), result_eq_198414)
    # Assigning a type to the variable 'if_condition_198415' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 9), 'if_condition_198415', if_condition_198415)
    # SSA begins for if statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_trust_ncg(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'fun' (line 500)
    fun_198417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 35), 'fun', False)
    # Getting the type of 'x0' (line 500)
    x0_198418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 40), 'x0', False)
    # Getting the type of 'args' (line 500)
    args_198419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 44), 'args', False)
    # Getting the type of 'jac' (line 500)
    jac_198420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 50), 'jac', False)
    # Getting the type of 'hess' (line 500)
    hess_198421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 55), 'hess', False)
    # Getting the type of 'hessp' (line 500)
    hessp_198422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 61), 'hessp', False)
    # Processing the call keyword arguments (line 500)
    # Getting the type of 'callback' (line 501)
    callback_198423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 44), 'callback', False)
    keyword_198424 = callback_198423
    # Getting the type of 'options' (line 501)
    options_198425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 56), 'options', False)
    kwargs_198426 = {'callback': keyword_198424, 'options_198425': options_198425}
    # Getting the type of '_minimize_trust_ncg' (line 500)
    _minimize_trust_ncg_198416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), '_minimize_trust_ncg', False)
    # Calling _minimize_trust_ncg(args, kwargs) (line 500)
    _minimize_trust_ncg_call_result_198427 = invoke(stypy.reporting.localization.Localization(__file__, 500, 15), _minimize_trust_ncg_198416, *[fun_198417, x0_198418, args_198419, jac_198420, hess_198421, hessp_198422], **kwargs_198426)
    
    # Assigning a type to the variable 'stypy_return_type' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'stypy_return_type', _minimize_trust_ncg_call_result_198427)
    # SSA branch for the else part of an if statement (line 499)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 502)
    meth_198428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 9), 'meth')
    str_198429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 17), 'str', 'trust-krylov')
    # Applying the binary operator '==' (line 502)
    result_eq_198430 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 9), '==', meth_198428, str_198429)
    
    # Testing the type of an if condition (line 502)
    if_condition_198431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 9), result_eq_198430)
    # Assigning a type to the variable 'if_condition_198431' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 9), 'if_condition_198431', if_condition_198431)
    # SSA begins for if statement (line 502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_trust_krylov(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'fun' (line 503)
    fun_198433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 38), 'fun', False)
    # Getting the type of 'x0' (line 503)
    x0_198434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 43), 'x0', False)
    # Getting the type of 'args' (line 503)
    args_198435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 47), 'args', False)
    # Getting the type of 'jac' (line 503)
    jac_198436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 53), 'jac', False)
    # Getting the type of 'hess' (line 503)
    hess_198437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 58), 'hess', False)
    # Getting the type of 'hessp' (line 503)
    hessp_198438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 64), 'hessp', False)
    # Processing the call keyword arguments (line 503)
    # Getting the type of 'callback' (line 504)
    callback_198439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 47), 'callback', False)
    keyword_198440 = callback_198439
    # Getting the type of 'options' (line 504)
    options_198441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 59), 'options', False)
    kwargs_198442 = {'options_198441': options_198441, 'callback': keyword_198440}
    # Getting the type of '_minimize_trust_krylov' (line 503)
    _minimize_trust_krylov_198432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 15), '_minimize_trust_krylov', False)
    # Calling _minimize_trust_krylov(args, kwargs) (line 503)
    _minimize_trust_krylov_call_result_198443 = invoke(stypy.reporting.localization.Localization(__file__, 503, 15), _minimize_trust_krylov_198432, *[fun_198433, x0_198434, args_198435, jac_198436, hess_198437, hessp_198438], **kwargs_198442)
    
    # Assigning a type to the variable 'stypy_return_type' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'stypy_return_type', _minimize_trust_krylov_call_result_198443)
    # SSA branch for the else part of an if statement (line 502)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 505)
    meth_198444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 9), 'meth')
    str_198445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 17), 'str', 'trust-exact')
    # Applying the binary operator '==' (line 505)
    result_eq_198446 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 9), '==', meth_198444, str_198445)
    
    # Testing the type of an if condition (line 505)
    if_condition_198447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 9), result_eq_198446)
    # Assigning a type to the variable 'if_condition_198447' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 9), 'if_condition_198447', if_condition_198447)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_trustregion_exact(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'fun' (line 506)
    fun_198449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 43), 'fun', False)
    # Getting the type of 'x0' (line 506)
    x0_198450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 48), 'x0', False)
    # Getting the type of 'args' (line 506)
    args_198451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 52), 'args', False)
    # Getting the type of 'jac' (line 506)
    jac_198452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 58), 'jac', False)
    # Getting the type of 'hess' (line 506)
    hess_198453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 63), 'hess', False)
    # Processing the call keyword arguments (line 506)
    # Getting the type of 'callback' (line 507)
    callback_198454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 52), 'callback', False)
    keyword_198455 = callback_198454
    # Getting the type of 'options' (line 507)
    options_198456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 64), 'options', False)
    kwargs_198457 = {'callback': keyword_198455, 'options_198456': options_198456}
    # Getting the type of '_minimize_trustregion_exact' (line 506)
    _minimize_trustregion_exact_198448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 15), '_minimize_trustregion_exact', False)
    # Calling _minimize_trustregion_exact(args, kwargs) (line 506)
    _minimize_trustregion_exact_call_result_198458 = invoke(stypy.reporting.localization.Localization(__file__, 506, 15), _minimize_trustregion_exact_198448, *[fun_198449, x0_198450, args_198451, jac_198452, hess_198453], **kwargs_198457)
    
    # Assigning a type to the variable 'stypy_return_type' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'stypy_return_type', _minimize_trustregion_exact_call_result_198458)
    # SSA branch for the else part of an if statement (line 505)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 509)
    # Processing the call arguments (line 509)
    str_198460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 25), 'str', 'Unknown solver %s')
    # Getting the type of 'method' (line 509)
    method_198461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 47), 'method', False)
    # Applying the binary operator '%' (line 509)
    result_mod_198462 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 25), '%', str_198460, method_198461)
    
    # Processing the call keyword arguments (line 509)
    kwargs_198463 = {}
    # Getting the type of 'ValueError' (line 509)
    ValueError_198459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 509)
    ValueError_call_result_198464 = invoke(stypy.reporting.localization.Localization(__file__, 509, 14), ValueError_198459, *[result_mod_198462], **kwargs_198463)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 509, 8), ValueError_call_result_198464, 'raise parameter', BaseException)
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 502)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 496)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 493)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 491)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 488)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 485)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 482)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 480)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 478)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 476)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 474)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 470)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'minimize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minimize' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_198465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_198465)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minimize'
    return stypy_return_type_198465

# Assigning a type to the variable 'minimize' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'minimize', minimize)

@norecursion
def minimize_scalar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 512)
    None_198466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 33), 'None')
    # Getting the type of 'None' (line 512)
    None_198467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 46), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 512)
    tuple_198468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 512)
    
    str_198469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 27), 'str', 'brent')
    # Getting the type of 'None' (line 513)
    None_198470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 40), 'None')
    # Getting the type of 'None' (line 513)
    None_198471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 54), 'None')
    defaults = [None_198466, None_198467, tuple_198468, str_198469, None_198470, None_198471]
    # Create a new context for function 'minimize_scalar'
    module_type_store = module_type_store.open_function_context('minimize_scalar', 512, 0, False)
    
    # Passed parameters checking function
    minimize_scalar.stypy_localization = localization
    minimize_scalar.stypy_type_of_self = None
    minimize_scalar.stypy_type_store = module_type_store
    minimize_scalar.stypy_function_name = 'minimize_scalar'
    minimize_scalar.stypy_param_names_list = ['fun', 'bracket', 'bounds', 'args', 'method', 'tol', 'options']
    minimize_scalar.stypy_varargs_param_name = None
    minimize_scalar.stypy_kwargs_param_name = None
    minimize_scalar.stypy_call_defaults = defaults
    minimize_scalar.stypy_call_varargs = varargs
    minimize_scalar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minimize_scalar', ['fun', 'bracket', 'bounds', 'args', 'method', 'tol', 'options'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minimize_scalar', localization, ['fun', 'bracket', 'bounds', 'args', 'method', 'tol', 'options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minimize_scalar(...)' code ##################

    str_198472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, (-1)), 'str', "Minimization of scalar function of one variable.\n\n    Parameters\n    ----------\n    fun : callable\n        Objective function.\n        Scalar function, must return a scalar.\n    bracket : sequence, optional\n        For methods 'brent' and 'golden', `bracket` defines the bracketing\n        interval and can either have three items ``(a, b, c)`` so that\n        ``a < b < c`` and ``fun(b) < fun(a), fun(c)`` or two items ``a`` and\n        ``c`` which are assumed to be a starting interval for a downhill\n        bracket search (see `bracket`); it doesn't always mean that the\n        obtained solution will satisfy ``a <= x <= c``.\n    bounds : sequence, optional\n        For method 'bounded', `bounds` is mandatory and must have two items\n        corresponding to the optimization bounds.\n    args : tuple, optional\n        Extra arguments passed to the objective function.\n    method : str or callable, optional\n        Type of solver.  Should be one of:\n\n            - 'Brent'     :ref:`(see here) <optimize.minimize_scalar-brent>`\n            - 'Bounded'   :ref:`(see here) <optimize.minimize_scalar-bounded>`\n            - 'Golden'    :ref:`(see here) <optimize.minimize_scalar-golden>`\n            - custom - a callable object (added in version 0.14.0), see below\n\n    tol : float, optional\n        Tolerance for termination. For detailed control, use solver-specific\n        options.\n    options : dict, optional\n        A dictionary of solver options.\n\n            maxiter : int\n                Maximum number of iterations to perform.\n            disp : bool\n                Set to True to print convergence messages.\n\n        See :func:`show_options()` for solver-specific options.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a ``OptimizeResult`` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.\n\n    See also\n    --------\n    minimize : Interface to minimization algorithms for scalar multivariate\n        functions\n    show_options : Additional options accepted by the solvers\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    'method' parameter. The default method is *Brent*.\n\n    Method :ref:`Brent <optimize.minimize_scalar-brent>` uses Brent's\n    algorithm to find a local minimum.  The algorithm uses inverse\n    parabolic interpolation when possible to speed up convergence of\n    the golden section method.\n\n    Method :ref:`Golden <optimize.minimize_scalar-golden>` uses the\n    golden section search technique. It uses analog of the bisection\n    method to decrease the bracketed interval. It is usually\n    preferable to use the *Brent* method.\n\n    Method :ref:`Bounded <optimize.minimize_scalar-bounded>` can\n    perform bounded minimization. It uses the Brent method to find a\n    local minimum in the interval x1 < xopt < x2.\n\n    **Custom minimizers**\n\n    It may be useful to pass a custom minimization method, for example\n    when using some library frontend to minimize_scalar.  You can simply\n    pass a callable as the ``method`` parameter.\n\n    The callable is called as ``method(fun, args, **kwargs, **options)``\n    where ``kwargs`` corresponds to any other parameters passed to `minimize`\n    (such as `bracket`, `tol`, etc.), except the `options` dict, which has\n    its contents also passed as `method` parameters pair by pair.  The method\n    shall return an ``OptimizeResult`` object.\n\n    The provided `method` callable must be able to accept (and possibly ignore)\n    arbitrary parameters; the set of parameters accepted by `minimize` may\n    expand in future versions and then these parameters will be passed to\n    the method.  You can find an example in the scipy.optimize tutorial.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    Consider the problem of minimizing the following function.\n\n    >>> def f(x):\n    ...     return (x - 2) * x * (x + 2)**2\n\n    Using the *Brent* method, we find the local minimum as:\n\n    >>> from scipy.optimize import minimize_scalar\n    >>> res = minimize_scalar(f)\n    >>> res.x\n    1.28077640403\n\n    Using the *Bounded* method, we find a local minimum with specified\n    bounds as:\n\n    >>> res = minimize_scalar(f, bounds=(-3, -1), method='bounded')\n    >>> res.x\n    -2.0000002026\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 629)
    # Getting the type of 'tuple' (line 629)
    tuple_198473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 28), 'tuple')
    # Getting the type of 'args' (line 629)
    args_198474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 22), 'args')
    
    (may_be_198475, more_types_in_union_198476) = may_not_be_subtype(tuple_198473, args_198474)

    if may_be_198475:

        if more_types_in_union_198476:
            # Runtime conditional SSA (line 629)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'args' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'args', remove_subtype_from_union(args_198474, tuple))
        
        # Assigning a Tuple to a Name (line 630):
        
        # Obtaining an instance of the builtin type 'tuple' (line 630)
        tuple_198477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 630)
        # Adding element type (line 630)
        # Getting the type of 'args' (line 630)
        args_198478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 16), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 16), tuple_198477, args_198478)
        
        # Assigning a type to the variable 'args' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'args', tuple_198477)

        if more_types_in_union_198476:
            # SSA join for if statement (line 629)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to callable(...): (line 632)
    # Processing the call arguments (line 632)
    # Getting the type of 'method' (line 632)
    method_198480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 'method', False)
    # Processing the call keyword arguments (line 632)
    kwargs_198481 = {}
    # Getting the type of 'callable' (line 632)
    callable_198479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 7), 'callable', False)
    # Calling callable(args, kwargs) (line 632)
    callable_call_result_198482 = invoke(stypy.reporting.localization.Localization(__file__, 632, 7), callable_198479, *[method_198480], **kwargs_198481)
    
    # Testing the type of an if condition (line 632)
    if_condition_198483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 632, 4), callable_call_result_198482)
    # Assigning a type to the variable 'if_condition_198483' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 4), 'if_condition_198483', if_condition_198483)
    # SSA begins for if statement (line 632)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 633):
    str_198484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 15), 'str', '_custom')
    # Assigning a type to the variable 'meth' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'meth', str_198484)
    # SSA branch for the else part of an if statement (line 632)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 635):
    
    # Call to lower(...): (line 635)
    # Processing the call keyword arguments (line 635)
    kwargs_198487 = {}
    # Getting the type of 'method' (line 635)
    method_198485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'method', False)
    # Obtaining the member 'lower' of a type (line 635)
    lower_198486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 15), method_198485, 'lower')
    # Calling lower(args, kwargs) (line 635)
    lower_call_result_198488 = invoke(stypy.reporting.localization.Localization(__file__, 635, 15), lower_198486, *[], **kwargs_198487)
    
    # Assigning a type to the variable 'meth' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'meth', lower_call_result_198488)
    # SSA join for if statement (line 632)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 636)
    # Getting the type of 'options' (line 636)
    options_198489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 7), 'options')
    # Getting the type of 'None' (line 636)
    None_198490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 18), 'None')
    
    (may_be_198491, more_types_in_union_198492) = may_be_none(options_198489, None_198490)

    if may_be_198491:

        if more_types_in_union_198492:
            # Runtime conditional SSA (line 636)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 637):
        
        # Obtaining an instance of the builtin type 'dict' (line 637)
        dict_198493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 637)
        
        # Assigning a type to the variable 'options' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'options', dict_198493)

        if more_types_in_union_198492:
            # SSA join for if statement (line 636)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 639)
    # Getting the type of 'tol' (line 639)
    tol_198494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'tol')
    # Getting the type of 'None' (line 639)
    None_198495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 18), 'None')
    
    (may_be_198496, more_types_in_union_198497) = may_not_be_none(tol_198494, None_198495)

    if may_be_198496:

        if more_types_in_union_198497:
            # Runtime conditional SSA (line 639)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 640):
        
        # Call to dict(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'options' (line 640)
        options_198499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 23), 'options', False)
        # Processing the call keyword arguments (line 640)
        kwargs_198500 = {}
        # Getting the type of 'dict' (line 640)
        dict_198498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 18), 'dict', False)
        # Calling dict(args, kwargs) (line 640)
        dict_call_result_198501 = invoke(stypy.reporting.localization.Localization(__file__, 640, 18), dict_198498, *[options_198499], **kwargs_198500)
        
        # Assigning a type to the variable 'options' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'options', dict_call_result_198501)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'meth' (line 641)
        meth_198502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 11), 'meth')
        str_198503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 19), 'str', 'bounded')
        # Applying the binary operator '==' (line 641)
        result_eq_198504 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 11), '==', meth_198502, str_198503)
        
        
        str_198505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 33), 'str', 'xatol')
        # Getting the type of 'options' (line 641)
        options_198506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 48), 'options')
        # Applying the binary operator 'notin' (line 641)
        result_contains_198507 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 33), 'notin', str_198505, options_198506)
        
        # Applying the binary operator 'and' (line 641)
        result_and_keyword_198508 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 11), 'and', result_eq_198504, result_contains_198507)
        
        # Testing the type of an if condition (line 641)
        if_condition_198509 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 641, 8), result_and_keyword_198508)
        # Assigning a type to the variable 'if_condition_198509' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'if_condition_198509', if_condition_198509)
        # SSA begins for if statement (line 641)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 642)
        # Processing the call arguments (line 642)
        str_198511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 17), 'str', "Method 'bounded' does not support relative tolerance in x; defaulting to absolute tolerance.")
        # Getting the type of 'RuntimeWarning' (line 643)
        RuntimeWarning_198512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 54), 'RuntimeWarning', False)
        # Processing the call keyword arguments (line 642)
        kwargs_198513 = {}
        # Getting the type of 'warn' (line 642)
        warn_198510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'warn', False)
        # Calling warn(args, kwargs) (line 642)
        warn_call_result_198514 = invoke(stypy.reporting.localization.Localization(__file__, 642, 12), warn_198510, *[str_198511, RuntimeWarning_198512], **kwargs_198513)
        
        
        # Assigning a Name to a Subscript (line 644):
        # Getting the type of 'tol' (line 644)
        tol_198515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 31), 'tol')
        # Getting the type of 'options' (line 644)
        options_198516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'options')
        str_198517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 20), 'str', 'xatol')
        # Storing an element on a container (line 644)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 12), options_198516, (str_198517, tol_198515))
        # SSA branch for the else part of an if statement (line 641)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'meth' (line 645)
        meth_198518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 13), 'meth')
        str_198519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 21), 'str', '_custom')
        # Applying the binary operator '==' (line 645)
        result_eq_198520 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 13), '==', meth_198518, str_198519)
        
        # Testing the type of an if condition (line 645)
        if_condition_198521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 645, 13), result_eq_198520)
        # Assigning a type to the variable 'if_condition_198521' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 13), 'if_condition_198521', if_condition_198521)
        # SSA begins for if statement (line 645)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 646)
        # Processing the call arguments (line 646)
        str_198524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 31), 'str', 'tol')
        # Getting the type of 'tol' (line 646)
        tol_198525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 38), 'tol', False)
        # Processing the call keyword arguments (line 646)
        kwargs_198526 = {}
        # Getting the type of 'options' (line 646)
        options_198522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 646)
        setdefault_198523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 12), options_198522, 'setdefault')
        # Calling setdefault(args, kwargs) (line 646)
        setdefault_call_result_198527 = invoke(stypy.reporting.localization.Localization(__file__, 646, 12), setdefault_198523, *[str_198524, tol_198525], **kwargs_198526)
        
        # SSA branch for the else part of an if statement (line 645)
        module_type_store.open_ssa_branch('else')
        
        # Call to setdefault(...): (line 648)
        # Processing the call arguments (line 648)
        str_198530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 31), 'str', 'xtol')
        # Getting the type of 'tol' (line 648)
        tol_198531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 39), 'tol', False)
        # Processing the call keyword arguments (line 648)
        kwargs_198532 = {}
        # Getting the type of 'options' (line 648)
        options_198528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'options', False)
        # Obtaining the member 'setdefault' of a type (line 648)
        setdefault_198529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 12), options_198528, 'setdefault')
        # Calling setdefault(args, kwargs) (line 648)
        setdefault_call_result_198533 = invoke(stypy.reporting.localization.Localization(__file__, 648, 12), setdefault_198529, *[str_198530, tol_198531], **kwargs_198532)
        
        # SSA join for if statement (line 645)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 641)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_198497:
            # SSA join for if statement (line 639)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'meth' (line 650)
    meth_198534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 7), 'meth')
    str_198535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 15), 'str', '_custom')
    # Applying the binary operator '==' (line 650)
    result_eq_198536 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 7), '==', meth_198534, str_198535)
    
    # Testing the type of an if condition (line 650)
    if_condition_198537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 4), result_eq_198536)
    # Assigning a type to the variable 'if_condition_198537' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'if_condition_198537', if_condition_198537)
    # SSA begins for if statement (line 650)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to method(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'fun' (line 651)
    fun_198539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 22), 'fun', False)
    # Processing the call keyword arguments (line 651)
    # Getting the type of 'args' (line 651)
    args_198540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 32), 'args', False)
    keyword_198541 = args_198540
    # Getting the type of 'bracket' (line 651)
    bracket_198542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 46), 'bracket', False)
    keyword_198543 = bracket_198542
    # Getting the type of 'bounds' (line 651)
    bounds_198544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 62), 'bounds', False)
    keyword_198545 = bounds_198544
    # Getting the type of 'options' (line 651)
    options_198546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 72), 'options', False)
    kwargs_198547 = {'args': keyword_198541, 'bounds': keyword_198545, 'options_198546': options_198546, 'bracket': keyword_198543}
    # Getting the type of 'method' (line 651)
    method_198538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 15), 'method', False)
    # Calling method(args, kwargs) (line 651)
    method_call_result_198548 = invoke(stypy.reporting.localization.Localization(__file__, 651, 15), method_198538, *[fun_198539], **kwargs_198547)
    
    # Assigning a type to the variable 'stypy_return_type' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'stypy_return_type', method_call_result_198548)
    # SSA branch for the else part of an if statement (line 650)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 652)
    meth_198549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 9), 'meth')
    str_198550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 17), 'str', 'brent')
    # Applying the binary operator '==' (line 652)
    result_eq_198551 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 9), '==', meth_198549, str_198550)
    
    # Testing the type of an if condition (line 652)
    if_condition_198552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 652, 9), result_eq_198551)
    # Assigning a type to the variable 'if_condition_198552' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 9), 'if_condition_198552', if_condition_198552)
    # SSA begins for if statement (line 652)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_scalar_brent(...): (line 653)
    # Processing the call arguments (line 653)
    # Getting the type of 'fun' (line 653)
    fun_198554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 38), 'fun', False)
    # Getting the type of 'bracket' (line 653)
    bracket_198555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 43), 'bracket', False)
    # Getting the type of 'args' (line 653)
    args_198556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 52), 'args', False)
    # Processing the call keyword arguments (line 653)
    # Getting the type of 'options' (line 653)
    options_198557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 60), 'options', False)
    kwargs_198558 = {'options_198557': options_198557}
    # Getting the type of '_minimize_scalar_brent' (line 653)
    _minimize_scalar_brent_198553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 15), '_minimize_scalar_brent', False)
    # Calling _minimize_scalar_brent(args, kwargs) (line 653)
    _minimize_scalar_brent_call_result_198559 = invoke(stypy.reporting.localization.Localization(__file__, 653, 15), _minimize_scalar_brent_198553, *[fun_198554, bracket_198555, args_198556], **kwargs_198558)
    
    # Assigning a type to the variable 'stypy_return_type' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'stypy_return_type', _minimize_scalar_brent_call_result_198559)
    # SSA branch for the else part of an if statement (line 652)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 654)
    meth_198560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 9), 'meth')
    str_198561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 17), 'str', 'bounded')
    # Applying the binary operator '==' (line 654)
    result_eq_198562 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 9), '==', meth_198560, str_198561)
    
    # Testing the type of an if condition (line 654)
    if_condition_198563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 9), result_eq_198562)
    # Assigning a type to the variable 'if_condition_198563' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 9), 'if_condition_198563', if_condition_198563)
    # SSA begins for if statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 655)
    # Getting the type of 'bounds' (line 655)
    bounds_198564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 11), 'bounds')
    # Getting the type of 'None' (line 655)
    None_198565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), 'None')
    
    (may_be_198566, more_types_in_union_198567) = may_be_none(bounds_198564, None_198565)

    if may_be_198566:

        if more_types_in_union_198567:
            # Runtime conditional SSA (line 655)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 656)
        # Processing the call arguments (line 656)
        str_198569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 29), 'str', 'The `bounds` parameter is mandatory for method `bounded`.')
        # Processing the call keyword arguments (line 656)
        kwargs_198570 = {}
        # Getting the type of 'ValueError' (line 656)
        ValueError_198568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 656)
        ValueError_call_result_198571 = invoke(stypy.reporting.localization.Localization(__file__, 656, 18), ValueError_198568, *[str_198569], **kwargs_198570)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 656, 12), ValueError_call_result_198571, 'raise parameter', BaseException)

        if more_types_in_union_198567:
            # SSA join for if statement (line 655)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to _minimize_scalar_bounded(...): (line 658)
    # Processing the call arguments (line 658)
    # Getting the type of 'fun' (line 658)
    fun_198573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 40), 'fun', False)
    # Getting the type of 'bounds' (line 658)
    bounds_198574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 45), 'bounds', False)
    # Getting the type of 'args' (line 658)
    args_198575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 53), 'args', False)
    # Processing the call keyword arguments (line 658)
    # Getting the type of 'options' (line 658)
    options_198576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 61), 'options', False)
    kwargs_198577 = {'options_198576': options_198576}
    # Getting the type of '_minimize_scalar_bounded' (line 658)
    _minimize_scalar_bounded_198572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 15), '_minimize_scalar_bounded', False)
    # Calling _minimize_scalar_bounded(args, kwargs) (line 658)
    _minimize_scalar_bounded_call_result_198578 = invoke(stypy.reporting.localization.Localization(__file__, 658, 15), _minimize_scalar_bounded_198572, *[fun_198573, bounds_198574, args_198575], **kwargs_198577)
    
    # Assigning a type to the variable 'stypy_return_type' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'stypy_return_type', _minimize_scalar_bounded_call_result_198578)
    # SSA branch for the else part of an if statement (line 654)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 659)
    meth_198579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 9), 'meth')
    str_198580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 17), 'str', 'golden')
    # Applying the binary operator '==' (line 659)
    result_eq_198581 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 9), '==', meth_198579, str_198580)
    
    # Testing the type of an if condition (line 659)
    if_condition_198582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 9), result_eq_198581)
    # Assigning a type to the variable 'if_condition_198582' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 9), 'if_condition_198582', if_condition_198582)
    # SSA begins for if statement (line 659)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_scalar_golden(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'fun' (line 660)
    fun_198584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 39), 'fun', False)
    # Getting the type of 'bracket' (line 660)
    bracket_198585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 44), 'bracket', False)
    # Getting the type of 'args' (line 660)
    args_198586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 53), 'args', False)
    # Processing the call keyword arguments (line 660)
    # Getting the type of 'options' (line 660)
    options_198587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 61), 'options', False)
    kwargs_198588 = {'options_198587': options_198587}
    # Getting the type of '_minimize_scalar_golden' (line 660)
    _minimize_scalar_golden_198583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 15), '_minimize_scalar_golden', False)
    # Calling _minimize_scalar_golden(args, kwargs) (line 660)
    _minimize_scalar_golden_call_result_198589 = invoke(stypy.reporting.localization.Localization(__file__, 660, 15), _minimize_scalar_golden_198583, *[fun_198584, bracket_198585, args_198586], **kwargs_198588)
    
    # Assigning a type to the variable 'stypy_return_type' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'stypy_return_type', _minimize_scalar_golden_call_result_198589)
    # SSA branch for the else part of an if statement (line 659)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 662)
    # Processing the call arguments (line 662)
    str_198591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 25), 'str', 'Unknown solver %s')
    # Getting the type of 'method' (line 662)
    method_198592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 47), 'method', False)
    # Applying the binary operator '%' (line 662)
    result_mod_198593 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 25), '%', str_198591, method_198592)
    
    # Processing the call keyword arguments (line 662)
    kwargs_198594 = {}
    # Getting the type of 'ValueError' (line 662)
    ValueError_198590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 662)
    ValueError_call_result_198595 = invoke(stypy.reporting.localization.Localization(__file__, 662, 14), ValueError_198590, *[result_mod_198593], **kwargs_198594)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 662, 8), ValueError_call_result_198595, 'raise parameter', BaseException)
    # SSA join for if statement (line 659)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 652)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 650)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'minimize_scalar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minimize_scalar' in the type store
    # Getting the type of 'stypy_return_type' (line 512)
    stypy_return_type_198596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_198596)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minimize_scalar'
    return stypy_return_type_198596

# Assigning a type to the variable 'minimize_scalar' (line 512)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 0), 'minimize_scalar', minimize_scalar)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
