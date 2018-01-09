
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: import inspect
3: import numpy as np
4: from .bdf import BDF
5: from .radau import Radau
6: from .rk import RK23, RK45
7: from .lsoda import LSODA
8: from scipy.optimize import OptimizeResult
9: from .common import EPS, OdeSolution
10: from .base import OdeSolver
11: 
12: 
13: METHODS = {'RK23': RK23,
14:            'RK45': RK45,
15:            'Radau': Radau,
16:            'BDF': BDF,
17:            'LSODA': LSODA}
18: 
19: 
20: MESSAGES = {0: "The solver successfully reached the interval end.",
21:             1: "A termination event occurred."}
22: 
23: 
24: class OdeResult(OptimizeResult):
25:     pass
26: 
27: 
28: def prepare_events(events):
29:     '''Standardize event functions and extract is_terminal and direction.'''
30:     if callable(events):
31:         events = (events,)
32: 
33:     if events is not None:
34:         is_terminal = np.empty(len(events), dtype=bool)
35:         direction = np.empty(len(events))
36:         for i, event in enumerate(events):
37:             try:
38:                 is_terminal[i] = event.terminal
39:             except AttributeError:
40:                 is_terminal[i] = False
41: 
42:             try:
43:                 direction[i] = event.direction
44:             except AttributeError:
45:                 direction[i] = 0
46:     else:
47:         is_terminal = None
48:         direction = None
49: 
50:     return events, is_terminal, direction
51: 
52: 
53: def solve_event_equation(event, sol, t_old, t):
54:     '''Solve an equation corresponding to an ODE event.
55: 
56:     The equation is ``event(t, y(t)) = 0``, here ``y(t)`` is known from an
57:     ODE solver using some sort of interpolation. It is solved by
58:     `scipy.optimize.brentq` with xtol=atol=4*EPS.
59: 
60:     Parameters
61:     ----------
62:     event : callable
63:         Function ``event(t, y)``.
64:     sol : callable
65:         Function ``sol(t)`` which evaluates an ODE solution between `t_old`
66:         and  `t`.
67:     t_old, t : float
68:         Previous and new values of time. They will be used as a bracketing
69:         interval.
70: 
71:     Returns
72:     -------
73:     root : float
74:         Found solution.
75:     '''
76:     from scipy.optimize import brentq
77:     return brentq(lambda t: event(t, sol(t)), t_old, t,
78:                   xtol=4 * EPS, rtol=4 * EPS)
79: 
80: 
81: def handle_events(sol, events, active_events, is_terminal, t_old, t):
82:     '''Helper function to handle events.
83: 
84:     Parameters
85:     ----------
86:     sol : DenseOutput
87:         Function ``sol(t)`` which evaluates an ODE solution between `t_old`
88:         and  `t`.
89:     events : list of callables, length n_events
90:         Event functions with signatures ``event(t, y)``.
91:     active_events : ndarray
92:         Indices of events which occurred.
93:     is_terminal : ndarray, shape (n_events,)
94:         Which events are terminal.
95:     t_old, t : float
96:         Previous and new values of time.
97: 
98:     Returns
99:     -------
100:     root_indices : ndarray
101:         Indices of events which take zero between `t_old` and `t` and before
102:         a possible termination.
103:     roots : ndarray
104:         Values of t at which events occurred.
105:     terminate : bool
106:         Whether a terminal event occurred.
107:     '''
108:     roots = []
109:     for event_index in active_events:
110:         roots.append(solve_event_equation(events[event_index], sol, t_old, t))
111: 
112:     roots = np.asarray(roots)
113: 
114:     if np.any(is_terminal[active_events]):
115:         if t > t_old:
116:             order = np.argsort(roots)
117:         else:
118:             order = np.argsort(-roots)
119:         active_events = active_events[order]
120:         roots = roots[order]
121:         t = np.nonzero(is_terminal[active_events])[0][0]
122:         active_events = active_events[:t + 1]
123:         roots = roots[:t + 1]
124:         terminate = True
125:     else:
126:         terminate = False
127: 
128:     return active_events, roots, terminate
129: 
130: 
131: def find_active_events(g, g_new, direction):
132:     '''Find which event occurred during an integration step.
133: 
134:     Parameters
135:     ----------
136:     g, g_new : array_like, shape (n_events,)
137:         Values of event functions at a current and next points.
138:     direction : ndarray, shape (n_events,)
139:         Event "direction" according to the definition in `solve_ivp`.
140: 
141:     Returns
142:     -------
143:     active_events : ndarray
144:         Indices of events which occurred during the step.
145:     '''
146:     g, g_new = np.asarray(g), np.asarray(g_new)
147:     up = (g <= 0) & (g_new >= 0)
148:     down = (g >= 0) & (g_new <= 0)
149:     either = up | down
150:     mask = (up & (direction > 0) |
151:             down & (direction < 0) |
152:             either & (direction == 0))
153: 
154:     return np.nonzero(mask)[0]
155: 
156: 
157: def solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
158:               events=None, vectorized=False, **options):
159:     '''Solve an initial value problem for a system of ODEs.
160: 
161:     This function numerically integrates a system of ordinary differential
162:     equations given an initial value::
163: 
164:         dy / dt = f(t, y)
165:         y(t0) = y0
166: 
167:     Here t is a 1-dimensional independent variable (time), y(t) is an
168:     n-dimensional vector-valued function (state) and an n-dimensional
169:     vector-valued function f(t, y) determines the differential equations.
170:     The goal is to find y(t) approximately satisfying the differential
171:     equations, given an initial value y(t0)=y0.
172: 
173:     Some of the solvers support integration in a complex domain, but note that
174:     for stiff ODE solvers the right hand side must be complex differentiable
175:     (satisfy Cauchy-Riemann equations [11]_). To solve a problem in a complex
176:     domain, pass y0 with a complex data type. Another option always available
177:     is to rewrite your problem for real and imaginary parts separately.
178: 
179:     Parameters
180:     ----------
181:     fun : callable
182:         Right-hand side of the system. The calling signature is ``fun(t, y)``.
183:         Here ``t`` is a scalar and there are two options for ndarray ``y``.
184:         It can either have shape (n,), then ``fun`` must return array_like with
185:         shape (n,). Or alternatively it can have shape (n, k), then ``fun``
186:         must return array_like with shape (n, k), i.e. each column
187:         corresponds to a single column in ``y``. The choice between the two
188:         options is determined by `vectorized` argument (see below). The
189:         vectorized implementation allows faster approximation of the Jacobian
190:         by finite differences (required for stiff solvers).
191:     t_span : 2-tuple of floats
192:         Interval of integration (t0, tf). The solver starts with t=t0 and
193:         integrates until it reaches t=tf.
194:     y0 : array_like, shape (n,)
195:         Initial state. For problems in a complex domain pass `y0` with a
196:         complex data type (even if the initial guess is purely real).
197:     method : string or `OdeSolver`, optional
198:         Integration method to use:
199: 
200:             * 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
201:               The error is controlled assuming 4th order accuracy, but steps
202:               are taken using a 5th oder accurate formula (local extrapolation
203:               is done). A quartic interpolation polynomial is used for the
204:               dense output [2]_. Can be applied in a complex domain.
205:             * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
206:               is controlled assuming 2nd order accuracy, but steps are taken
207:               using a 3rd oder accurate formula (local extrapolation is done).
208:               A cubic Hermit polynomial is used for the dense output.
209:               Can be applied in a complex domain.
210:             * 'Radau': Implicit Runge-Kutta method of Radau IIA family of
211:               order 5 [4]_. The error is controlled for a 3rd order accurate
212:               embedded formula. A cubic polynomial which satisfies the
213:               collocation conditions is used for the dense output.
214:             * 'BDF': Implicit multi-step variable order (1 to 5) method based
215:               on a Backward Differentiation Formulas for the derivative
216:               approximation [5]_. An implementation approach follows the one
217:               described in [6]_. A quasi-constant step scheme is used
218:               and accuracy enhancement using NDF modification is also
219:               implemented. Can be applied in a complex domain.
220:             * 'LSODA': Adams/BDF method with automatic stiffness detection and
221:               switching [7]_, [8]_. This is a wrapper of the Fortran solver
222:               from ODEPACK.
223: 
224:         You should use 'RK45' or 'RK23' methods for non-stiff problems and
225:         'Radau' or 'BDF' for stiff problems [9]_. If not sure, first try to run
226:         'RK45' and if it does unusual many iterations or diverges then your
227:         problem is likely to be stiff and you should use 'Radau' or 'BDF'.
228:         'LSODA' can also be a good universal choice, but it might be somewhat
229:         less  convenient to work with as it wraps an old Fortran code.
230: 
231:         You can also pass an arbitrary class derived from `OdeSolver` which
232:         implements the solver.
233:     dense_output : bool, optional
234:         Whether to compute a continuous solution. Default is False.
235:     t_eval : array_like or None, optional
236:         Times at which to store the computed solution, must be sorted and lie
237:         within `t_span`. If None (default), use points selected by a solver.
238:     events : callable, list of callables or None, optional
239:         Events to track. Events are defined by functions which take
240:         a zero value at a point of an event. Each function must have a
241:         signature ``event(t, y)`` and return float, the solver will find an
242:         accurate value of ``t`` at which ``event(t, y(t)) = 0`` using a root
243:         finding algorithm. Additionally each ``event`` function might have
244:         attributes:
245: 
246:             * terminal: bool, whether to terminate integration if this
247:               event occurs. Implicitly False if not assigned.
248:             * direction: float, direction of crossing a zero. If `direction`
249:               is positive then `event` must go from negative to positive, and
250:               vice-versa if `direction` is negative. If 0, then either way will
251:               count. Implicitly 0 if not assigned.
252: 
253:         You can assign attributes like ``event.terminal = True`` to any
254:         function in Python. If None (default), events won't be tracked.
255:     vectorized : bool, optional
256:         Whether `fun` is implemented in a vectorized fashion. Default is False.
257:     options
258:         Options passed to a chosen solver constructor. All options available
259:         for already implemented solvers are listed below.
260:     max_step : float, optional
261:         Maximum allowed step size. Default is np.inf, i.e. step is not
262:         bounded and determined solely by the solver.
263:     rtol, atol : float and array_like, optional
264:         Relative and absolute tolerances. The solver keeps the local error
265:         estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
266:         relative accuracy (number of correct digits). But if a component of `y`
267:         is approximately below `atol` then the error only needs to fall within
268:         the same `atol` threshold, and the number of correct digits is not
269:         guaranteed. If components of y have different scales, it might be
270:         beneficial to set different `atol` values for different components by
271:         passing array_like with shape (n,) for `atol`. Default values are
272:         1e-3 for `rtol` and 1e-6 for `atol`.
273:     jac : {None, array_like, sparse_matrix, callable}, optional
274:         Jacobian matrix of the right-hand side of the system with respect to
275:         y, required by 'Radau', 'BDF' and 'LSODA' methods. The Jacobian matrix
276:         has shape (n, n) and its element (i, j) is equal to ``d f_i / d y_j``.
277:         There are 3 ways to define the Jacobian:
278: 
279:             * If array_like or sparse_matrix, then the Jacobian is assumed to
280:               be constant. Not supported by 'LSODA'.
281:             * If callable, then the Jacobian is assumed to depend on both
282:               t and y, and will be called as ``jac(t, y)`` as necessary.
283:               For 'Radau' and 'BDF' methods the return value might be a sparse
284:               matrix.
285:             * If None (default), then the Jacobian will be approximated by
286:               finite differences.
287: 
288:         It is generally recommended to provide the Jacobian rather than
289:         relying on a finite difference approximation.
290:     jac_sparsity : {None, array_like, sparse matrix}, optional
291:         Defines a sparsity structure of the Jacobian matrix for a finite
292:         difference approximation, its shape must be (n, n). If the Jacobian has
293:         only few non-zero elements in *each* row, providing the sparsity
294:         structure will greatly speed up the computations [10]_. A zero
295:         entry means that a corresponding element in the Jacobian is identically
296:         zero. If None (default), the Jacobian is assumed to be dense.
297:         Not supported by 'LSODA', see `lband` and `uband` instead.
298:     lband, uband : int or None
299:         Parameters defining the Jacobian matrix bandwidth for 'LSODA' method.
300:         The Jacobian bandwidth means that
301:         ``jac[i, j] != 0 only for i - lband <= j <= i + uband``. Setting these
302:         requires your jac routine to return the Jacobian in the packed format:
303:         the returned array must have ``n`` columns and ``uband + lband + 1``
304:         rows in which Jacobian diagonals are written. Specifically
305:         ``jac_packed[uband + i - j , j] = jac[i, j]``. The same format is used
306:         in `scipy.linalg.solve_banded` (check for an illustration).
307:         These parameters can be also used with ``jac=None`` to reduce the
308:         number of Jacobian elements estimated by finite differences.
309:     min_step, first_step : float, optional
310:         The minimum allowed step size and the initial step size respectively
311:         for 'LSODA' method. By default `min_step` is zero and `first_step` is
312:         selected automatically.
313: 
314:     Returns
315:     -------
316:     Bunch object with the following fields defined:
317:     t : ndarray, shape (n_points,)
318:         Time points.
319:     y : ndarray, shape (n, n_points)
320:         Solution values at `t`.
321:     sol : `OdeSolution` or None
322:         Found solution as `OdeSolution` instance, None if `dense_output` was
323:         set to False.
324:     t_events : list of ndarray or None
325:         Contains arrays with times at each a corresponding event was detected,
326:         the length of the list equals to the number of events. None if `events`
327:         was None.
328:     nfev : int
329:         Number of the system rhs evaluations.
330:     njev : int
331:         Number of the Jacobian evaluations.
332:     nlu : int
333:         Number of LU decompositions.
334:     status : int
335:         Reason for algorithm termination:
336: 
337:             * -1: Integration step failed.
338:             * 0: The solver successfully reached the interval end.
339:             * 1: A termination event occurred.
340: 
341:     message : string
342:         Verbal description of the termination reason.
343:     success : bool
344:         True if the solver reached the interval end or a termination event
345:         occurred (``status >= 0``).
346: 
347:     References
348:     ----------
349:     .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
350:            formulae", Journal of Computational and Applied Mathematics, Vol. 6,
351:            No. 1, pp. 19-26, 1980.
352:     .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
353:            of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
354:     .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
355:            Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
356:     .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
357:            Stiff and Differential-Algebraic Problems", Sec. IV.8.
358:     .. [5] `Backward Differentiation Formula
359:             <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
360:             on Wikipedia.
361:     .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
362:            COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
363:     .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
364:            Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
365:            pp. 55-64, 1983.
366:     .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
367:            nonstiff systems of ordinary differential equations", SIAM Journal
368:            on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
369:            1983.
370:     .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
371:            Wikipedia.
372:     .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
373:             sparse Jacobian matrices", Journal of the Institute of Mathematics
374:             and its Applications, 13, pp. 117-120, 1974.
375:     .. [11] `Cauchy-Riemann equations
376:              <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
377:              Wikipedia.
378: 
379:     Examples
380:     --------
381:     Basic exponential decay showing automatically chosen time points.
382:     
383:     >>> from scipy.integrate import solve_ivp
384:     >>> def exponential_decay(t, y): return -0.5 * y
385:     >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
386:     >>> print(sol.t)
387:     [  0.           0.11487653   1.26364188   3.06061781   4.85759374
388:        6.65456967   8.4515456   10.        ]
389:     >>> print(sol.y)
390:     [[ 2.          1.88836035  1.06327177  0.43319312  0.17648948  0.0719045
391:        0.02929499  0.01350938]
392:      [ 4.          3.7767207   2.12654355  0.86638624  0.35297895  0.143809
393:        0.05858998  0.02701876]
394:      [ 8.          7.5534414   4.25308709  1.73277247  0.7059579   0.287618
395:        0.11717996  0.05403753]]
396:        
397:     Specifying points where the solution is desired.
398:     
399:     >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], 
400:     ...                 t_eval=[0, 1, 2, 4, 10])
401:     >>> print(sol.t)
402:     [ 0  1  2  4 10]
403:     >>> print(sol.y)
404:     [[ 2.          1.21305369  0.73534021  0.27066736  0.01350938]
405:      [ 4.          2.42610739  1.47068043  0.54133472  0.02701876]
406:      [ 8.          4.85221478  2.94136085  1.08266944  0.05403753]]
407: 
408:     Cannon fired upward with terminal event upon impact. The ``terminal`` and 
409:     ``direction`` fields of an event are applied by monkey patching a function.
410:     Here ``y[0]`` is position and ``y[1]`` is velocity. The projectile starts at
411:     position 0 with velocity +10. Note that the integration never reaches t=100
412:     because the event is terminal.
413:     
414:     >>> def upward_cannon(t, y): return [y[1], -0.5]
415:     >>> def hit_ground(t, y): return y[1]
416:     >>> hit_ground.terminal = True
417:     >>> hit_ground.direction = -1
418:     >>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)
419:     >>> print(sol.t_events)
420:     [array([ 20.])]
421:     >>> print(sol.t)
422:     [  0.00000000e+00   9.99900010e-05   1.09989001e-03   1.10988901e-02
423:        1.11088891e-01   1.11098890e+00   1.11099890e+01   2.00000000e+01]
424:     '''
425:     if method not in METHODS and not (
426:             inspect.isclass(method) and issubclass(method, OdeSolver)):
427:         raise ValueError("`method` must be one of {} or OdeSolver class."
428:                          .format(METHODS))
429: 
430:     t0, tf = float(t_span[0]), float(t_span[1])
431: 
432:     if t_eval is not None:
433:         t_eval = np.asarray(t_eval)
434:         if t_eval.ndim != 1:
435:             raise ValueError("`t_eval` must be 1-dimensional.")
436: 
437:         if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
438:             raise ValueError("Values in `t_eval` are not within `t_span`.")
439: 
440:         d = np.diff(t_eval)
441:         if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
442:             raise ValueError("Values in `t_eval` are not properly sorted.")
443: 
444:         if tf > t0:
445:             t_eval_i = 0
446:         else:
447:             # Make order of t_eval decreasing to use np.searchsorted.
448:             t_eval = t_eval[::-1]
449:             # This will be an upper bound for slices.
450:             t_eval_i = t_eval.shape[0]
451: 
452:     if method in METHODS:
453:         method = METHODS[method]
454: 
455:     solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)
456: 
457:     if t_eval is None:
458:         ts = [t0]
459:         ys = [y0]
460:     else:
461:         ts = []
462:         ys = []
463: 
464:     interpolants = []
465: 
466:     events, is_terminal, event_dir = prepare_events(events)
467: 
468:     if events is not None:
469:         g = [event(t0, y0) for event in events]
470:         t_events = [[] for _ in range(len(events))]
471:     else:
472:         t_events = None
473: 
474:     status = None
475:     while status is None:
476:         message = solver.step()
477: 
478:         if solver.status == 'finished':
479:             status = 0
480:         elif solver.status == 'failed':
481:             status = -1
482:             break
483: 
484:         t_old = solver.t_old
485:         t = solver.t
486:         y = solver.y
487: 
488:         if dense_output:
489:             sol = solver.dense_output()
490:             interpolants.append(sol)
491:         else:
492:             sol = None
493: 
494:         if events is not None:
495:             g_new = [event(t, y) for event in events]
496:             active_events = find_active_events(g, g_new, event_dir)
497:             if active_events.size > 0:
498:                 if sol is None:
499:                     sol = solver.dense_output()
500: 
501:                 root_indices, roots, terminate = handle_events(
502:                     sol, events, active_events, is_terminal, t_old, t)
503: 
504:                 for e, te in zip(root_indices, roots):
505:                     t_events[e].append(te)
506: 
507:                 if terminate:
508:                     status = 1
509:                     t = roots[-1]
510:                     y = sol(t)
511: 
512:             g = g_new
513: 
514:         if t_eval is None:
515:             ts.append(t)
516:             ys.append(y)
517:         else:
518:             # The value in t_eval equal to t will be included.
519:             if solver.direction > 0:
520:                 t_eval_i_new = np.searchsorted(t_eval, t, side='right')
521:                 t_eval_step = t_eval[t_eval_i:t_eval_i_new]
522:             else:
523:                 t_eval_i_new = np.searchsorted(t_eval, t, side='left')
524:                 # It has to be done with two slice operations, because
525:                 # you can't slice to 0-th element inclusive using backward
526:                 # slicing.
527:                 t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]
528: 
529:             if t_eval_step.size > 0:
530:                 if sol is None:
531:                     sol = solver.dense_output()
532:                 ts.append(t_eval_step)
533:                 ys.append(sol(t_eval_step))
534:                 t_eval_i = t_eval_i_new
535: 
536:     message = MESSAGES.get(status, message)
537: 
538:     if t_events is not None:
539:         t_events = [np.asarray(te) for te in t_events]
540: 
541:     if t_eval is None:
542:         ts = np.array(ts)
543:         ys = np.vstack(ys).T
544:     else:
545:         ts = np.hstack(ts)
546:         ys = np.hstack(ys)
547: 
548:     if dense_output:
549:         sol = OdeSolution(ts, interpolants)
550:     else:
551:         sol = None
552: 
553:     return OdeResult(t=ts, y=ys, sol=sol, t_events=t_events, nfev=solver.nfev,
554:                      njev=solver.njev, nlu=solver.nlu, status=status,
555:                      message=message, success=status >= 0)
556: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import inspect' statement (line 2)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_55575 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_55575) is not StypyTypeError):

    if (import_55575 != 'pyd_module'):
        __import__(import_55575)
        sys_modules_55576 = sys.modules[import_55575]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_55576.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_55575)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.integrate._ivp.bdf import BDF' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_55577 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.bdf')

if (type(import_55577) is not StypyTypeError):

    if (import_55577 != 'pyd_module'):
        __import__(import_55577)
        sys_modules_55578 = sys.modules[import_55577]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.bdf', sys_modules_55578.module_type_store, module_type_store, ['BDF'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_55578, sys_modules_55578.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.bdf import BDF

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.bdf', None, module_type_store, ['BDF'], [BDF])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.bdf' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.bdf', import_55577)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.integrate._ivp.radau import Radau' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_55579 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._ivp.radau')

if (type(import_55579) is not StypyTypeError):

    if (import_55579 != 'pyd_module'):
        __import__(import_55579)
        sys_modules_55580 = sys.modules[import_55579]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._ivp.radau', sys_modules_55580.module_type_store, module_type_store, ['Radau'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_55580, sys_modules_55580.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.radau import Radau

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._ivp.radau', None, module_type_store, ['Radau'], [Radau])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.radau' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._ivp.radau', import_55579)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.integrate._ivp.rk import RK23, RK45' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_55581 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate._ivp.rk')

if (type(import_55581) is not StypyTypeError):

    if (import_55581 != 'pyd_module'):
        __import__(import_55581)
        sys_modules_55582 = sys.modules[import_55581]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate._ivp.rk', sys_modules_55582.module_type_store, module_type_store, ['RK23', 'RK45'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_55582, sys_modules_55582.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.rk import RK23, RK45

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate._ivp.rk', None, module_type_store, ['RK23', 'RK45'], [RK23, RK45])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.rk' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate._ivp.rk', import_55581)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.integrate._ivp.lsoda import LSODA' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_55583 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.lsoda')

if (type(import_55583) is not StypyTypeError):

    if (import_55583 != 'pyd_module'):
        __import__(import_55583)
        sys_modules_55584 = sys.modules[import_55583]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.lsoda', sys_modules_55584.module_type_store, module_type_store, ['LSODA'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_55584, sys_modules_55584.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.lsoda import LSODA

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.lsoda', None, module_type_store, ['LSODA'], [LSODA])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.lsoda' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.lsoda', import_55583)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize import OptimizeResult' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_55585 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize')

if (type(import_55585) is not StypyTypeError):

    if (import_55585 != 'pyd_module'):
        __import__(import_55585)
        sys_modules_55586 = sys.modules[import_55585]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', sys_modules_55586.module_type_store, module_type_store, ['OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_55586, sys_modules_55586.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult'], [OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', import_55585)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.integrate._ivp.common import EPS, OdeSolution' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_55587 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.common')

if (type(import_55587) is not StypyTypeError):

    if (import_55587 != 'pyd_module'):
        __import__(import_55587)
        sys_modules_55588 = sys.modules[import_55587]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.common', sys_modules_55588.module_type_store, module_type_store, ['EPS', 'OdeSolution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_55588, sys_modules_55588.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.common import EPS, OdeSolution

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.common', None, module_type_store, ['EPS', 'OdeSolution'], [EPS, OdeSolution])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.common' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.common', import_55587)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.integrate._ivp.base import OdeSolver' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_55589 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.integrate._ivp.base')

if (type(import_55589) is not StypyTypeError):

    if (import_55589 != 'pyd_module'):
        __import__(import_55589)
        sys_modules_55590 = sys.modules[import_55589]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.integrate._ivp.base', sys_modules_55590.module_type_store, module_type_store, ['OdeSolver'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_55590, sys_modules_55590.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.base import OdeSolver

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.integrate._ivp.base', None, module_type_store, ['OdeSolver'], [OdeSolver])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.base' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.integrate._ivp.base', import_55589)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')


# Assigning a Dict to a Name (line 13):

# Assigning a Dict to a Name (line 13):

# Obtaining an instance of the builtin type 'dict' (line 13)
dict_55591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 13)
# Adding element type (key, value) (line 13)
str_55592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'RK23')
# Getting the type of 'RK23' (line 13)
RK23_55593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'RK23')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), dict_55591, (str_55592, RK23_55593))
# Adding element type (key, value) (line 13)
str_55594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'RK45')
# Getting the type of 'RK45' (line 14)
RK45_55595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'RK45')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), dict_55591, (str_55594, RK45_55595))
# Adding element type (key, value) (line 13)
str_55596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'Radau')
# Getting the type of 'Radau' (line 15)
Radau_55597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'Radau')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), dict_55591, (str_55596, Radau_55597))
# Adding element type (key, value) (line 13)
str_55598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'BDF')
# Getting the type of 'BDF' (line 16)
BDF_55599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'BDF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), dict_55591, (str_55598, BDF_55599))
# Adding element type (key, value) (line 13)
str_55600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'LSODA')
# Getting the type of 'LSODA' (line 17)
LSODA_55601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'LSODA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), dict_55591, (str_55600, LSODA_55601))

# Assigning a type to the variable 'METHODS' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'METHODS', dict_55591)

# Assigning a Dict to a Name (line 20):

# Assigning a Dict to a Name (line 20):

# Obtaining an instance of the builtin type 'dict' (line 20)
dict_55602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 20)
# Adding element type (key, value) (line 20)
int_55603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
str_55604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'str', 'The solver successfully reached the interval end.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 11), dict_55602, (int_55603, str_55604))
# Adding element type (key, value) (line 20)
int_55605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'int')
str_55606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'str', 'A termination event occurred.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 11), dict_55602, (int_55605, str_55606))

# Assigning a type to the variable 'MESSAGES' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'MESSAGES', dict_55602)
# Declaration of the 'OdeResult' class
# Getting the type of 'OptimizeResult' (line 24)
OptimizeResult_55607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'OptimizeResult')

class OdeResult(OptimizeResult_55607, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 24, 0, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeResult.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'OdeResult' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'OdeResult', OdeResult)

@norecursion
def prepare_events(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'prepare_events'
    module_type_store = module_type_store.open_function_context('prepare_events', 28, 0, False)
    
    # Passed parameters checking function
    prepare_events.stypy_localization = localization
    prepare_events.stypy_type_of_self = None
    prepare_events.stypy_type_store = module_type_store
    prepare_events.stypy_function_name = 'prepare_events'
    prepare_events.stypy_param_names_list = ['events']
    prepare_events.stypy_varargs_param_name = None
    prepare_events.stypy_kwargs_param_name = None
    prepare_events.stypy_call_defaults = defaults
    prepare_events.stypy_call_varargs = varargs
    prepare_events.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'prepare_events', ['events'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'prepare_events', localization, ['events'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'prepare_events(...)' code ##################

    str_55608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'Standardize event functions and extract is_terminal and direction.')
    
    
    # Call to callable(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'events' (line 30)
    events_55610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'events', False)
    # Processing the call keyword arguments (line 30)
    kwargs_55611 = {}
    # Getting the type of 'callable' (line 30)
    callable_55609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'callable', False)
    # Calling callable(args, kwargs) (line 30)
    callable_call_result_55612 = invoke(stypy.reporting.localization.Localization(__file__, 30, 7), callable_55609, *[events_55610], **kwargs_55611)
    
    # Testing the type of an if condition (line 30)
    if_condition_55613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), callable_call_result_55612)
    # Assigning a type to the variable 'if_condition_55613' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_55613', if_condition_55613)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 31):
    
    # Assigning a Tuple to a Name (line 31):
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_55614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    # Getting the type of 'events' (line 31)
    events_55615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'events')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 18), tuple_55614, events_55615)
    
    # Assigning a type to the variable 'events' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'events', tuple_55614)
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'events' (line 33)
    events_55616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'events')
    # Getting the type of 'None' (line 33)
    None_55617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'None')
    
    (may_be_55618, more_types_in_union_55619) = may_not_be_none(events_55616, None_55617)

    if may_be_55618:

        if more_types_in_union_55619:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 34):
        
        # Assigning a Call to a Name (line 34):
        
        # Call to empty(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to len(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'events' (line 34)
        events_55623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'events', False)
        # Processing the call keyword arguments (line 34)
        kwargs_55624 = {}
        # Getting the type of 'len' (line 34)
        len_55622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'len', False)
        # Calling len(args, kwargs) (line 34)
        len_call_result_55625 = invoke(stypy.reporting.localization.Localization(__file__, 34, 31), len_55622, *[events_55623], **kwargs_55624)
        
        # Processing the call keyword arguments (line 34)
        # Getting the type of 'bool' (line 34)
        bool_55626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 50), 'bool', False)
        keyword_55627 = bool_55626
        kwargs_55628 = {'dtype': keyword_55627}
        # Getting the type of 'np' (line 34)
        np_55620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'np', False)
        # Obtaining the member 'empty' of a type (line 34)
        empty_55621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 22), np_55620, 'empty')
        # Calling empty(args, kwargs) (line 34)
        empty_call_result_55629 = invoke(stypy.reporting.localization.Localization(__file__, 34, 22), empty_55621, *[len_call_result_55625], **kwargs_55628)
        
        # Assigning a type to the variable 'is_terminal' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'is_terminal', empty_call_result_55629)
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to empty(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Call to len(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'events' (line 35)
        events_55633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'events', False)
        # Processing the call keyword arguments (line 35)
        kwargs_55634 = {}
        # Getting the type of 'len' (line 35)
        len_55632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'len', False)
        # Calling len(args, kwargs) (line 35)
        len_call_result_55635 = invoke(stypy.reporting.localization.Localization(__file__, 35, 29), len_55632, *[events_55633], **kwargs_55634)
        
        # Processing the call keyword arguments (line 35)
        kwargs_55636 = {}
        # Getting the type of 'np' (line 35)
        np_55630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'np', False)
        # Obtaining the member 'empty' of a type (line 35)
        empty_55631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), np_55630, 'empty')
        # Calling empty(args, kwargs) (line 35)
        empty_call_result_55637 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), empty_55631, *[len_call_result_55635], **kwargs_55636)
        
        # Assigning a type to the variable 'direction' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'direction', empty_call_result_55637)
        
        
        # Call to enumerate(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'events' (line 36)
        events_55639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'events', False)
        # Processing the call keyword arguments (line 36)
        kwargs_55640 = {}
        # Getting the type of 'enumerate' (line 36)
        enumerate_55638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 36)
        enumerate_call_result_55641 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), enumerate_55638, *[events_55639], **kwargs_55640)
        
        # Testing the type of a for loop iterable (line 36)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 36, 8), enumerate_call_result_55641)
        # Getting the type of the for loop variable (line 36)
        for_loop_var_55642 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 36, 8), enumerate_call_result_55641)
        # Assigning a type to the variable 'i' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), for_loop_var_55642))
        # Assigning a type to the variable 'event' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'event', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), for_loop_var_55642))
        # SSA begins for a for statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Subscript (line 38):
        
        # Assigning a Attribute to a Subscript (line 38):
        # Getting the type of 'event' (line 38)
        event_55643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'event')
        # Obtaining the member 'terminal' of a type (line 38)
        terminal_55644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 33), event_55643, 'terminal')
        # Getting the type of 'is_terminal' (line 38)
        is_terminal_55645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'is_terminal')
        # Getting the type of 'i' (line 38)
        i_55646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'i')
        # Storing an element on a container (line 38)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 16), is_terminal_55645, (i_55646, terminal_55644))
        # SSA branch for the except part of a try statement (line 37)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 37)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Subscript (line 40):
        
        # Assigning a Name to a Subscript (line 40):
        # Getting the type of 'False' (line 40)
        False_55647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'False')
        # Getting the type of 'is_terminal' (line 40)
        is_terminal_55648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'is_terminal')
        # Getting the type of 'i' (line 40)
        i_55649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'i')
        # Storing an element on a container (line 40)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 16), is_terminal_55648, (i_55649, False_55647))
        # SSA join for try-except statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Subscript (line 43):
        
        # Assigning a Attribute to a Subscript (line 43):
        # Getting the type of 'event' (line 43)
        event_55650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'event')
        # Obtaining the member 'direction' of a type (line 43)
        direction_55651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 31), event_55650, 'direction')
        # Getting the type of 'direction' (line 43)
        direction_55652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'direction')
        # Getting the type of 'i' (line 43)
        i_55653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'i')
        # Storing an element on a container (line 43)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 16), direction_55652, (i_55653, direction_55651))
        # SSA branch for the except part of a try statement (line 42)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 42)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Subscript (line 45):
        
        # Assigning a Num to a Subscript (line 45):
        int_55654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'int')
        # Getting the type of 'direction' (line 45)
        direction_55655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'direction')
        # Getting the type of 'i' (line 45)
        i_55656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'i')
        # Storing an element on a container (line 45)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 16), direction_55655, (i_55656, int_55654))
        # SSA join for try-except statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_55619:
            # Runtime conditional SSA for else branch (line 33)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_55618) or more_types_in_union_55619):
        
        # Assigning a Name to a Name (line 47):
        
        # Assigning a Name to a Name (line 47):
        # Getting the type of 'None' (line 47)
        None_55657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'None')
        # Assigning a type to the variable 'is_terminal' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'is_terminal', None_55657)
        
        # Assigning a Name to a Name (line 48):
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'None' (line 48)
        None_55658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'None')
        # Assigning a type to the variable 'direction' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'direction', None_55658)

        if (may_be_55618 and more_types_in_union_55619):
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_55659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    # Getting the type of 'events' (line 50)
    events_55660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'events')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_55659, events_55660)
    # Adding element type (line 50)
    # Getting the type of 'is_terminal' (line 50)
    is_terminal_55661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'is_terminal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_55659, is_terminal_55661)
    # Adding element type (line 50)
    # Getting the type of 'direction' (line 50)
    direction_55662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'direction')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_55659, direction_55662)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', tuple_55659)
    
    # ################# End of 'prepare_events(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'prepare_events' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_55663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55663)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'prepare_events'
    return stypy_return_type_55663

# Assigning a type to the variable 'prepare_events' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'prepare_events', prepare_events)

@norecursion
def solve_event_equation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_event_equation'
    module_type_store = module_type_store.open_function_context('solve_event_equation', 53, 0, False)
    
    # Passed parameters checking function
    solve_event_equation.stypy_localization = localization
    solve_event_equation.stypy_type_of_self = None
    solve_event_equation.stypy_type_store = module_type_store
    solve_event_equation.stypy_function_name = 'solve_event_equation'
    solve_event_equation.stypy_param_names_list = ['event', 'sol', 't_old', 't']
    solve_event_equation.stypy_varargs_param_name = None
    solve_event_equation.stypy_kwargs_param_name = None
    solve_event_equation.stypy_call_defaults = defaults
    solve_event_equation.stypy_call_varargs = varargs
    solve_event_equation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_event_equation', ['event', 'sol', 't_old', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_event_equation', localization, ['event', 'sol', 't_old', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_event_equation(...)' code ##################

    str_55664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', 'Solve an equation corresponding to an ODE event.\n\n    The equation is ``event(t, y(t)) = 0``, here ``y(t)`` is known from an\n    ODE solver using some sort of interpolation. It is solved by\n    `scipy.optimize.brentq` with xtol=atol=4*EPS.\n\n    Parameters\n    ----------\n    event : callable\n        Function ``event(t, y)``.\n    sol : callable\n        Function ``sol(t)`` which evaluates an ODE solution between `t_old`\n        and  `t`.\n    t_old, t : float\n        Previous and new values of time. They will be used as a bracketing\n        interval.\n\n    Returns\n    -------\n    root : float\n        Found solution.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 76, 4))
    
    # 'from scipy.optimize import brentq' statement (line 76)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
    import_55665 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 76, 4), 'scipy.optimize')

    if (type(import_55665) is not StypyTypeError):

        if (import_55665 != 'pyd_module'):
            __import__(import_55665)
            sys_modules_55666 = sys.modules[import_55665]
            import_from_module(stypy.reporting.localization.Localization(__file__, 76, 4), 'scipy.optimize', sys_modules_55666.module_type_store, module_type_store, ['brentq'])
            nest_module(stypy.reporting.localization.Localization(__file__, 76, 4), __file__, sys_modules_55666, sys_modules_55666.module_type_store, module_type_store)
        else:
            from scipy.optimize import brentq

            import_from_module(stypy.reporting.localization.Localization(__file__, 76, 4), 'scipy.optimize', None, module_type_store, ['brentq'], [brentq])

    else:
        # Assigning a type to the variable 'scipy.optimize' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'scipy.optimize', import_55665)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
    
    
    # Call to brentq(...): (line 77)
    # Processing the call arguments (line 77)

    @norecursion
    def _stypy_temp_lambda_48(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_48'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_48', 77, 18, True)
        # Passed parameters checking function
        _stypy_temp_lambda_48.stypy_localization = localization
        _stypy_temp_lambda_48.stypy_type_of_self = None
        _stypy_temp_lambda_48.stypy_type_store = module_type_store
        _stypy_temp_lambda_48.stypy_function_name = '_stypy_temp_lambda_48'
        _stypy_temp_lambda_48.stypy_param_names_list = ['t']
        _stypy_temp_lambda_48.stypy_varargs_param_name = None
        _stypy_temp_lambda_48.stypy_kwargs_param_name = None
        _stypy_temp_lambda_48.stypy_call_defaults = defaults
        _stypy_temp_lambda_48.stypy_call_varargs = varargs
        _stypy_temp_lambda_48.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_48', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_48', ['t'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to event(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 't' (line 77)
        t_55669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 't', False)
        
        # Call to sol(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 't' (line 77)
        t_55671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 41), 't', False)
        # Processing the call keyword arguments (line 77)
        kwargs_55672 = {}
        # Getting the type of 'sol' (line 77)
        sol_55670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'sol', False)
        # Calling sol(args, kwargs) (line 77)
        sol_call_result_55673 = invoke(stypy.reporting.localization.Localization(__file__, 77, 37), sol_55670, *[t_55671], **kwargs_55672)
        
        # Processing the call keyword arguments (line 77)
        kwargs_55674 = {}
        # Getting the type of 'event' (line 77)
        event_55668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'event', False)
        # Calling event(args, kwargs) (line 77)
        event_call_result_55675 = invoke(stypy.reporting.localization.Localization(__file__, 77, 28), event_55668, *[t_55669, sol_call_result_55673], **kwargs_55674)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'stypy_return_type', event_call_result_55675)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_48' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_55676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55676)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_48'
        return stypy_return_type_55676

    # Assigning a type to the variable '_stypy_temp_lambda_48' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), '_stypy_temp_lambda_48', _stypy_temp_lambda_48)
    # Getting the type of '_stypy_temp_lambda_48' (line 77)
    _stypy_temp_lambda_48_55677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), '_stypy_temp_lambda_48')
    # Getting the type of 't_old' (line 77)
    t_old_55678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 46), 't_old', False)
    # Getting the type of 't' (line 77)
    t_55679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 53), 't', False)
    # Processing the call keyword arguments (line 77)
    int_55680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
    # Getting the type of 'EPS' (line 78)
    EPS_55681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'EPS', False)
    # Applying the binary operator '*' (line 78)
    result_mul_55682 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 23), '*', int_55680, EPS_55681)
    
    keyword_55683 = result_mul_55682
    int_55684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 37), 'int')
    # Getting the type of 'EPS' (line 78)
    EPS_55685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 41), 'EPS', False)
    # Applying the binary operator '*' (line 78)
    result_mul_55686 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 37), '*', int_55684, EPS_55685)
    
    keyword_55687 = result_mul_55686
    kwargs_55688 = {'xtol': keyword_55683, 'rtol': keyword_55687}
    # Getting the type of 'brentq' (line 77)
    brentq_55667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'brentq', False)
    # Calling brentq(args, kwargs) (line 77)
    brentq_call_result_55689 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), brentq_55667, *[_stypy_temp_lambda_48_55677, t_old_55678, t_55679], **kwargs_55688)
    
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', brentq_call_result_55689)
    
    # ################# End of 'solve_event_equation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_event_equation' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_55690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55690)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_event_equation'
    return stypy_return_type_55690

# Assigning a type to the variable 'solve_event_equation' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'solve_event_equation', solve_event_equation)

@norecursion
def handle_events(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'handle_events'
    module_type_store = module_type_store.open_function_context('handle_events', 81, 0, False)
    
    # Passed parameters checking function
    handle_events.stypy_localization = localization
    handle_events.stypy_type_of_self = None
    handle_events.stypy_type_store = module_type_store
    handle_events.stypy_function_name = 'handle_events'
    handle_events.stypy_param_names_list = ['sol', 'events', 'active_events', 'is_terminal', 't_old', 't']
    handle_events.stypy_varargs_param_name = None
    handle_events.stypy_kwargs_param_name = None
    handle_events.stypy_call_defaults = defaults
    handle_events.stypy_call_varargs = varargs
    handle_events.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'handle_events', ['sol', 'events', 'active_events', 'is_terminal', 't_old', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'handle_events', localization, ['sol', 'events', 'active_events', 'is_terminal', 't_old', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'handle_events(...)' code ##################

    str_55691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, (-1)), 'str', 'Helper function to handle events.\n\n    Parameters\n    ----------\n    sol : DenseOutput\n        Function ``sol(t)`` which evaluates an ODE solution between `t_old`\n        and  `t`.\n    events : list of callables, length n_events\n        Event functions with signatures ``event(t, y)``.\n    active_events : ndarray\n        Indices of events which occurred.\n    is_terminal : ndarray, shape (n_events,)\n        Which events are terminal.\n    t_old, t : float\n        Previous and new values of time.\n\n    Returns\n    -------\n    root_indices : ndarray\n        Indices of events which take zero between `t_old` and `t` and before\n        a possible termination.\n    roots : ndarray\n        Values of t at which events occurred.\n    terminate : bool\n        Whether a terminal event occurred.\n    ')
    
    # Assigning a List to a Name (line 108):
    
    # Assigning a List to a Name (line 108):
    
    # Obtaining an instance of the builtin type 'list' (line 108)
    list_55692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 108)
    
    # Assigning a type to the variable 'roots' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'roots', list_55692)
    
    # Getting the type of 'active_events' (line 109)
    active_events_55693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'active_events')
    # Testing the type of a for loop iterable (line 109)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 4), active_events_55693)
    # Getting the type of the for loop variable (line 109)
    for_loop_var_55694 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 4), active_events_55693)
    # Assigning a type to the variable 'event_index' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'event_index', for_loop_var_55694)
    # SSA begins for a for statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Call to solve_event_equation(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    # Getting the type of 'event_index' (line 110)
    event_index_55698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 49), 'event_index', False)
    # Getting the type of 'events' (line 110)
    events_55699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 42), 'events', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___55700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 42), events_55699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_55701 = invoke(stypy.reporting.localization.Localization(__file__, 110, 42), getitem___55700, event_index_55698)
    
    # Getting the type of 'sol' (line 110)
    sol_55702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 63), 'sol', False)
    # Getting the type of 't_old' (line 110)
    t_old_55703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 68), 't_old', False)
    # Getting the type of 't' (line 110)
    t_55704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 75), 't', False)
    # Processing the call keyword arguments (line 110)
    kwargs_55705 = {}
    # Getting the type of 'solve_event_equation' (line 110)
    solve_event_equation_55697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'solve_event_equation', False)
    # Calling solve_event_equation(args, kwargs) (line 110)
    solve_event_equation_call_result_55706 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), solve_event_equation_55697, *[subscript_call_result_55701, sol_55702, t_old_55703, t_55704], **kwargs_55705)
    
    # Processing the call keyword arguments (line 110)
    kwargs_55707 = {}
    # Getting the type of 'roots' (line 110)
    roots_55695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'roots', False)
    # Obtaining the member 'append' of a type (line 110)
    append_55696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), roots_55695, 'append')
    # Calling append(args, kwargs) (line 110)
    append_call_result_55708 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), append_55696, *[solve_event_equation_call_result_55706], **kwargs_55707)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to asarray(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'roots' (line 112)
    roots_55711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'roots', False)
    # Processing the call keyword arguments (line 112)
    kwargs_55712 = {}
    # Getting the type of 'np' (line 112)
    np_55709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 112)
    asarray_55710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), np_55709, 'asarray')
    # Calling asarray(args, kwargs) (line 112)
    asarray_call_result_55713 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), asarray_55710, *[roots_55711], **kwargs_55712)
    
    # Assigning a type to the variable 'roots' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'roots', asarray_call_result_55713)
    
    
    # Call to any(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Obtaining the type of the subscript
    # Getting the type of 'active_events' (line 114)
    active_events_55716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'active_events', False)
    # Getting the type of 'is_terminal' (line 114)
    is_terminal_55717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'is_terminal', False)
    # Obtaining the member '__getitem__' of a type (line 114)
    getitem___55718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 14), is_terminal_55717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 114)
    subscript_call_result_55719 = invoke(stypy.reporting.localization.Localization(__file__, 114, 14), getitem___55718, active_events_55716)
    
    # Processing the call keyword arguments (line 114)
    kwargs_55720 = {}
    # Getting the type of 'np' (line 114)
    np_55714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 114)
    any_55715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 7), np_55714, 'any')
    # Calling any(args, kwargs) (line 114)
    any_call_result_55721 = invoke(stypy.reporting.localization.Localization(__file__, 114, 7), any_55715, *[subscript_call_result_55719], **kwargs_55720)
    
    # Testing the type of an if condition (line 114)
    if_condition_55722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), any_call_result_55721)
    # Assigning a type to the variable 'if_condition_55722' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_55722', if_condition_55722)
    # SSA begins for if statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 't' (line 115)
    t_55723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 't')
    # Getting the type of 't_old' (line 115)
    t_old_55724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 't_old')
    # Applying the binary operator '>' (line 115)
    result_gt_55725 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 11), '>', t_55723, t_old_55724)
    
    # Testing the type of an if condition (line 115)
    if_condition_55726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), result_gt_55725)
    # Assigning a type to the variable 'if_condition_55726' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_55726', if_condition_55726)
    # SSA begins for if statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 116):
    
    # Assigning a Call to a Name (line 116):
    
    # Call to argsort(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'roots' (line 116)
    roots_55729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'roots', False)
    # Processing the call keyword arguments (line 116)
    kwargs_55730 = {}
    # Getting the type of 'np' (line 116)
    np_55727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'np', False)
    # Obtaining the member 'argsort' of a type (line 116)
    argsort_55728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 20), np_55727, 'argsort')
    # Calling argsort(args, kwargs) (line 116)
    argsort_call_result_55731 = invoke(stypy.reporting.localization.Localization(__file__, 116, 20), argsort_55728, *[roots_55729], **kwargs_55730)
    
    # Assigning a type to the variable 'order' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'order', argsort_call_result_55731)
    # SSA branch for the else part of an if statement (line 115)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 118):
    
    # Assigning a Call to a Name (line 118):
    
    # Call to argsort(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Getting the type of 'roots' (line 118)
    roots_55734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 32), 'roots', False)
    # Applying the 'usub' unary operator (line 118)
    result___neg___55735 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 31), 'usub', roots_55734)
    
    # Processing the call keyword arguments (line 118)
    kwargs_55736 = {}
    # Getting the type of 'np' (line 118)
    np_55732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'np', False)
    # Obtaining the member 'argsort' of a type (line 118)
    argsort_55733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 20), np_55732, 'argsort')
    # Calling argsort(args, kwargs) (line 118)
    argsort_call_result_55737 = invoke(stypy.reporting.localization.Localization(__file__, 118, 20), argsort_55733, *[result___neg___55735], **kwargs_55736)
    
    # Assigning a type to the variable 'order' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'order', argsort_call_result_55737)
    # SSA join for if statement (line 115)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 119):
    
    # Assigning a Subscript to a Name (line 119):
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 119)
    order_55738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'order')
    # Getting the type of 'active_events' (line 119)
    active_events_55739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'active_events')
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___55740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 24), active_events_55739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_55741 = invoke(stypy.reporting.localization.Localization(__file__, 119, 24), getitem___55740, order_55738)
    
    # Assigning a type to the variable 'active_events' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'active_events', subscript_call_result_55741)
    
    # Assigning a Subscript to a Name (line 120):
    
    # Assigning a Subscript to a Name (line 120):
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 120)
    order_55742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'order')
    # Getting the type of 'roots' (line 120)
    roots_55743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'roots')
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___55744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), roots_55743, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_55745 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), getitem___55744, order_55742)
    
    # Assigning a type to the variable 'roots' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'roots', subscript_call_result_55745)
    
    # Assigning a Subscript to a Name (line 121):
    
    # Assigning a Subscript to a Name (line 121):
    
    # Obtaining the type of the subscript
    int_55746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 54), 'int')
    
    # Obtaining the type of the subscript
    int_55747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 51), 'int')
    
    # Call to nonzero(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Obtaining the type of the subscript
    # Getting the type of 'active_events' (line 121)
    active_events_55750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'active_events', False)
    # Getting the type of 'is_terminal' (line 121)
    is_terminal_55751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'is_terminal', False)
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___55752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 23), is_terminal_55751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_55753 = invoke(stypy.reporting.localization.Localization(__file__, 121, 23), getitem___55752, active_events_55750)
    
    # Processing the call keyword arguments (line 121)
    kwargs_55754 = {}
    # Getting the type of 'np' (line 121)
    np_55748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 121)
    nonzero_55749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), np_55748, 'nonzero')
    # Calling nonzero(args, kwargs) (line 121)
    nonzero_call_result_55755 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), nonzero_55749, *[subscript_call_result_55753], **kwargs_55754)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___55756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), nonzero_call_result_55755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_55757 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), getitem___55756, int_55747)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___55758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), subscript_call_result_55757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_55759 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), getitem___55758, int_55746)
    
    # Assigning a type to the variable 't' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 't', subscript_call_result_55759)
    
    # Assigning a Subscript to a Name (line 122):
    
    # Assigning a Subscript to a Name (line 122):
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 122)
    t_55760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 't')
    int_55761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 43), 'int')
    # Applying the binary operator '+' (line 122)
    result_add_55762 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 39), '+', t_55760, int_55761)
    
    slice_55763 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 24), None, result_add_55762, None)
    # Getting the type of 'active_events' (line 122)
    active_events_55764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'active_events')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___55765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 24), active_events_55764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_55766 = invoke(stypy.reporting.localization.Localization(__file__, 122, 24), getitem___55765, slice_55763)
    
    # Assigning a type to the variable 'active_events' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'active_events', subscript_call_result_55766)
    
    # Assigning a Subscript to a Name (line 123):
    
    # Assigning a Subscript to a Name (line 123):
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 123)
    t_55767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 't')
    int_55768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 27), 'int')
    # Applying the binary operator '+' (line 123)
    result_add_55769 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 23), '+', t_55767, int_55768)
    
    slice_55770 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 123, 16), None, result_add_55769, None)
    # Getting the type of 'roots' (line 123)
    roots_55771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'roots')
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___55772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), roots_55771, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_55773 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), getitem___55772, slice_55770)
    
    # Assigning a type to the variable 'roots' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'roots', subscript_call_result_55773)
    
    # Assigning a Name to a Name (line 124):
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'True' (line 124)
    True_55774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'True')
    # Assigning a type to the variable 'terminate' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'terminate', True_55774)
    # SSA branch for the else part of an if statement (line 114)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 126):
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'False' (line 126)
    False_55775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'False')
    # Assigning a type to the variable 'terminate' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'terminate', False_55775)
    # SSA join for if statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 128)
    tuple_55776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 128)
    # Adding element type (line 128)
    # Getting the type of 'active_events' (line 128)
    active_events_55777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'active_events')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 11), tuple_55776, active_events_55777)
    # Adding element type (line 128)
    # Getting the type of 'roots' (line 128)
    roots_55778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'roots')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 11), tuple_55776, roots_55778)
    # Adding element type (line 128)
    # Getting the type of 'terminate' (line 128)
    terminate_55779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 33), 'terminate')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 11), tuple_55776, terminate_55779)
    
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', tuple_55776)
    
    # ################# End of 'handle_events(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'handle_events' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_55780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'handle_events'
    return stypy_return_type_55780

# Assigning a type to the variable 'handle_events' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'handle_events', handle_events)

@norecursion
def find_active_events(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_active_events'
    module_type_store = module_type_store.open_function_context('find_active_events', 131, 0, False)
    
    # Passed parameters checking function
    find_active_events.stypy_localization = localization
    find_active_events.stypy_type_of_self = None
    find_active_events.stypy_type_store = module_type_store
    find_active_events.stypy_function_name = 'find_active_events'
    find_active_events.stypy_param_names_list = ['g', 'g_new', 'direction']
    find_active_events.stypy_varargs_param_name = None
    find_active_events.stypy_kwargs_param_name = None
    find_active_events.stypy_call_defaults = defaults
    find_active_events.stypy_call_varargs = varargs
    find_active_events.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_active_events', ['g', 'g_new', 'direction'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_active_events', localization, ['g', 'g_new', 'direction'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_active_events(...)' code ##################

    str_55781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, (-1)), 'str', 'Find which event occurred during an integration step.\n\n    Parameters\n    ----------\n    g, g_new : array_like, shape (n_events,)\n        Values of event functions at a current and next points.\n    direction : ndarray, shape (n_events,)\n        Event "direction" according to the definition in `solve_ivp`.\n\n    Returns\n    -------\n    active_events : ndarray\n        Indices of events which occurred during the step.\n    ')
    
    # Assigning a Tuple to a Tuple (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to asarray(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'g' (line 146)
    g_55784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'g', False)
    # Processing the call keyword arguments (line 146)
    kwargs_55785 = {}
    # Getting the type of 'np' (line 146)
    np_55782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'np', False)
    # Obtaining the member 'asarray' of a type (line 146)
    asarray_55783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), np_55782, 'asarray')
    # Calling asarray(args, kwargs) (line 146)
    asarray_call_result_55786 = invoke(stypy.reporting.localization.Localization(__file__, 146, 15), asarray_55783, *[g_55784], **kwargs_55785)
    
    # Assigning a type to the variable 'tuple_assignment_55565' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'tuple_assignment_55565', asarray_call_result_55786)
    
    # Assigning a Call to a Name (line 146):
    
    # Call to asarray(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'g_new' (line 146)
    g_new_55789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 41), 'g_new', False)
    # Processing the call keyword arguments (line 146)
    kwargs_55790 = {}
    # Getting the type of 'np' (line 146)
    np_55787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'np', False)
    # Obtaining the member 'asarray' of a type (line 146)
    asarray_55788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 30), np_55787, 'asarray')
    # Calling asarray(args, kwargs) (line 146)
    asarray_call_result_55791 = invoke(stypy.reporting.localization.Localization(__file__, 146, 30), asarray_55788, *[g_new_55789], **kwargs_55790)
    
    # Assigning a type to the variable 'tuple_assignment_55566' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'tuple_assignment_55566', asarray_call_result_55791)
    
    # Assigning a Name to a Name (line 146):
    # Getting the type of 'tuple_assignment_55565' (line 146)
    tuple_assignment_55565_55792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'tuple_assignment_55565')
    # Assigning a type to the variable 'g' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'g', tuple_assignment_55565_55792)
    
    # Assigning a Name to a Name (line 146):
    # Getting the type of 'tuple_assignment_55566' (line 146)
    tuple_assignment_55566_55793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'tuple_assignment_55566')
    # Assigning a type to the variable 'g_new' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 7), 'g_new', tuple_assignment_55566_55793)
    
    # Assigning a BinOp to a Name (line 147):
    
    # Assigning a BinOp to a Name (line 147):
    
    # Getting the type of 'g' (line 147)
    g_55794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 10), 'g')
    int_55795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 15), 'int')
    # Applying the binary operator '<=' (line 147)
    result_le_55796 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 10), '<=', g_55794, int_55795)
    
    
    # Getting the type of 'g_new' (line 147)
    g_new_55797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'g_new')
    int_55798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'int')
    # Applying the binary operator '>=' (line 147)
    result_ge_55799 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 21), '>=', g_new_55797, int_55798)
    
    # Applying the binary operator '&' (line 147)
    result_and__55800 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 9), '&', result_le_55796, result_ge_55799)
    
    # Assigning a type to the variable 'up' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'up', result_and__55800)
    
    # Assigning a BinOp to a Name (line 148):
    
    # Assigning a BinOp to a Name (line 148):
    
    # Getting the type of 'g' (line 148)
    g_55801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'g')
    int_55802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 17), 'int')
    # Applying the binary operator '>=' (line 148)
    result_ge_55803 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 12), '>=', g_55801, int_55802)
    
    
    # Getting the type of 'g_new' (line 148)
    g_new_55804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'g_new')
    int_55805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 32), 'int')
    # Applying the binary operator '<=' (line 148)
    result_le_55806 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 23), '<=', g_new_55804, int_55805)
    
    # Applying the binary operator '&' (line 148)
    result_and__55807 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 11), '&', result_ge_55803, result_le_55806)
    
    # Assigning a type to the variable 'down' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'down', result_and__55807)
    
    # Assigning a BinOp to a Name (line 149):
    
    # Assigning a BinOp to a Name (line 149):
    # Getting the type of 'up' (line 149)
    up_55808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 13), 'up')
    # Getting the type of 'down' (line 149)
    down_55809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'down')
    # Applying the binary operator '|' (line 149)
    result_or__55810 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 13), '|', up_55808, down_55809)
    
    # Assigning a type to the variable 'either' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'either', result_or__55810)
    
    # Assigning a BinOp to a Name (line 150):
    
    # Assigning a BinOp to a Name (line 150):
    # Getting the type of 'up' (line 150)
    up_55811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'up')
    
    # Getting the type of 'direction' (line 150)
    direction_55812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'direction')
    int_55813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 30), 'int')
    # Applying the binary operator '>' (line 150)
    result_gt_55814 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 18), '>', direction_55812, int_55813)
    
    # Applying the binary operator '&' (line 150)
    result_and__55815 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '&', up_55811, result_gt_55814)
    
    # Getting the type of 'down' (line 151)
    down_55816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'down')
    
    # Getting the type of 'direction' (line 151)
    direction_55817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'direction')
    int_55818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 32), 'int')
    # Applying the binary operator '<' (line 151)
    result_lt_55819 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 20), '<', direction_55817, int_55818)
    
    # Applying the binary operator '&' (line 151)
    result_and__55820 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 12), '&', down_55816, result_lt_55819)
    
    # Applying the binary operator '|' (line 150)
    result_or__55821 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '|', result_and__55815, result_and__55820)
    
    # Getting the type of 'either' (line 152)
    either_55822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'either')
    
    # Getting the type of 'direction' (line 152)
    direction_55823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'direction')
    int_55824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 35), 'int')
    # Applying the binary operator '==' (line 152)
    result_eq_55825 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 22), '==', direction_55823, int_55824)
    
    # Applying the binary operator '&' (line 152)
    result_and__55826 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 12), '&', either_55822, result_eq_55825)
    
    # Applying the binary operator '|' (line 151)
    result_or__55827 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 35), '|', result_or__55821, result_and__55826)
    
    # Assigning a type to the variable 'mask' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'mask', result_or__55827)
    
    # Obtaining the type of the subscript
    int_55828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'int')
    
    # Call to nonzero(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'mask' (line 154)
    mask_55831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'mask', False)
    # Processing the call keyword arguments (line 154)
    kwargs_55832 = {}
    # Getting the type of 'np' (line 154)
    np_55829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 154)
    nonzero_55830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 11), np_55829, 'nonzero')
    # Calling nonzero(args, kwargs) (line 154)
    nonzero_call_result_55833 = invoke(stypy.reporting.localization.Localization(__file__, 154, 11), nonzero_55830, *[mask_55831], **kwargs_55832)
    
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___55834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 11), nonzero_call_result_55833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_55835 = invoke(stypy.reporting.localization.Localization(__file__, 154, 11), getitem___55834, int_55828)
    
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type', subscript_call_result_55835)
    
    # ################# End of 'find_active_events(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_active_events' in the type store
    # Getting the type of 'stypy_return_type' (line 131)
    stypy_return_type_55836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55836)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_active_events'
    return stypy_return_type_55836

# Assigning a type to the variable 'find_active_events' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'find_active_events', find_active_events)

@norecursion
def solve_ivp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_55837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 38), 'str', 'RK45')
    # Getting the type of 'None' (line 157)
    None_55838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 53), 'None')
    # Getting the type of 'False' (line 157)
    False_55839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 72), 'False')
    # Getting the type of 'None' (line 158)
    None_55840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'None')
    # Getting the type of 'False' (line 158)
    False_55841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'False')
    defaults = [str_55837, None_55838, False_55839, None_55840, False_55841]
    # Create a new context for function 'solve_ivp'
    module_type_store = module_type_store.open_function_context('solve_ivp', 157, 0, False)
    
    # Passed parameters checking function
    solve_ivp.stypy_localization = localization
    solve_ivp.stypy_type_of_self = None
    solve_ivp.stypy_type_store = module_type_store
    solve_ivp.stypy_function_name = 'solve_ivp'
    solve_ivp.stypy_param_names_list = ['fun', 't_span', 'y0', 'method', 't_eval', 'dense_output', 'events', 'vectorized']
    solve_ivp.stypy_varargs_param_name = None
    solve_ivp.stypy_kwargs_param_name = 'options'
    solve_ivp.stypy_call_defaults = defaults
    solve_ivp.stypy_call_varargs = varargs
    solve_ivp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_ivp', ['fun', 't_span', 'y0', 'method', 't_eval', 'dense_output', 'events', 'vectorized'], None, 'options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_ivp', localization, ['fun', 't_span', 'y0', 'method', 't_eval', 'dense_output', 'events', 'vectorized'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_ivp(...)' code ##################

    str_55842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, (-1)), 'str', 'Solve an initial value problem for a system of ODEs.\n\n    This function numerically integrates a system of ordinary differential\n    equations given an initial value::\n\n        dy / dt = f(t, y)\n        y(t0) = y0\n\n    Here t is a 1-dimensional independent variable (time), y(t) is an\n    n-dimensional vector-valued function (state) and an n-dimensional\n    vector-valued function f(t, y) determines the differential equations.\n    The goal is to find y(t) approximately satisfying the differential\n    equations, given an initial value y(t0)=y0.\n\n    Some of the solvers support integration in a complex domain, but note that\n    for stiff ODE solvers the right hand side must be complex differentiable\n    (satisfy Cauchy-Riemann equations [11]_). To solve a problem in a complex\n    domain, pass y0 with a complex data type. Another option always available\n    is to rewrite your problem for real and imaginary parts separately.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(t, y)``.\n        Here ``t`` is a scalar and there are two options for ndarray ``y``.\n        It can either have shape (n,), then ``fun`` must return array_like with\n        shape (n,). Or alternatively it can have shape (n, k), then ``fun``\n        must return array_like with shape (n, k), i.e. each column\n        corresponds to a single column in ``y``. The choice between the two\n        options is determined by `vectorized` argument (see below). The\n        vectorized implementation allows faster approximation of the Jacobian\n        by finite differences (required for stiff solvers).\n    t_span : 2-tuple of floats\n        Interval of integration (t0, tf). The solver starts with t=t0 and\n        integrates until it reaches t=tf.\n    y0 : array_like, shape (n,)\n        Initial state. For problems in a complex domain pass `y0` with a\n        complex data type (even if the initial guess is purely real).\n    method : string or `OdeSolver`, optional\n        Integration method to use:\n\n            * \'RK45\' (default): Explicit Runge-Kutta method of order 5(4) [1]_.\n              The error is controlled assuming 4th order accuracy, but steps\n              are taken using a 5th oder accurate formula (local extrapolation\n              is done). A quartic interpolation polynomial is used for the\n              dense output [2]_. Can be applied in a complex domain.\n            * \'RK23\': Explicit Runge-Kutta method of order 3(2) [3]_. The error\n              is controlled assuming 2nd order accuracy, but steps are taken\n              using a 3rd oder accurate formula (local extrapolation is done).\n              A cubic Hermit polynomial is used for the dense output.\n              Can be applied in a complex domain.\n            * \'Radau\': Implicit Runge-Kutta method of Radau IIA family of\n              order 5 [4]_. The error is controlled for a 3rd order accurate\n              embedded formula. A cubic polynomial which satisfies the\n              collocation conditions is used for the dense output.\n            * \'BDF\': Implicit multi-step variable order (1 to 5) method based\n              on a Backward Differentiation Formulas for the derivative\n              approximation [5]_. An implementation approach follows the one\n              described in [6]_. A quasi-constant step scheme is used\n              and accuracy enhancement using NDF modification is also\n              implemented. Can be applied in a complex domain.\n            * \'LSODA\': Adams/BDF method with automatic stiffness detection and\n              switching [7]_, [8]_. This is a wrapper of the Fortran solver\n              from ODEPACK.\n\n        You should use \'RK45\' or \'RK23\' methods for non-stiff problems and\n        \'Radau\' or \'BDF\' for stiff problems [9]_. If not sure, first try to run\n        \'RK45\' and if it does unusual many iterations or diverges then your\n        problem is likely to be stiff and you should use \'Radau\' or \'BDF\'.\n        \'LSODA\' can also be a good universal choice, but it might be somewhat\n        less  convenient to work with as it wraps an old Fortran code.\n\n        You can also pass an arbitrary class derived from `OdeSolver` which\n        implements the solver.\n    dense_output : bool, optional\n        Whether to compute a continuous solution. Default is False.\n    t_eval : array_like or None, optional\n        Times at which to store the computed solution, must be sorted and lie\n        within `t_span`. If None (default), use points selected by a solver.\n    events : callable, list of callables or None, optional\n        Events to track. Events are defined by functions which take\n        a zero value at a point of an event. Each function must have a\n        signature ``event(t, y)`` and return float, the solver will find an\n        accurate value of ``t`` at which ``event(t, y(t)) = 0`` using a root\n        finding algorithm. Additionally each ``event`` function might have\n        attributes:\n\n            * terminal: bool, whether to terminate integration if this\n              event occurs. Implicitly False if not assigned.\n            * direction: float, direction of crossing a zero. If `direction`\n              is positive then `event` must go from negative to positive, and\n              vice-versa if `direction` is negative. If 0, then either way will\n              count. Implicitly 0 if not assigned.\n\n        You can assign attributes like ``event.terminal = True`` to any\n        function in Python. If None (default), events won\'t be tracked.\n    vectorized : bool, optional\n        Whether `fun` is implemented in a vectorized fashion. Default is False.\n    options\n        Options passed to a chosen solver constructor. All options available\n        for already implemented solvers are listed below.\n    max_step : float, optional\n        Maximum allowed step size. Default is np.inf, i.e. step is not\n        bounded and determined solely by the solver.\n    rtol, atol : float and array_like, optional\n        Relative and absolute tolerances. The solver keeps the local error\n        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a\n        relative accuracy (number of correct digits). But if a component of `y`\n        is approximately below `atol` then the error only needs to fall within\n        the same `atol` threshold, and the number of correct digits is not\n        guaranteed. If components of y have different scales, it might be\n        beneficial to set different `atol` values for different components by\n        passing array_like with shape (n,) for `atol`. Default values are\n        1e-3 for `rtol` and 1e-6 for `atol`.\n    jac : {None, array_like, sparse_matrix, callable}, optional\n        Jacobian matrix of the right-hand side of the system with respect to\n        y, required by \'Radau\', \'BDF\' and \'LSODA\' methods. The Jacobian matrix\n        has shape (n, n) and its element (i, j) is equal to ``d f_i / d y_j``.\n        There are 3 ways to define the Jacobian:\n\n            * If array_like or sparse_matrix, then the Jacobian is assumed to\n              be constant. Not supported by \'LSODA\'.\n            * If callable, then the Jacobian is assumed to depend on both\n              t and y, and will be called as ``jac(t, y)`` as necessary.\n              For \'Radau\' and \'BDF\' methods the return value might be a sparse\n              matrix.\n            * If None (default), then the Jacobian will be approximated by\n              finite differences.\n\n        It is generally recommended to provide the Jacobian rather than\n        relying on a finite difference approximation.\n    jac_sparsity : {None, array_like, sparse matrix}, optional\n        Defines a sparsity structure of the Jacobian matrix for a finite\n        difference approximation, its shape must be (n, n). If the Jacobian has\n        only few non-zero elements in *each* row, providing the sparsity\n        structure will greatly speed up the computations [10]_. A zero\n        entry means that a corresponding element in the Jacobian is identically\n        zero. If None (default), the Jacobian is assumed to be dense.\n        Not supported by \'LSODA\', see `lband` and `uband` instead.\n    lband, uband : int or None\n        Parameters defining the Jacobian matrix bandwidth for \'LSODA\' method.\n        The Jacobian bandwidth means that\n        ``jac[i, j] != 0 only for i - lband <= j <= i + uband``. Setting these\n        requires your jac routine to return the Jacobian in the packed format:\n        the returned array must have ``n`` columns and ``uband + lband + 1``\n        rows in which Jacobian diagonals are written. Specifically\n        ``jac_packed[uband + i - j , j] = jac[i, j]``. The same format is used\n        in `scipy.linalg.solve_banded` (check for an illustration).\n        These parameters can be also used with ``jac=None`` to reduce the\n        number of Jacobian elements estimated by finite differences.\n    min_step, first_step : float, optional\n        The minimum allowed step size and the initial step size respectively\n        for \'LSODA\' method. By default `min_step` is zero and `first_step` is\n        selected automatically.\n\n    Returns\n    -------\n    Bunch object with the following fields defined:\n    t : ndarray, shape (n_points,)\n        Time points.\n    y : ndarray, shape (n, n_points)\n        Solution values at `t`.\n    sol : `OdeSolution` or None\n        Found solution as `OdeSolution` instance, None if `dense_output` was\n        set to False.\n    t_events : list of ndarray or None\n        Contains arrays with times at each a corresponding event was detected,\n        the length of the list equals to the number of events. None if `events`\n        was None.\n    nfev : int\n        Number of the system rhs evaluations.\n    njev : int\n        Number of the Jacobian evaluations.\n    nlu : int\n        Number of LU decompositions.\n    status : int\n        Reason for algorithm termination:\n\n            * -1: Integration step failed.\n            * 0: The solver successfully reached the interval end.\n            * 1: A termination event occurred.\n\n    message : string\n        Verbal description of the termination reason.\n    success : bool\n        True if the solver reached the interval end or a termination event\n        occurred (``status >= 0``).\n\n    References\n    ----------\n    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta\n           formulae", Journal of Computational and Applied Mathematics, Vol. 6,\n           No. 1, pp. 19-26, 1980.\n    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics\n           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.\n    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",\n           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.\n    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:\n           Stiff and Differential-Algebraic Problems", Sec. IV.8.\n    .. [5] `Backward Differentiation Formula\n            <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_\n            on Wikipedia.\n    .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.\n           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.\n    .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE\n           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,\n           pp. 55-64, 1983.\n    .. [8] L. Petzold, "Automatic selection of methods for solving stiff and\n           nonstiff systems of ordinary differential equations", SIAM Journal\n           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,\n           1983.\n    .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on\n           Wikipedia.\n    .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of\n            sparse Jacobian matrices", Journal of the Institute of Mathematics\n            and its Applications, 13, pp. 117-120, 1974.\n    .. [11] `Cauchy-Riemann equations\n             <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on\n             Wikipedia.\n\n    Examples\n    --------\n    Basic exponential decay showing automatically chosen time points.\n    \n    >>> from scipy.integrate import solve_ivp\n    >>> def exponential_decay(t, y): return -0.5 * y\n    >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])\n    >>> print(sol.t)\n    [  0.           0.11487653   1.26364188   3.06061781   4.85759374\n       6.65456967   8.4515456   10.        ]\n    >>> print(sol.y)\n    [[ 2.          1.88836035  1.06327177  0.43319312  0.17648948  0.0719045\n       0.02929499  0.01350938]\n     [ 4.          3.7767207   2.12654355  0.86638624  0.35297895  0.143809\n       0.05858998  0.02701876]\n     [ 8.          7.5534414   4.25308709  1.73277247  0.7059579   0.287618\n       0.11717996  0.05403753]]\n       \n    Specifying points where the solution is desired.\n    \n    >>> sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], \n    ...                 t_eval=[0, 1, 2, 4, 10])\n    >>> print(sol.t)\n    [ 0  1  2  4 10]\n    >>> print(sol.y)\n    [[ 2.          1.21305369  0.73534021  0.27066736  0.01350938]\n     [ 4.          2.42610739  1.47068043  0.54133472  0.02701876]\n     [ 8.          4.85221478  2.94136085  1.08266944  0.05403753]]\n\n    Cannon fired upward with terminal event upon impact. The ``terminal`` and \n    ``direction`` fields of an event are applied by monkey patching a function.\n    Here ``y[0]`` is position and ``y[1]`` is velocity. The projectile starts at\n    position 0 with velocity +10. Note that the integration never reaches t=100\n    because the event is terminal.\n    \n    >>> def upward_cannon(t, y): return [y[1], -0.5]\n    >>> def hit_ground(t, y): return y[1]\n    >>> hit_ground.terminal = True\n    >>> hit_ground.direction = -1\n    >>> sol = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)\n    >>> print(sol.t_events)\n    [array([ 20.])]\n    >>> print(sol.t)\n    [  0.00000000e+00   9.99900010e-05   1.09989001e-03   1.10988901e-02\n       1.11088891e-01   1.11098890e+00   1.11099890e+01   2.00000000e+01]\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 425)
    method_55843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 7), 'method')
    # Getting the type of 'METHODS' (line 425)
    METHODS_55844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 21), 'METHODS')
    # Applying the binary operator 'notin' (line 425)
    result_contains_55845 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 7), 'notin', method_55843, METHODS_55844)
    
    
    
    # Evaluating a boolean operation
    
    # Call to isclass(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'method' (line 426)
    method_55848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 28), 'method', False)
    # Processing the call keyword arguments (line 426)
    kwargs_55849 = {}
    # Getting the type of 'inspect' (line 426)
    inspect_55846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 426)
    isclass_55847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), inspect_55846, 'isclass')
    # Calling isclass(args, kwargs) (line 426)
    isclass_call_result_55850 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), isclass_55847, *[method_55848], **kwargs_55849)
    
    
    # Call to issubclass(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'method' (line 426)
    method_55852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 51), 'method', False)
    # Getting the type of 'OdeSolver' (line 426)
    OdeSolver_55853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 59), 'OdeSolver', False)
    # Processing the call keyword arguments (line 426)
    kwargs_55854 = {}
    # Getting the type of 'issubclass' (line 426)
    issubclass_55851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 40), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 426)
    issubclass_call_result_55855 = invoke(stypy.reporting.localization.Localization(__file__, 426, 40), issubclass_55851, *[method_55852, OdeSolver_55853], **kwargs_55854)
    
    # Applying the binary operator 'and' (line 426)
    result_and_keyword_55856 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 12), 'and', isclass_call_result_55850, issubclass_call_result_55855)
    
    # Applying the 'not' unary operator (line 425)
    result_not__55857 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 33), 'not', result_and_keyword_55856)
    
    # Applying the binary operator 'and' (line 425)
    result_and_keyword_55858 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 7), 'and', result_contains_55845, result_not__55857)
    
    # Testing the type of an if condition (line 425)
    if_condition_55859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 4), result_and_keyword_55858)
    # Assigning a type to the variable 'if_condition_55859' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'if_condition_55859', if_condition_55859)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 427)
    # Processing the call arguments (line 427)
    
    # Call to format(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'METHODS' (line 428)
    METHODS_55863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 33), 'METHODS', False)
    # Processing the call keyword arguments (line 427)
    kwargs_55864 = {}
    str_55861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 25), 'str', '`method` must be one of {} or OdeSolver class.')
    # Obtaining the member 'format' of a type (line 427)
    format_55862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 25), str_55861, 'format')
    # Calling format(args, kwargs) (line 427)
    format_call_result_55865 = invoke(stypy.reporting.localization.Localization(__file__, 427, 25), format_55862, *[METHODS_55863], **kwargs_55864)
    
    # Processing the call keyword arguments (line 427)
    kwargs_55866 = {}
    # Getting the type of 'ValueError' (line 427)
    ValueError_55860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 427)
    ValueError_call_result_55867 = invoke(stypy.reporting.localization.Localization(__file__, 427, 14), ValueError_55860, *[format_call_result_55865], **kwargs_55866)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 427, 8), ValueError_call_result_55867, 'raise parameter', BaseException)
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to float(...): (line 430)
    # Processing the call arguments (line 430)
    
    # Obtaining the type of the subscript
    int_55869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 26), 'int')
    # Getting the type of 't_span' (line 430)
    t_span_55870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 't_span', False)
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___55871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 19), t_span_55870, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_55872 = invoke(stypy.reporting.localization.Localization(__file__, 430, 19), getitem___55871, int_55869)
    
    # Processing the call keyword arguments (line 430)
    kwargs_55873 = {}
    # Getting the type of 'float' (line 430)
    float_55868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 13), 'float', False)
    # Calling float(args, kwargs) (line 430)
    float_call_result_55874 = invoke(stypy.reporting.localization.Localization(__file__, 430, 13), float_55868, *[subscript_call_result_55872], **kwargs_55873)
    
    # Assigning a type to the variable 'tuple_assignment_55567' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'tuple_assignment_55567', float_call_result_55874)
    
    # Assigning a Call to a Name (line 430):
    
    # Call to float(...): (line 430)
    # Processing the call arguments (line 430)
    
    # Obtaining the type of the subscript
    int_55876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 44), 'int')
    # Getting the type of 't_span' (line 430)
    t_span_55877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 37), 't_span', False)
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___55878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 37), t_span_55877, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 430)
    subscript_call_result_55879 = invoke(stypy.reporting.localization.Localization(__file__, 430, 37), getitem___55878, int_55876)
    
    # Processing the call keyword arguments (line 430)
    kwargs_55880 = {}
    # Getting the type of 'float' (line 430)
    float_55875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'float', False)
    # Calling float(args, kwargs) (line 430)
    float_call_result_55881 = invoke(stypy.reporting.localization.Localization(__file__, 430, 31), float_55875, *[subscript_call_result_55879], **kwargs_55880)
    
    # Assigning a type to the variable 'tuple_assignment_55568' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'tuple_assignment_55568', float_call_result_55881)
    
    # Assigning a Name to a Name (line 430):
    # Getting the type of 'tuple_assignment_55567' (line 430)
    tuple_assignment_55567_55882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'tuple_assignment_55567')
    # Assigning a type to the variable 't0' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 't0', tuple_assignment_55567_55882)
    
    # Assigning a Name to a Name (line 430):
    # Getting the type of 'tuple_assignment_55568' (line 430)
    tuple_assignment_55568_55883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'tuple_assignment_55568')
    # Assigning a type to the variable 'tf' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tf', tuple_assignment_55568_55883)
    
    # Type idiom detected: calculating its left and rigth part (line 432)
    # Getting the type of 't_eval' (line 432)
    t_eval_55884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 't_eval')
    # Getting the type of 'None' (line 432)
    None_55885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 21), 'None')
    
    (may_be_55886, more_types_in_union_55887) = may_not_be_none(t_eval_55884, None_55885)

    if may_be_55886:

        if more_types_in_union_55887:
            # Runtime conditional SSA (line 432)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 433):
        
        # Assigning a Call to a Name (line 433):
        
        # Call to asarray(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 't_eval' (line 433)
        t_eval_55890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 28), 't_eval', False)
        # Processing the call keyword arguments (line 433)
        kwargs_55891 = {}
        # Getting the type of 'np' (line 433)
        np_55888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 17), 'np', False)
        # Obtaining the member 'asarray' of a type (line 433)
        asarray_55889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 17), np_55888, 'asarray')
        # Calling asarray(args, kwargs) (line 433)
        asarray_call_result_55892 = invoke(stypy.reporting.localization.Localization(__file__, 433, 17), asarray_55889, *[t_eval_55890], **kwargs_55891)
        
        # Assigning a type to the variable 't_eval' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 't_eval', asarray_call_result_55892)
        
        
        # Getting the type of 't_eval' (line 434)
        t_eval_55893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 11), 't_eval')
        # Obtaining the member 'ndim' of a type (line 434)
        ndim_55894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 11), t_eval_55893, 'ndim')
        int_55895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 26), 'int')
        # Applying the binary operator '!=' (line 434)
        result_ne_55896 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 11), '!=', ndim_55894, int_55895)
        
        # Testing the type of an if condition (line 434)
        if_condition_55897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 8), result_ne_55896)
        # Assigning a type to the variable 'if_condition_55897' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'if_condition_55897', if_condition_55897)
        # SSA begins for if statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 435)
        # Processing the call arguments (line 435)
        str_55899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 29), 'str', '`t_eval` must be 1-dimensional.')
        # Processing the call keyword arguments (line 435)
        kwargs_55900 = {}
        # Getting the type of 'ValueError' (line 435)
        ValueError_55898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 435)
        ValueError_call_result_55901 = invoke(stypy.reporting.localization.Localization(__file__, 435, 18), ValueError_55898, *[str_55899], **kwargs_55900)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 435, 12), ValueError_call_result_55901, 'raise parameter', BaseException)
        # SSA join for if statement (line 434)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to any(...): (line 437)
        # Processing the call arguments (line 437)
        
        # Getting the type of 't_eval' (line 437)
        t_eval_55904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 18), 't_eval', False)
        
        # Call to min(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 't0' (line 437)
        t0_55906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 31), 't0', False)
        # Getting the type of 'tf' (line 437)
        tf_55907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 35), 'tf', False)
        # Processing the call keyword arguments (line 437)
        kwargs_55908 = {}
        # Getting the type of 'min' (line 437)
        min_55905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 27), 'min', False)
        # Calling min(args, kwargs) (line 437)
        min_call_result_55909 = invoke(stypy.reporting.localization.Localization(__file__, 437, 27), min_55905, *[t0_55906, tf_55907], **kwargs_55908)
        
        # Applying the binary operator '<' (line 437)
        result_lt_55910 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 18), '<', t_eval_55904, min_call_result_55909)
        
        # Processing the call keyword arguments (line 437)
        kwargs_55911 = {}
        # Getting the type of 'np' (line 437)
        np_55902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 437)
        any_55903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 11), np_55902, 'any')
        # Calling any(args, kwargs) (line 437)
        any_call_result_55912 = invoke(stypy.reporting.localization.Localization(__file__, 437, 11), any_55903, *[result_lt_55910], **kwargs_55911)
        
        
        # Call to any(...): (line 437)
        # Processing the call arguments (line 437)
        
        # Getting the type of 't_eval' (line 437)
        t_eval_55915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 50), 't_eval', False)
        
        # Call to max(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 't0' (line 437)
        t0_55917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 63), 't0', False)
        # Getting the type of 'tf' (line 437)
        tf_55918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 67), 'tf', False)
        # Processing the call keyword arguments (line 437)
        kwargs_55919 = {}
        # Getting the type of 'max' (line 437)
        max_55916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 59), 'max', False)
        # Calling max(args, kwargs) (line 437)
        max_call_result_55920 = invoke(stypy.reporting.localization.Localization(__file__, 437, 59), max_55916, *[t0_55917, tf_55918], **kwargs_55919)
        
        # Applying the binary operator '>' (line 437)
        result_gt_55921 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 50), '>', t_eval_55915, max_call_result_55920)
        
        # Processing the call keyword arguments (line 437)
        kwargs_55922 = {}
        # Getting the type of 'np' (line 437)
        np_55913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 43), 'np', False)
        # Obtaining the member 'any' of a type (line 437)
        any_55914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 43), np_55913, 'any')
        # Calling any(args, kwargs) (line 437)
        any_call_result_55923 = invoke(stypy.reporting.localization.Localization(__file__, 437, 43), any_55914, *[result_gt_55921], **kwargs_55922)
        
        # Applying the binary operator 'or' (line 437)
        result_or_keyword_55924 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 11), 'or', any_call_result_55912, any_call_result_55923)
        
        # Testing the type of an if condition (line 437)
        if_condition_55925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 8), result_or_keyword_55924)
        # Assigning a type to the variable 'if_condition_55925' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'if_condition_55925', if_condition_55925)
        # SSA begins for if statement (line 437)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 438)
        # Processing the call arguments (line 438)
        str_55927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 29), 'str', 'Values in `t_eval` are not within `t_span`.')
        # Processing the call keyword arguments (line 438)
        kwargs_55928 = {}
        # Getting the type of 'ValueError' (line 438)
        ValueError_55926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 438)
        ValueError_call_result_55929 = invoke(stypy.reporting.localization.Localization(__file__, 438, 18), ValueError_55926, *[str_55927], **kwargs_55928)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 438, 12), ValueError_call_result_55929, 'raise parameter', BaseException)
        # SSA join for if statement (line 437)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 440):
        
        # Assigning a Call to a Name (line 440):
        
        # Call to diff(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 't_eval' (line 440)
        t_eval_55932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 20), 't_eval', False)
        # Processing the call keyword arguments (line 440)
        kwargs_55933 = {}
        # Getting the type of 'np' (line 440)
        np_55930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'np', False)
        # Obtaining the member 'diff' of a type (line 440)
        diff_55931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 12), np_55930, 'diff')
        # Calling diff(args, kwargs) (line 440)
        diff_call_result_55934 = invoke(stypy.reporting.localization.Localization(__file__, 440, 12), diff_55931, *[t_eval_55932], **kwargs_55933)
        
        # Assigning a type to the variable 'd' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'd', diff_call_result_55934)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'tf' (line 441)
        tf_55935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'tf')
        # Getting the type of 't0' (line 441)
        t0_55936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 't0')
        # Applying the binary operator '>' (line 441)
        result_gt_55937 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 11), '>', tf_55935, t0_55936)
        
        
        # Call to any(...): (line 441)
        # Processing the call arguments (line 441)
        
        # Getting the type of 'd' (line 441)
        d_55940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 'd', False)
        int_55941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 35), 'int')
        # Applying the binary operator '<=' (line 441)
        result_le_55942 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 30), '<=', d_55940, int_55941)
        
        # Processing the call keyword arguments (line 441)
        kwargs_55943 = {}
        # Getting the type of 'np' (line 441)
        np_55938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'np', False)
        # Obtaining the member 'any' of a type (line 441)
        any_55939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 23), np_55938, 'any')
        # Calling any(args, kwargs) (line 441)
        any_call_result_55944 = invoke(stypy.reporting.localization.Localization(__file__, 441, 23), any_55939, *[result_le_55942], **kwargs_55943)
        
        # Applying the binary operator 'and' (line 441)
        result_and_keyword_55945 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 11), 'and', result_gt_55937, any_call_result_55944)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'tf' (line 441)
        tf_55946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 41), 'tf')
        # Getting the type of 't0' (line 441)
        t0_55947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 46), 't0')
        # Applying the binary operator '<' (line 441)
        result_lt_55948 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 41), '<', tf_55946, t0_55947)
        
        
        # Call to any(...): (line 441)
        # Processing the call arguments (line 441)
        
        # Getting the type of 'd' (line 441)
        d_55951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 60), 'd', False)
        int_55952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 65), 'int')
        # Applying the binary operator '>=' (line 441)
        result_ge_55953 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 60), '>=', d_55951, int_55952)
        
        # Processing the call keyword arguments (line 441)
        kwargs_55954 = {}
        # Getting the type of 'np' (line 441)
        np_55949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 53), 'np', False)
        # Obtaining the member 'any' of a type (line 441)
        any_55950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 53), np_55949, 'any')
        # Calling any(args, kwargs) (line 441)
        any_call_result_55955 = invoke(stypy.reporting.localization.Localization(__file__, 441, 53), any_55950, *[result_ge_55953], **kwargs_55954)
        
        # Applying the binary operator 'and' (line 441)
        result_and_keyword_55956 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 41), 'and', result_lt_55948, any_call_result_55955)
        
        # Applying the binary operator 'or' (line 441)
        result_or_keyword_55957 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 11), 'or', result_and_keyword_55945, result_and_keyword_55956)
        
        # Testing the type of an if condition (line 441)
        if_condition_55958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 8), result_or_keyword_55957)
        # Assigning a type to the variable 'if_condition_55958' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'if_condition_55958', if_condition_55958)
        # SSA begins for if statement (line 441)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 442)
        # Processing the call arguments (line 442)
        str_55960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 29), 'str', 'Values in `t_eval` are not properly sorted.')
        # Processing the call keyword arguments (line 442)
        kwargs_55961 = {}
        # Getting the type of 'ValueError' (line 442)
        ValueError_55959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 442)
        ValueError_call_result_55962 = invoke(stypy.reporting.localization.Localization(__file__, 442, 18), ValueError_55959, *[str_55960], **kwargs_55961)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 442, 12), ValueError_call_result_55962, 'raise parameter', BaseException)
        # SSA join for if statement (line 441)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'tf' (line 444)
        tf_55963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'tf')
        # Getting the type of 't0' (line 444)
        t0_55964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 't0')
        # Applying the binary operator '>' (line 444)
        result_gt_55965 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 11), '>', tf_55963, t0_55964)
        
        # Testing the type of an if condition (line 444)
        if_condition_55966 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 8), result_gt_55965)
        # Assigning a type to the variable 'if_condition_55966' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'if_condition_55966', if_condition_55966)
        # SSA begins for if statement (line 444)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 445):
        
        # Assigning a Num to a Name (line 445):
        int_55967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 23), 'int')
        # Assigning a type to the variable 't_eval_i' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 't_eval_i', int_55967)
        # SSA branch for the else part of an if statement (line 444)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 448):
        
        # Assigning a Subscript to a Name (line 448):
        
        # Obtaining the type of the subscript
        int_55968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 30), 'int')
        slice_55969 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 448, 21), None, None, int_55968)
        # Getting the type of 't_eval' (line 448)
        t_eval_55970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 21), 't_eval')
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___55971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 21), t_eval_55970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 448)
        subscript_call_result_55972 = invoke(stypy.reporting.localization.Localization(__file__, 448, 21), getitem___55971, slice_55969)
        
        # Assigning a type to the variable 't_eval' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 't_eval', subscript_call_result_55972)
        
        # Assigning a Subscript to a Name (line 450):
        
        # Assigning a Subscript to a Name (line 450):
        
        # Obtaining the type of the subscript
        int_55973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 36), 'int')
        # Getting the type of 't_eval' (line 450)
        t_eval_55974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 23), 't_eval')
        # Obtaining the member 'shape' of a type (line 450)
        shape_55975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 23), t_eval_55974, 'shape')
        # Obtaining the member '__getitem__' of a type (line 450)
        getitem___55976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 23), shape_55975, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 450)
        subscript_call_result_55977 = invoke(stypy.reporting.localization.Localization(__file__, 450, 23), getitem___55976, int_55973)
        
        # Assigning a type to the variable 't_eval_i' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 't_eval_i', subscript_call_result_55977)
        # SSA join for if statement (line 444)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_55887:
            # SSA join for if statement (line 432)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'method' (line 452)
    method_55978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 7), 'method')
    # Getting the type of 'METHODS' (line 452)
    METHODS_55979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 17), 'METHODS')
    # Applying the binary operator 'in' (line 452)
    result_contains_55980 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 7), 'in', method_55978, METHODS_55979)
    
    # Testing the type of an if condition (line 452)
    if_condition_55981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 452, 4), result_contains_55980)
    # Assigning a type to the variable 'if_condition_55981' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'if_condition_55981', if_condition_55981)
    # SSA begins for if statement (line 452)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 453):
    
    # Assigning a Subscript to a Name (line 453):
    
    # Obtaining the type of the subscript
    # Getting the type of 'method' (line 453)
    method_55982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 25), 'method')
    # Getting the type of 'METHODS' (line 453)
    METHODS_55983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 17), 'METHODS')
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___55984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 17), METHODS_55983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_55985 = invoke(stypy.reporting.localization.Localization(__file__, 453, 17), getitem___55984, method_55982)
    
    # Assigning a type to the variable 'method' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'method', subscript_call_result_55985)
    # SSA join for if statement (line 452)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 455):
    
    # Assigning a Call to a Name (line 455):
    
    # Call to method(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'fun' (line 455)
    fun_55987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'fun', False)
    # Getting the type of 't0' (line 455)
    t0_55988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 25), 't0', False)
    # Getting the type of 'y0' (line 455)
    y0_55989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 29), 'y0', False)
    # Getting the type of 'tf' (line 455)
    tf_55990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'tf', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'vectorized' (line 455)
    vectorized_55991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 48), 'vectorized', False)
    keyword_55992 = vectorized_55991
    # Getting the type of 'options' (line 455)
    options_55993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 62), 'options', False)
    kwargs_55994 = {'vectorized': keyword_55992, 'options_55993': options_55993}
    # Getting the type of 'method' (line 455)
    method_55986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 13), 'method', False)
    # Calling method(args, kwargs) (line 455)
    method_call_result_55995 = invoke(stypy.reporting.localization.Localization(__file__, 455, 13), method_55986, *[fun_55987, t0_55988, y0_55989, tf_55990], **kwargs_55994)
    
    # Assigning a type to the variable 'solver' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'solver', method_call_result_55995)
    
    # Type idiom detected: calculating its left and rigth part (line 457)
    # Getting the type of 't_eval' (line 457)
    t_eval_55996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 7), 't_eval')
    # Getting the type of 'None' (line 457)
    None_55997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 17), 'None')
    
    (may_be_55998, more_types_in_union_55999) = may_be_none(t_eval_55996, None_55997)

    if may_be_55998:

        if more_types_in_union_55999:
            # Runtime conditional SSA (line 457)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 458):
        
        # Assigning a List to a Name (line 458):
        
        # Obtaining an instance of the builtin type 'list' (line 458)
        list_56000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 458)
        # Adding element type (line 458)
        # Getting the type of 't0' (line 458)
        t0_56001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 14), 't0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 13), list_56000, t0_56001)
        
        # Assigning a type to the variable 'ts' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'ts', list_56000)
        
        # Assigning a List to a Name (line 459):
        
        # Assigning a List to a Name (line 459):
        
        # Obtaining an instance of the builtin type 'list' (line 459)
        list_56002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 459)
        # Adding element type (line 459)
        # Getting the type of 'y0' (line 459)
        y0_56003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 14), 'y0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 13), list_56002, y0_56003)
        
        # Assigning a type to the variable 'ys' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'ys', list_56002)

        if more_types_in_union_55999:
            # Runtime conditional SSA for else branch (line 457)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_55998) or more_types_in_union_55999):
        
        # Assigning a List to a Name (line 461):
        
        # Assigning a List to a Name (line 461):
        
        # Obtaining an instance of the builtin type 'list' (line 461)
        list_56004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 461)
        
        # Assigning a type to the variable 'ts' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'ts', list_56004)
        
        # Assigning a List to a Name (line 462):
        
        # Assigning a List to a Name (line 462):
        
        # Obtaining an instance of the builtin type 'list' (line 462)
        list_56005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 462)
        
        # Assigning a type to the variable 'ys' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'ys', list_56005)

        if (may_be_55998 and more_types_in_union_55999):
            # SSA join for if statement (line 457)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 464):
    
    # Assigning a List to a Name (line 464):
    
    # Obtaining an instance of the builtin type 'list' (line 464)
    list_56006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 464)
    
    # Assigning a type to the variable 'interpolants' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'interpolants', list_56006)
    
    # Assigning a Call to a Tuple (line 466):
    
    # Assigning a Subscript to a Name (line 466):
    
    # Obtaining the type of the subscript
    int_56007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 4), 'int')
    
    # Call to prepare_events(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'events' (line 466)
    events_56009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 52), 'events', False)
    # Processing the call keyword arguments (line 466)
    kwargs_56010 = {}
    # Getting the type of 'prepare_events' (line 466)
    prepare_events_56008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 37), 'prepare_events', False)
    # Calling prepare_events(args, kwargs) (line 466)
    prepare_events_call_result_56011 = invoke(stypy.reporting.localization.Localization(__file__, 466, 37), prepare_events_56008, *[events_56009], **kwargs_56010)
    
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___56012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 4), prepare_events_call_result_56011, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_56013 = invoke(stypy.reporting.localization.Localization(__file__, 466, 4), getitem___56012, int_56007)
    
    # Assigning a type to the variable 'tuple_var_assignment_55569' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'tuple_var_assignment_55569', subscript_call_result_56013)
    
    # Assigning a Subscript to a Name (line 466):
    
    # Obtaining the type of the subscript
    int_56014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 4), 'int')
    
    # Call to prepare_events(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'events' (line 466)
    events_56016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 52), 'events', False)
    # Processing the call keyword arguments (line 466)
    kwargs_56017 = {}
    # Getting the type of 'prepare_events' (line 466)
    prepare_events_56015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 37), 'prepare_events', False)
    # Calling prepare_events(args, kwargs) (line 466)
    prepare_events_call_result_56018 = invoke(stypy.reporting.localization.Localization(__file__, 466, 37), prepare_events_56015, *[events_56016], **kwargs_56017)
    
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___56019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 4), prepare_events_call_result_56018, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_56020 = invoke(stypy.reporting.localization.Localization(__file__, 466, 4), getitem___56019, int_56014)
    
    # Assigning a type to the variable 'tuple_var_assignment_55570' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'tuple_var_assignment_55570', subscript_call_result_56020)
    
    # Assigning a Subscript to a Name (line 466):
    
    # Obtaining the type of the subscript
    int_56021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 4), 'int')
    
    # Call to prepare_events(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'events' (line 466)
    events_56023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 52), 'events', False)
    # Processing the call keyword arguments (line 466)
    kwargs_56024 = {}
    # Getting the type of 'prepare_events' (line 466)
    prepare_events_56022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 37), 'prepare_events', False)
    # Calling prepare_events(args, kwargs) (line 466)
    prepare_events_call_result_56025 = invoke(stypy.reporting.localization.Localization(__file__, 466, 37), prepare_events_56022, *[events_56023], **kwargs_56024)
    
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___56026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 4), prepare_events_call_result_56025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_56027 = invoke(stypy.reporting.localization.Localization(__file__, 466, 4), getitem___56026, int_56021)
    
    # Assigning a type to the variable 'tuple_var_assignment_55571' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'tuple_var_assignment_55571', subscript_call_result_56027)
    
    # Assigning a Name to a Name (line 466):
    # Getting the type of 'tuple_var_assignment_55569' (line 466)
    tuple_var_assignment_55569_56028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'tuple_var_assignment_55569')
    # Assigning a type to the variable 'events' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'events', tuple_var_assignment_55569_56028)
    
    # Assigning a Name to a Name (line 466):
    # Getting the type of 'tuple_var_assignment_55570' (line 466)
    tuple_var_assignment_55570_56029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'tuple_var_assignment_55570')
    # Assigning a type to the variable 'is_terminal' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'is_terminal', tuple_var_assignment_55570_56029)
    
    # Assigning a Name to a Name (line 466):
    # Getting the type of 'tuple_var_assignment_55571' (line 466)
    tuple_var_assignment_55571_56030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'tuple_var_assignment_55571')
    # Assigning a type to the variable 'event_dir' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 25), 'event_dir', tuple_var_assignment_55571_56030)
    
    # Type idiom detected: calculating its left and rigth part (line 468)
    # Getting the type of 'events' (line 468)
    events_56031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'events')
    # Getting the type of 'None' (line 468)
    None_56032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 21), 'None')
    
    (may_be_56033, more_types_in_union_56034) = may_not_be_none(events_56031, None_56032)

    if may_be_56033:

        if more_types_in_union_56034:
            # Runtime conditional SSA (line 468)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a ListComp to a Name (line 469):
        
        # Assigning a ListComp to a Name (line 469):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'events' (line 469)
        events_56040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 40), 'events')
        comprehension_56041 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 13), events_56040)
        # Assigning a type to the variable 'event' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 13), 'event', comprehension_56041)
        
        # Call to event(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 't0' (line 469)
        t0_56036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 't0', False)
        # Getting the type of 'y0' (line 469)
        y0_56037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 23), 'y0', False)
        # Processing the call keyword arguments (line 469)
        kwargs_56038 = {}
        # Getting the type of 'event' (line 469)
        event_56035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 13), 'event', False)
        # Calling event(args, kwargs) (line 469)
        event_call_result_56039 = invoke(stypy.reporting.localization.Localization(__file__, 469, 13), event_56035, *[t0_56036, y0_56037], **kwargs_56038)
        
        list_56042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 13), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 13), list_56042, event_call_result_56039)
        # Assigning a type to the variable 'g' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'g', list_56042)
        
        # Assigning a ListComp to a Name (line 470):
        
        # Assigning a ListComp to a Name (line 470):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 470)
        # Processing the call arguments (line 470)
        
        # Call to len(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'events' (line 470)
        events_56046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 42), 'events', False)
        # Processing the call keyword arguments (line 470)
        kwargs_56047 = {}
        # Getting the type of 'len' (line 470)
        len_56045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 38), 'len', False)
        # Calling len(args, kwargs) (line 470)
        len_call_result_56048 = invoke(stypy.reporting.localization.Localization(__file__, 470, 38), len_56045, *[events_56046], **kwargs_56047)
        
        # Processing the call keyword arguments (line 470)
        kwargs_56049 = {}
        # Getting the type of 'range' (line 470)
        range_56044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 32), 'range', False)
        # Calling range(args, kwargs) (line 470)
        range_call_result_56050 = invoke(stypy.reporting.localization.Localization(__file__, 470, 32), range_56044, *[len_call_result_56048], **kwargs_56049)
        
        comprehension_56051 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 20), range_call_result_56050)
        # Assigning a type to the variable '_' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), '_', comprehension_56051)
        
        # Obtaining an instance of the builtin type 'list' (line 470)
        list_56043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 470)
        
        list_56052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 20), list_56052, list_56043)
        # Assigning a type to the variable 't_events' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 't_events', list_56052)

        if more_types_in_union_56034:
            # Runtime conditional SSA for else branch (line 468)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_56033) or more_types_in_union_56034):
        
        # Assigning a Name to a Name (line 472):
        
        # Assigning a Name to a Name (line 472):
        # Getting the type of 'None' (line 472)
        None_56053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 19), 'None')
        # Assigning a type to the variable 't_events' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 't_events', None_56053)

        if (may_be_56033 and more_types_in_union_56034):
            # SSA join for if statement (line 468)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 474):
    
    # Assigning a Name to a Name (line 474):
    # Getting the type of 'None' (line 474)
    None_56054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 13), 'None')
    # Assigning a type to the variable 'status' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'status', None_56054)
    
    
    # Getting the type of 'status' (line 475)
    status_56055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 10), 'status')
    # Getting the type of 'None' (line 475)
    None_56056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'None')
    # Applying the binary operator 'is' (line 475)
    result_is__56057 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 10), 'is', status_56055, None_56056)
    
    # Testing the type of an if condition (line 475)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 475, 4), result_is__56057)
    # SSA begins for while statement (line 475)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to step(...): (line 476)
    # Processing the call keyword arguments (line 476)
    kwargs_56060 = {}
    # Getting the type of 'solver' (line 476)
    solver_56058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 18), 'solver', False)
    # Obtaining the member 'step' of a type (line 476)
    step_56059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 18), solver_56058, 'step')
    # Calling step(args, kwargs) (line 476)
    step_call_result_56061 = invoke(stypy.reporting.localization.Localization(__file__, 476, 18), step_56059, *[], **kwargs_56060)
    
    # Assigning a type to the variable 'message' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'message', step_call_result_56061)
    
    
    # Getting the type of 'solver' (line 478)
    solver_56062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'solver')
    # Obtaining the member 'status' of a type (line 478)
    status_56063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 11), solver_56062, 'status')
    str_56064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 28), 'str', 'finished')
    # Applying the binary operator '==' (line 478)
    result_eq_56065 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 11), '==', status_56063, str_56064)
    
    # Testing the type of an if condition (line 478)
    if_condition_56066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 8), result_eq_56065)
    # Assigning a type to the variable 'if_condition_56066' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'if_condition_56066', if_condition_56066)
    # SSA begins for if statement (line 478)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 479):
    
    # Assigning a Num to a Name (line 479):
    int_56067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 21), 'int')
    # Assigning a type to the variable 'status' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'status', int_56067)
    # SSA branch for the else part of an if statement (line 478)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'solver' (line 480)
    solver_56068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 13), 'solver')
    # Obtaining the member 'status' of a type (line 480)
    status_56069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 13), solver_56068, 'status')
    str_56070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 30), 'str', 'failed')
    # Applying the binary operator '==' (line 480)
    result_eq_56071 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 13), '==', status_56069, str_56070)
    
    # Testing the type of an if condition (line 480)
    if_condition_56072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 13), result_eq_56071)
    # Assigning a type to the variable 'if_condition_56072' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 13), 'if_condition_56072', if_condition_56072)
    # SSA begins for if statement (line 480)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 481):
    
    # Assigning a Num to a Name (line 481):
    int_56073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 21), 'int')
    # Assigning a type to the variable 'status' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'status', int_56073)
    # SSA join for if statement (line 480)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 478)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 484):
    
    # Assigning a Attribute to a Name (line 484):
    # Getting the type of 'solver' (line 484)
    solver_56074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'solver')
    # Obtaining the member 't_old' of a type (line 484)
    t_old_56075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 16), solver_56074, 't_old')
    # Assigning a type to the variable 't_old' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 't_old', t_old_56075)
    
    # Assigning a Attribute to a Name (line 485):
    
    # Assigning a Attribute to a Name (line 485):
    # Getting the type of 'solver' (line 485)
    solver_56076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'solver')
    # Obtaining the member 't' of a type (line 485)
    t_56077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 12), solver_56076, 't')
    # Assigning a type to the variable 't' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 't', t_56077)
    
    # Assigning a Attribute to a Name (line 486):
    
    # Assigning a Attribute to a Name (line 486):
    # Getting the type of 'solver' (line 486)
    solver_56078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'solver')
    # Obtaining the member 'y' of a type (line 486)
    y_56079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 12), solver_56078, 'y')
    # Assigning a type to the variable 'y' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'y', y_56079)
    
    # Getting the type of 'dense_output' (line 488)
    dense_output_56080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 11), 'dense_output')
    # Testing the type of an if condition (line 488)
    if_condition_56081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 8), dense_output_56080)
    # Assigning a type to the variable 'if_condition_56081' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'if_condition_56081', if_condition_56081)
    # SSA begins for if statement (line 488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 489):
    
    # Assigning a Call to a Name (line 489):
    
    # Call to dense_output(...): (line 489)
    # Processing the call keyword arguments (line 489)
    kwargs_56084 = {}
    # Getting the type of 'solver' (line 489)
    solver_56082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 18), 'solver', False)
    # Obtaining the member 'dense_output' of a type (line 489)
    dense_output_56083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 18), solver_56082, 'dense_output')
    # Calling dense_output(args, kwargs) (line 489)
    dense_output_call_result_56085 = invoke(stypy.reporting.localization.Localization(__file__, 489, 18), dense_output_56083, *[], **kwargs_56084)
    
    # Assigning a type to the variable 'sol' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'sol', dense_output_call_result_56085)
    
    # Call to append(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'sol' (line 490)
    sol_56088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 32), 'sol', False)
    # Processing the call keyword arguments (line 490)
    kwargs_56089 = {}
    # Getting the type of 'interpolants' (line 490)
    interpolants_56086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'interpolants', False)
    # Obtaining the member 'append' of a type (line 490)
    append_56087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), interpolants_56086, 'append')
    # Calling append(args, kwargs) (line 490)
    append_call_result_56090 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), append_56087, *[sol_56088], **kwargs_56089)
    
    # SSA branch for the else part of an if statement (line 488)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 492):
    
    # Assigning a Name to a Name (line 492):
    # Getting the type of 'None' (line 492)
    None_56091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 18), 'None')
    # Assigning a type to the variable 'sol' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'sol', None_56091)
    # SSA join for if statement (line 488)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 494)
    # Getting the type of 'events' (line 494)
    events_56092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'events')
    # Getting the type of 'None' (line 494)
    None_56093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 25), 'None')
    
    (may_be_56094, more_types_in_union_56095) = may_not_be_none(events_56092, None_56093)

    if may_be_56094:

        if more_types_in_union_56095:
            # Runtime conditional SSA (line 494)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a ListComp to a Name (line 495):
        
        # Assigning a ListComp to a Name (line 495):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'events' (line 495)
        events_56101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 46), 'events')
        comprehension_56102 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 21), events_56101)
        # Assigning a type to the variable 'event' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 21), 'event', comprehension_56102)
        
        # Call to event(...): (line 495)
        # Processing the call arguments (line 495)
        # Getting the type of 't' (line 495)
        t_56097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 27), 't', False)
        # Getting the type of 'y' (line 495)
        y_56098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 30), 'y', False)
        # Processing the call keyword arguments (line 495)
        kwargs_56099 = {}
        # Getting the type of 'event' (line 495)
        event_56096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 21), 'event', False)
        # Calling event(args, kwargs) (line 495)
        event_call_result_56100 = invoke(stypy.reporting.localization.Localization(__file__, 495, 21), event_56096, *[t_56097, y_56098], **kwargs_56099)
        
        list_56103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 21), list_56103, event_call_result_56100)
        # Assigning a type to the variable 'g_new' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'g_new', list_56103)
        
        # Assigning a Call to a Name (line 496):
        
        # Assigning a Call to a Name (line 496):
        
        # Call to find_active_events(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'g' (line 496)
        g_56105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 47), 'g', False)
        # Getting the type of 'g_new' (line 496)
        g_new_56106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 50), 'g_new', False)
        # Getting the type of 'event_dir' (line 496)
        event_dir_56107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 57), 'event_dir', False)
        # Processing the call keyword arguments (line 496)
        kwargs_56108 = {}
        # Getting the type of 'find_active_events' (line 496)
        find_active_events_56104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 28), 'find_active_events', False)
        # Calling find_active_events(args, kwargs) (line 496)
        find_active_events_call_result_56109 = invoke(stypy.reporting.localization.Localization(__file__, 496, 28), find_active_events_56104, *[g_56105, g_new_56106, event_dir_56107], **kwargs_56108)
        
        # Assigning a type to the variable 'active_events' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'active_events', find_active_events_call_result_56109)
        
        
        # Getting the type of 'active_events' (line 497)
        active_events_56110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), 'active_events')
        # Obtaining the member 'size' of a type (line 497)
        size_56111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 15), active_events_56110, 'size')
        int_56112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 36), 'int')
        # Applying the binary operator '>' (line 497)
        result_gt_56113 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 15), '>', size_56111, int_56112)
        
        # Testing the type of an if condition (line 497)
        if_condition_56114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 12), result_gt_56113)
        # Assigning a type to the variable 'if_condition_56114' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'if_condition_56114', if_condition_56114)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 498)
        # Getting the type of 'sol' (line 498)
        sol_56115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 19), 'sol')
        # Getting the type of 'None' (line 498)
        None_56116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 26), 'None')
        
        (may_be_56117, more_types_in_union_56118) = may_be_none(sol_56115, None_56116)

        if may_be_56117:

            if more_types_in_union_56118:
                # Runtime conditional SSA (line 498)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 499):
            
            # Assigning a Call to a Name (line 499):
            
            # Call to dense_output(...): (line 499)
            # Processing the call keyword arguments (line 499)
            kwargs_56121 = {}
            # Getting the type of 'solver' (line 499)
            solver_56119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 26), 'solver', False)
            # Obtaining the member 'dense_output' of a type (line 499)
            dense_output_56120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 26), solver_56119, 'dense_output')
            # Calling dense_output(args, kwargs) (line 499)
            dense_output_call_result_56122 = invoke(stypy.reporting.localization.Localization(__file__, 499, 26), dense_output_56120, *[], **kwargs_56121)
            
            # Assigning a type to the variable 'sol' (line 499)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'sol', dense_output_call_result_56122)

            if more_types_in_union_56118:
                # SSA join for if statement (line 498)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 501):
        
        # Assigning a Subscript to a Name (line 501):
        
        # Obtaining the type of the subscript
        int_56123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 16), 'int')
        
        # Call to handle_events(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'sol' (line 502)
        sol_56125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 20), 'sol', False)
        # Getting the type of 'events' (line 502)
        events_56126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'events', False)
        # Getting the type of 'active_events' (line 502)
        active_events_56127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 33), 'active_events', False)
        # Getting the type of 'is_terminal' (line 502)
        is_terminal_56128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 48), 'is_terminal', False)
        # Getting the type of 't_old' (line 502)
        t_old_56129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 61), 't_old', False)
        # Getting the type of 't' (line 502)
        t_56130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 68), 't', False)
        # Processing the call keyword arguments (line 501)
        kwargs_56131 = {}
        # Getting the type of 'handle_events' (line 501)
        handle_events_56124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 49), 'handle_events', False)
        # Calling handle_events(args, kwargs) (line 501)
        handle_events_call_result_56132 = invoke(stypy.reporting.localization.Localization(__file__, 501, 49), handle_events_56124, *[sol_56125, events_56126, active_events_56127, is_terminal_56128, t_old_56129, t_56130], **kwargs_56131)
        
        # Obtaining the member '__getitem__' of a type (line 501)
        getitem___56133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 16), handle_events_call_result_56132, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 501)
        subscript_call_result_56134 = invoke(stypy.reporting.localization.Localization(__file__, 501, 16), getitem___56133, int_56123)
        
        # Assigning a type to the variable 'tuple_var_assignment_55572' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_55572', subscript_call_result_56134)
        
        # Assigning a Subscript to a Name (line 501):
        
        # Obtaining the type of the subscript
        int_56135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 16), 'int')
        
        # Call to handle_events(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'sol' (line 502)
        sol_56137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 20), 'sol', False)
        # Getting the type of 'events' (line 502)
        events_56138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'events', False)
        # Getting the type of 'active_events' (line 502)
        active_events_56139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 33), 'active_events', False)
        # Getting the type of 'is_terminal' (line 502)
        is_terminal_56140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 48), 'is_terminal', False)
        # Getting the type of 't_old' (line 502)
        t_old_56141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 61), 't_old', False)
        # Getting the type of 't' (line 502)
        t_56142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 68), 't', False)
        # Processing the call keyword arguments (line 501)
        kwargs_56143 = {}
        # Getting the type of 'handle_events' (line 501)
        handle_events_56136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 49), 'handle_events', False)
        # Calling handle_events(args, kwargs) (line 501)
        handle_events_call_result_56144 = invoke(stypy.reporting.localization.Localization(__file__, 501, 49), handle_events_56136, *[sol_56137, events_56138, active_events_56139, is_terminal_56140, t_old_56141, t_56142], **kwargs_56143)
        
        # Obtaining the member '__getitem__' of a type (line 501)
        getitem___56145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 16), handle_events_call_result_56144, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 501)
        subscript_call_result_56146 = invoke(stypy.reporting.localization.Localization(__file__, 501, 16), getitem___56145, int_56135)
        
        # Assigning a type to the variable 'tuple_var_assignment_55573' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_55573', subscript_call_result_56146)
        
        # Assigning a Subscript to a Name (line 501):
        
        # Obtaining the type of the subscript
        int_56147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 16), 'int')
        
        # Call to handle_events(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'sol' (line 502)
        sol_56149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 20), 'sol', False)
        # Getting the type of 'events' (line 502)
        events_56150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'events', False)
        # Getting the type of 'active_events' (line 502)
        active_events_56151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 33), 'active_events', False)
        # Getting the type of 'is_terminal' (line 502)
        is_terminal_56152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 48), 'is_terminal', False)
        # Getting the type of 't_old' (line 502)
        t_old_56153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 61), 't_old', False)
        # Getting the type of 't' (line 502)
        t_56154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 68), 't', False)
        # Processing the call keyword arguments (line 501)
        kwargs_56155 = {}
        # Getting the type of 'handle_events' (line 501)
        handle_events_56148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 49), 'handle_events', False)
        # Calling handle_events(args, kwargs) (line 501)
        handle_events_call_result_56156 = invoke(stypy.reporting.localization.Localization(__file__, 501, 49), handle_events_56148, *[sol_56149, events_56150, active_events_56151, is_terminal_56152, t_old_56153, t_56154], **kwargs_56155)
        
        # Obtaining the member '__getitem__' of a type (line 501)
        getitem___56157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 16), handle_events_call_result_56156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 501)
        subscript_call_result_56158 = invoke(stypy.reporting.localization.Localization(__file__, 501, 16), getitem___56157, int_56147)
        
        # Assigning a type to the variable 'tuple_var_assignment_55574' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_55574', subscript_call_result_56158)
        
        # Assigning a Name to a Name (line 501):
        # Getting the type of 'tuple_var_assignment_55572' (line 501)
        tuple_var_assignment_55572_56159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_55572')
        # Assigning a type to the variable 'root_indices' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'root_indices', tuple_var_assignment_55572_56159)
        
        # Assigning a Name to a Name (line 501):
        # Getting the type of 'tuple_var_assignment_55573' (line 501)
        tuple_var_assignment_55573_56160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_55573')
        # Assigning a type to the variable 'roots' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 30), 'roots', tuple_var_assignment_55573_56160)
        
        # Assigning a Name to a Name (line 501):
        # Getting the type of 'tuple_var_assignment_55574' (line 501)
        tuple_var_assignment_55574_56161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'tuple_var_assignment_55574')
        # Assigning a type to the variable 'terminate' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 37), 'terminate', tuple_var_assignment_55574_56161)
        
        
        # Call to zip(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'root_indices' (line 504)
        root_indices_56163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 33), 'root_indices', False)
        # Getting the type of 'roots' (line 504)
        roots_56164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 47), 'roots', False)
        # Processing the call keyword arguments (line 504)
        kwargs_56165 = {}
        # Getting the type of 'zip' (line 504)
        zip_56162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 29), 'zip', False)
        # Calling zip(args, kwargs) (line 504)
        zip_call_result_56166 = invoke(stypy.reporting.localization.Localization(__file__, 504, 29), zip_56162, *[root_indices_56163, roots_56164], **kwargs_56165)
        
        # Testing the type of a for loop iterable (line 504)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 504, 16), zip_call_result_56166)
        # Getting the type of the for loop variable (line 504)
        for_loop_var_56167 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 504, 16), zip_call_result_56166)
        # Assigning a type to the variable 'e' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 16), 'e', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 16), for_loop_var_56167))
        # Assigning a type to the variable 'te' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 16), 'te', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 16), for_loop_var_56167))
        # SSA begins for a for statement (line 504)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'te' (line 505)
        te_56173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 39), 'te', False)
        # Processing the call keyword arguments (line 505)
        kwargs_56174 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'e' (line 505)
        e_56168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 29), 'e', False)
        # Getting the type of 't_events' (line 505)
        t_events_56169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 20), 't_events', False)
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___56170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 20), t_events_56169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_56171 = invoke(stypy.reporting.localization.Localization(__file__, 505, 20), getitem___56170, e_56168)
        
        # Obtaining the member 'append' of a type (line 505)
        append_56172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 20), subscript_call_result_56171, 'append')
        # Calling append(args, kwargs) (line 505)
        append_call_result_56175 = invoke(stypy.reporting.localization.Localization(__file__, 505, 20), append_56172, *[te_56173], **kwargs_56174)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'terminate' (line 507)
        terminate_56176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 19), 'terminate')
        # Testing the type of an if condition (line 507)
        if_condition_56177 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 16), terminate_56176)
        # Assigning a type to the variable 'if_condition_56177' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'if_condition_56177', if_condition_56177)
        # SSA begins for if statement (line 507)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 508):
        
        # Assigning a Num to a Name (line 508):
        int_56178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 29), 'int')
        # Assigning a type to the variable 'status' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 20), 'status', int_56178)
        
        # Assigning a Subscript to a Name (line 509):
        
        # Assigning a Subscript to a Name (line 509):
        
        # Obtaining the type of the subscript
        int_56179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 30), 'int')
        # Getting the type of 'roots' (line 509)
        roots_56180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 24), 'roots')
        # Obtaining the member '__getitem__' of a type (line 509)
        getitem___56181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 24), roots_56180, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 509)
        subscript_call_result_56182 = invoke(stypy.reporting.localization.Localization(__file__, 509, 24), getitem___56181, int_56179)
        
        # Assigning a type to the variable 't' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 20), 't', subscript_call_result_56182)
        
        # Assigning a Call to a Name (line 510):
        
        # Assigning a Call to a Name (line 510):
        
        # Call to sol(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 't' (line 510)
        t_56184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 28), 't', False)
        # Processing the call keyword arguments (line 510)
        kwargs_56185 = {}
        # Getting the type of 'sol' (line 510)
        sol_56183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 24), 'sol', False)
        # Calling sol(args, kwargs) (line 510)
        sol_call_result_56186 = invoke(stypy.reporting.localization.Localization(__file__, 510, 24), sol_56183, *[t_56184], **kwargs_56185)
        
        # Assigning a type to the variable 'y' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 20), 'y', sol_call_result_56186)
        # SSA join for if statement (line 507)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 512):
        
        # Assigning a Name to a Name (line 512):
        # Getting the type of 'g_new' (line 512)
        g_new_56187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 16), 'g_new')
        # Assigning a type to the variable 'g' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'g', g_new_56187)

        if more_types_in_union_56095:
            # SSA join for if statement (line 494)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 514)
    # Getting the type of 't_eval' (line 514)
    t_eval_56188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 't_eval')
    # Getting the type of 'None' (line 514)
    None_56189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 21), 'None')
    
    (may_be_56190, more_types_in_union_56191) = may_be_none(t_eval_56188, None_56189)

    if may_be_56190:

        if more_types_in_union_56191:
            # Runtime conditional SSA (line 514)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 515)
        # Processing the call arguments (line 515)
        # Getting the type of 't' (line 515)
        t_56194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 22), 't', False)
        # Processing the call keyword arguments (line 515)
        kwargs_56195 = {}
        # Getting the type of 'ts' (line 515)
        ts_56192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'ts', False)
        # Obtaining the member 'append' of a type (line 515)
        append_56193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 12), ts_56192, 'append')
        # Calling append(args, kwargs) (line 515)
        append_call_result_56196 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), append_56193, *[t_56194], **kwargs_56195)
        
        
        # Call to append(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'y' (line 516)
        y_56199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 22), 'y', False)
        # Processing the call keyword arguments (line 516)
        kwargs_56200 = {}
        # Getting the type of 'ys' (line 516)
        ys_56197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'ys', False)
        # Obtaining the member 'append' of a type (line 516)
        append_56198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), ys_56197, 'append')
        # Calling append(args, kwargs) (line 516)
        append_call_result_56201 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), append_56198, *[y_56199], **kwargs_56200)
        

        if more_types_in_union_56191:
            # Runtime conditional SSA for else branch (line 514)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_56190) or more_types_in_union_56191):
        
        
        # Getting the type of 'solver' (line 519)
        solver_56202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'solver')
        # Obtaining the member 'direction' of a type (line 519)
        direction_56203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 15), solver_56202, 'direction')
        int_56204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 34), 'int')
        # Applying the binary operator '>' (line 519)
        result_gt_56205 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 15), '>', direction_56203, int_56204)
        
        # Testing the type of an if condition (line 519)
        if_condition_56206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 519, 12), result_gt_56205)
        # Assigning a type to the variable 'if_condition_56206' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'if_condition_56206', if_condition_56206)
        # SSA begins for if statement (line 519)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 520):
        
        # Assigning a Call to a Name (line 520):
        
        # Call to searchsorted(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 't_eval' (line 520)
        t_eval_56209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 47), 't_eval', False)
        # Getting the type of 't' (line 520)
        t_56210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 55), 't', False)
        # Processing the call keyword arguments (line 520)
        str_56211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 63), 'str', 'right')
        keyword_56212 = str_56211
        kwargs_56213 = {'side': keyword_56212}
        # Getting the type of 'np' (line 520)
        np_56207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 31), 'np', False)
        # Obtaining the member 'searchsorted' of a type (line 520)
        searchsorted_56208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 31), np_56207, 'searchsorted')
        # Calling searchsorted(args, kwargs) (line 520)
        searchsorted_call_result_56214 = invoke(stypy.reporting.localization.Localization(__file__, 520, 31), searchsorted_56208, *[t_eval_56209, t_56210], **kwargs_56213)
        
        # Assigning a type to the variable 't_eval_i_new' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 't_eval_i_new', searchsorted_call_result_56214)
        
        # Assigning a Subscript to a Name (line 521):
        
        # Assigning a Subscript to a Name (line 521):
        
        # Obtaining the type of the subscript
        # Getting the type of 't_eval_i' (line 521)
        t_eval_i_56215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 37), 't_eval_i')
        # Getting the type of 't_eval_i_new' (line 521)
        t_eval_i_new_56216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 46), 't_eval_i_new')
        slice_56217 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 521, 30), t_eval_i_56215, t_eval_i_new_56216, None)
        # Getting the type of 't_eval' (line 521)
        t_eval_56218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 30), 't_eval')
        # Obtaining the member '__getitem__' of a type (line 521)
        getitem___56219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 30), t_eval_56218, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 521)
        subscript_call_result_56220 = invoke(stypy.reporting.localization.Localization(__file__, 521, 30), getitem___56219, slice_56217)
        
        # Assigning a type to the variable 't_eval_step' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 16), 't_eval_step', subscript_call_result_56220)
        # SSA branch for the else part of an if statement (line 519)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 523):
        
        # Assigning a Call to a Name (line 523):
        
        # Call to searchsorted(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 't_eval' (line 523)
        t_eval_56223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 47), 't_eval', False)
        # Getting the type of 't' (line 523)
        t_56224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 55), 't', False)
        # Processing the call keyword arguments (line 523)
        str_56225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 63), 'str', 'left')
        keyword_56226 = str_56225
        kwargs_56227 = {'side': keyword_56226}
        # Getting the type of 'np' (line 523)
        np_56221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 31), 'np', False)
        # Obtaining the member 'searchsorted' of a type (line 523)
        searchsorted_56222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 31), np_56221, 'searchsorted')
        # Calling searchsorted(args, kwargs) (line 523)
        searchsorted_call_result_56228 = invoke(stypy.reporting.localization.Localization(__file__, 523, 31), searchsorted_56222, *[t_eval_56223, t_56224], **kwargs_56227)
        
        # Assigning a type to the variable 't_eval_i_new' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 't_eval_i_new', searchsorted_call_result_56228)
        
        # Assigning a Subscript to a Name (line 527):
        
        # Assigning a Subscript to a Name (line 527):
        
        # Obtaining the type of the subscript
        int_56229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 62), 'int')
        slice_56230 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 527, 30), None, None, int_56229)
        
        # Obtaining the type of the subscript
        # Getting the type of 't_eval_i_new' (line 527)
        t_eval_i_new_56231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 37), 't_eval_i_new')
        # Getting the type of 't_eval_i' (line 527)
        t_eval_i_56232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 50), 't_eval_i')
        slice_56233 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 527, 30), t_eval_i_new_56231, t_eval_i_56232, None)
        # Getting the type of 't_eval' (line 527)
        t_eval_56234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 30), 't_eval')
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___56235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 30), t_eval_56234, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_56236 = invoke(stypy.reporting.localization.Localization(__file__, 527, 30), getitem___56235, slice_56233)
        
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___56237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 30), subscript_call_result_56236, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_56238 = invoke(stypy.reporting.localization.Localization(__file__, 527, 30), getitem___56237, slice_56230)
        
        # Assigning a type to the variable 't_eval_step' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 16), 't_eval_step', subscript_call_result_56238)
        # SSA join for if statement (line 519)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 't_eval_step' (line 529)
        t_eval_step_56239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 15), 't_eval_step')
        # Obtaining the member 'size' of a type (line 529)
        size_56240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 15), t_eval_step_56239, 'size')
        int_56241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 34), 'int')
        # Applying the binary operator '>' (line 529)
        result_gt_56242 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 15), '>', size_56240, int_56241)
        
        # Testing the type of an if condition (line 529)
        if_condition_56243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 12), result_gt_56242)
        # Assigning a type to the variable 'if_condition_56243' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'if_condition_56243', if_condition_56243)
        # SSA begins for if statement (line 529)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 530)
        # Getting the type of 'sol' (line 530)
        sol_56244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 19), 'sol')
        # Getting the type of 'None' (line 530)
        None_56245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 26), 'None')
        
        (may_be_56246, more_types_in_union_56247) = may_be_none(sol_56244, None_56245)

        if may_be_56246:

            if more_types_in_union_56247:
                # Runtime conditional SSA (line 530)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 531):
            
            # Assigning a Call to a Name (line 531):
            
            # Call to dense_output(...): (line 531)
            # Processing the call keyword arguments (line 531)
            kwargs_56250 = {}
            # Getting the type of 'solver' (line 531)
            solver_56248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 26), 'solver', False)
            # Obtaining the member 'dense_output' of a type (line 531)
            dense_output_56249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 26), solver_56248, 'dense_output')
            # Calling dense_output(args, kwargs) (line 531)
            dense_output_call_result_56251 = invoke(stypy.reporting.localization.Localization(__file__, 531, 26), dense_output_56249, *[], **kwargs_56250)
            
            # Assigning a type to the variable 'sol' (line 531)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 20), 'sol', dense_output_call_result_56251)

            if more_types_in_union_56247:
                # SSA join for if statement (line 530)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 't_eval_step' (line 532)
        t_eval_step_56254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 26), 't_eval_step', False)
        # Processing the call keyword arguments (line 532)
        kwargs_56255 = {}
        # Getting the type of 'ts' (line 532)
        ts_56252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'ts', False)
        # Obtaining the member 'append' of a type (line 532)
        append_56253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 16), ts_56252, 'append')
        # Calling append(args, kwargs) (line 532)
        append_call_result_56256 = invoke(stypy.reporting.localization.Localization(__file__, 532, 16), append_56253, *[t_eval_step_56254], **kwargs_56255)
        
        
        # Call to append(...): (line 533)
        # Processing the call arguments (line 533)
        
        # Call to sol(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 't_eval_step' (line 533)
        t_eval_step_56260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 30), 't_eval_step', False)
        # Processing the call keyword arguments (line 533)
        kwargs_56261 = {}
        # Getting the type of 'sol' (line 533)
        sol_56259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 26), 'sol', False)
        # Calling sol(args, kwargs) (line 533)
        sol_call_result_56262 = invoke(stypy.reporting.localization.Localization(__file__, 533, 26), sol_56259, *[t_eval_step_56260], **kwargs_56261)
        
        # Processing the call keyword arguments (line 533)
        kwargs_56263 = {}
        # Getting the type of 'ys' (line 533)
        ys_56257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 16), 'ys', False)
        # Obtaining the member 'append' of a type (line 533)
        append_56258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 16), ys_56257, 'append')
        # Calling append(args, kwargs) (line 533)
        append_call_result_56264 = invoke(stypy.reporting.localization.Localization(__file__, 533, 16), append_56258, *[sol_call_result_56262], **kwargs_56263)
        
        
        # Assigning a Name to a Name (line 534):
        
        # Assigning a Name to a Name (line 534):
        # Getting the type of 't_eval_i_new' (line 534)
        t_eval_i_new_56265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 27), 't_eval_i_new')
        # Assigning a type to the variable 't_eval_i' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 't_eval_i', t_eval_i_new_56265)
        # SSA join for if statement (line 529)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_56190 and more_types_in_union_56191):
            # SSA join for if statement (line 514)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for while statement (line 475)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 536):
    
    # Assigning a Call to a Name (line 536):
    
    # Call to get(...): (line 536)
    # Processing the call arguments (line 536)
    # Getting the type of 'status' (line 536)
    status_56268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 27), 'status', False)
    # Getting the type of 'message' (line 536)
    message_56269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 35), 'message', False)
    # Processing the call keyword arguments (line 536)
    kwargs_56270 = {}
    # Getting the type of 'MESSAGES' (line 536)
    MESSAGES_56266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 14), 'MESSAGES', False)
    # Obtaining the member 'get' of a type (line 536)
    get_56267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 14), MESSAGES_56266, 'get')
    # Calling get(args, kwargs) (line 536)
    get_call_result_56271 = invoke(stypy.reporting.localization.Localization(__file__, 536, 14), get_56267, *[status_56268, message_56269], **kwargs_56270)
    
    # Assigning a type to the variable 'message' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'message', get_call_result_56271)
    
    # Type idiom detected: calculating its left and rigth part (line 538)
    # Getting the type of 't_events' (line 538)
    t_events_56272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 't_events')
    # Getting the type of 'None' (line 538)
    None_56273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 23), 'None')
    
    (may_be_56274, more_types_in_union_56275) = may_not_be_none(t_events_56272, None_56273)

    if may_be_56274:

        if more_types_in_union_56275:
            # Runtime conditional SSA (line 538)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a ListComp to a Name (line 539):
        
        # Assigning a ListComp to a Name (line 539):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 't_events' (line 539)
        t_events_56281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 45), 't_events')
        comprehension_56282 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 20), t_events_56281)
        # Assigning a type to the variable 'te' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'te', comprehension_56282)
        
        # Call to asarray(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'te' (line 539)
        te_56278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 31), 'te', False)
        # Processing the call keyword arguments (line 539)
        kwargs_56279 = {}
        # Getting the type of 'np' (line 539)
        np_56276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'np', False)
        # Obtaining the member 'asarray' of a type (line 539)
        asarray_56277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 20), np_56276, 'asarray')
        # Calling asarray(args, kwargs) (line 539)
        asarray_call_result_56280 = invoke(stypy.reporting.localization.Localization(__file__, 539, 20), asarray_56277, *[te_56278], **kwargs_56279)
        
        list_56283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 20), list_56283, asarray_call_result_56280)
        # Assigning a type to the variable 't_events' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 't_events', list_56283)

        if more_types_in_union_56275:
            # SSA join for if statement (line 538)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 541)
    # Getting the type of 't_eval' (line 541)
    t_eval_56284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 7), 't_eval')
    # Getting the type of 'None' (line 541)
    None_56285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 17), 'None')
    
    (may_be_56286, more_types_in_union_56287) = may_be_none(t_eval_56284, None_56285)

    if may_be_56286:

        if more_types_in_union_56287:
            # Runtime conditional SSA (line 541)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 542):
        
        # Assigning a Call to a Name (line 542):
        
        # Call to array(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'ts' (line 542)
        ts_56290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 22), 'ts', False)
        # Processing the call keyword arguments (line 542)
        kwargs_56291 = {}
        # Getting the type of 'np' (line 542)
        np_56288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 542)
        array_56289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 13), np_56288, 'array')
        # Calling array(args, kwargs) (line 542)
        array_call_result_56292 = invoke(stypy.reporting.localization.Localization(__file__, 542, 13), array_56289, *[ts_56290], **kwargs_56291)
        
        # Assigning a type to the variable 'ts' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'ts', array_call_result_56292)
        
        # Assigning a Attribute to a Name (line 543):
        
        # Assigning a Attribute to a Name (line 543):
        
        # Call to vstack(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'ys' (line 543)
        ys_56295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 23), 'ys', False)
        # Processing the call keyword arguments (line 543)
        kwargs_56296 = {}
        # Getting the type of 'np' (line 543)
        np_56293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 13), 'np', False)
        # Obtaining the member 'vstack' of a type (line 543)
        vstack_56294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 13), np_56293, 'vstack')
        # Calling vstack(args, kwargs) (line 543)
        vstack_call_result_56297 = invoke(stypy.reporting.localization.Localization(__file__, 543, 13), vstack_56294, *[ys_56295], **kwargs_56296)
        
        # Obtaining the member 'T' of a type (line 543)
        T_56298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 13), vstack_call_result_56297, 'T')
        # Assigning a type to the variable 'ys' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'ys', T_56298)

        if more_types_in_union_56287:
            # Runtime conditional SSA for else branch (line 541)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_56286) or more_types_in_union_56287):
        
        # Assigning a Call to a Name (line 545):
        
        # Assigning a Call to a Name (line 545):
        
        # Call to hstack(...): (line 545)
        # Processing the call arguments (line 545)
        # Getting the type of 'ts' (line 545)
        ts_56301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 23), 'ts', False)
        # Processing the call keyword arguments (line 545)
        kwargs_56302 = {}
        # Getting the type of 'np' (line 545)
        np_56299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 13), 'np', False)
        # Obtaining the member 'hstack' of a type (line 545)
        hstack_56300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 13), np_56299, 'hstack')
        # Calling hstack(args, kwargs) (line 545)
        hstack_call_result_56303 = invoke(stypy.reporting.localization.Localization(__file__, 545, 13), hstack_56300, *[ts_56301], **kwargs_56302)
        
        # Assigning a type to the variable 'ts' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'ts', hstack_call_result_56303)
        
        # Assigning a Call to a Name (line 546):
        
        # Assigning a Call to a Name (line 546):
        
        # Call to hstack(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'ys' (line 546)
        ys_56306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'ys', False)
        # Processing the call keyword arguments (line 546)
        kwargs_56307 = {}
        # Getting the type of 'np' (line 546)
        np_56304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 13), 'np', False)
        # Obtaining the member 'hstack' of a type (line 546)
        hstack_56305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 13), np_56304, 'hstack')
        # Calling hstack(args, kwargs) (line 546)
        hstack_call_result_56308 = invoke(stypy.reporting.localization.Localization(__file__, 546, 13), hstack_56305, *[ys_56306], **kwargs_56307)
        
        # Assigning a type to the variable 'ys' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'ys', hstack_call_result_56308)

        if (may_be_56286 and more_types_in_union_56287):
            # SSA join for if statement (line 541)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'dense_output' (line 548)
    dense_output_56309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 7), 'dense_output')
    # Testing the type of an if condition (line 548)
    if_condition_56310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 4), dense_output_56309)
    # Assigning a type to the variable 'if_condition_56310' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'if_condition_56310', if_condition_56310)
    # SSA begins for if statement (line 548)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 549):
    
    # Assigning a Call to a Name (line 549):
    
    # Call to OdeSolution(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'ts' (line 549)
    ts_56312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 26), 'ts', False)
    # Getting the type of 'interpolants' (line 549)
    interpolants_56313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 30), 'interpolants', False)
    # Processing the call keyword arguments (line 549)
    kwargs_56314 = {}
    # Getting the type of 'OdeSolution' (line 549)
    OdeSolution_56311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 14), 'OdeSolution', False)
    # Calling OdeSolution(args, kwargs) (line 549)
    OdeSolution_call_result_56315 = invoke(stypy.reporting.localization.Localization(__file__, 549, 14), OdeSolution_56311, *[ts_56312, interpolants_56313], **kwargs_56314)
    
    # Assigning a type to the variable 'sol' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'sol', OdeSolution_call_result_56315)
    # SSA branch for the else part of an if statement (line 548)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 551):
    
    # Assigning a Name to a Name (line 551):
    # Getting the type of 'None' (line 551)
    None_56316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 14), 'None')
    # Assigning a type to the variable 'sol' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'sol', None_56316)
    # SSA join for if statement (line 548)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to OdeResult(...): (line 553)
    # Processing the call keyword arguments (line 553)
    # Getting the type of 'ts' (line 553)
    ts_56318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 23), 'ts', False)
    keyword_56319 = ts_56318
    # Getting the type of 'ys' (line 553)
    ys_56320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 29), 'ys', False)
    keyword_56321 = ys_56320
    # Getting the type of 'sol' (line 553)
    sol_56322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 37), 'sol', False)
    keyword_56323 = sol_56322
    # Getting the type of 't_events' (line 553)
    t_events_56324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 51), 't_events', False)
    keyword_56325 = t_events_56324
    # Getting the type of 'solver' (line 553)
    solver_56326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 66), 'solver', False)
    # Obtaining the member 'nfev' of a type (line 553)
    nfev_56327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 66), solver_56326, 'nfev')
    keyword_56328 = nfev_56327
    # Getting the type of 'solver' (line 554)
    solver_56329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 26), 'solver', False)
    # Obtaining the member 'njev' of a type (line 554)
    njev_56330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 26), solver_56329, 'njev')
    keyword_56331 = njev_56330
    # Getting the type of 'solver' (line 554)
    solver_56332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 43), 'solver', False)
    # Obtaining the member 'nlu' of a type (line 554)
    nlu_56333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 43), solver_56332, 'nlu')
    keyword_56334 = nlu_56333
    # Getting the type of 'status' (line 554)
    status_56335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 62), 'status', False)
    keyword_56336 = status_56335
    # Getting the type of 'message' (line 555)
    message_56337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 29), 'message', False)
    keyword_56338 = message_56337
    
    # Getting the type of 'status' (line 555)
    status_56339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 46), 'status', False)
    int_56340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 56), 'int')
    # Applying the binary operator '>=' (line 555)
    result_ge_56341 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 46), '>=', status_56339, int_56340)
    
    keyword_56342 = result_ge_56341
    kwargs_56343 = {'status': keyword_56336, 'nlu': keyword_56334, 'success': keyword_56342, 't_events': keyword_56325, 'sol': keyword_56323, 'nfev': keyword_56328, 't': keyword_56319, 'y': keyword_56321, 'message': keyword_56338, 'njev': keyword_56331}
    # Getting the type of 'OdeResult' (line 553)
    OdeResult_56317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 11), 'OdeResult', False)
    # Calling OdeResult(args, kwargs) (line 553)
    OdeResult_call_result_56344 = invoke(stypy.reporting.localization.Localization(__file__, 553, 11), OdeResult_56317, *[], **kwargs_56343)
    
    # Assigning a type to the variable 'stypy_return_type' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'stypy_return_type', OdeResult_call_result_56344)
    
    # ################# End of 'solve_ivp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_ivp' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_56345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_56345)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_ivp'
    return stypy_return_type_56345

# Assigning a type to the variable 'solve_ivp' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'solve_ivp', solve_ivp)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
