
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A top-level linear programming interface. Currently this interface only
3: solves linear programming problems via the Simplex Method.
4: 
5: .. versionadded:: 0.15.0
6: 
7: Functions
8: ---------
9: .. autosummary::
10:    :toctree: generated/
11: 
12:     linprog
13:     linprog_verbose_callback
14:     linprog_terse_callback
15: 
16: '''
17: 
18: from __future__ import division, print_function, absolute_import
19: 
20: import numpy as np
21: from .optimize import OptimizeResult, _check_unknown_options
22: from ._linprog_ip import _linprog_ip
23: 
24: __all__ = ['linprog', 'linprog_verbose_callback', 'linprog_terse_callback']
25: 
26: __docformat__ = "restructuredtext en"
27: 
28: 
29: def linprog_verbose_callback(xk, **kwargs):
30:     '''
31:     A sample callback function demonstrating the linprog callback interface.
32:     This callback produces detailed output to sys.stdout before each iteration
33:     and after the final iteration of the simplex algorithm.
34: 
35:     Parameters
36:     ----------
37:     xk : array_like
38:         The current solution vector.
39:     **kwargs : dict
40:         A dictionary containing the following parameters:
41: 
42:         tableau : array_like
43:             The current tableau of the simplex algorithm.
44:             Its structure is defined in _solve_simplex.
45:         phase : int
46:             The current Phase of the simplex algorithm (1 or 2)
47:         nit : int
48:             The current iteration number.
49:         pivot : tuple(int, int)
50:             The index of the tableau selected as the next pivot,
51:             or nan if no pivot exists
52:         basis : array(int)
53:             A list of the current basic variables.
54:             Each element contains the name of a basic variable and its value.
55:         complete : bool
56:             True if the simplex algorithm has completed
57:             (and this is the final call to callback), otherwise False.
58:     '''
59:     tableau = kwargs["tableau"]
60:     nit = kwargs["nit"]
61:     pivrow, pivcol = kwargs["pivot"]
62:     phase = kwargs["phase"]
63:     basis = kwargs["basis"]
64:     complete = kwargs["complete"]
65: 
66:     saved_printoptions = np.get_printoptions()
67:     np.set_printoptions(linewidth=500,
68:                         formatter={'float': lambda x: "{0: 12.4f}".format(x)})
69:     if complete:
70:         print("--------- Iteration Complete - Phase {0:d} -------\n".format(phase))
71:         print("Tableau:")
72:     elif nit == 0:
73:         print("--------- Initial Tableau - Phase {0:d} ----------\n".format(phase))
74: 
75:     else:
76:         print("--------- Iteration {0:d}  - Phase {1:d} --------\n".format(nit, phase))
77:         print("Tableau:")
78: 
79:     if nit >= 0:
80:         print("" + str(tableau) + "\n")
81:         if not complete:
82:             print("Pivot Element: T[{0:.0f}, {1:.0f}]\n".format(pivrow, pivcol))
83:         print("Basic Variables:", basis)
84:         print()
85:         print("Current Solution:")
86:         print("x = ", xk)
87:         print()
88:         print("Current Objective Value:")
89:         print("f = ", -tableau[-1, -1])
90:         print()
91:     np.set_printoptions(**saved_printoptions)
92: 
93: 
94: def linprog_terse_callback(xk, **kwargs):
95:     '''
96:     A sample callback function demonstrating the linprog callback interface.
97:     This callback produces brief output to sys.stdout before each iteration
98:     and after the final iteration of the simplex algorithm.
99: 
100:     Parameters
101:     ----------
102:     xk : array_like
103:         The current solution vector.
104:     **kwargs : dict
105:         A dictionary containing the following parameters:
106: 
107:         tableau : array_like
108:             The current tableau of the simplex algorithm.
109:             Its structure is defined in _solve_simplex.
110:         vars : tuple(str, ...)
111:             Column headers for each column in tableau.
112:             "x[i]" for actual variables, "s[i]" for slack surplus variables,
113:             "a[i]" for artificial variables, and "RHS" for the constraint
114:             RHS vector.
115:         phase : int
116:             The current Phase of the simplex algorithm (1 or 2)
117:         nit : int
118:             The current iteration number.
119:         pivot : tuple(int, int)
120:             The index of the tableau selected as the next pivot,
121:             or nan if no pivot exists
122:         basics : list[tuple(int, float)]
123:             A list of the current basic variables.
124:             Each element contains the index of a basic variable and
125:             its value.
126:         complete : bool
127:             True if the simplex algorithm has completed
128:             (and this is the final call to callback), otherwise False.
129:     '''
130:     nit = kwargs["nit"]
131: 
132:     if nit == 0:
133:         print("Iter:   X:")
134:     print("{0: <5d}   ".format(nit), end="")
135:     print(xk)
136: 
137: 
138: def _pivot_col(T, tol=1.0E-12, bland=False):
139:     '''
140:     Given a linear programming simplex tableau, determine the column
141:     of the variable to enter the basis.
142: 
143:     Parameters
144:     ----------
145:     T : 2D ndarray
146:         The simplex tableau.
147:     tol : float
148:         Elements in the objective row larger than -tol will not be considered
149:         for pivoting.  Nominally this value is zero, but numerical issues
150:         cause a tolerance about zero to be necessary.
151:     bland : bool
152:         If True, use Bland's rule for selection of the column (select the
153:         first column with a negative coefficient in the objective row,
154:         regardless of magnitude).
155: 
156:     Returns
157:     -------
158:     status: bool
159:         True if a suitable pivot column was found, otherwise False.
160:         A return of False indicates that the linear programming simplex
161:         algorithm is complete.
162:     col: int
163:         The index of the column of the pivot element.
164:         If status is False, col will be returned as nan.
165:     '''
166:     ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
167:     if ma.count() == 0:
168:         return False, np.nan
169:     if bland:
170:         return True, np.where(ma.mask == False)[0][0]
171:     return True, np.ma.where(ma == ma.min())[0][0]
172: 
173: 
174: def _pivot_row(T, pivcol, phase, tol=1.0E-12):
175:     '''
176:     Given a linear programming simplex tableau, determine the row for the
177:     pivot operation.
178: 
179:     Parameters
180:     ----------
181:     T : 2D ndarray
182:         The simplex tableau.
183:     pivcol : int
184:         The index of the pivot column.
185:     phase : int
186:         The phase of the simplex algorithm (1 or 2).
187:     tol : float
188:         Elements in the pivot column smaller than tol will not be considered
189:         for pivoting.  Nominally this value is zero, but numerical issues
190:         cause a tolerance about zero to be necessary.
191: 
192:     Returns
193:     -------
194:     status: bool
195:         True if a suitable pivot row was found, otherwise False.  A return
196:         of False indicates that the linear programming problem is unbounded.
197:     row: int
198:         The index of the row of the pivot element.  If status is False, row
199:         will be returned as nan.
200:     '''
201:     if phase == 1:
202:         k = 2
203:     else:
204:         k = 1
205:     ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
206:     if ma.count() == 0:
207:         return False, np.nan
208:     mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
209:     q = mb / ma
210:     return True, np.ma.where(q == q.min())[0][0]
211: 
212: 
213: def _solve_simplex(T, n, basis, maxiter=1000, phase=2, callback=None,
214:                    tol=1.0E-12, nit0=0, bland=False):
215:     '''
216:     Solve a linear programming problem in "standard maximization form" using
217:     the Simplex Method.
218: 
219:     Minimize :math:`f = c^T x`
220: 
221:     subject to
222: 
223:     .. math::
224: 
225:         Ax = b
226:         x_i >= 0
227:         b_j >= 0
228: 
229:     Parameters
230:     ----------
231:     T : array_like
232:         A 2-D array representing the simplex T corresponding to the
233:         maximization problem.  It should have the form:
234: 
235:         [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
236:          [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
237:          .
238:          .
239:          .
240:          [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
241:          [c[0],   c[1], ...,   c[n_total],    0]]
242: 
243:         for a Phase 2 problem, or the form:
244: 
245:         [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
246:          [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
247:          .
248:          .
249:          .
250:          [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
251:          [c[0],   c[1], ...,   c[n_total],   0],
252:          [c'[0],  c'[1], ...,  c'[n_total],  0]]
253: 
254:          for a Phase 1 problem (a Problem in which a basic feasible solution is
255:          sought prior to maximizing the actual objective.  T is modified in
256:          place by _solve_simplex.
257:     n : int
258:         The number of true variables in the problem.
259:     basis : array
260:         An array of the indices of the basic variables, such that basis[i]
261:         contains the column corresponding to the basic variable for row i.
262:         Basis is modified in place by _solve_simplex
263:     maxiter : int
264:         The maximum number of iterations to perform before aborting the
265:         optimization.
266:     phase : int
267:         The phase of the optimization being executed.  In phase 1 a basic
268:         feasible solution is sought and the T has an additional row
269:         representing an alternate objective function.
270:     callback : callable, optional
271:         If a callback function is provided, it will be called within each
272:         iteration of the simplex algorithm. The callback must have the
273:         signature `callback(xk, **kwargs)` where xk is the current solution
274:         vector and kwargs is a dictionary containing the following::
275:         "T" : The current Simplex algorithm T
276:         "nit" : The current iteration.
277:         "pivot" : The pivot (row, column) used for the next iteration.
278:         "phase" : Whether the algorithm is in Phase 1 or Phase 2.
279:         "basis" : The indices of the columns of the basic variables.
280:     tol : float
281:         The tolerance which determines when a solution is "close enough" to
282:         zero in Phase 1 to be considered a basic feasible solution or close
283:         enough to positive to serve as an optimal solution.
284:     nit0 : int
285:         The initial iteration number used to keep an accurate iteration total
286:         in a two-phase problem.
287:     bland : bool
288:         If True, choose pivots using Bland's rule [3].  In problems which
289:         fail to converge due to cycling, using Bland's rule can provide
290:         convergence at the expense of a less optimal path about the simplex.
291: 
292:     Returns
293:     -------
294:     res : OptimizeResult
295:         The optimization result represented as a ``OptimizeResult`` object.
296:         Important attributes are: ``x`` the solution array, ``success`` a
297:         Boolean flag indicating if the optimizer exited successfully and
298:         ``message`` which describes the cause of the termination. Possible
299:         values for the ``status`` attribute are:
300:          0 : Optimization terminated successfully
301:          1 : Iteration limit reached
302:          2 : Problem appears to be infeasible
303:          3 : Problem appears to be unbounded
304: 
305:         See `OptimizeResult` for a description of other attributes.
306:     '''
307:     nit = nit0
308:     complete = False
309: 
310:     if phase == 1:
311:         m = T.shape[0]-2
312:     elif phase == 2:
313:         m = T.shape[0]-1
314:     else:
315:         raise ValueError("Argument 'phase' to _solve_simplex must be 1 or 2")
316: 
317:     if phase == 2:
318:         # Check if any artificial variables are still in the basis.
319:         # If yes, check if any coefficients from this row and a column
320:         # corresponding to one of the non-artificial variable is non-zero.
321:         # If found, pivot at this term. If not, start phase 2.
322:         # Do this for all artificial variables in the basis.
323:         # Ref: "An Introduction to Linear Programming and Game Theory"
324:         # by Paul R. Thie, Gerard E. Keough, 3rd Ed,
325:         # Chapter 3.7 Redundant Systems (pag 102)
326:         for pivrow in [row for row in range(basis.size)
327:                        if basis[row] > T.shape[1] - 2]:
328:             non_zero_row = [col for col in range(T.shape[1] - 1)
329:                             if T[pivrow, col] != 0]
330:             if len(non_zero_row) > 0:
331:                 pivcol = non_zero_row[0]
332:                 # variable represented by pivcol enters
333:                 # variable in basis[pivrow] leaves
334:                 basis[pivrow] = pivcol
335:                 pivval = T[pivrow][pivcol]
336:                 T[pivrow, :] = T[pivrow, :] / pivval
337:                 for irow in range(T.shape[0]):
338:                     if irow != pivrow:
339:                         T[irow, :] = T[irow, :] - T[pivrow, :]*T[irow, pivcol]
340:                 nit += 1
341: 
342:     if len(basis[:m]) == 0:
343:         solution = np.zeros(T.shape[1] - 1, dtype=np.float64)
344:     else:
345:         solution = np.zeros(max(T.shape[1] - 1, max(basis[:m]) + 1),
346:                             dtype=np.float64)
347: 
348:     while not complete:
349:         # Find the pivot column
350:         pivcol_found, pivcol = _pivot_col(T, tol, bland)
351:         if not pivcol_found:
352:             pivcol = np.nan
353:             pivrow = np.nan
354:             status = 0
355:             complete = True
356:         else:
357:             # Find the pivot row
358:             pivrow_found, pivrow = _pivot_row(T, pivcol, phase, tol)
359:             if not pivrow_found:
360:                 status = 3
361:                 complete = True
362: 
363:         if callback is not None:
364:             solution[:] = 0
365:             solution[basis[:m]] = T[:m, -1]
366:             callback(solution[:n], **{"tableau": T,
367:                                       "phase": phase,
368:                                       "nit": nit,
369:                                       "pivot": (pivrow, pivcol),
370:                                       "basis": basis,
371:                                       "complete": complete and phase == 2})
372: 
373:         if not complete:
374:             if nit >= maxiter:
375:                 # Iteration limit exceeded
376:                 status = 1
377:                 complete = True
378:             else:
379:                 # variable represented by pivcol enters
380:                 # variable in basis[pivrow] leaves
381:                 basis[pivrow] = pivcol
382:                 pivval = T[pivrow][pivcol]
383:                 T[pivrow, :] = T[pivrow, :] / pivval
384:                 for irow in range(T.shape[0]):
385:                     if irow != pivrow:
386:                         T[irow, :] = T[irow, :] - T[pivrow, :]*T[irow, pivcol]
387:                 nit += 1
388: 
389:     return nit, status
390: 
391: 
392: def _linprog_simplex(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
393:                      bounds=None, maxiter=1000, disp=False, callback=None,
394:                      tol=1.0E-12, bland=False, **unknown_options):
395:     '''
396:     Solve the following linear programming problem via a two-phase
397:     simplex algorithm.::
398: 
399:         minimize:     c^T * x
400: 
401:         subject to:   A_ub * x <= b_ub
402:                       A_eq * x == b_eq
403: 
404:     Parameters
405:     ----------
406:     c : array_like
407:         Coefficients of the linear objective function to be minimized.
408:     A_ub : array_like
409:         2-D array which, when matrix-multiplied by ``x``, gives the values of
410:         the upper-bound inequality constraints at ``x``.
411:     b_ub : array_like
412:         1-D array of values representing the upper-bound of each inequality
413:         constraint (row) in ``A_ub``.
414:     A_eq : array_like
415:         2-D array which, when matrix-multiplied by ``x``, gives the values of
416:         the equality constraints at ``x``.
417:     b_eq : array_like
418:         1-D array of values representing the RHS of each equality constraint
419:         (row) in ``A_eq``.
420:     bounds : array_like
421:         The bounds for each independent variable in the solution, which can
422:         take one of three forms::
423: 
424:         None : The default bounds, all variables are non-negative.
425:         (lb, ub) : If a 2-element sequence is provided, the same
426:                   lower bound (lb) and upper bound (ub) will be applied
427:                   to all variables.
428:         [(lb_0, ub_0), (lb_1, ub_1), ...] : If an n x 2 sequence is provided,
429:                   each variable x_i will be bounded by lb[i] and ub[i].
430:         Infinite bounds are specified using -np.inf (negative)
431:         or np.inf (positive).
432: 
433:     callback : callable
434:         If a callback function is provide, it will be called within each
435:         iteration of the simplex algorithm. The callback must have the
436:         signature ``callback(xk, **kwargs)`` where ``xk`` is the current s
437:         olution vector and kwargs is a dictionary containing the following::
438: 
439:         "tableau" : The current Simplex algorithm tableau
440:         "nit" : The current iteration.
441:         "pivot" : The pivot (row, column) used for the next iteration.
442:         "phase" : Whether the algorithm is in Phase 1 or Phase 2.
443:         "bv" : A structured array containing a string representation of each
444:                basic variable and its current value.
445: 
446:     Options
447:     -------
448:     maxiter : int
449:        The maximum number of iterations to perform.
450:     disp : bool
451:         If True, print exit status message to sys.stdout
452:     tol : float
453:         The tolerance which determines when a solution is "close enough" to
454:         zero in Phase 1 to be considered a basic feasible solution or close
455:         enough to positive to serve as an optimal solution.
456:     bland : bool
457:         If True, use Bland's anti-cycling rule [3] to choose pivots to
458:         prevent cycling.  If False, choose pivots which should lead to a
459:         converged solution more quickly.  The latter method is subject to
460:         cycling (non-convergence) in rare instances.
461: 
462:     Returns
463:     -------
464:     A `scipy.optimize.OptimizeResult` consisting of the following fields:
465: 
466:         x : ndarray
467:             The independent variable vector which optimizes the linear
468:             programming problem.
469:         fun : float
470:             Value of the objective function.
471:         slack : ndarray
472:             The values of the slack variables.  Each slack variable corresponds
473:             to an inequality constraint.  If the slack is zero, then the
474:             corresponding constraint is active.
475:         success : bool
476:             Returns True if the algorithm succeeded in finding an optimal
477:             solution.
478:         status : int
479:             An integer representing the exit status of the optimization::
480: 
481:              0 : Optimization terminated successfully
482:              1 : Iteration limit reached
483:              2 : Problem appears to be infeasible
484:              3 : Problem appears to be unbounded
485: 
486:         nit : int
487:             The number of iterations performed.
488:         message : str
489:             A string descriptor of the exit status of the optimization.
490: 
491:     Examples
492:     --------
493:     Consider the following problem:
494: 
495:     Minimize: f = -1*x[0] + 4*x[1]
496: 
497:     Subject to: -3*x[0] + 1*x[1] <= 6
498:                  1*x[0] + 2*x[1] <= 4
499:                             x[1] >= -3
500: 
501:     where:  -inf <= x[0] <= inf
502: 
503:     This problem deviates from the standard linear programming problem.  In
504:     standard form, linear programming problems assume the variables x are
505:     non-negative.  Since the variables don't have standard bounds where
506:     0 <= x <= inf, the bounds of the variables must be explicitly set.
507: 
508:     There are two upper-bound constraints, which can be expressed as
509: 
510:     dot(A_ub, x) <= b_ub
511: 
512:     The input for this problem is as follows:
513: 
514:     >>> from scipy.optimize import linprog
515:     >>> c = [-1, 4]
516:     >>> A = [[-3, 1], [1, 2]]
517:     >>> b = [6, 4]
518:     >>> x0_bnds = (None, None)
519:     >>> x1_bnds = (-3, None)
520:     >>> res = linprog(c, A, b, bounds=(x0_bnds, x1_bnds))
521:     >>> print(res)
522:          fun: -22.0
523:      message: 'Optimization terminated successfully.'
524:          nit: 1
525:        slack: array([ 39.,   0.])
526:       status: 0
527:      success: True
528:            x: array([ 10.,  -3.])
529: 
530:     References
531:     ----------
532:     .. [1] Dantzig, George B., Linear programming and extensions. Rand
533:            Corporation Research Study Princeton Univ. Press, Princeton, NJ,
534:            1963
535:     .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
536:            Mathematical Programming", McGraw-Hill, Chapter 4.
537:     .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
538:            Mathematics of Operations Research (2), 1977: pp. 103-107.
539:     '''
540:     _check_unknown_options(unknown_options)
541: 
542:     status = 0
543:     messages = {0: "Optimization terminated successfully.",
544:                 1: "Iteration limit reached.",
545:                 2: "Optimization failed. Unable to find a feasible"
546:                    " starting point.",
547:                 3: "Optimization failed. The problem appears to be unbounded.",
548:                 4: "Optimization failed. Singular matrix encountered."}
549:     have_floor_variable = False
550: 
551:     cc = np.asarray(c)
552: 
553:     # The initial value of the objective function element in the tableau
554:     f0 = 0
555: 
556:     # The number of variables as given by c
557:     n = len(c)
558: 
559:     # Convert the input arguments to arrays (sized to zero if not provided)
560:     Aeq = np.asarray(A_eq) if A_eq is not None else np.empty([0, len(cc)])
561:     Aub = np.asarray(A_ub) if A_ub is not None else np.empty([0, len(cc)])
562:     beq = np.ravel(np.asarray(b_eq)) if b_eq is not None else np.empty([0])
563:     bub = np.ravel(np.asarray(b_ub)) if b_ub is not None else np.empty([0])
564: 
565:     # Analyze the bounds and determine what modifications to be made to
566:     # the constraints in order to accommodate them.
567:     L = np.zeros(n, dtype=np.float64)
568:     U = np.ones(n, dtype=np.float64)*np.inf
569:     if bounds is None or len(bounds) == 0:
570:         pass
571:     elif len(bounds) == 2 and not hasattr(bounds[0], '__len__'):
572:         # All bounds are the same
573:         a = bounds[0] if bounds[0] is not None else -np.inf
574:         b = bounds[1] if bounds[1] is not None else np.inf
575:         L = np.asarray(n*[a], dtype=np.float64)
576:         U = np.asarray(n*[b], dtype=np.float64)
577:     else:
578:         if len(bounds) != n:
579:             status = -1
580:             message = ("Invalid input for linprog with method = 'simplex'.  "
581:                        "Length of bounds is inconsistent with the length of c")
582:         else:
583:             try:
584:                 for i in range(n):
585:                     if len(bounds[i]) != 2:
586:                         raise IndexError()
587:                     L[i] = bounds[i][0] if bounds[i][0] is not None else -np.inf
588:                     U[i] = bounds[i][1] if bounds[i][1] is not None else np.inf
589:             except IndexError:
590:                 status = -1
591:                 message = ("Invalid input for linprog with "
592:                            "method = 'simplex'.  bounds must be a n x 2 "
593:                            "sequence/array where n = len(c).")
594: 
595:     if np.any(L == -np.inf):
596:         # If any lower-bound constraint is a free variable
597:         # add the first column variable as the "floor" variable which
598:         # accommodates the most negative variable in the problem.
599:         n = n + 1
600:         L = np.concatenate([np.array([0]), L])
601:         U = np.concatenate([np.array([np.inf]), U])
602:         cc = np.concatenate([np.array([0]), cc])
603:         Aeq = np.hstack([np.zeros([Aeq.shape[0], 1]), Aeq])
604:         Aub = np.hstack([np.zeros([Aub.shape[0], 1]), Aub])
605:         have_floor_variable = True
606: 
607:     # Now before we deal with any variables with lower bounds < 0,
608:     # deal with finite bounds which can be simply added as new constraints.
609:     # Also validate bounds inputs here.
610:     for i in range(n):
611:         if(L[i] > U[i]):
612:             status = -1
613:             message = ("Invalid input for linprog with method = 'simplex'.  "
614:                        "Lower bound %d is greater than upper bound%d" % (i, i))
615: 
616:         if np.isinf(L[i]) and L[i] > 0:
617:             status = -1
618:             message = ("Invalid input for linprog with method = 'simplex'.  "
619:                        "Lower bound may not be +infinity")
620: 
621:         if np.isinf(U[i]) and U[i] < 0:
622:             status = -1
623:             message = ("Invalid input for linprog with method = 'simplex'.  "
624:                        "Upper bound may not be -infinity")
625: 
626:         if np.isfinite(L[i]) and L[i] > 0:
627:             # Add a new lower-bound (negative upper-bound) constraint
628:             Aub = np.vstack([Aub, np.zeros(n)])
629:             Aub[-1, i] = -1
630:             bub = np.concatenate([bub, np.array([-L[i]])])
631:             L[i] = 0
632: 
633:         if np.isfinite(U[i]):
634:             # Add a new upper-bound constraint
635:             Aub = np.vstack([Aub, np.zeros(n)])
636:             Aub[-1, i] = 1
637:             bub = np.concatenate([bub, np.array([U[i]])])
638:             U[i] = np.inf
639: 
640:     # Now find negative lower bounds (finite or infinite) which require a
641:     # change of variables or free variables and handle them appropriately
642:     for i in range(0, n):
643:         if L[i] < 0:
644:             if np.isfinite(L[i]) and L[i] < 0:
645:                 # Add a change of variables for x[i]
646:                 # For each row in the constraint matrices, we take the
647:                 # coefficient from column i in A,
648:                 # and subtract the product of that and L[i] to the RHS b
649:                 beq = beq - Aeq[:, i] * L[i]
650:                 bub = bub - Aub[:, i] * L[i]
651:                 # We now have a nonzero initial value for the objective
652:                 # function as well.
653:                 f0 = f0 - cc[i] * L[i]
654:             else:
655:                 # This is an unrestricted variable, let x[i] = u[i] - v[0]
656:                 # where v is the first column in all matrices.
657:                 Aeq[:, 0] = Aeq[:, 0] - Aeq[:, i]
658:                 Aub[:, 0] = Aub[:, 0] - Aub[:, i]
659:                 cc[0] = cc[0] - cc[i]
660: 
661:         if np.isinf(U[i]):
662:             if U[i] < 0:
663:                 status = -1
664:                 message = ("Invalid input for linprog with "
665:                            "method = 'simplex'.  Upper bound may not be -inf.")
666: 
667:     # The number of upper bound constraints (rows in A_ub and elements in b_ub)
668:     mub = len(bub)
669: 
670:     # The number of equality constraints (rows in A_eq and elements in b_eq)
671:     meq = len(beq)
672: 
673:     # The total number of constraints
674:     m = mub+meq
675: 
676:     # The number of slack variables (one for each upper-bound constraints)
677:     n_slack = mub
678: 
679:     # The number of artificial variables (one for each lower-bound and equality
680:     # constraint)
681:     n_artificial = meq + np.count_nonzero(bub < 0)
682: 
683:     try:
684:         Aub_rows, Aub_cols = Aub.shape
685:     except ValueError:
686:         raise ValueError("Invalid input.  A_ub must be two-dimensional")
687: 
688:     try:
689:         Aeq_rows, Aeq_cols = Aeq.shape
690:     except ValueError:
691:         raise ValueError("Invalid input.  A_eq must be two-dimensional")
692: 
693:     if Aeq_rows != meq:
694:         status = -1
695:         message = ("Invalid input for linprog with method = 'simplex'.  "
696:                    "The number of rows in A_eq must be equal "
697:                    "to the number of values in b_eq")
698: 
699:     if Aub_rows != mub:
700:         status = -1
701:         message = ("Invalid input for linprog with method = 'simplex'.  "
702:                    "The number of rows in A_ub must be equal "
703:                    "to the number of values in b_ub")
704: 
705:     if Aeq_cols > 0 and Aeq_cols != n:
706:         status = -1
707:         message = ("Invalid input for linprog with method = 'simplex'.  "
708:                    "Number of columns in A_eq must be equal "
709:                    "to the size of c")
710: 
711:     if Aub_cols > 0 and Aub_cols != n:
712:         status = -1
713:         message = ("Invalid input for linprog with method = 'simplex'.  "
714:                    "Number of columns in A_ub must be equal to the size of c")
715: 
716:     if status != 0:
717:         # Invalid inputs provided
718:         raise ValueError(message)
719: 
720:     # Create the tableau
721:     T = np.zeros([m+2, n+n_slack+n_artificial+1])
722: 
723:     # Insert objective into tableau
724:     T[-2, :n] = cc
725:     T[-2, -1] = f0
726: 
727:     b = T[:-2, -1]
728: 
729:     if meq > 0:
730:         # Add Aeq to the tableau
731:         T[:meq, :n] = Aeq
732:         # Add beq to the tableau
733:         b[:meq] = beq
734:     if mub > 0:
735:         # Add Aub to the tableau
736:         T[meq:meq+mub, :n] = Aub
737:         # At bub to the tableau
738:         b[meq:meq+mub] = bub
739:         # Add the slack variables to the tableau
740:         np.fill_diagonal(T[meq:m, n:n+n_slack], 1)
741: 
742:     # Further set up the tableau.
743:     # If a row corresponds to an equality constraint or a negative b (a lower
744:     # bound constraint), then an artificial variable is added for that row.
745:     # Also, if b is negative, first flip the signs in that constraint.
746:     slcount = 0
747:     avcount = 0
748:     basis = np.zeros(m, dtype=int)
749:     r_artificial = np.zeros(n_artificial, dtype=int)
750:     for i in range(m):
751:         if i < meq or b[i] < 0:
752:             # basic variable i is in column n+n_slack+avcount
753:             basis[i] = n+n_slack+avcount
754:             r_artificial[avcount] = i
755:             avcount += 1
756:             if b[i] < 0:
757:                 b[i] *= -1
758:                 T[i, :-1] *= -1
759:             T[i, basis[i]] = 1
760:             T[-1, basis[i]] = 1
761:         else:
762:             # basic variable i is in column n+slcount
763:             basis[i] = n+slcount
764:             slcount += 1
765: 
766:     # Make the artificial variables basic feasible variables by subtracting
767:     # each row with an artificial variable from the Phase 1 objective
768:     for r in r_artificial:
769:         T[-1, :] = T[-1, :] - T[r, :]
770: 
771:     nit1, status = _solve_simplex(T, n, basis, phase=1, callback=callback,
772:                                   maxiter=maxiter, tol=tol, bland=bland)
773: 
774:     # if pseudo objective is zero, remove the last row from the tableau and
775:     # proceed to phase 2
776:     if abs(T[-1, -1]) < tol:
777:         # Remove the pseudo-objective row from the tableau
778:         T = T[:-1, :]
779:         # Remove the artificial variable columns from the tableau
780:         T = np.delete(T, np.s_[n+n_slack:n+n_slack+n_artificial], 1)
781:     else:
782:         # Failure to find a feasible starting point
783:         status = 2
784: 
785:     if status != 0:
786:         message = messages[status]
787:         if disp:
788:             print(message)
789:         return OptimizeResult(x=np.nan, fun=-T[-1, -1], nit=nit1,
790:                               status=status, message=message, success=False)
791: 
792:     # Phase 2
793:     nit2, status = _solve_simplex(T, n, basis, maxiter=maxiter-nit1, phase=2,
794:                                   callback=callback, tol=tol, nit0=nit1,
795:                                   bland=bland)
796: 
797:     solution = np.zeros(n+n_slack+n_artificial)
798:     solution[basis[:m]] = T[:m, -1]
799:     x = solution[:n]
800:     slack = solution[n:n+n_slack]
801: 
802:     # For those variables with finite negative lower bounds,
803:     # reverse the change of variables
804:     masked_L = np.ma.array(L, mask=np.isinf(L), fill_value=0.0).filled()
805:     x = x + masked_L
806: 
807:     # For those variables with infinite negative lower bounds,
808:     # take x[i] as the difference between x[i] and the floor variable.
809:     if have_floor_variable:
810:         for i in range(1, n):
811:             if np.isinf(L[i]):
812:                 x[i] -= x[0]
813:         x = x[1:]
814: 
815:     # Optimization complete at this point
816:     obj = -T[-1, -1]
817: 
818:     if status in (0, 1):
819:         if disp:
820:             print(messages[status])
821:             print("         Current function value: {0: <12.6f}".format(obj))
822:             print("         Iterations: {0:d}".format(nit2))
823:     else:
824:         if disp:
825:             print(messages[status])
826:             print("         Iterations: {0:d}".format(nit2))
827: 
828:     return OptimizeResult(x=x, fun=obj, nit=int(nit2), status=status,
829:                           slack=slack, message=messages[status],
830:                           success=(status == 0))
831: 
832: 
833: def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
834:             bounds=None, method='simplex', callback=None,
835:             options=None):
836:     '''
837:     Minimize a linear objective function subject to linear
838:     equality and inequality constraints.
839: 
840:     Linear Programming is intended to solve the following problem form::
841: 
842:         Minimize:     c^T * x
843: 
844:         Subject to:   A_ub * x <= b_ub
845:                       A_eq * x == b_eq
846: 
847:     Parameters
848:     ----------
849:     c : array_like
850:         Coefficients of the linear objective function to be minimized.
851:     A_ub : array_like, optional
852:         2-D array which, when matrix-multiplied by ``x``, gives the values of
853:         the upper-bound inequality constraints at ``x``.
854:     b_ub : array_like, optional
855:         1-D array of values representing the upper-bound of each inequality
856:         constraint (row) in ``A_ub``.
857:     A_eq : array_like, optional
858:         2-D array which, when matrix-multiplied by ``x``, gives the values of
859:         the equality constraints at ``x``.
860:     b_eq : array_like, optional
861:         1-D array of values representing the RHS of each equality constraint
862:         (row) in ``A_eq``.
863:     bounds : sequence, optional
864:         ``(min, max)`` pairs for each element in ``x``, defining
865:         the bounds on that parameter. Use None for one of ``min`` or
866:         ``max`` when there is no bound in that direction. By default
867:         bounds are ``(0, None)`` (non-negative)
868:         If a sequence containing a single tuple is provided, then ``min`` and
869:         ``max`` will be applied to all variables in the problem.
870:     method : str, optional
871:         Type of solver.  :ref:`'simplex' <optimize.linprog-simplex>`
872:         and :ref:`'interior-point' <optimize.linprog-interior-point>`
873:         are supported.
874:     callback : callable, optional (simplex only)
875:         If a callback function is provide, it will be called within each
876:         iteration of the simplex algorithm. The callback must have the
877:         signature ``callback(xk, **kwargs)`` where ``xk`` is the current
878:         solution vector and ``kwargs`` is a dictionary containing the
879:         following::
880: 
881:             "tableau" : The current Simplex algorithm tableau
882:             "nit" : The current iteration.
883:             "pivot" : The pivot (row, column) used for the next iteration.
884:             "phase" : Whether the algorithm is in Phase 1 or Phase 2.
885:             "basis" : The indices of the columns of the basic variables.
886: 
887:     options : dict, optional
888:         A dictionary of solver options. All methods accept the following
889:         generic options:
890: 
891:             maxiter : int
892:                 Maximum number of iterations to perform.
893:             disp : bool
894:                 Set to True to print convergence messages.
895: 
896:         For method-specific options, see :func:`show_options('linprog')`.
897: 
898:     Returns
899:     -------
900:     A `scipy.optimize.OptimizeResult` consisting of the following fields:
901: 
902:         x : ndarray
903:             The independent variable vector which optimizes the linear
904:             programming problem.
905:         fun : float
906:             Value of the objective function.
907:         slack : ndarray
908:             The values of the slack variables.  Each slack variable corresponds
909:             to an inequality constraint.  If the slack is zero, then the
910:             corresponding constraint is active.
911:         success : bool
912:             Returns True if the algorithm succeeded in finding an optimal
913:             solution.
914:         status : int
915:             An integer representing the exit status of the optimization::
916: 
917:                  0 : Optimization terminated successfully
918:                  1 : Iteration limit reached
919:                  2 : Problem appears to be infeasible
920:                  3 : Problem appears to be unbounded
921: 
922:         nit : int
923:             The number of iterations performed.
924:         message : str
925:             A string descriptor of the exit status of the optimization.
926: 
927:     See Also
928:     --------
929:     show_options : Additional options accepted by the solvers
930: 
931:     Notes
932:     -----
933:     This section describes the available solvers that can be selected by the
934:     'method' parameter. The default method
935:     is :ref:`Simplex <optimize.linprog-simplex>`.
936:     :ref:`Interior point <optimize.linprog-interior-point>` is also available.
937: 
938:     Method *simplex* uses the simplex algorithm (as it relates to linear
939:     programming, NOT the Nelder-Mead simplex) [1]_, [2]_. This algorithm
940:     should be reasonably reliable and fast for small problems.
941: 
942:     .. versionadded:: 0.15.0
943: 
944:     Method *interior-point* uses the primal-dual path following algorithm
945:     as outlined in [4]_. This algorithm is intended to provide a faster
946:     and more reliable alternative to *simplex*, especially for large,
947:     sparse problems. Note, however, that the solution returned may be slightly
948:     less accurate than that of the simplex method and may not correspond with a
949:     vertex of the polytope defined by the constraints.
950: 
951:     References
952:     ----------
953:     .. [1] Dantzig, George B., Linear programming and extensions. Rand
954:            Corporation Research Study Princeton Univ. Press, Princeton, NJ,
955:            1963
956:     .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
957:            Mathematical Programming", McGraw-Hill, Chapter 4.
958:     .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
959:            Mathematics of Operations Research (2), 1977: pp. 103-107.
960:     .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
961:            optimizer for linear programming: an implementation of the
962:            homogeneous algorithm." High performance optimization. Springer US,
963:            2000. 197-232.
964:     .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
965:            large-scale linear programming." Optimization Methods and Software
966:            6.3 (1995): 219-227.
967:     .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
968:            Programming based on Newton's Method." Unpublished Course Notes,
969:            March 2004. Available 2/25/2017 at
970:            https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
971:     .. [7] Fourer, Robert. "Solving Linear Programs by Interior-Point Methods."
972:            Unpublished Course Notes, August 26, 2005. Available 2/25/2017 at
973:            http://www.4er.org/CourseNotes/Book%20B/B-III.pdf
974:     .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
975:            programming." Mathematical Programming 71.2 (1995): 221-245.
976:     .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
977:            programming." Athena Scientific 1 (1997): 997.
978:     .. [10] Andersen, Erling D., et al. Implementation of interior point
979:             methods for large scale linear programming. HEC/Universite de
980:             Geneve, 1996.
981: 
982:     Examples
983:     --------
984:     Consider the following problem:
985: 
986:     Minimize: f = -1*x[0] + 4*x[1]
987: 
988:     Subject to: -3*x[0] + 1*x[1] <= 6
989:                  1*x[0] + 2*x[1] <= 4
990:                             x[1] >= -3
991: 
992:     where:  -inf <= x[0] <= inf
993: 
994:     This problem deviates from the standard linear programming problem.
995:     In standard form, linear programming problems assume the variables x are
996:     non-negative.  Since the variables don't have standard bounds where
997:     0 <= x <= inf, the bounds of the variables must be explicitly set.
998: 
999:     There are two upper-bound constraints, which can be expressed as
1000: 
1001:     dot(A_ub, x) <= b_ub
1002: 
1003:     The input for this problem is as follows:
1004: 
1005:     >>> c = [-1, 4]
1006:     >>> A = [[-3, 1], [1, 2]]
1007:     >>> b = [6, 4]
1008:     >>> x0_bounds = (None, None)
1009:     >>> x1_bounds = (-3, None)
1010:     >>> from scipy.optimize import linprog
1011:     >>> res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),
1012:     ...               options={"disp": True})
1013:     Optimization terminated successfully.
1014:          Current function value: -22.000000
1015:          Iterations: 1
1016:     >>> print(res)
1017:          fun: -22.0
1018:      message: 'Optimization terminated successfully.'
1019:          nit: 1
1020:        slack: array([ 39.,   0.])
1021:       status: 0
1022:      success: True
1023:            x: array([ 10.,  -3.])
1024: 
1025:     Note the actual objective value is 11.428571.  In this case we minimized
1026:     the negative of the objective function.
1027: 
1028:     '''
1029:     meth = method.lower()
1030:     if options is None:
1031:         options = {}
1032: 
1033:     if meth == 'simplex':
1034:         return _linprog_simplex(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
1035:                                 bounds=bounds, callback=callback, **options)
1036:     elif meth == 'interior-point':
1037:         return _linprog_ip(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
1038:                            bounds=bounds, callback=callback, **options)
1039:     else:
1040:         raise ValueError('Unknown solver %s' % method)
1041: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_190487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', '\nA top-level linear programming interface. Currently this interface only\nsolves linear programming problems via the Simplex Method.\n\n.. versionadded:: 0.15.0\n\nFunctions\n---------\n.. autosummary::\n   :toctree: generated/\n\n    linprog\n    linprog_verbose_callback\n    linprog_terse_callback\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import numpy' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_190488 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy')

if (type(import_190488) is not StypyTypeError):

    if (import_190488 != 'pyd_module'):
        __import__(import_190488)
        sys_modules_190489 = sys.modules[import_190488]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', sys_modules_190489.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy', import_190488)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.optimize.optimize import OptimizeResult, _check_unknown_options' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_190490 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize.optimize')

if (type(import_190490) is not StypyTypeError):

    if (import_190490 != 'pyd_module'):
        __import__(import_190490)
        sys_modules_190491 = sys.modules[import_190490]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize.optimize', sys_modules_190491.module_type_store, module_type_store, ['OptimizeResult', '_check_unknown_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_190491, sys_modules_190491.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import OptimizeResult, _check_unknown_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize.optimize', None, module_type_store, ['OptimizeResult', '_check_unknown_options'], [OptimizeResult, _check_unknown_options])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize.optimize', import_190490)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from scipy.optimize._linprog_ip import _linprog_ip' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_190492 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.optimize._linprog_ip')

if (type(import_190492) is not StypyTypeError):

    if (import_190492 != 'pyd_module'):
        __import__(import_190492)
        sys_modules_190493 = sys.modules[import_190492]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.optimize._linprog_ip', sys_modules_190493.module_type_store, module_type_store, ['_linprog_ip'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_190493, sys_modules_190493.module_type_store, module_type_store)
    else:
        from scipy.optimize._linprog_ip import _linprog_ip

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.optimize._linprog_ip', None, module_type_store, ['_linprog_ip'], [_linprog_ip])

else:
    # Assigning a type to the variable 'scipy.optimize._linprog_ip' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.optimize._linprog_ip', import_190492)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 24):

# Assigning a List to a Name (line 24):
__all__ = ['linprog', 'linprog_verbose_callback', 'linprog_terse_callback']
module_type_store.set_exportable_members(['linprog', 'linprog_verbose_callback', 'linprog_terse_callback'])

# Obtaining an instance of the builtin type 'list' (line 24)
list_190494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
str_190495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'str', 'linprog')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_190494, str_190495)
# Adding element type (line 24)
str_190496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'str', 'linprog_verbose_callback')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_190494, str_190496)
# Adding element type (line 24)
str_190497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 50), 'str', 'linprog_terse_callback')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_190494, str_190497)

# Assigning a type to the variable '__all__' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '__all__', list_190494)

# Assigning a Str to a Name (line 26):

# Assigning a Str to a Name (line 26):
str_190498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__docformat__', str_190498)

@norecursion
def linprog_verbose_callback(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'linprog_verbose_callback'
    module_type_store = module_type_store.open_function_context('linprog_verbose_callback', 29, 0, False)
    
    # Passed parameters checking function
    linprog_verbose_callback.stypy_localization = localization
    linprog_verbose_callback.stypy_type_of_self = None
    linprog_verbose_callback.stypy_type_store = module_type_store
    linprog_verbose_callback.stypy_function_name = 'linprog_verbose_callback'
    linprog_verbose_callback.stypy_param_names_list = ['xk']
    linprog_verbose_callback.stypy_varargs_param_name = None
    linprog_verbose_callback.stypy_kwargs_param_name = 'kwargs'
    linprog_verbose_callback.stypy_call_defaults = defaults
    linprog_verbose_callback.stypy_call_varargs = varargs
    linprog_verbose_callback.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'linprog_verbose_callback', ['xk'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'linprog_verbose_callback', localization, ['xk'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'linprog_verbose_callback(...)' code ##################

    str_190499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', '\n    A sample callback function demonstrating the linprog callback interface.\n    This callback produces detailed output to sys.stdout before each iteration\n    and after the final iteration of the simplex algorithm.\n\n    Parameters\n    ----------\n    xk : array_like\n        The current solution vector.\n    **kwargs : dict\n        A dictionary containing the following parameters:\n\n        tableau : array_like\n            The current tableau of the simplex algorithm.\n            Its structure is defined in _solve_simplex.\n        phase : int\n            The current Phase of the simplex algorithm (1 or 2)\n        nit : int\n            The current iteration number.\n        pivot : tuple(int, int)\n            The index of the tableau selected as the next pivot,\n            or nan if no pivot exists\n        basis : array(int)\n            A list of the current basic variables.\n            Each element contains the name of a basic variable and its value.\n        complete : bool\n            True if the simplex algorithm has completed\n            (and this is the final call to callback), otherwise False.\n    ')
    
    # Assigning a Subscript to a Name (line 59):
    
    # Assigning a Subscript to a Name (line 59):
    
    # Obtaining the type of the subscript
    str_190500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'str', 'tableau')
    # Getting the type of 'kwargs' (line 59)
    kwargs_190501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 14), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___190502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 14), kwargs_190501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_190503 = invoke(stypy.reporting.localization.Localization(__file__, 59, 14), getitem___190502, str_190500)
    
    # Assigning a type to the variable 'tableau' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tableau', subscript_call_result_190503)
    
    # Assigning a Subscript to a Name (line 60):
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    str_190504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 17), 'str', 'nit')
    # Getting the type of 'kwargs' (line 60)
    kwargs_190505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 10), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___190506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 10), kwargs_190505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_190507 = invoke(stypy.reporting.localization.Localization(__file__, 60, 10), getitem___190506, str_190504)
    
    # Assigning a type to the variable 'nit' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'nit', subscript_call_result_190507)
    
    # Assigning a Subscript to a Tuple (line 61):
    
    # Assigning a Subscript to a Name (line 61):
    
    # Obtaining the type of the subscript
    int_190508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'int')
    
    # Obtaining the type of the subscript
    str_190509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'str', 'pivot')
    # Getting the type of 'kwargs' (line 61)
    kwargs_190510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___190511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 21), kwargs_190510, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_190512 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), getitem___190511, str_190509)
    
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___190513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), subscript_call_result_190512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_190514 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), getitem___190513, int_190508)
    
    # Assigning a type to the variable 'tuple_var_assignment_190473' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_190473', subscript_call_result_190514)
    
    # Assigning a Subscript to a Name (line 61):
    
    # Obtaining the type of the subscript
    int_190515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'int')
    
    # Obtaining the type of the subscript
    str_190516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'str', 'pivot')
    # Getting the type of 'kwargs' (line 61)
    kwargs_190517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___190518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 21), kwargs_190517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_190519 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), getitem___190518, str_190516)
    
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___190520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), subscript_call_result_190519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_190521 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), getitem___190520, int_190515)
    
    # Assigning a type to the variable 'tuple_var_assignment_190474' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_190474', subscript_call_result_190521)
    
    # Assigning a Name to a Name (line 61):
    # Getting the type of 'tuple_var_assignment_190473' (line 61)
    tuple_var_assignment_190473_190522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_190473')
    # Assigning a type to the variable 'pivrow' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'pivrow', tuple_var_assignment_190473_190522)
    
    # Assigning a Name to a Name (line 61):
    # Getting the type of 'tuple_var_assignment_190474' (line 61)
    tuple_var_assignment_190474_190523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'tuple_var_assignment_190474')
    # Assigning a type to the variable 'pivcol' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'pivcol', tuple_var_assignment_190474_190523)
    
    # Assigning a Subscript to a Name (line 62):
    
    # Assigning a Subscript to a Name (line 62):
    
    # Obtaining the type of the subscript
    str_190524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'str', 'phase')
    # Getting the type of 'kwargs' (line 62)
    kwargs_190525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___190526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), kwargs_190525, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_190527 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), getitem___190526, str_190524)
    
    # Assigning a type to the variable 'phase' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'phase', subscript_call_result_190527)
    
    # Assigning a Subscript to a Name (line 63):
    
    # Assigning a Subscript to a Name (line 63):
    
    # Obtaining the type of the subscript
    str_190528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'str', 'basis')
    # Getting the type of 'kwargs' (line 63)
    kwargs_190529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___190530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), kwargs_190529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_190531 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), getitem___190530, str_190528)
    
    # Assigning a type to the variable 'basis' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'basis', subscript_call_result_190531)
    
    # Assigning a Subscript to a Name (line 64):
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    str_190532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'str', 'complete')
    # Getting the type of 'kwargs' (line 64)
    kwargs_190533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___190534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), kwargs_190533, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_190535 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), getitem___190534, str_190532)
    
    # Assigning a type to the variable 'complete' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'complete', subscript_call_result_190535)
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to get_printoptions(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_190538 = {}
    # Getting the type of 'np' (line 66)
    np_190536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'np', False)
    # Obtaining the member 'get_printoptions' of a type (line 66)
    get_printoptions_190537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), np_190536, 'get_printoptions')
    # Calling get_printoptions(args, kwargs) (line 66)
    get_printoptions_call_result_190539 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), get_printoptions_190537, *[], **kwargs_190538)
    
    # Assigning a type to the variable 'saved_printoptions' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'saved_printoptions', get_printoptions_call_result_190539)
    
    # Call to set_printoptions(...): (line 67)
    # Processing the call keyword arguments (line 67)
    int_190542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'int')
    keyword_190543 = int_190542
    
    # Obtaining an instance of the builtin type 'dict' (line 68)
    dict_190544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 34), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 68)
    # Adding element type (key, value) (line 68)
    str_190545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 35), 'str', 'float')

    @norecursion
    def _stypy_temp_lambda_62(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_62'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_62', 68, 44, True)
        # Passed parameters checking function
        _stypy_temp_lambda_62.stypy_localization = localization
        _stypy_temp_lambda_62.stypy_type_of_self = None
        _stypy_temp_lambda_62.stypy_type_store = module_type_store
        _stypy_temp_lambda_62.stypy_function_name = '_stypy_temp_lambda_62'
        _stypy_temp_lambda_62.stypy_param_names_list = ['x']
        _stypy_temp_lambda_62.stypy_varargs_param_name = None
        _stypy_temp_lambda_62.stypy_kwargs_param_name = None
        _stypy_temp_lambda_62.stypy_call_defaults = defaults
        _stypy_temp_lambda_62.stypy_call_varargs = varargs
        _stypy_temp_lambda_62.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_62', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_62', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to format(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'x' (line 68)
        x_190548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 74), 'x', False)
        # Processing the call keyword arguments (line 68)
        kwargs_190549 = {}
        str_190546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 54), 'str', '{0: 12.4f}')
        # Obtaining the member 'format' of a type (line 68)
        format_190547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 54), str_190546, 'format')
        # Calling format(args, kwargs) (line 68)
        format_call_result_190550 = invoke(stypy.reporting.localization.Localization(__file__, 68, 54), format_190547, *[x_190548], **kwargs_190549)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 44), 'stypy_return_type', format_call_result_190550)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_62' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_190551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 44), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190551)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_62'
        return stypy_return_type_190551

    # Assigning a type to the variable '_stypy_temp_lambda_62' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 44), '_stypy_temp_lambda_62', _stypy_temp_lambda_62)
    # Getting the type of '_stypy_temp_lambda_62' (line 68)
    _stypy_temp_lambda_62_190552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 44), '_stypy_temp_lambda_62')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 34), dict_190544, (str_190545, _stypy_temp_lambda_62_190552))
    
    keyword_190553 = dict_190544
    kwargs_190554 = {'linewidth': keyword_190543, 'formatter': keyword_190553}
    # Getting the type of 'np' (line 67)
    np_190540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'np', False)
    # Obtaining the member 'set_printoptions' of a type (line 67)
    set_printoptions_190541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), np_190540, 'set_printoptions')
    # Calling set_printoptions(args, kwargs) (line 67)
    set_printoptions_call_result_190555 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), set_printoptions_190541, *[], **kwargs_190554)
    
    
    # Getting the type of 'complete' (line 69)
    complete_190556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 7), 'complete')
    # Testing the type of an if condition (line 69)
    if_condition_190557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 4), complete_190556)
    # Assigning a type to the variable 'if_condition_190557' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'if_condition_190557', if_condition_190557)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 70)
    # Processing the call arguments (line 70)
    
    # Call to format(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'phase' (line 70)
    phase_190561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 76), 'phase', False)
    # Processing the call keyword arguments (line 70)
    kwargs_190562 = {}
    str_190559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 14), 'str', '--------- Iteration Complete - Phase {0:d} -------\n')
    # Obtaining the member 'format' of a type (line 70)
    format_190560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 14), str_190559, 'format')
    # Calling format(args, kwargs) (line 70)
    format_call_result_190563 = invoke(stypy.reporting.localization.Localization(__file__, 70, 14), format_190560, *[phase_190561], **kwargs_190562)
    
    # Processing the call keyword arguments (line 70)
    kwargs_190564 = {}
    # Getting the type of 'print' (line 70)
    print_190558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'print', False)
    # Calling print(args, kwargs) (line 70)
    print_call_result_190565 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), print_190558, *[format_call_result_190563], **kwargs_190564)
    
    
    # Call to print(...): (line 71)
    # Processing the call arguments (line 71)
    str_190567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 14), 'str', 'Tableau:')
    # Processing the call keyword arguments (line 71)
    kwargs_190568 = {}
    # Getting the type of 'print' (line 71)
    print_190566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'print', False)
    # Calling print(args, kwargs) (line 71)
    print_call_result_190569 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), print_190566, *[str_190567], **kwargs_190568)
    
    # SSA branch for the else part of an if statement (line 69)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'nit' (line 72)
    nit_190570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 9), 'nit')
    int_190571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 16), 'int')
    # Applying the binary operator '==' (line 72)
    result_eq_190572 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 9), '==', nit_190570, int_190571)
    
    # Testing the type of an if condition (line 72)
    if_condition_190573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 9), result_eq_190572)
    # Assigning a type to the variable 'if_condition_190573' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 9), 'if_condition_190573', if_condition_190573)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Call to format(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'phase' (line 73)
    phase_190577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 76), 'phase', False)
    # Processing the call keyword arguments (line 73)
    kwargs_190578 = {}
    str_190575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 14), 'str', '--------- Initial Tableau - Phase {0:d} ----------\n')
    # Obtaining the member 'format' of a type (line 73)
    format_190576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 14), str_190575, 'format')
    # Calling format(args, kwargs) (line 73)
    format_call_result_190579 = invoke(stypy.reporting.localization.Localization(__file__, 73, 14), format_190576, *[phase_190577], **kwargs_190578)
    
    # Processing the call keyword arguments (line 73)
    kwargs_190580 = {}
    # Getting the type of 'print' (line 73)
    print_190574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'print', False)
    # Calling print(args, kwargs) (line 73)
    print_call_result_190581 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), print_190574, *[format_call_result_190579], **kwargs_190580)
    
    # SSA branch for the else part of an if statement (line 72)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Call to format(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'nit' (line 76)
    nit_190585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 75), 'nit', False)
    # Getting the type of 'phase' (line 76)
    phase_190586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 80), 'phase', False)
    # Processing the call keyword arguments (line 76)
    kwargs_190587 = {}
    str_190583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'str', '--------- Iteration {0:d}  - Phase {1:d} --------\n')
    # Obtaining the member 'format' of a type (line 76)
    format_190584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 14), str_190583, 'format')
    # Calling format(args, kwargs) (line 76)
    format_call_result_190588 = invoke(stypy.reporting.localization.Localization(__file__, 76, 14), format_190584, *[nit_190585, phase_190586], **kwargs_190587)
    
    # Processing the call keyword arguments (line 76)
    kwargs_190589 = {}
    # Getting the type of 'print' (line 76)
    print_190582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'print', False)
    # Calling print(args, kwargs) (line 76)
    print_call_result_190590 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), print_190582, *[format_call_result_190588], **kwargs_190589)
    
    
    # Call to print(...): (line 77)
    # Processing the call arguments (line 77)
    str_190592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 14), 'str', 'Tableau:')
    # Processing the call keyword arguments (line 77)
    kwargs_190593 = {}
    # Getting the type of 'print' (line 77)
    print_190591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'print', False)
    # Calling print(args, kwargs) (line 77)
    print_call_result_190594 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), print_190591, *[str_190592], **kwargs_190593)
    
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'nit' (line 79)
    nit_190595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'nit')
    int_190596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 14), 'int')
    # Applying the binary operator '>=' (line 79)
    result_ge_190597 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 7), '>=', nit_190595, int_190596)
    
    # Testing the type of an if condition (line 79)
    if_condition_190598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), result_ge_190597)
    # Assigning a type to the variable 'if_condition_190598' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_190598', if_condition_190598)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 80)
    # Processing the call arguments (line 80)
    str_190600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 14), 'str', '')
    
    # Call to str(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'tableau' (line 80)
    tableau_190602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'tableau', False)
    # Processing the call keyword arguments (line 80)
    kwargs_190603 = {}
    # Getting the type of 'str' (line 80)
    str_190601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'str', False)
    # Calling str(args, kwargs) (line 80)
    str_call_result_190604 = invoke(stypy.reporting.localization.Localization(__file__, 80, 19), str_190601, *[tableau_190602], **kwargs_190603)
    
    # Applying the binary operator '+' (line 80)
    result_add_190605 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 14), '+', str_190600, str_call_result_190604)
    
    str_190606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 34), 'str', '\n')
    # Applying the binary operator '+' (line 80)
    result_add_190607 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 32), '+', result_add_190605, str_190606)
    
    # Processing the call keyword arguments (line 80)
    kwargs_190608 = {}
    # Getting the type of 'print' (line 80)
    print_190599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'print', False)
    # Calling print(args, kwargs) (line 80)
    print_call_result_190609 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), print_190599, *[result_add_190607], **kwargs_190608)
    
    
    
    # Getting the type of 'complete' (line 81)
    complete_190610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'complete')
    # Applying the 'not' unary operator (line 81)
    result_not__190611 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), 'not', complete_190610)
    
    # Testing the type of an if condition (line 81)
    if_condition_190612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), result_not__190611)
    # Assigning a type to the variable 'if_condition_190612' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_190612', if_condition_190612)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Call to format(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'pivrow' (line 82)
    pivrow_190616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 64), 'pivrow', False)
    # Getting the type of 'pivcol' (line 82)
    pivcol_190617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 72), 'pivcol', False)
    # Processing the call keyword arguments (line 82)
    kwargs_190618 = {}
    str_190614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'str', 'Pivot Element: T[{0:.0f}, {1:.0f}]\n')
    # Obtaining the member 'format' of a type (line 82)
    format_190615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 18), str_190614, 'format')
    # Calling format(args, kwargs) (line 82)
    format_call_result_190619 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), format_190615, *[pivrow_190616, pivcol_190617], **kwargs_190618)
    
    # Processing the call keyword arguments (line 82)
    kwargs_190620 = {}
    # Getting the type of 'print' (line 82)
    print_190613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'print', False)
    # Calling print(args, kwargs) (line 82)
    print_call_result_190621 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), print_190613, *[format_call_result_190619], **kwargs_190620)
    
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 83)
    # Processing the call arguments (line 83)
    str_190623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 14), 'str', 'Basic Variables:')
    # Getting the type of 'basis' (line 83)
    basis_190624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'basis', False)
    # Processing the call keyword arguments (line 83)
    kwargs_190625 = {}
    # Getting the type of 'print' (line 83)
    print_190622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'print', False)
    # Calling print(args, kwargs) (line 83)
    print_call_result_190626 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), print_190622, *[str_190623, basis_190624], **kwargs_190625)
    
    
    # Call to print(...): (line 84)
    # Processing the call keyword arguments (line 84)
    kwargs_190628 = {}
    # Getting the type of 'print' (line 84)
    print_190627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'print', False)
    # Calling print(args, kwargs) (line 84)
    print_call_result_190629 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), print_190627, *[], **kwargs_190628)
    
    
    # Call to print(...): (line 85)
    # Processing the call arguments (line 85)
    str_190631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 14), 'str', 'Current Solution:')
    # Processing the call keyword arguments (line 85)
    kwargs_190632 = {}
    # Getting the type of 'print' (line 85)
    print_190630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'print', False)
    # Calling print(args, kwargs) (line 85)
    print_call_result_190633 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), print_190630, *[str_190631], **kwargs_190632)
    
    
    # Call to print(...): (line 86)
    # Processing the call arguments (line 86)
    str_190635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 14), 'str', 'x = ')
    # Getting the type of 'xk' (line 86)
    xk_190636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'xk', False)
    # Processing the call keyword arguments (line 86)
    kwargs_190637 = {}
    # Getting the type of 'print' (line 86)
    print_190634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'print', False)
    # Calling print(args, kwargs) (line 86)
    print_call_result_190638 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), print_190634, *[str_190635, xk_190636], **kwargs_190637)
    
    
    # Call to print(...): (line 87)
    # Processing the call keyword arguments (line 87)
    kwargs_190640 = {}
    # Getting the type of 'print' (line 87)
    print_190639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'print', False)
    # Calling print(args, kwargs) (line 87)
    print_call_result_190641 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), print_190639, *[], **kwargs_190640)
    
    
    # Call to print(...): (line 88)
    # Processing the call arguments (line 88)
    str_190643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 14), 'str', 'Current Objective Value:')
    # Processing the call keyword arguments (line 88)
    kwargs_190644 = {}
    # Getting the type of 'print' (line 88)
    print_190642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'print', False)
    # Calling print(args, kwargs) (line 88)
    print_call_result_190645 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), print_190642, *[str_190643], **kwargs_190644)
    
    
    # Call to print(...): (line 89)
    # Processing the call arguments (line 89)
    str_190647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 14), 'str', 'f = ')
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 89)
    tuple_190648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 89)
    # Adding element type (line 89)
    int_190649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 31), tuple_190648, int_190649)
    # Adding element type (line 89)
    int_190650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 31), tuple_190648, int_190650)
    
    # Getting the type of 'tableau' (line 89)
    tableau_190651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'tableau', False)
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___190652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 23), tableau_190651, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_190653 = invoke(stypy.reporting.localization.Localization(__file__, 89, 23), getitem___190652, tuple_190648)
    
    # Applying the 'usub' unary operator (line 89)
    result___neg___190654 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 22), 'usub', subscript_call_result_190653)
    
    # Processing the call keyword arguments (line 89)
    kwargs_190655 = {}
    # Getting the type of 'print' (line 89)
    print_190646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'print', False)
    # Calling print(args, kwargs) (line 89)
    print_call_result_190656 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), print_190646, *[str_190647, result___neg___190654], **kwargs_190655)
    
    
    # Call to print(...): (line 90)
    # Processing the call keyword arguments (line 90)
    kwargs_190658 = {}
    # Getting the type of 'print' (line 90)
    print_190657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'print', False)
    # Calling print(args, kwargs) (line 90)
    print_call_result_190659 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), print_190657, *[], **kwargs_190658)
    
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to set_printoptions(...): (line 91)
    # Processing the call keyword arguments (line 91)
    # Getting the type of 'saved_printoptions' (line 91)
    saved_printoptions_190662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'saved_printoptions', False)
    kwargs_190663 = {'saved_printoptions_190662': saved_printoptions_190662}
    # Getting the type of 'np' (line 91)
    np_190660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'np', False)
    # Obtaining the member 'set_printoptions' of a type (line 91)
    set_printoptions_190661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), np_190660, 'set_printoptions')
    # Calling set_printoptions(args, kwargs) (line 91)
    set_printoptions_call_result_190664 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), set_printoptions_190661, *[], **kwargs_190663)
    
    
    # ################# End of 'linprog_verbose_callback(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'linprog_verbose_callback' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_190665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190665)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'linprog_verbose_callback'
    return stypy_return_type_190665

# Assigning a type to the variable 'linprog_verbose_callback' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'linprog_verbose_callback', linprog_verbose_callback)

@norecursion
def linprog_terse_callback(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'linprog_terse_callback'
    module_type_store = module_type_store.open_function_context('linprog_terse_callback', 94, 0, False)
    
    # Passed parameters checking function
    linprog_terse_callback.stypy_localization = localization
    linprog_terse_callback.stypy_type_of_self = None
    linprog_terse_callback.stypy_type_store = module_type_store
    linprog_terse_callback.stypy_function_name = 'linprog_terse_callback'
    linprog_terse_callback.stypy_param_names_list = ['xk']
    linprog_terse_callback.stypy_varargs_param_name = None
    linprog_terse_callback.stypy_kwargs_param_name = 'kwargs'
    linprog_terse_callback.stypy_call_defaults = defaults
    linprog_terse_callback.stypy_call_varargs = varargs
    linprog_terse_callback.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'linprog_terse_callback', ['xk'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'linprog_terse_callback', localization, ['xk'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'linprog_terse_callback(...)' code ##################

    str_190666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', '\n    A sample callback function demonstrating the linprog callback interface.\n    This callback produces brief output to sys.stdout before each iteration\n    and after the final iteration of the simplex algorithm.\n\n    Parameters\n    ----------\n    xk : array_like\n        The current solution vector.\n    **kwargs : dict\n        A dictionary containing the following parameters:\n\n        tableau : array_like\n            The current tableau of the simplex algorithm.\n            Its structure is defined in _solve_simplex.\n        vars : tuple(str, ...)\n            Column headers for each column in tableau.\n            "x[i]" for actual variables, "s[i]" for slack surplus variables,\n            "a[i]" for artificial variables, and "RHS" for the constraint\n            RHS vector.\n        phase : int\n            The current Phase of the simplex algorithm (1 or 2)\n        nit : int\n            The current iteration number.\n        pivot : tuple(int, int)\n            The index of the tableau selected as the next pivot,\n            or nan if no pivot exists\n        basics : list[tuple(int, float)]\n            A list of the current basic variables.\n            Each element contains the index of a basic variable and\n            its value.\n        complete : bool\n            True if the simplex algorithm has completed\n            (and this is the final call to callback), otherwise False.\n    ')
    
    # Assigning a Subscript to a Name (line 130):
    
    # Assigning a Subscript to a Name (line 130):
    
    # Obtaining the type of the subscript
    str_190667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 17), 'str', 'nit')
    # Getting the type of 'kwargs' (line 130)
    kwargs_190668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 10), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___190669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 10), kwargs_190668, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_190670 = invoke(stypy.reporting.localization.Localization(__file__, 130, 10), getitem___190669, str_190667)
    
    # Assigning a type to the variable 'nit' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'nit', subscript_call_result_190670)
    
    
    # Getting the type of 'nit' (line 132)
    nit_190671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 7), 'nit')
    int_190672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 14), 'int')
    # Applying the binary operator '==' (line 132)
    result_eq_190673 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 7), '==', nit_190671, int_190672)
    
    # Testing the type of an if condition (line 132)
    if_condition_190674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 4), result_eq_190673)
    # Assigning a type to the variable 'if_condition_190674' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'if_condition_190674', if_condition_190674)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 133)
    # Processing the call arguments (line 133)
    str_190676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 14), 'str', 'Iter:   X:')
    # Processing the call keyword arguments (line 133)
    kwargs_190677 = {}
    # Getting the type of 'print' (line 133)
    print_190675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'print', False)
    # Calling print(args, kwargs) (line 133)
    print_call_result_190678 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), print_190675, *[str_190676], **kwargs_190677)
    
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 134)
    # Processing the call arguments (line 134)
    
    # Call to format(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'nit' (line 134)
    nit_190682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'nit', False)
    # Processing the call keyword arguments (line 134)
    kwargs_190683 = {}
    str_190680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 10), 'str', '{0: <5d}   ')
    # Obtaining the member 'format' of a type (line 134)
    format_190681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 10), str_190680, 'format')
    # Calling format(args, kwargs) (line 134)
    format_call_result_190684 = invoke(stypy.reporting.localization.Localization(__file__, 134, 10), format_190681, *[nit_190682], **kwargs_190683)
    
    # Processing the call keyword arguments (line 134)
    str_190685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 41), 'str', '')
    keyword_190686 = str_190685
    kwargs_190687 = {'end': keyword_190686}
    # Getting the type of 'print' (line 134)
    print_190679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'print', False)
    # Calling print(args, kwargs) (line 134)
    print_call_result_190688 = invoke(stypy.reporting.localization.Localization(__file__, 134, 4), print_190679, *[format_call_result_190684], **kwargs_190687)
    
    
    # Call to print(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'xk' (line 135)
    xk_190690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 10), 'xk', False)
    # Processing the call keyword arguments (line 135)
    kwargs_190691 = {}
    # Getting the type of 'print' (line 135)
    print_190689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'print', False)
    # Calling print(args, kwargs) (line 135)
    print_call_result_190692 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), print_190689, *[xk_190690], **kwargs_190691)
    
    
    # ################# End of 'linprog_terse_callback(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'linprog_terse_callback' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_190693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190693)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'linprog_terse_callback'
    return stypy_return_type_190693

# Assigning a type to the variable 'linprog_terse_callback' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'linprog_terse_callback', linprog_terse_callback)

@norecursion
def _pivot_col(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_190694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 22), 'float')
    # Getting the type of 'False' (line 138)
    False_190695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 37), 'False')
    defaults = [float_190694, False_190695]
    # Create a new context for function '_pivot_col'
    module_type_store = module_type_store.open_function_context('_pivot_col', 138, 0, False)
    
    # Passed parameters checking function
    _pivot_col.stypy_localization = localization
    _pivot_col.stypy_type_of_self = None
    _pivot_col.stypy_type_store = module_type_store
    _pivot_col.stypy_function_name = '_pivot_col'
    _pivot_col.stypy_param_names_list = ['T', 'tol', 'bland']
    _pivot_col.stypy_varargs_param_name = None
    _pivot_col.stypy_kwargs_param_name = None
    _pivot_col.stypy_call_defaults = defaults
    _pivot_col.stypy_call_varargs = varargs
    _pivot_col.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_pivot_col', ['T', 'tol', 'bland'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_pivot_col', localization, ['T', 'tol', 'bland'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_pivot_col(...)' code ##################

    str_190696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, (-1)), 'str', "\n    Given a linear programming simplex tableau, determine the column\n    of the variable to enter the basis.\n\n    Parameters\n    ----------\n    T : 2D ndarray\n        The simplex tableau.\n    tol : float\n        Elements in the objective row larger than -tol will not be considered\n        for pivoting.  Nominally this value is zero, but numerical issues\n        cause a tolerance about zero to be necessary.\n    bland : bool\n        If True, use Bland's rule for selection of the column (select the\n        first column with a negative coefficient in the objective row,\n        regardless of magnitude).\n\n    Returns\n    -------\n    status: bool\n        True if a suitable pivot column was found, otherwise False.\n        A return of False indicates that the linear programming simplex\n        algorithm is complete.\n    col: int\n        The index of the column of the pivot element.\n        If status is False, col will be returned as nan.\n    ")
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to masked_where(...): (line 166)
    # Processing the call arguments (line 166)
    
    
    # Obtaining the type of the subscript
    int_190700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'int')
    int_190701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 35), 'int')
    slice_190702 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 28), None, int_190701, None)
    # Getting the type of 'T' (line 166)
    T_190703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___190704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), T_190703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_190705 = invoke(stypy.reporting.localization.Localization(__file__, 166, 28), getitem___190704, (int_190700, slice_190702))
    
    
    # Getting the type of 'tol' (line 166)
    tol_190706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 43), 'tol', False)
    # Applying the 'usub' unary operator (line 166)
    result___neg___190707 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 42), 'usub', tol_190706)
    
    # Applying the binary operator '>=' (line 166)
    result_ge_190708 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 28), '>=', subscript_call_result_190705, result___neg___190707)
    
    
    # Obtaining the type of the subscript
    int_190709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 50), 'int')
    int_190710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 55), 'int')
    slice_190711 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 48), None, int_190710, None)
    # Getting the type of 'T' (line 166)
    T_190712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___190713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 48), T_190712, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_190714 = invoke(stypy.reporting.localization.Localization(__file__, 166, 48), getitem___190713, (int_190709, slice_190711))
    
    # Processing the call keyword arguments (line 166)
    # Getting the type of 'False' (line 166)
    False_190715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 65), 'False', False)
    keyword_190716 = False_190715
    kwargs_190717 = {'copy': keyword_190716}
    # Getting the type of 'np' (line 166)
    np_190697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 9), 'np', False)
    # Obtaining the member 'ma' of a type (line 166)
    ma_190698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 9), np_190697, 'ma')
    # Obtaining the member 'masked_where' of a type (line 166)
    masked_where_190699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 9), ma_190698, 'masked_where')
    # Calling masked_where(args, kwargs) (line 166)
    masked_where_call_result_190718 = invoke(stypy.reporting.localization.Localization(__file__, 166, 9), masked_where_190699, *[result_ge_190708, subscript_call_result_190714], **kwargs_190717)
    
    # Assigning a type to the variable 'ma' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'ma', masked_where_call_result_190718)
    
    
    
    # Call to count(...): (line 167)
    # Processing the call keyword arguments (line 167)
    kwargs_190721 = {}
    # Getting the type of 'ma' (line 167)
    ma_190719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 7), 'ma', False)
    # Obtaining the member 'count' of a type (line 167)
    count_190720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 7), ma_190719, 'count')
    # Calling count(args, kwargs) (line 167)
    count_call_result_190722 = invoke(stypy.reporting.localization.Localization(__file__, 167, 7), count_190720, *[], **kwargs_190721)
    
    int_190723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 21), 'int')
    # Applying the binary operator '==' (line 167)
    result_eq_190724 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 7), '==', count_call_result_190722, int_190723)
    
    # Testing the type of an if condition (line 167)
    if_condition_190725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 4), result_eq_190724)
    # Assigning a type to the variable 'if_condition_190725' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'if_condition_190725', if_condition_190725)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 168)
    tuple_190726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 168)
    # Adding element type (line 168)
    # Getting the type of 'False' (line 168)
    False_190727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 15), tuple_190726, False_190727)
    # Adding element type (line 168)
    # Getting the type of 'np' (line 168)
    np_190728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'np')
    # Obtaining the member 'nan' of a type (line 168)
    nan_190729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 22), np_190728, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 15), tuple_190726, nan_190729)
    
    # Assigning a type to the variable 'stypy_return_type' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'stypy_return_type', tuple_190726)
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'bland' (line 169)
    bland_190730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'bland')
    # Testing the type of an if condition (line 169)
    if_condition_190731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), bland_190730)
    # Assigning a type to the variable 'if_condition_190731' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_190731', if_condition_190731)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 170)
    tuple_190732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 170)
    # Adding element type (line 170)
    # Getting the type of 'True' (line 170)
    True_190733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 15), tuple_190732, True_190733)
    # Adding element type (line 170)
    
    # Obtaining the type of the subscript
    int_190734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 51), 'int')
    
    # Obtaining the type of the subscript
    int_190735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 48), 'int')
    
    # Call to where(...): (line 170)
    # Processing the call arguments (line 170)
    
    # Getting the type of 'ma' (line 170)
    ma_190738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'ma', False)
    # Obtaining the member 'mask' of a type (line 170)
    mask_190739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 30), ma_190738, 'mask')
    # Getting the type of 'False' (line 170)
    False_190740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'False', False)
    # Applying the binary operator '==' (line 170)
    result_eq_190741 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 30), '==', mask_190739, False_190740)
    
    # Processing the call keyword arguments (line 170)
    kwargs_190742 = {}
    # Getting the type of 'np' (line 170)
    np_190736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'np', False)
    # Obtaining the member 'where' of a type (line 170)
    where_190737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 21), np_190736, 'where')
    # Calling where(args, kwargs) (line 170)
    where_call_result_190743 = invoke(stypy.reporting.localization.Localization(__file__, 170, 21), where_190737, *[result_eq_190741], **kwargs_190742)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___190744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 21), where_call_result_190743, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_190745 = invoke(stypy.reporting.localization.Localization(__file__, 170, 21), getitem___190744, int_190735)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___190746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 21), subscript_call_result_190745, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_190747 = invoke(stypy.reporting.localization.Localization(__file__, 170, 21), getitem___190746, int_190734)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 15), tuple_190732, subscript_call_result_190747)
    
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stypy_return_type', tuple_190732)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 171)
    tuple_190748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 171)
    # Adding element type (line 171)
    # Getting the type of 'True' (line 171)
    True_190749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 11), tuple_190748, True_190749)
    # Adding element type (line 171)
    
    # Obtaining the type of the subscript
    int_190750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 48), 'int')
    
    # Obtaining the type of the subscript
    int_190751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 45), 'int')
    
    # Call to where(...): (line 171)
    # Processing the call arguments (line 171)
    
    # Getting the type of 'ma' (line 171)
    ma_190755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'ma', False)
    
    # Call to min(...): (line 171)
    # Processing the call keyword arguments (line 171)
    kwargs_190758 = {}
    # Getting the type of 'ma' (line 171)
    ma_190756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'ma', False)
    # Obtaining the member 'min' of a type (line 171)
    min_190757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 35), ma_190756, 'min')
    # Calling min(args, kwargs) (line 171)
    min_call_result_190759 = invoke(stypy.reporting.localization.Localization(__file__, 171, 35), min_190757, *[], **kwargs_190758)
    
    # Applying the binary operator '==' (line 171)
    result_eq_190760 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 29), '==', ma_190755, min_call_result_190759)
    
    # Processing the call keyword arguments (line 171)
    kwargs_190761 = {}
    # Getting the type of 'np' (line 171)
    np_190752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'np', False)
    # Obtaining the member 'ma' of a type (line 171)
    ma_190753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), np_190752, 'ma')
    # Obtaining the member 'where' of a type (line 171)
    where_190754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), ma_190753, 'where')
    # Calling where(args, kwargs) (line 171)
    where_call_result_190762 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), where_190754, *[result_eq_190760], **kwargs_190761)
    
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___190763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), where_call_result_190762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_190764 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), getitem___190763, int_190751)
    
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___190765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), subscript_call_result_190764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_190766 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), getitem___190765, int_190750)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 11), tuple_190748, subscript_call_result_190766)
    
    # Assigning a type to the variable 'stypy_return_type' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type', tuple_190748)
    
    # ################# End of '_pivot_col(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_pivot_col' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_190767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190767)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_pivot_col'
    return stypy_return_type_190767

# Assigning a type to the variable '_pivot_col' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), '_pivot_col', _pivot_col)

@norecursion
def _pivot_row(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_190768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 37), 'float')
    defaults = [float_190768]
    # Create a new context for function '_pivot_row'
    module_type_store = module_type_store.open_function_context('_pivot_row', 174, 0, False)
    
    # Passed parameters checking function
    _pivot_row.stypy_localization = localization
    _pivot_row.stypy_type_of_self = None
    _pivot_row.stypy_type_store = module_type_store
    _pivot_row.stypy_function_name = '_pivot_row'
    _pivot_row.stypy_param_names_list = ['T', 'pivcol', 'phase', 'tol']
    _pivot_row.stypy_varargs_param_name = None
    _pivot_row.stypy_kwargs_param_name = None
    _pivot_row.stypy_call_defaults = defaults
    _pivot_row.stypy_call_varargs = varargs
    _pivot_row.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_pivot_row', ['T', 'pivcol', 'phase', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_pivot_row', localization, ['T', 'pivcol', 'phase', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_pivot_row(...)' code ##################

    str_190769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, (-1)), 'str', '\n    Given a linear programming simplex tableau, determine the row for the\n    pivot operation.\n\n    Parameters\n    ----------\n    T : 2D ndarray\n        The simplex tableau.\n    pivcol : int\n        The index of the pivot column.\n    phase : int\n        The phase of the simplex algorithm (1 or 2).\n    tol : float\n        Elements in the pivot column smaller than tol will not be considered\n        for pivoting.  Nominally this value is zero, but numerical issues\n        cause a tolerance about zero to be necessary.\n\n    Returns\n    -------\n    status: bool\n        True if a suitable pivot row was found, otherwise False.  A return\n        of False indicates that the linear programming problem is unbounded.\n    row: int\n        The index of the row of the pivot element.  If status is False, row\n        will be returned as nan.\n    ')
    
    
    # Getting the type of 'phase' (line 201)
    phase_190770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 7), 'phase')
    int_190771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 16), 'int')
    # Applying the binary operator '==' (line 201)
    result_eq_190772 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 7), '==', phase_190770, int_190771)
    
    # Testing the type of an if condition (line 201)
    if_condition_190773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 4), result_eq_190772)
    # Assigning a type to the variable 'if_condition_190773' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'if_condition_190773', if_condition_190773)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 202):
    
    # Assigning a Num to a Name (line 202):
    int_190774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 12), 'int')
    # Assigning a type to the variable 'k' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'k', int_190774)
    # SSA branch for the else part of an if statement (line 201)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 204):
    
    # Assigning a Num to a Name (line 204):
    int_190775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 12), 'int')
    # Assigning a type to the variable 'k' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'k', int_190775)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to masked_where(...): (line 205)
    # Processing the call arguments (line 205)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'k' (line 205)
    k_190779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 32), 'k', False)
    # Applying the 'usub' unary operator (line 205)
    result___neg___190780 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 31), 'usub', k_190779)
    
    slice_190781 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 28), None, result___neg___190780, None)
    # Getting the type of 'pivcol' (line 205)
    pivcol_190782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 35), 'pivcol', False)
    # Getting the type of 'T' (line 205)
    T_190783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 28), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___190784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 28), T_190783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_190785 = invoke(stypy.reporting.localization.Localization(__file__, 205, 28), getitem___190784, (slice_190781, pivcol_190782))
    
    # Getting the type of 'tol' (line 205)
    tol_190786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 46), 'tol', False)
    # Applying the binary operator '<=' (line 205)
    result_le_190787 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 28), '<=', subscript_call_result_190785, tol_190786)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'k' (line 205)
    k_190788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 55), 'k', False)
    # Applying the 'usub' unary operator (line 205)
    result___neg___190789 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 54), 'usub', k_190788)
    
    slice_190790 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 51), None, result___neg___190789, None)
    # Getting the type of 'pivcol' (line 205)
    pivcol_190791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 58), 'pivcol', False)
    # Getting the type of 'T' (line 205)
    T_190792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 51), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___190793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 51), T_190792, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_190794 = invoke(stypy.reporting.localization.Localization(__file__, 205, 51), getitem___190793, (slice_190790, pivcol_190791))
    
    # Processing the call keyword arguments (line 205)
    # Getting the type of 'False' (line 205)
    False_190795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 72), 'False', False)
    keyword_190796 = False_190795
    kwargs_190797 = {'copy': keyword_190796}
    # Getting the type of 'np' (line 205)
    np_190776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 9), 'np', False)
    # Obtaining the member 'ma' of a type (line 205)
    ma_190777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 9), np_190776, 'ma')
    # Obtaining the member 'masked_where' of a type (line 205)
    masked_where_190778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 9), ma_190777, 'masked_where')
    # Calling masked_where(args, kwargs) (line 205)
    masked_where_call_result_190798 = invoke(stypy.reporting.localization.Localization(__file__, 205, 9), masked_where_190778, *[result_le_190787, subscript_call_result_190794], **kwargs_190797)
    
    # Assigning a type to the variable 'ma' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'ma', masked_where_call_result_190798)
    
    
    
    # Call to count(...): (line 206)
    # Processing the call keyword arguments (line 206)
    kwargs_190801 = {}
    # Getting the type of 'ma' (line 206)
    ma_190799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'ma', False)
    # Obtaining the member 'count' of a type (line 206)
    count_190800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 7), ma_190799, 'count')
    # Calling count(args, kwargs) (line 206)
    count_call_result_190802 = invoke(stypy.reporting.localization.Localization(__file__, 206, 7), count_190800, *[], **kwargs_190801)
    
    int_190803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 21), 'int')
    # Applying the binary operator '==' (line 206)
    result_eq_190804 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 7), '==', count_call_result_190802, int_190803)
    
    # Testing the type of an if condition (line 206)
    if_condition_190805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 4), result_eq_190804)
    # Assigning a type to the variable 'if_condition_190805' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'if_condition_190805', if_condition_190805)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 207)
    tuple_190806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 207)
    # Adding element type (line 207)
    # Getting the type of 'False' (line 207)
    False_190807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 15), tuple_190806, False_190807)
    # Adding element type (line 207)
    # Getting the type of 'np' (line 207)
    np_190808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'np')
    # Obtaining the member 'nan' of a type (line 207)
    nan_190809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 22), np_190808, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 15), tuple_190806, nan_190809)
    
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', tuple_190806)
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to masked_where(...): (line 208)
    # Processing the call arguments (line 208)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'k' (line 208)
    k_190813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), 'k', False)
    # Applying the 'usub' unary operator (line 208)
    result___neg___190814 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 31), 'usub', k_190813)
    
    slice_190815 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 208, 28), None, result___neg___190814, None)
    # Getting the type of 'pivcol' (line 208)
    pivcol_190816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 35), 'pivcol', False)
    # Getting the type of 'T' (line 208)
    T_190817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 28), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___190818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 28), T_190817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_190819 = invoke(stypy.reporting.localization.Localization(__file__, 208, 28), getitem___190818, (slice_190815, pivcol_190816))
    
    # Getting the type of 'tol' (line 208)
    tol_190820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 46), 'tol', False)
    # Applying the binary operator '<=' (line 208)
    result_le_190821 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 28), '<=', subscript_call_result_190819, tol_190820)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'k' (line 208)
    k_190822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 55), 'k', False)
    # Applying the 'usub' unary operator (line 208)
    result___neg___190823 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 54), 'usub', k_190822)
    
    slice_190824 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 208, 51), None, result___neg___190823, None)
    int_190825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 58), 'int')
    # Getting the type of 'T' (line 208)
    T_190826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 51), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___190827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 51), T_190826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_190828 = invoke(stypy.reporting.localization.Localization(__file__, 208, 51), getitem___190827, (slice_190824, int_190825))
    
    # Processing the call keyword arguments (line 208)
    # Getting the type of 'False' (line 208)
    False_190829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 68), 'False', False)
    keyword_190830 = False_190829
    kwargs_190831 = {'copy': keyword_190830}
    # Getting the type of 'np' (line 208)
    np_190810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 9), 'np', False)
    # Obtaining the member 'ma' of a type (line 208)
    ma_190811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 9), np_190810, 'ma')
    # Obtaining the member 'masked_where' of a type (line 208)
    masked_where_190812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 9), ma_190811, 'masked_where')
    # Calling masked_where(args, kwargs) (line 208)
    masked_where_call_result_190832 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), masked_where_190812, *[result_le_190821, subscript_call_result_190828], **kwargs_190831)
    
    # Assigning a type to the variable 'mb' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'mb', masked_where_call_result_190832)
    
    # Assigning a BinOp to a Name (line 209):
    
    # Assigning a BinOp to a Name (line 209):
    # Getting the type of 'mb' (line 209)
    mb_190833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'mb')
    # Getting the type of 'ma' (line 209)
    ma_190834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'ma')
    # Applying the binary operator 'div' (line 209)
    result_div_190835 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 8), 'div', mb_190833, ma_190834)
    
    # Assigning a type to the variable 'q' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'q', result_div_190835)
    
    # Obtaining an instance of the builtin type 'tuple' (line 210)
    tuple_190836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of 'True' (line 210)
    True_190837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 11), tuple_190836, True_190837)
    # Adding element type (line 210)
    
    # Obtaining the type of the subscript
    int_190838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 46), 'int')
    
    # Obtaining the type of the subscript
    int_190839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 43), 'int')
    
    # Call to where(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Getting the type of 'q' (line 210)
    q_190843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 29), 'q', False)
    
    # Call to min(...): (line 210)
    # Processing the call keyword arguments (line 210)
    kwargs_190846 = {}
    # Getting the type of 'q' (line 210)
    q_190844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 34), 'q', False)
    # Obtaining the member 'min' of a type (line 210)
    min_190845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 34), q_190844, 'min')
    # Calling min(args, kwargs) (line 210)
    min_call_result_190847 = invoke(stypy.reporting.localization.Localization(__file__, 210, 34), min_190845, *[], **kwargs_190846)
    
    # Applying the binary operator '==' (line 210)
    result_eq_190848 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 29), '==', q_190843, min_call_result_190847)
    
    # Processing the call keyword arguments (line 210)
    kwargs_190849 = {}
    # Getting the type of 'np' (line 210)
    np_190840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'np', False)
    # Obtaining the member 'ma' of a type (line 210)
    ma_190841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 17), np_190840, 'ma')
    # Obtaining the member 'where' of a type (line 210)
    where_190842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 17), ma_190841, 'where')
    # Calling where(args, kwargs) (line 210)
    where_call_result_190850 = invoke(stypy.reporting.localization.Localization(__file__, 210, 17), where_190842, *[result_eq_190848], **kwargs_190849)
    
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___190851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 17), where_call_result_190850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_190852 = invoke(stypy.reporting.localization.Localization(__file__, 210, 17), getitem___190851, int_190839)
    
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___190853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 17), subscript_call_result_190852, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_190854 = invoke(stypy.reporting.localization.Localization(__file__, 210, 17), getitem___190853, int_190838)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 11), tuple_190836, subscript_call_result_190854)
    
    # Assigning a type to the variable 'stypy_return_type' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type', tuple_190836)
    
    # ################# End of '_pivot_row(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_pivot_row' in the type store
    # Getting the type of 'stypy_return_type' (line 174)
    stypy_return_type_190855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190855)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_pivot_row'
    return stypy_return_type_190855

# Assigning a type to the variable '_pivot_row' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), '_pivot_row', _pivot_row)

@norecursion
def _solve_simplex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_190856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 40), 'int')
    int_190857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 52), 'int')
    # Getting the type of 'None' (line 213)
    None_190858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 64), 'None')
    float_190859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 23), 'float')
    int_190860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 37), 'int')
    # Getting the type of 'False' (line 214)
    False_190861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 46), 'False')
    defaults = [int_190856, int_190857, None_190858, float_190859, int_190860, False_190861]
    # Create a new context for function '_solve_simplex'
    module_type_store = module_type_store.open_function_context('_solve_simplex', 213, 0, False)
    
    # Passed parameters checking function
    _solve_simplex.stypy_localization = localization
    _solve_simplex.stypy_type_of_self = None
    _solve_simplex.stypy_type_store = module_type_store
    _solve_simplex.stypy_function_name = '_solve_simplex'
    _solve_simplex.stypy_param_names_list = ['T', 'n', 'basis', 'maxiter', 'phase', 'callback', 'tol', 'nit0', 'bland']
    _solve_simplex.stypy_varargs_param_name = None
    _solve_simplex.stypy_kwargs_param_name = None
    _solve_simplex.stypy_call_defaults = defaults
    _solve_simplex.stypy_call_varargs = varargs
    _solve_simplex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_solve_simplex', ['T', 'n', 'basis', 'maxiter', 'phase', 'callback', 'tol', 'nit0', 'bland'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_solve_simplex', localization, ['T', 'n', 'basis', 'maxiter', 'phase', 'callback', 'tol', 'nit0', 'bland'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_solve_simplex(...)' code ##################

    str_190862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, (-1)), 'str', '\n    Solve a linear programming problem in "standard maximization form" using\n    the Simplex Method.\n\n    Minimize :math:`f = c^T x`\n\n    subject to\n\n    .. math::\n\n        Ax = b\n        x_i >= 0\n        b_j >= 0\n\n    Parameters\n    ----------\n    T : array_like\n        A 2-D array representing the simplex T corresponding to the\n        maximization problem.  It should have the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],    0]]\n\n        for a Phase 2 problem, or the form:\n\n        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],\n         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],\n         .\n         .\n         .\n         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],\n         [c[0],   c[1], ...,   c[n_total],   0],\n         [c\'[0],  c\'[1], ...,  c\'[n_total],  0]]\n\n         for a Phase 1 problem (a Problem in which a basic feasible solution is\n         sought prior to maximizing the actual objective.  T is modified in\n         place by _solve_simplex.\n    n : int\n        The number of true variables in the problem.\n    basis : array\n        An array of the indices of the basic variables, such that basis[i]\n        contains the column corresponding to the basic variable for row i.\n        Basis is modified in place by _solve_simplex\n    maxiter : int\n        The maximum number of iterations to perform before aborting the\n        optimization.\n    phase : int\n        The phase of the optimization being executed.  In phase 1 a basic\n        feasible solution is sought and the T has an additional row\n        representing an alternate objective function.\n    callback : callable, optional\n        If a callback function is provided, it will be called within each\n        iteration of the simplex algorithm. The callback must have the\n        signature `callback(xk, **kwargs)` where xk is the current solution\n        vector and kwargs is a dictionary containing the following::\n        "T" : The current Simplex algorithm T\n        "nit" : The current iteration.\n        "pivot" : The pivot (row, column) used for the next iteration.\n        "phase" : Whether the algorithm is in Phase 1 or Phase 2.\n        "basis" : The indices of the columns of the basic variables.\n    tol : float\n        The tolerance which determines when a solution is "close enough" to\n        zero in Phase 1 to be considered a basic feasible solution or close\n        enough to positive to serve as an optimal solution.\n    nit0 : int\n        The initial iteration number used to keep an accurate iteration total\n        in a two-phase problem.\n    bland : bool\n        If True, choose pivots using Bland\'s rule [3].  In problems which\n        fail to converge due to cycling, using Bland\'s rule can provide\n        convergence at the expense of a less optimal path about the simplex.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a ``OptimizeResult`` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully and\n        ``message`` which describes the cause of the termination. Possible\n        values for the ``status`` attribute are:\n         0 : Optimization terminated successfully\n         1 : Iteration limit reached\n         2 : Problem appears to be infeasible\n         3 : Problem appears to be unbounded\n\n        See `OptimizeResult` for a description of other attributes.\n    ')
    
    # Assigning a Name to a Name (line 307):
    
    # Assigning a Name to a Name (line 307):
    # Getting the type of 'nit0' (line 307)
    nit0_190863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 10), 'nit0')
    # Assigning a type to the variable 'nit' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'nit', nit0_190863)
    
    # Assigning a Name to a Name (line 308):
    
    # Assigning a Name to a Name (line 308):
    # Getting the type of 'False' (line 308)
    False_190864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 15), 'False')
    # Assigning a type to the variable 'complete' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'complete', False_190864)
    
    
    # Getting the type of 'phase' (line 310)
    phase_190865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 7), 'phase')
    int_190866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 16), 'int')
    # Applying the binary operator '==' (line 310)
    result_eq_190867 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 7), '==', phase_190865, int_190866)
    
    # Testing the type of an if condition (line 310)
    if_condition_190868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 4), result_eq_190867)
    # Assigning a type to the variable 'if_condition_190868' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'if_condition_190868', if_condition_190868)
    # SSA begins for if statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 311):
    
    # Assigning a BinOp to a Name (line 311):
    
    # Obtaining the type of the subscript
    int_190869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 20), 'int')
    # Getting the type of 'T' (line 311)
    T_190870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'T')
    # Obtaining the member 'shape' of a type (line 311)
    shape_190871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), T_190870, 'shape')
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___190872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), shape_190871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_190873 = invoke(stypy.reporting.localization.Localization(__file__, 311, 12), getitem___190872, int_190869)
    
    int_190874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 23), 'int')
    # Applying the binary operator '-' (line 311)
    result_sub_190875 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 12), '-', subscript_call_result_190873, int_190874)
    
    # Assigning a type to the variable 'm' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'm', result_sub_190875)
    # SSA branch for the else part of an if statement (line 310)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'phase' (line 312)
    phase_190876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 9), 'phase')
    int_190877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 18), 'int')
    # Applying the binary operator '==' (line 312)
    result_eq_190878 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 9), '==', phase_190876, int_190877)
    
    # Testing the type of an if condition (line 312)
    if_condition_190879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 9), result_eq_190878)
    # Assigning a type to the variable 'if_condition_190879' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 9), 'if_condition_190879', if_condition_190879)
    # SSA begins for if statement (line 312)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 313):
    
    # Assigning a BinOp to a Name (line 313):
    
    # Obtaining the type of the subscript
    int_190880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 20), 'int')
    # Getting the type of 'T' (line 313)
    T_190881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'T')
    # Obtaining the member 'shape' of a type (line 313)
    shape_190882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 12), T_190881, 'shape')
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___190883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 12), shape_190882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_190884 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), getitem___190883, int_190880)
    
    int_190885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 23), 'int')
    # Applying the binary operator '-' (line 313)
    result_sub_190886 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 12), '-', subscript_call_result_190884, int_190885)
    
    # Assigning a type to the variable 'm' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'm', result_sub_190886)
    # SSA branch for the else part of an if statement (line 312)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 315)
    # Processing the call arguments (line 315)
    str_190888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 25), 'str', "Argument 'phase' to _solve_simplex must be 1 or 2")
    # Processing the call keyword arguments (line 315)
    kwargs_190889 = {}
    # Getting the type of 'ValueError' (line 315)
    ValueError_190887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 315)
    ValueError_call_result_190890 = invoke(stypy.reporting.localization.Localization(__file__, 315, 14), ValueError_190887, *[str_190888], **kwargs_190889)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 315, 8), ValueError_call_result_190890, 'raise parameter', BaseException)
    # SSA join for if statement (line 312)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'phase' (line 317)
    phase_190891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 7), 'phase')
    int_190892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 16), 'int')
    # Applying the binary operator '==' (line 317)
    result_eq_190893 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 7), '==', phase_190891, int_190892)
    
    # Testing the type of an if condition (line 317)
    if_condition_190894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 4), result_eq_190893)
    # Assigning a type to the variable 'if_condition_190894' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'if_condition_190894', if_condition_190894)
    # SSA begins for if statement (line 317)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'basis' (line 326)
    basis_190909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 44), 'basis', False)
    # Obtaining the member 'size' of a type (line 326)
    size_190910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 44), basis_190909, 'size')
    # Processing the call keyword arguments (line 326)
    kwargs_190911 = {}
    # Getting the type of 'range' (line 326)
    range_190908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 38), 'range', False)
    # Calling range(args, kwargs) (line 326)
    range_call_result_190912 = invoke(stypy.reporting.localization.Localization(__file__, 326, 38), range_190908, *[size_190910], **kwargs_190911)
    
    comprehension_190913 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 23), range_call_result_190912)
    # Assigning a type to the variable 'row' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 23), 'row', comprehension_190913)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 327)
    row_190896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), 'row')
    # Getting the type of 'basis' (line 327)
    basis_190897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'basis')
    # Obtaining the member '__getitem__' of a type (line 327)
    getitem___190898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 26), basis_190897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 327)
    subscript_call_result_190899 = invoke(stypy.reporting.localization.Localization(__file__, 327, 26), getitem___190898, row_190896)
    
    
    # Obtaining the type of the subscript
    int_190900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 47), 'int')
    # Getting the type of 'T' (line 327)
    T_190901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 39), 'T')
    # Obtaining the member 'shape' of a type (line 327)
    shape_190902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 39), T_190901, 'shape')
    # Obtaining the member '__getitem__' of a type (line 327)
    getitem___190903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 39), shape_190902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 327)
    subscript_call_result_190904 = invoke(stypy.reporting.localization.Localization(__file__, 327, 39), getitem___190903, int_190900)
    
    int_190905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 52), 'int')
    # Applying the binary operator '-' (line 327)
    result_sub_190906 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 39), '-', subscript_call_result_190904, int_190905)
    
    # Applying the binary operator '>' (line 327)
    result_gt_190907 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 26), '>', subscript_call_result_190899, result_sub_190906)
    
    # Getting the type of 'row' (line 326)
    row_190895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 23), 'row')
    list_190914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 23), list_190914, row_190895)
    # Testing the type of a for loop iterable (line 326)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 326, 8), list_190914)
    # Getting the type of the for loop variable (line 326)
    for_loop_var_190915 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 326, 8), list_190914)
    # Assigning a type to the variable 'pivrow' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'pivrow', for_loop_var_190915)
    # SSA begins for a for statement (line 326)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a ListComp to a Name (line 328):
    
    # Assigning a ListComp to a Name (line 328):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 328)
    # Processing the call arguments (line 328)
    
    # Obtaining the type of the subscript
    int_190926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 57), 'int')
    # Getting the type of 'T' (line 328)
    T_190927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 49), 'T', False)
    # Obtaining the member 'shape' of a type (line 328)
    shape_190928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 49), T_190927, 'shape')
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___190929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 49), shape_190928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_190930 = invoke(stypy.reporting.localization.Localization(__file__, 328, 49), getitem___190929, int_190926)
    
    int_190931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 62), 'int')
    # Applying the binary operator '-' (line 328)
    result_sub_190932 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 49), '-', subscript_call_result_190930, int_190931)
    
    # Processing the call keyword arguments (line 328)
    kwargs_190933 = {}
    # Getting the type of 'range' (line 328)
    range_190925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 43), 'range', False)
    # Calling range(args, kwargs) (line 328)
    range_call_result_190934 = invoke(stypy.reporting.localization.Localization(__file__, 328, 43), range_190925, *[result_sub_190932], **kwargs_190933)
    
    comprehension_190935 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 28), range_call_result_190934)
    # Assigning a type to the variable 'col' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'col', comprehension_190935)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 329)
    tuple_190917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 329)
    # Adding element type (line 329)
    # Getting the type of 'pivrow' (line 329)
    pivrow_190918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'pivrow')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 33), tuple_190917, pivrow_190918)
    # Adding element type (line 329)
    # Getting the type of 'col' (line 329)
    col_190919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 41), 'col')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 33), tuple_190917, col_190919)
    
    # Getting the type of 'T' (line 329)
    T_190920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 31), 'T')
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___190921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 31), T_190920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 329)
    subscript_call_result_190922 = invoke(stypy.reporting.localization.Localization(__file__, 329, 31), getitem___190921, tuple_190917)
    
    int_190923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 49), 'int')
    # Applying the binary operator '!=' (line 329)
    result_ne_190924 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 31), '!=', subscript_call_result_190922, int_190923)
    
    # Getting the type of 'col' (line 328)
    col_190916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'col')
    list_190936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 28), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 28), list_190936, col_190916)
    # Assigning a type to the variable 'non_zero_row' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'non_zero_row', list_190936)
    
    
    
    # Call to len(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'non_zero_row' (line 330)
    non_zero_row_190938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'non_zero_row', False)
    # Processing the call keyword arguments (line 330)
    kwargs_190939 = {}
    # Getting the type of 'len' (line 330)
    len_190937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'len', False)
    # Calling len(args, kwargs) (line 330)
    len_call_result_190940 = invoke(stypy.reporting.localization.Localization(__file__, 330, 15), len_190937, *[non_zero_row_190938], **kwargs_190939)
    
    int_190941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 35), 'int')
    # Applying the binary operator '>' (line 330)
    result_gt_190942 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 15), '>', len_call_result_190940, int_190941)
    
    # Testing the type of an if condition (line 330)
    if_condition_190943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 12), result_gt_190942)
    # Assigning a type to the variable 'if_condition_190943' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'if_condition_190943', if_condition_190943)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 331):
    
    # Assigning a Subscript to a Name (line 331):
    
    # Obtaining the type of the subscript
    int_190944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 38), 'int')
    # Getting the type of 'non_zero_row' (line 331)
    non_zero_row_190945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 25), 'non_zero_row')
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___190946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 25), non_zero_row_190945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_190947 = invoke(stypy.reporting.localization.Localization(__file__, 331, 25), getitem___190946, int_190944)
    
    # Assigning a type to the variable 'pivcol' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'pivcol', subscript_call_result_190947)
    
    # Assigning a Name to a Subscript (line 334):
    
    # Assigning a Name to a Subscript (line 334):
    # Getting the type of 'pivcol' (line 334)
    pivcol_190948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'pivcol')
    # Getting the type of 'basis' (line 334)
    basis_190949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'basis')
    # Getting the type of 'pivrow' (line 334)
    pivrow_190950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 22), 'pivrow')
    # Storing an element on a container (line 334)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 16), basis_190949, (pivrow_190950, pivcol_190948))
    
    # Assigning a Subscript to a Name (line 335):
    
    # Assigning a Subscript to a Name (line 335):
    
    # Obtaining the type of the subscript
    # Getting the type of 'pivcol' (line 335)
    pivcol_190951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 35), 'pivcol')
    
    # Obtaining the type of the subscript
    # Getting the type of 'pivrow' (line 335)
    pivrow_190952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 27), 'pivrow')
    # Getting the type of 'T' (line 335)
    T_190953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'T')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___190954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 25), T_190953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_190955 = invoke(stypy.reporting.localization.Localization(__file__, 335, 25), getitem___190954, pivrow_190952)
    
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___190956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 25), subscript_call_result_190955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_190957 = invoke(stypy.reporting.localization.Localization(__file__, 335, 25), getitem___190956, pivcol_190951)
    
    # Assigning a type to the variable 'pivval' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'pivval', subscript_call_result_190957)
    
    # Assigning a BinOp to a Subscript (line 336):
    
    # Assigning a BinOp to a Subscript (line 336):
    
    # Obtaining the type of the subscript
    # Getting the type of 'pivrow' (line 336)
    pivrow_190958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 33), 'pivrow')
    slice_190959 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 336, 31), None, None, None)
    # Getting the type of 'T' (line 336)
    T_190960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 31), 'T')
    # Obtaining the member '__getitem__' of a type (line 336)
    getitem___190961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 31), T_190960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 336)
    subscript_call_result_190962 = invoke(stypy.reporting.localization.Localization(__file__, 336, 31), getitem___190961, (pivrow_190958, slice_190959))
    
    # Getting the type of 'pivval' (line 336)
    pivval_190963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 46), 'pivval')
    # Applying the binary operator 'div' (line 336)
    result_div_190964 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 31), 'div', subscript_call_result_190962, pivval_190963)
    
    # Getting the type of 'T' (line 336)
    T_190965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'T')
    # Getting the type of 'pivrow' (line 336)
    pivrow_190966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 18), 'pivrow')
    slice_190967 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 336, 16), None, None, None)
    # Storing an element on a container (line 336)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 16), T_190965, ((pivrow_190966, slice_190967), result_div_190964))
    
    
    # Call to range(...): (line 337)
    # Processing the call arguments (line 337)
    
    # Obtaining the type of the subscript
    int_190969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 42), 'int')
    # Getting the type of 'T' (line 337)
    T_190970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 34), 'T', False)
    # Obtaining the member 'shape' of a type (line 337)
    shape_190971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 34), T_190970, 'shape')
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___190972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 34), shape_190971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_190973 = invoke(stypy.reporting.localization.Localization(__file__, 337, 34), getitem___190972, int_190969)
    
    # Processing the call keyword arguments (line 337)
    kwargs_190974 = {}
    # Getting the type of 'range' (line 337)
    range_190968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 28), 'range', False)
    # Calling range(args, kwargs) (line 337)
    range_call_result_190975 = invoke(stypy.reporting.localization.Localization(__file__, 337, 28), range_190968, *[subscript_call_result_190973], **kwargs_190974)
    
    # Testing the type of a for loop iterable (line 337)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 337, 16), range_call_result_190975)
    # Getting the type of the for loop variable (line 337)
    for_loop_var_190976 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 337, 16), range_call_result_190975)
    # Assigning a type to the variable 'irow' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'irow', for_loop_var_190976)
    # SSA begins for a for statement (line 337)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'irow' (line 338)
    irow_190977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 23), 'irow')
    # Getting the type of 'pivrow' (line 338)
    pivrow_190978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 31), 'pivrow')
    # Applying the binary operator '!=' (line 338)
    result_ne_190979 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 23), '!=', irow_190977, pivrow_190978)
    
    # Testing the type of an if condition (line 338)
    if_condition_190980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 20), result_ne_190979)
    # Assigning a type to the variable 'if_condition_190980' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 20), 'if_condition_190980', if_condition_190980)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 339):
    
    # Assigning a BinOp to a Subscript (line 339):
    
    # Obtaining the type of the subscript
    # Getting the type of 'irow' (line 339)
    irow_190981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 39), 'irow')
    slice_190982 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 339, 37), None, None, None)
    # Getting the type of 'T' (line 339)
    T_190983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 37), 'T')
    # Obtaining the member '__getitem__' of a type (line 339)
    getitem___190984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 37), T_190983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 339)
    subscript_call_result_190985 = invoke(stypy.reporting.localization.Localization(__file__, 339, 37), getitem___190984, (irow_190981, slice_190982))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'pivrow' (line 339)
    pivrow_190986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 52), 'pivrow')
    slice_190987 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 339, 50), None, None, None)
    # Getting the type of 'T' (line 339)
    T_190988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 50), 'T')
    # Obtaining the member '__getitem__' of a type (line 339)
    getitem___190989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 50), T_190988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 339)
    subscript_call_result_190990 = invoke(stypy.reporting.localization.Localization(__file__, 339, 50), getitem___190989, (pivrow_190986, slice_190987))
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 339)
    tuple_190991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 65), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 339)
    # Adding element type (line 339)
    # Getting the type of 'irow' (line 339)
    irow_190992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 65), 'irow')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 65), tuple_190991, irow_190992)
    # Adding element type (line 339)
    # Getting the type of 'pivcol' (line 339)
    pivcol_190993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 71), 'pivcol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 65), tuple_190991, pivcol_190993)
    
    # Getting the type of 'T' (line 339)
    T_190994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 63), 'T')
    # Obtaining the member '__getitem__' of a type (line 339)
    getitem___190995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 63), T_190994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 339)
    subscript_call_result_190996 = invoke(stypy.reporting.localization.Localization(__file__, 339, 63), getitem___190995, tuple_190991)
    
    # Applying the binary operator '*' (line 339)
    result_mul_190997 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 50), '*', subscript_call_result_190990, subscript_call_result_190996)
    
    # Applying the binary operator '-' (line 339)
    result_sub_190998 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 37), '-', subscript_call_result_190985, result_mul_190997)
    
    # Getting the type of 'T' (line 339)
    T_190999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'T')
    # Getting the type of 'irow' (line 339)
    irow_191000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 26), 'irow')
    slice_191001 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 339, 24), None, None, None)
    # Storing an element on a container (line 339)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 24), T_190999, ((irow_191000, slice_191001), result_sub_190998))
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'nit' (line 340)
    nit_191002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'nit')
    int_191003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 23), 'int')
    # Applying the binary operator '+=' (line 340)
    result_iadd_191004 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 16), '+=', nit_191002, int_191003)
    # Assigning a type to the variable 'nit' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'nit', result_iadd_191004)
    
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 317)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 342)
    # Processing the call arguments (line 342)
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 342)
    m_191006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 18), 'm', False)
    slice_191007 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 342, 11), None, m_191006, None)
    # Getting the type of 'basis' (line 342)
    basis_191008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 11), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___191009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 11), basis_191008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_191010 = invoke(stypy.reporting.localization.Localization(__file__, 342, 11), getitem___191009, slice_191007)
    
    # Processing the call keyword arguments (line 342)
    kwargs_191011 = {}
    # Getting the type of 'len' (line 342)
    len_191005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 7), 'len', False)
    # Calling len(args, kwargs) (line 342)
    len_call_result_191012 = invoke(stypy.reporting.localization.Localization(__file__, 342, 7), len_191005, *[subscript_call_result_191010], **kwargs_191011)
    
    int_191013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 25), 'int')
    # Applying the binary operator '==' (line 342)
    result_eq_191014 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 7), '==', len_call_result_191012, int_191013)
    
    # Testing the type of an if condition (line 342)
    if_condition_191015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 4), result_eq_191014)
    # Assigning a type to the variable 'if_condition_191015' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'if_condition_191015', if_condition_191015)
    # SSA begins for if statement (line 342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 343):
    
    # Assigning a Call to a Name (line 343):
    
    # Call to zeros(...): (line 343)
    # Processing the call arguments (line 343)
    
    # Obtaining the type of the subscript
    int_191018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 36), 'int')
    # Getting the type of 'T' (line 343)
    T_191019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 28), 'T', False)
    # Obtaining the member 'shape' of a type (line 343)
    shape_191020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 28), T_191019, 'shape')
    # Obtaining the member '__getitem__' of a type (line 343)
    getitem___191021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 28), shape_191020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 343)
    subscript_call_result_191022 = invoke(stypy.reporting.localization.Localization(__file__, 343, 28), getitem___191021, int_191018)
    
    int_191023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 41), 'int')
    # Applying the binary operator '-' (line 343)
    result_sub_191024 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 28), '-', subscript_call_result_191022, int_191023)
    
    # Processing the call keyword arguments (line 343)
    # Getting the type of 'np' (line 343)
    np_191025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 50), 'np', False)
    # Obtaining the member 'float64' of a type (line 343)
    float64_191026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 50), np_191025, 'float64')
    keyword_191027 = float64_191026
    kwargs_191028 = {'dtype': keyword_191027}
    # Getting the type of 'np' (line 343)
    np_191016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 343)
    zeros_191017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 19), np_191016, 'zeros')
    # Calling zeros(args, kwargs) (line 343)
    zeros_call_result_191029 = invoke(stypy.reporting.localization.Localization(__file__, 343, 19), zeros_191017, *[result_sub_191024], **kwargs_191028)
    
    # Assigning a type to the variable 'solution' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'solution', zeros_call_result_191029)
    # SSA branch for the else part of an if statement (line 342)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 345):
    
    # Assigning a Call to a Name (line 345):
    
    # Call to zeros(...): (line 345)
    # Processing the call arguments (line 345)
    
    # Call to max(...): (line 345)
    # Processing the call arguments (line 345)
    
    # Obtaining the type of the subscript
    int_191033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 40), 'int')
    # Getting the type of 'T' (line 345)
    T_191034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 32), 'T', False)
    # Obtaining the member 'shape' of a type (line 345)
    shape_191035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 32), T_191034, 'shape')
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___191036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 32), shape_191035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_191037 = invoke(stypy.reporting.localization.Localization(__file__, 345, 32), getitem___191036, int_191033)
    
    int_191038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 45), 'int')
    # Applying the binary operator '-' (line 345)
    result_sub_191039 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 32), '-', subscript_call_result_191037, int_191038)
    
    
    # Call to max(...): (line 345)
    # Processing the call arguments (line 345)
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 345)
    m_191041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 59), 'm', False)
    slice_191042 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 345, 52), None, m_191041, None)
    # Getting the type of 'basis' (line 345)
    basis_191043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 52), 'basis', False)
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___191044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 52), basis_191043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_191045 = invoke(stypy.reporting.localization.Localization(__file__, 345, 52), getitem___191044, slice_191042)
    
    # Processing the call keyword arguments (line 345)
    kwargs_191046 = {}
    # Getting the type of 'max' (line 345)
    max_191040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 48), 'max', False)
    # Calling max(args, kwargs) (line 345)
    max_call_result_191047 = invoke(stypy.reporting.localization.Localization(__file__, 345, 48), max_191040, *[subscript_call_result_191045], **kwargs_191046)
    
    int_191048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 65), 'int')
    # Applying the binary operator '+' (line 345)
    result_add_191049 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 48), '+', max_call_result_191047, int_191048)
    
    # Processing the call keyword arguments (line 345)
    kwargs_191050 = {}
    # Getting the type of 'max' (line 345)
    max_191032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 28), 'max', False)
    # Calling max(args, kwargs) (line 345)
    max_call_result_191051 = invoke(stypy.reporting.localization.Localization(__file__, 345, 28), max_191032, *[result_sub_191039, result_add_191049], **kwargs_191050)
    
    # Processing the call keyword arguments (line 345)
    # Getting the type of 'np' (line 346)
    np_191052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 34), 'np', False)
    # Obtaining the member 'float64' of a type (line 346)
    float64_191053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 34), np_191052, 'float64')
    keyword_191054 = float64_191053
    kwargs_191055 = {'dtype': keyword_191054}
    # Getting the type of 'np' (line 345)
    np_191030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 345)
    zeros_191031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 19), np_191030, 'zeros')
    # Calling zeros(args, kwargs) (line 345)
    zeros_call_result_191056 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), zeros_191031, *[max_call_result_191051], **kwargs_191055)
    
    # Assigning a type to the variable 'solution' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'solution', zeros_call_result_191056)
    # SSA join for if statement (line 342)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'complete' (line 348)
    complete_191057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 14), 'complete')
    # Applying the 'not' unary operator (line 348)
    result_not__191058 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 10), 'not', complete_191057)
    
    # Testing the type of an if condition (line 348)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 4), result_not__191058)
    # SSA begins for while statement (line 348)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 350):
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    int_191059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
    
    # Call to _pivot_col(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'T' (line 350)
    T_191061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 42), 'T', False)
    # Getting the type of 'tol' (line 350)
    tol_191062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 45), 'tol', False)
    # Getting the type of 'bland' (line 350)
    bland_191063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 50), 'bland', False)
    # Processing the call keyword arguments (line 350)
    kwargs_191064 = {}
    # Getting the type of '_pivot_col' (line 350)
    _pivot_col_191060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 31), '_pivot_col', False)
    # Calling _pivot_col(args, kwargs) (line 350)
    _pivot_col_call_result_191065 = invoke(stypy.reporting.localization.Localization(__file__, 350, 31), _pivot_col_191060, *[T_191061, tol_191062, bland_191063], **kwargs_191064)
    
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___191066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), _pivot_col_call_result_191065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_191067 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___191066, int_191059)
    
    # Assigning a type to the variable 'tuple_var_assignment_190475' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_190475', subscript_call_result_191067)
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    int_191068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
    
    # Call to _pivot_col(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'T' (line 350)
    T_191070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 42), 'T', False)
    # Getting the type of 'tol' (line 350)
    tol_191071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 45), 'tol', False)
    # Getting the type of 'bland' (line 350)
    bland_191072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 50), 'bland', False)
    # Processing the call keyword arguments (line 350)
    kwargs_191073 = {}
    # Getting the type of '_pivot_col' (line 350)
    _pivot_col_191069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 31), '_pivot_col', False)
    # Calling _pivot_col(args, kwargs) (line 350)
    _pivot_col_call_result_191074 = invoke(stypy.reporting.localization.Localization(__file__, 350, 31), _pivot_col_191069, *[T_191070, tol_191071, bland_191072], **kwargs_191073)
    
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___191075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), _pivot_col_call_result_191074, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_191076 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___191075, int_191068)
    
    # Assigning a type to the variable 'tuple_var_assignment_190476' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_190476', subscript_call_result_191076)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_var_assignment_190475' (line 350)
    tuple_var_assignment_190475_191077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_190475')
    # Assigning a type to the variable 'pivcol_found' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'pivcol_found', tuple_var_assignment_190475_191077)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_var_assignment_190476' (line 350)
    tuple_var_assignment_190476_191078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_190476')
    # Assigning a type to the variable 'pivcol' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'pivcol', tuple_var_assignment_190476_191078)
    
    
    # Getting the type of 'pivcol_found' (line 351)
    pivcol_found_191079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), 'pivcol_found')
    # Applying the 'not' unary operator (line 351)
    result_not__191080 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 11), 'not', pivcol_found_191079)
    
    # Testing the type of an if condition (line 351)
    if_condition_191081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 8), result_not__191080)
    # Assigning a type to the variable 'if_condition_191081' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'if_condition_191081', if_condition_191081)
    # SSA begins for if statement (line 351)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 352):
    
    # Assigning a Attribute to a Name (line 352):
    # Getting the type of 'np' (line 352)
    np_191082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 21), 'np')
    # Obtaining the member 'nan' of a type (line 352)
    nan_191083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 21), np_191082, 'nan')
    # Assigning a type to the variable 'pivcol' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'pivcol', nan_191083)
    
    # Assigning a Attribute to a Name (line 353):
    
    # Assigning a Attribute to a Name (line 353):
    # Getting the type of 'np' (line 353)
    np_191084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'np')
    # Obtaining the member 'nan' of a type (line 353)
    nan_191085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 21), np_191084, 'nan')
    # Assigning a type to the variable 'pivrow' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'pivrow', nan_191085)
    
    # Assigning a Num to a Name (line 354):
    
    # Assigning a Num to a Name (line 354):
    int_191086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 21), 'int')
    # Assigning a type to the variable 'status' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'status', int_191086)
    
    # Assigning a Name to a Name (line 355):
    
    # Assigning a Name to a Name (line 355):
    # Getting the type of 'True' (line 355)
    True_191087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 23), 'True')
    # Assigning a type to the variable 'complete' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'complete', True_191087)
    # SSA branch for the else part of an if statement (line 351)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 358):
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_191088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
    
    # Call to _pivot_row(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'T' (line 358)
    T_191090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 46), 'T', False)
    # Getting the type of 'pivcol' (line 358)
    pivcol_191091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 49), 'pivcol', False)
    # Getting the type of 'phase' (line 358)
    phase_191092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 57), 'phase', False)
    # Getting the type of 'tol' (line 358)
    tol_191093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 64), 'tol', False)
    # Processing the call keyword arguments (line 358)
    kwargs_191094 = {}
    # Getting the type of '_pivot_row' (line 358)
    _pivot_row_191089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 35), '_pivot_row', False)
    # Calling _pivot_row(args, kwargs) (line 358)
    _pivot_row_call_result_191095 = invoke(stypy.reporting.localization.Localization(__file__, 358, 35), _pivot_row_191089, *[T_191090, pivcol_191091, phase_191092, tol_191093], **kwargs_191094)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___191096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), _pivot_row_call_result_191095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_191097 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), getitem___191096, int_191088)
    
    # Assigning a type to the variable 'tuple_var_assignment_190477' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_190477', subscript_call_result_191097)
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_191098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'int')
    
    # Call to _pivot_row(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'T' (line 358)
    T_191100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 46), 'T', False)
    # Getting the type of 'pivcol' (line 358)
    pivcol_191101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 49), 'pivcol', False)
    # Getting the type of 'phase' (line 358)
    phase_191102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 57), 'phase', False)
    # Getting the type of 'tol' (line 358)
    tol_191103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 64), 'tol', False)
    # Processing the call keyword arguments (line 358)
    kwargs_191104 = {}
    # Getting the type of '_pivot_row' (line 358)
    _pivot_row_191099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 35), '_pivot_row', False)
    # Calling _pivot_row(args, kwargs) (line 358)
    _pivot_row_call_result_191105 = invoke(stypy.reporting.localization.Localization(__file__, 358, 35), _pivot_row_191099, *[T_191100, pivcol_191101, phase_191102, tol_191103], **kwargs_191104)
    
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___191106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), _pivot_row_call_result_191105, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_191107 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), getitem___191106, int_191098)
    
    # Assigning a type to the variable 'tuple_var_assignment_190478' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_190478', subscript_call_result_191107)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_190477' (line 358)
    tuple_var_assignment_190477_191108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_190477')
    # Assigning a type to the variable 'pivrow_found' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'pivrow_found', tuple_var_assignment_190477_191108)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_190478' (line 358)
    tuple_var_assignment_190478_191109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'tuple_var_assignment_190478')
    # Assigning a type to the variable 'pivrow' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 26), 'pivrow', tuple_var_assignment_190478_191109)
    
    
    # Getting the type of 'pivrow_found' (line 359)
    pivrow_found_191110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'pivrow_found')
    # Applying the 'not' unary operator (line 359)
    result_not__191111 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 15), 'not', pivrow_found_191110)
    
    # Testing the type of an if condition (line 359)
    if_condition_191112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 12), result_not__191111)
    # Assigning a type to the variable 'if_condition_191112' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'if_condition_191112', if_condition_191112)
    # SSA begins for if statement (line 359)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 360):
    
    # Assigning a Num to a Name (line 360):
    int_191113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 25), 'int')
    # Assigning a type to the variable 'status' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'status', int_191113)
    
    # Assigning a Name to a Name (line 361):
    
    # Assigning a Name to a Name (line 361):
    # Getting the type of 'True' (line 361)
    True_191114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'True')
    # Assigning a type to the variable 'complete' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'complete', True_191114)
    # SSA join for if statement (line 359)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 351)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 363)
    # Getting the type of 'callback' (line 363)
    callback_191115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'callback')
    # Getting the type of 'None' (line 363)
    None_191116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'None')
    
    (may_be_191117, more_types_in_union_191118) = may_not_be_none(callback_191115, None_191116)

    if may_be_191117:

        if more_types_in_union_191118:
            # Runtime conditional SSA (line 363)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Subscript (line 364):
        
        # Assigning a Num to a Subscript (line 364):
        int_191119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 26), 'int')
        # Getting the type of 'solution' (line 364)
        solution_191120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'solution')
        slice_191121 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 364, 12), None, None, None)
        # Storing an element on a container (line 364)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 12), solution_191120, (slice_191121, int_191119))
        
        # Assigning a Subscript to a Subscript (line 365):
        
        # Assigning a Subscript to a Subscript (line 365):
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 365)
        m_191122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 37), 'm')
        slice_191123 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 365, 34), None, m_191122, None)
        int_191124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 40), 'int')
        # Getting the type of 'T' (line 365)
        T_191125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 34), 'T')
        # Obtaining the member '__getitem__' of a type (line 365)
        getitem___191126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 34), T_191125, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 365)
        subscript_call_result_191127 = invoke(stypy.reporting.localization.Localization(__file__, 365, 34), getitem___191126, (slice_191123, int_191124))
        
        # Getting the type of 'solution' (line 365)
        solution_191128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'solution')
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 365)
        m_191129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 28), 'm')
        slice_191130 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 365, 21), None, m_191129, None)
        # Getting the type of 'basis' (line 365)
        basis_191131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 21), 'basis')
        # Obtaining the member '__getitem__' of a type (line 365)
        getitem___191132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 21), basis_191131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 365)
        subscript_call_result_191133 = invoke(stypy.reporting.localization.Localization(__file__, 365, 21), getitem___191132, slice_191130)
        
        # Storing an element on a container (line 365)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 12), solution_191128, (subscript_call_result_191133, subscript_call_result_191127))
        
        # Call to callback(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 366)
        n_191135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 31), 'n', False)
        slice_191136 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 366, 21), None, n_191135, None)
        # Getting the type of 'solution' (line 366)
        solution_191137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 21), 'solution', False)
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___191138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 21), solution_191137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_191139 = invoke(stypy.reporting.localization.Localization(__file__, 366, 21), getitem___191138, slice_191136)
        
        # Processing the call keyword arguments (line 366)
        
        # Obtaining an instance of the builtin type 'dict' (line 366)
        dict_191140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 37), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 366)
        # Adding element type (key, value) (line 366)
        str_191141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 38), 'str', 'tableau')
        # Getting the type of 'T' (line 366)
        T_191142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 49), 'T', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 37), dict_191140, (str_191141, T_191142))
        # Adding element type (key, value) (line 366)
        str_191143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 38), 'str', 'phase')
        # Getting the type of 'phase' (line 367)
        phase_191144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 47), 'phase', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 37), dict_191140, (str_191143, phase_191144))
        # Adding element type (key, value) (line 366)
        str_191145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 38), 'str', 'nit')
        # Getting the type of 'nit' (line 368)
        nit_191146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 45), 'nit', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 37), dict_191140, (str_191145, nit_191146))
        # Adding element type (key, value) (line 366)
        str_191147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 38), 'str', 'pivot')
        
        # Obtaining an instance of the builtin type 'tuple' (line 369)
        tuple_191148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 369)
        # Adding element type (line 369)
        # Getting the type of 'pivrow' (line 369)
        pivrow_191149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 48), 'pivrow', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 48), tuple_191148, pivrow_191149)
        # Adding element type (line 369)
        # Getting the type of 'pivcol' (line 369)
        pivcol_191150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 56), 'pivcol', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 48), tuple_191148, pivcol_191150)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 37), dict_191140, (str_191147, tuple_191148))
        # Adding element type (key, value) (line 366)
        str_191151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 38), 'str', 'basis')
        # Getting the type of 'basis' (line 370)
        basis_191152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 47), 'basis', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 37), dict_191140, (str_191151, basis_191152))
        # Adding element type (key, value) (line 366)
        str_191153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 38), 'str', 'complete')
        
        # Evaluating a boolean operation
        # Getting the type of 'complete' (line 371)
        complete_191154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 50), 'complete', False)
        
        # Getting the type of 'phase' (line 371)
        phase_191155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 63), 'phase', False)
        int_191156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 72), 'int')
        # Applying the binary operator '==' (line 371)
        result_eq_191157 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 63), '==', phase_191155, int_191156)
        
        # Applying the binary operator 'and' (line 371)
        result_and_keyword_191158 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 50), 'and', complete_191154, result_eq_191157)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 37), dict_191140, (str_191153, result_and_keyword_191158))
        
        kwargs_191159 = {'dict_191140': dict_191140}
        # Getting the type of 'callback' (line 366)
        callback_191134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'callback', False)
        # Calling callback(args, kwargs) (line 366)
        callback_call_result_191160 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), callback_191134, *[subscript_call_result_191139], **kwargs_191159)
        

        if more_types_in_union_191118:
            # SSA join for if statement (line 363)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'complete' (line 373)
    complete_191161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'complete')
    # Applying the 'not' unary operator (line 373)
    result_not__191162 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 11), 'not', complete_191161)
    
    # Testing the type of an if condition (line 373)
    if_condition_191163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 8), result_not__191162)
    # Assigning a type to the variable 'if_condition_191163' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'if_condition_191163', if_condition_191163)
    # SSA begins for if statement (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'nit' (line 374)
    nit_191164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'nit')
    # Getting the type of 'maxiter' (line 374)
    maxiter_191165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 22), 'maxiter')
    # Applying the binary operator '>=' (line 374)
    result_ge_191166 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 15), '>=', nit_191164, maxiter_191165)
    
    # Testing the type of an if condition (line 374)
    if_condition_191167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 12), result_ge_191166)
    # Assigning a type to the variable 'if_condition_191167' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'if_condition_191167', if_condition_191167)
    # SSA begins for if statement (line 374)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 376):
    
    # Assigning a Num to a Name (line 376):
    int_191168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 25), 'int')
    # Assigning a type to the variable 'status' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'status', int_191168)
    
    # Assigning a Name to a Name (line 377):
    
    # Assigning a Name to a Name (line 377):
    # Getting the type of 'True' (line 377)
    True_191169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 27), 'True')
    # Assigning a type to the variable 'complete' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'complete', True_191169)
    # SSA branch for the else part of an if statement (line 374)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 381):
    
    # Assigning a Name to a Subscript (line 381):
    # Getting the type of 'pivcol' (line 381)
    pivcol_191170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 32), 'pivcol')
    # Getting the type of 'basis' (line 381)
    basis_191171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'basis')
    # Getting the type of 'pivrow' (line 381)
    pivrow_191172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 22), 'pivrow')
    # Storing an element on a container (line 381)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 16), basis_191171, (pivrow_191172, pivcol_191170))
    
    # Assigning a Subscript to a Name (line 382):
    
    # Assigning a Subscript to a Name (line 382):
    
    # Obtaining the type of the subscript
    # Getting the type of 'pivcol' (line 382)
    pivcol_191173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 35), 'pivcol')
    
    # Obtaining the type of the subscript
    # Getting the type of 'pivrow' (line 382)
    pivrow_191174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 27), 'pivrow')
    # Getting the type of 'T' (line 382)
    T_191175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 25), 'T')
    # Obtaining the member '__getitem__' of a type (line 382)
    getitem___191176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 25), T_191175, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 382)
    subscript_call_result_191177 = invoke(stypy.reporting.localization.Localization(__file__, 382, 25), getitem___191176, pivrow_191174)
    
    # Obtaining the member '__getitem__' of a type (line 382)
    getitem___191178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 25), subscript_call_result_191177, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 382)
    subscript_call_result_191179 = invoke(stypy.reporting.localization.Localization(__file__, 382, 25), getitem___191178, pivcol_191173)
    
    # Assigning a type to the variable 'pivval' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'pivval', subscript_call_result_191179)
    
    # Assigning a BinOp to a Subscript (line 383):
    
    # Assigning a BinOp to a Subscript (line 383):
    
    # Obtaining the type of the subscript
    # Getting the type of 'pivrow' (line 383)
    pivrow_191180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 33), 'pivrow')
    slice_191181 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 383, 31), None, None, None)
    # Getting the type of 'T' (line 383)
    T_191182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 31), 'T')
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___191183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 31), T_191182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_191184 = invoke(stypy.reporting.localization.Localization(__file__, 383, 31), getitem___191183, (pivrow_191180, slice_191181))
    
    # Getting the type of 'pivval' (line 383)
    pivval_191185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 46), 'pivval')
    # Applying the binary operator 'div' (line 383)
    result_div_191186 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 31), 'div', subscript_call_result_191184, pivval_191185)
    
    # Getting the type of 'T' (line 383)
    T_191187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'T')
    # Getting the type of 'pivrow' (line 383)
    pivrow_191188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 18), 'pivrow')
    slice_191189 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 383, 16), None, None, None)
    # Storing an element on a container (line 383)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 16), T_191187, ((pivrow_191188, slice_191189), result_div_191186))
    
    
    # Call to range(...): (line 384)
    # Processing the call arguments (line 384)
    
    # Obtaining the type of the subscript
    int_191191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 42), 'int')
    # Getting the type of 'T' (line 384)
    T_191192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 34), 'T', False)
    # Obtaining the member 'shape' of a type (line 384)
    shape_191193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 34), T_191192, 'shape')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___191194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 34), shape_191193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_191195 = invoke(stypy.reporting.localization.Localization(__file__, 384, 34), getitem___191194, int_191191)
    
    # Processing the call keyword arguments (line 384)
    kwargs_191196 = {}
    # Getting the type of 'range' (line 384)
    range_191190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), 'range', False)
    # Calling range(args, kwargs) (line 384)
    range_call_result_191197 = invoke(stypy.reporting.localization.Localization(__file__, 384, 28), range_191190, *[subscript_call_result_191195], **kwargs_191196)
    
    # Testing the type of a for loop iterable (line 384)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 384, 16), range_call_result_191197)
    # Getting the type of the for loop variable (line 384)
    for_loop_var_191198 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 384, 16), range_call_result_191197)
    # Assigning a type to the variable 'irow' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'irow', for_loop_var_191198)
    # SSA begins for a for statement (line 384)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'irow' (line 385)
    irow_191199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'irow')
    # Getting the type of 'pivrow' (line 385)
    pivrow_191200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 31), 'pivrow')
    # Applying the binary operator '!=' (line 385)
    result_ne_191201 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 23), '!=', irow_191199, pivrow_191200)
    
    # Testing the type of an if condition (line 385)
    if_condition_191202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 20), result_ne_191201)
    # Assigning a type to the variable 'if_condition_191202' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'if_condition_191202', if_condition_191202)
    # SSA begins for if statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 386):
    
    # Assigning a BinOp to a Subscript (line 386):
    
    # Obtaining the type of the subscript
    # Getting the type of 'irow' (line 386)
    irow_191203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 39), 'irow')
    slice_191204 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 386, 37), None, None, None)
    # Getting the type of 'T' (line 386)
    T_191205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 37), 'T')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___191206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 37), T_191205, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_191207 = invoke(stypy.reporting.localization.Localization(__file__, 386, 37), getitem___191206, (irow_191203, slice_191204))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'pivrow' (line 386)
    pivrow_191208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 52), 'pivrow')
    slice_191209 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 386, 50), None, None, None)
    # Getting the type of 'T' (line 386)
    T_191210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 50), 'T')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___191211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 50), T_191210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_191212 = invoke(stypy.reporting.localization.Localization(__file__, 386, 50), getitem___191211, (pivrow_191208, slice_191209))
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 386)
    tuple_191213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 65), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 386)
    # Adding element type (line 386)
    # Getting the type of 'irow' (line 386)
    irow_191214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 65), 'irow')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 65), tuple_191213, irow_191214)
    # Adding element type (line 386)
    # Getting the type of 'pivcol' (line 386)
    pivcol_191215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 71), 'pivcol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 65), tuple_191213, pivcol_191215)
    
    # Getting the type of 'T' (line 386)
    T_191216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 63), 'T')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___191217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 63), T_191216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_191218 = invoke(stypy.reporting.localization.Localization(__file__, 386, 63), getitem___191217, tuple_191213)
    
    # Applying the binary operator '*' (line 386)
    result_mul_191219 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 50), '*', subscript_call_result_191212, subscript_call_result_191218)
    
    # Applying the binary operator '-' (line 386)
    result_sub_191220 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 37), '-', subscript_call_result_191207, result_mul_191219)
    
    # Getting the type of 'T' (line 386)
    T_191221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'T')
    # Getting the type of 'irow' (line 386)
    irow_191222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 26), 'irow')
    slice_191223 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 386, 24), None, None, None)
    # Storing an element on a container (line 386)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 24), T_191221, ((irow_191222, slice_191223), result_sub_191220))
    # SSA join for if statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'nit' (line 387)
    nit_191224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'nit')
    int_191225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 23), 'int')
    # Applying the binary operator '+=' (line 387)
    result_iadd_191226 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 16), '+=', nit_191224, int_191225)
    # Assigning a type to the variable 'nit' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'nit', result_iadd_191226)
    
    # SSA join for if statement (line 374)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 373)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 348)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 389)
    tuple_191227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 389)
    # Adding element type (line 389)
    # Getting the type of 'nit' (line 389)
    nit_191228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'nit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 11), tuple_191227, nit_191228)
    # Adding element type (line 389)
    # Getting the type of 'status' (line 389)
    status_191229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'status')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 11), tuple_191227, status_191229)
    
    # Assigning a type to the variable 'stypy_return_type' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'stypy_return_type', tuple_191227)
    
    # ################# End of '_solve_simplex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_solve_simplex' in the type store
    # Getting the type of 'stypy_return_type' (line 213)
    stypy_return_type_191230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_191230)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_solve_simplex'
    return stypy_return_type_191230

# Assigning a type to the variable '_solve_simplex' (line 213)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), '_solve_simplex', _solve_simplex)

@norecursion
def _linprog_simplex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 392)
    None_191231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 29), 'None')
    # Getting the type of 'None' (line 392)
    None_191232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 40), 'None')
    # Getting the type of 'None' (line 392)
    None_191233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 51), 'None')
    # Getting the type of 'None' (line 392)
    None_191234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 62), 'None')
    # Getting the type of 'None' (line 393)
    None_191235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 28), 'None')
    int_191236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 42), 'int')
    # Getting the type of 'False' (line 393)
    False_191237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 53), 'False')
    # Getting the type of 'None' (line 393)
    None_191238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 69), 'None')
    float_191239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 25), 'float')
    # Getting the type of 'False' (line 394)
    False_191240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 40), 'False')
    defaults = [None_191231, None_191232, None_191233, None_191234, None_191235, int_191236, False_191237, None_191238, float_191239, False_191240]
    # Create a new context for function '_linprog_simplex'
    module_type_store = module_type_store.open_function_context('_linprog_simplex', 392, 0, False)
    
    # Passed parameters checking function
    _linprog_simplex.stypy_localization = localization
    _linprog_simplex.stypy_type_of_self = None
    _linprog_simplex.stypy_type_store = module_type_store
    _linprog_simplex.stypy_function_name = '_linprog_simplex'
    _linprog_simplex.stypy_param_names_list = ['c', 'A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds', 'maxiter', 'disp', 'callback', 'tol', 'bland']
    _linprog_simplex.stypy_varargs_param_name = None
    _linprog_simplex.stypy_kwargs_param_name = 'unknown_options'
    _linprog_simplex.stypy_call_defaults = defaults
    _linprog_simplex.stypy_call_varargs = varargs
    _linprog_simplex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_linprog_simplex', ['c', 'A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds', 'maxiter', 'disp', 'callback', 'tol', 'bland'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_linprog_simplex', localization, ['c', 'A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds', 'maxiter', 'disp', 'callback', 'tol', 'bland'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_linprog_simplex(...)' code ##################

    str_191241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, (-1)), 'str', '\n    Solve the following linear programming problem via a two-phase\n    simplex algorithm.::\n\n        minimize:     c^T * x\n\n        subject to:   A_ub * x <= b_ub\n                      A_eq * x == b_eq\n\n    Parameters\n    ----------\n    c : array_like\n        Coefficients of the linear objective function to be minimized.\n    A_ub : array_like\n        2-D array which, when matrix-multiplied by ``x``, gives the values of\n        the upper-bound inequality constraints at ``x``.\n    b_ub : array_like\n        1-D array of values representing the upper-bound of each inequality\n        constraint (row) in ``A_ub``.\n    A_eq : array_like\n        2-D array which, when matrix-multiplied by ``x``, gives the values of\n        the equality constraints at ``x``.\n    b_eq : array_like\n        1-D array of values representing the RHS of each equality constraint\n        (row) in ``A_eq``.\n    bounds : array_like\n        The bounds for each independent variable in the solution, which can\n        take one of three forms::\n\n        None : The default bounds, all variables are non-negative.\n        (lb, ub) : If a 2-element sequence is provided, the same\n                  lower bound (lb) and upper bound (ub) will be applied\n                  to all variables.\n        [(lb_0, ub_0), (lb_1, ub_1), ...] : If an n x 2 sequence is provided,\n                  each variable x_i will be bounded by lb[i] and ub[i].\n        Infinite bounds are specified using -np.inf (negative)\n        or np.inf (positive).\n\n    callback : callable\n        If a callback function is provide, it will be called within each\n        iteration of the simplex algorithm. The callback must have the\n        signature ``callback(xk, **kwargs)`` where ``xk`` is the current s\n        olution vector and kwargs is a dictionary containing the following::\n\n        "tableau" : The current Simplex algorithm tableau\n        "nit" : The current iteration.\n        "pivot" : The pivot (row, column) used for the next iteration.\n        "phase" : Whether the algorithm is in Phase 1 or Phase 2.\n        "bv" : A structured array containing a string representation of each\n               basic variable and its current value.\n\n    Options\n    -------\n    maxiter : int\n       The maximum number of iterations to perform.\n    disp : bool\n        If True, print exit status message to sys.stdout\n    tol : float\n        The tolerance which determines when a solution is "close enough" to\n        zero in Phase 1 to be considered a basic feasible solution or close\n        enough to positive to serve as an optimal solution.\n    bland : bool\n        If True, use Bland\'s anti-cycling rule [3] to choose pivots to\n        prevent cycling.  If False, choose pivots which should lead to a\n        converged solution more quickly.  The latter method is subject to\n        cycling (non-convergence) in rare instances.\n\n    Returns\n    -------\n    A `scipy.optimize.OptimizeResult` consisting of the following fields:\n\n        x : ndarray\n            The independent variable vector which optimizes the linear\n            programming problem.\n        fun : float\n            Value of the objective function.\n        slack : ndarray\n            The values of the slack variables.  Each slack variable corresponds\n            to an inequality constraint.  If the slack is zero, then the\n            corresponding constraint is active.\n        success : bool\n            Returns True if the algorithm succeeded in finding an optimal\n            solution.\n        status : int\n            An integer representing the exit status of the optimization::\n\n             0 : Optimization terminated successfully\n             1 : Iteration limit reached\n             2 : Problem appears to be infeasible\n             3 : Problem appears to be unbounded\n\n        nit : int\n            The number of iterations performed.\n        message : str\n            A string descriptor of the exit status of the optimization.\n\n    Examples\n    --------\n    Consider the following problem:\n\n    Minimize: f = -1*x[0] + 4*x[1]\n\n    Subject to: -3*x[0] + 1*x[1] <= 6\n                 1*x[0] + 2*x[1] <= 4\n                            x[1] >= -3\n\n    where:  -inf <= x[0] <= inf\n\n    This problem deviates from the standard linear programming problem.  In\n    standard form, linear programming problems assume the variables x are\n    non-negative.  Since the variables don\'t have standard bounds where\n    0 <= x <= inf, the bounds of the variables must be explicitly set.\n\n    There are two upper-bound constraints, which can be expressed as\n\n    dot(A_ub, x) <= b_ub\n\n    The input for this problem is as follows:\n\n    >>> from scipy.optimize import linprog\n    >>> c = [-1, 4]\n    >>> A = [[-3, 1], [1, 2]]\n    >>> b = [6, 4]\n    >>> x0_bnds = (None, None)\n    >>> x1_bnds = (-3, None)\n    >>> res = linprog(c, A, b, bounds=(x0_bnds, x1_bnds))\n    >>> print(res)\n         fun: -22.0\n     message: \'Optimization terminated successfully.\'\n         nit: 1\n       slack: array([ 39.,   0.])\n      status: 0\n     success: True\n           x: array([ 10.,  -3.])\n\n    References\n    ----------\n    .. [1] Dantzig, George B., Linear programming and extensions. Rand\n           Corporation Research Study Princeton Univ. Press, Princeton, NJ,\n           1963\n    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to\n           Mathematical Programming", McGraw-Hill, Chapter 4.\n    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.\n           Mathematics of Operations Research (2), 1977: pp. 103-107.\n    ')
    
    # Call to _check_unknown_options(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'unknown_options' (line 540)
    unknown_options_191243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 540)
    kwargs_191244 = {}
    # Getting the type of '_check_unknown_options' (line 540)
    _check_unknown_options_191242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 540)
    _check_unknown_options_call_result_191245 = invoke(stypy.reporting.localization.Localization(__file__, 540, 4), _check_unknown_options_191242, *[unknown_options_191243], **kwargs_191244)
    
    
    # Assigning a Num to a Name (line 542):
    
    # Assigning a Num to a Name (line 542):
    int_191246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 13), 'int')
    # Assigning a type to the variable 'status' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'status', int_191246)
    
    # Assigning a Dict to a Name (line 543):
    
    # Assigning a Dict to a Name (line 543):
    
    # Obtaining an instance of the builtin type 'dict' (line 543)
    dict_191247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 543)
    # Adding element type (key, value) (line 543)
    int_191248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 16), 'int')
    str_191249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 19), 'str', 'Optimization terminated successfully.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 15), dict_191247, (int_191248, str_191249))
    # Adding element type (key, value) (line 543)
    int_191250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 16), 'int')
    str_191251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 19), 'str', 'Iteration limit reached.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 15), dict_191247, (int_191250, str_191251))
    # Adding element type (key, value) (line 543)
    int_191252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 16), 'int')
    str_191253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 19), 'str', 'Optimization failed. Unable to find a feasible starting point.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 15), dict_191247, (int_191252, str_191253))
    # Adding element type (key, value) (line 543)
    int_191254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 16), 'int')
    str_191255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 19), 'str', 'Optimization failed. The problem appears to be unbounded.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 15), dict_191247, (int_191254, str_191255))
    # Adding element type (key, value) (line 543)
    int_191256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 16), 'int')
    str_191257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 19), 'str', 'Optimization failed. Singular matrix encountered.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 15), dict_191247, (int_191256, str_191257))
    
    # Assigning a type to the variable 'messages' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'messages', dict_191247)
    
    # Assigning a Name to a Name (line 549):
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'False' (line 549)
    False_191258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 26), 'False')
    # Assigning a type to the variable 'have_floor_variable' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'have_floor_variable', False_191258)
    
    # Assigning a Call to a Name (line 551):
    
    # Assigning a Call to a Name (line 551):
    
    # Call to asarray(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'c' (line 551)
    c_191261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'c', False)
    # Processing the call keyword arguments (line 551)
    kwargs_191262 = {}
    # Getting the type of 'np' (line 551)
    np_191259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 9), 'np', False)
    # Obtaining the member 'asarray' of a type (line 551)
    asarray_191260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 9), np_191259, 'asarray')
    # Calling asarray(args, kwargs) (line 551)
    asarray_call_result_191263 = invoke(stypy.reporting.localization.Localization(__file__, 551, 9), asarray_191260, *[c_191261], **kwargs_191262)
    
    # Assigning a type to the variable 'cc' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'cc', asarray_call_result_191263)
    
    # Assigning a Num to a Name (line 554):
    
    # Assigning a Num to a Name (line 554):
    int_191264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 9), 'int')
    # Assigning a type to the variable 'f0' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'f0', int_191264)
    
    # Assigning a Call to a Name (line 557):
    
    # Assigning a Call to a Name (line 557):
    
    # Call to len(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'c' (line 557)
    c_191266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'c', False)
    # Processing the call keyword arguments (line 557)
    kwargs_191267 = {}
    # Getting the type of 'len' (line 557)
    len_191265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'len', False)
    # Calling len(args, kwargs) (line 557)
    len_call_result_191268 = invoke(stypy.reporting.localization.Localization(__file__, 557, 8), len_191265, *[c_191266], **kwargs_191267)
    
    # Assigning a type to the variable 'n' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'n', len_call_result_191268)
    
    # Assigning a IfExp to a Name (line 560):
    
    # Assigning a IfExp to a Name (line 560):
    
    
    # Getting the type of 'A_eq' (line 560)
    A_eq_191269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 30), 'A_eq')
    # Getting the type of 'None' (line 560)
    None_191270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'None')
    # Applying the binary operator 'isnot' (line 560)
    result_is_not_191271 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 30), 'isnot', A_eq_191269, None_191270)
    
    # Testing the type of an if expression (line 560)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 10), result_is_not_191271)
    # SSA begins for if expression (line 560)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to asarray(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'A_eq' (line 560)
    A_eq_191274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 21), 'A_eq', False)
    # Processing the call keyword arguments (line 560)
    kwargs_191275 = {}
    # Getting the type of 'np' (line 560)
    np_191272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 560)
    asarray_191273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 10), np_191272, 'asarray')
    # Calling asarray(args, kwargs) (line 560)
    asarray_call_result_191276 = invoke(stypy.reporting.localization.Localization(__file__, 560, 10), asarray_191273, *[A_eq_191274], **kwargs_191275)
    
    # SSA branch for the else part of an if expression (line 560)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to empty(...): (line 560)
    # Processing the call arguments (line 560)
    
    # Obtaining an instance of the builtin type 'list' (line 560)
    list_191279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 560)
    # Adding element type (line 560)
    int_191280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 61), list_191279, int_191280)
    # Adding element type (line 560)
    
    # Call to len(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'cc' (line 560)
    cc_191282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 69), 'cc', False)
    # Processing the call keyword arguments (line 560)
    kwargs_191283 = {}
    # Getting the type of 'len' (line 560)
    len_191281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 65), 'len', False)
    # Calling len(args, kwargs) (line 560)
    len_call_result_191284 = invoke(stypy.reporting.localization.Localization(__file__, 560, 65), len_191281, *[cc_191282], **kwargs_191283)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 61), list_191279, len_call_result_191284)
    
    # Processing the call keyword arguments (line 560)
    kwargs_191285 = {}
    # Getting the type of 'np' (line 560)
    np_191277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 52), 'np', False)
    # Obtaining the member 'empty' of a type (line 560)
    empty_191278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 52), np_191277, 'empty')
    # Calling empty(args, kwargs) (line 560)
    empty_call_result_191286 = invoke(stypy.reporting.localization.Localization(__file__, 560, 52), empty_191278, *[list_191279], **kwargs_191285)
    
    # SSA join for if expression (line 560)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_191287 = union_type.UnionType.add(asarray_call_result_191276, empty_call_result_191286)
    
    # Assigning a type to the variable 'Aeq' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'Aeq', if_exp_191287)
    
    # Assigning a IfExp to a Name (line 561):
    
    # Assigning a IfExp to a Name (line 561):
    
    
    # Getting the type of 'A_ub' (line 561)
    A_ub_191288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 30), 'A_ub')
    # Getting the type of 'None' (line 561)
    None_191289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 42), 'None')
    # Applying the binary operator 'isnot' (line 561)
    result_is_not_191290 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 30), 'isnot', A_ub_191288, None_191289)
    
    # Testing the type of an if expression (line 561)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 561, 10), result_is_not_191290)
    # SSA begins for if expression (line 561)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to asarray(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'A_ub' (line 561)
    A_ub_191293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 21), 'A_ub', False)
    # Processing the call keyword arguments (line 561)
    kwargs_191294 = {}
    # Getting the type of 'np' (line 561)
    np_191291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 561)
    asarray_191292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 10), np_191291, 'asarray')
    # Calling asarray(args, kwargs) (line 561)
    asarray_call_result_191295 = invoke(stypy.reporting.localization.Localization(__file__, 561, 10), asarray_191292, *[A_ub_191293], **kwargs_191294)
    
    # SSA branch for the else part of an if expression (line 561)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to empty(...): (line 561)
    # Processing the call arguments (line 561)
    
    # Obtaining an instance of the builtin type 'list' (line 561)
    list_191298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 561)
    # Adding element type (line 561)
    int_191299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 61), list_191298, int_191299)
    # Adding element type (line 561)
    
    # Call to len(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'cc' (line 561)
    cc_191301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 69), 'cc', False)
    # Processing the call keyword arguments (line 561)
    kwargs_191302 = {}
    # Getting the type of 'len' (line 561)
    len_191300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 65), 'len', False)
    # Calling len(args, kwargs) (line 561)
    len_call_result_191303 = invoke(stypy.reporting.localization.Localization(__file__, 561, 65), len_191300, *[cc_191301], **kwargs_191302)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 61), list_191298, len_call_result_191303)
    
    # Processing the call keyword arguments (line 561)
    kwargs_191304 = {}
    # Getting the type of 'np' (line 561)
    np_191296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 52), 'np', False)
    # Obtaining the member 'empty' of a type (line 561)
    empty_191297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 52), np_191296, 'empty')
    # Calling empty(args, kwargs) (line 561)
    empty_call_result_191305 = invoke(stypy.reporting.localization.Localization(__file__, 561, 52), empty_191297, *[list_191298], **kwargs_191304)
    
    # SSA join for if expression (line 561)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_191306 = union_type.UnionType.add(asarray_call_result_191295, empty_call_result_191305)
    
    # Assigning a type to the variable 'Aub' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'Aub', if_exp_191306)
    
    # Assigning a IfExp to a Name (line 562):
    
    # Assigning a IfExp to a Name (line 562):
    
    
    # Getting the type of 'b_eq' (line 562)
    b_eq_191307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 40), 'b_eq')
    # Getting the type of 'None' (line 562)
    None_191308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 52), 'None')
    # Applying the binary operator 'isnot' (line 562)
    result_is_not_191309 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 40), 'isnot', b_eq_191307, None_191308)
    
    # Testing the type of an if expression (line 562)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 10), result_is_not_191309)
    # SSA begins for if expression (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to ravel(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Call to asarray(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'b_eq' (line 562)
    b_eq_191314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 30), 'b_eq', False)
    # Processing the call keyword arguments (line 562)
    kwargs_191315 = {}
    # Getting the type of 'np' (line 562)
    np_191312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 19), 'np', False)
    # Obtaining the member 'asarray' of a type (line 562)
    asarray_191313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 19), np_191312, 'asarray')
    # Calling asarray(args, kwargs) (line 562)
    asarray_call_result_191316 = invoke(stypy.reporting.localization.Localization(__file__, 562, 19), asarray_191313, *[b_eq_191314], **kwargs_191315)
    
    # Processing the call keyword arguments (line 562)
    kwargs_191317 = {}
    # Getting the type of 'np' (line 562)
    np_191310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 10), 'np', False)
    # Obtaining the member 'ravel' of a type (line 562)
    ravel_191311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 10), np_191310, 'ravel')
    # Calling ravel(args, kwargs) (line 562)
    ravel_call_result_191318 = invoke(stypy.reporting.localization.Localization(__file__, 562, 10), ravel_191311, *[asarray_call_result_191316], **kwargs_191317)
    
    # SSA branch for the else part of an if expression (line 562)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to empty(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Obtaining an instance of the builtin type 'list' (line 562)
    list_191321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 71), 'list')
    # Adding type elements to the builtin type 'list' instance (line 562)
    # Adding element type (line 562)
    int_191322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 71), list_191321, int_191322)
    
    # Processing the call keyword arguments (line 562)
    kwargs_191323 = {}
    # Getting the type of 'np' (line 562)
    np_191319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 62), 'np', False)
    # Obtaining the member 'empty' of a type (line 562)
    empty_191320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 62), np_191319, 'empty')
    # Calling empty(args, kwargs) (line 562)
    empty_call_result_191324 = invoke(stypy.reporting.localization.Localization(__file__, 562, 62), empty_191320, *[list_191321], **kwargs_191323)
    
    # SSA join for if expression (line 562)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_191325 = union_type.UnionType.add(ravel_call_result_191318, empty_call_result_191324)
    
    # Assigning a type to the variable 'beq' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'beq', if_exp_191325)
    
    # Assigning a IfExp to a Name (line 563):
    
    # Assigning a IfExp to a Name (line 563):
    
    
    # Getting the type of 'b_ub' (line 563)
    b_ub_191326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 40), 'b_ub')
    # Getting the type of 'None' (line 563)
    None_191327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 52), 'None')
    # Applying the binary operator 'isnot' (line 563)
    result_is_not_191328 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 40), 'isnot', b_ub_191326, None_191327)
    
    # Testing the type of an if expression (line 563)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 563, 10), result_is_not_191328)
    # SSA begins for if expression (line 563)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to ravel(...): (line 563)
    # Processing the call arguments (line 563)
    
    # Call to asarray(...): (line 563)
    # Processing the call arguments (line 563)
    # Getting the type of 'b_ub' (line 563)
    b_ub_191333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 30), 'b_ub', False)
    # Processing the call keyword arguments (line 563)
    kwargs_191334 = {}
    # Getting the type of 'np' (line 563)
    np_191331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 19), 'np', False)
    # Obtaining the member 'asarray' of a type (line 563)
    asarray_191332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 19), np_191331, 'asarray')
    # Calling asarray(args, kwargs) (line 563)
    asarray_call_result_191335 = invoke(stypy.reporting.localization.Localization(__file__, 563, 19), asarray_191332, *[b_ub_191333], **kwargs_191334)
    
    # Processing the call keyword arguments (line 563)
    kwargs_191336 = {}
    # Getting the type of 'np' (line 563)
    np_191329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 10), 'np', False)
    # Obtaining the member 'ravel' of a type (line 563)
    ravel_191330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 10), np_191329, 'ravel')
    # Calling ravel(args, kwargs) (line 563)
    ravel_call_result_191337 = invoke(stypy.reporting.localization.Localization(__file__, 563, 10), ravel_191330, *[asarray_call_result_191335], **kwargs_191336)
    
    # SSA branch for the else part of an if expression (line 563)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to empty(...): (line 563)
    # Processing the call arguments (line 563)
    
    # Obtaining an instance of the builtin type 'list' (line 563)
    list_191340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 71), 'list')
    # Adding type elements to the builtin type 'list' instance (line 563)
    # Adding element type (line 563)
    int_191341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 71), list_191340, int_191341)
    
    # Processing the call keyword arguments (line 563)
    kwargs_191342 = {}
    # Getting the type of 'np' (line 563)
    np_191338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 62), 'np', False)
    # Obtaining the member 'empty' of a type (line 563)
    empty_191339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 62), np_191338, 'empty')
    # Calling empty(args, kwargs) (line 563)
    empty_call_result_191343 = invoke(stypy.reporting.localization.Localization(__file__, 563, 62), empty_191339, *[list_191340], **kwargs_191342)
    
    # SSA join for if expression (line 563)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_191344 = union_type.UnionType.add(ravel_call_result_191337, empty_call_result_191343)
    
    # Assigning a type to the variable 'bub' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'bub', if_exp_191344)
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to zeros(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'n' (line 567)
    n_191347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 17), 'n', False)
    # Processing the call keyword arguments (line 567)
    # Getting the type of 'np' (line 567)
    np_191348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 26), 'np', False)
    # Obtaining the member 'float64' of a type (line 567)
    float64_191349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 26), np_191348, 'float64')
    keyword_191350 = float64_191349
    kwargs_191351 = {'dtype': keyword_191350}
    # Getting the type of 'np' (line 567)
    np_191345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 567)
    zeros_191346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 8), np_191345, 'zeros')
    # Calling zeros(args, kwargs) (line 567)
    zeros_call_result_191352 = invoke(stypy.reporting.localization.Localization(__file__, 567, 8), zeros_191346, *[n_191347], **kwargs_191351)
    
    # Assigning a type to the variable 'L' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'L', zeros_call_result_191352)
    
    # Assigning a BinOp to a Name (line 568):
    
    # Assigning a BinOp to a Name (line 568):
    
    # Call to ones(...): (line 568)
    # Processing the call arguments (line 568)
    # Getting the type of 'n' (line 568)
    n_191355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 16), 'n', False)
    # Processing the call keyword arguments (line 568)
    # Getting the type of 'np' (line 568)
    np_191356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 25), 'np', False)
    # Obtaining the member 'float64' of a type (line 568)
    float64_191357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 25), np_191356, 'float64')
    keyword_191358 = float64_191357
    kwargs_191359 = {'dtype': keyword_191358}
    # Getting the type of 'np' (line 568)
    np_191353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 568)
    ones_191354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 8), np_191353, 'ones')
    # Calling ones(args, kwargs) (line 568)
    ones_call_result_191360 = invoke(stypy.reporting.localization.Localization(__file__, 568, 8), ones_191354, *[n_191355], **kwargs_191359)
    
    # Getting the type of 'np' (line 568)
    np_191361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 37), 'np')
    # Obtaining the member 'inf' of a type (line 568)
    inf_191362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 37), np_191361, 'inf')
    # Applying the binary operator '*' (line 568)
    result_mul_191363 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 8), '*', ones_call_result_191360, inf_191362)
    
    # Assigning a type to the variable 'U' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'U', result_mul_191363)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'bounds' (line 569)
    bounds_191364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 7), 'bounds')
    # Getting the type of 'None' (line 569)
    None_191365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 17), 'None')
    # Applying the binary operator 'is' (line 569)
    result_is__191366 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 7), 'is', bounds_191364, None_191365)
    
    
    
    # Call to len(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'bounds' (line 569)
    bounds_191368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 29), 'bounds', False)
    # Processing the call keyword arguments (line 569)
    kwargs_191369 = {}
    # Getting the type of 'len' (line 569)
    len_191367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 25), 'len', False)
    # Calling len(args, kwargs) (line 569)
    len_call_result_191370 = invoke(stypy.reporting.localization.Localization(__file__, 569, 25), len_191367, *[bounds_191368], **kwargs_191369)
    
    int_191371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 40), 'int')
    # Applying the binary operator '==' (line 569)
    result_eq_191372 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 25), '==', len_call_result_191370, int_191371)
    
    # Applying the binary operator 'or' (line 569)
    result_or_keyword_191373 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 7), 'or', result_is__191366, result_eq_191372)
    
    # Testing the type of an if condition (line 569)
    if_condition_191374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 4), result_or_keyword_191373)
    # Assigning a type to the variable 'if_condition_191374' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'if_condition_191374', if_condition_191374)
    # SSA begins for if statement (line 569)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 569)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'bounds' (line 571)
    bounds_191376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 13), 'bounds', False)
    # Processing the call keyword arguments (line 571)
    kwargs_191377 = {}
    # Getting the type of 'len' (line 571)
    len_191375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 9), 'len', False)
    # Calling len(args, kwargs) (line 571)
    len_call_result_191378 = invoke(stypy.reporting.localization.Localization(__file__, 571, 9), len_191375, *[bounds_191376], **kwargs_191377)
    
    int_191379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 24), 'int')
    # Applying the binary operator '==' (line 571)
    result_eq_191380 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 9), '==', len_call_result_191378, int_191379)
    
    
    
    # Call to hasattr(...): (line 571)
    # Processing the call arguments (line 571)
    
    # Obtaining the type of the subscript
    int_191382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 49), 'int')
    # Getting the type of 'bounds' (line 571)
    bounds_191383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 42), 'bounds', False)
    # Obtaining the member '__getitem__' of a type (line 571)
    getitem___191384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 42), bounds_191383, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 571)
    subscript_call_result_191385 = invoke(stypy.reporting.localization.Localization(__file__, 571, 42), getitem___191384, int_191382)
    
    str_191386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 53), 'str', '__len__')
    # Processing the call keyword arguments (line 571)
    kwargs_191387 = {}
    # Getting the type of 'hasattr' (line 571)
    hasattr_191381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 34), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 571)
    hasattr_call_result_191388 = invoke(stypy.reporting.localization.Localization(__file__, 571, 34), hasattr_191381, *[subscript_call_result_191385, str_191386], **kwargs_191387)
    
    # Applying the 'not' unary operator (line 571)
    result_not__191389 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 30), 'not', hasattr_call_result_191388)
    
    # Applying the binary operator 'and' (line 571)
    result_and_keyword_191390 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 9), 'and', result_eq_191380, result_not__191389)
    
    # Testing the type of an if condition (line 571)
    if_condition_191391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 9), result_and_keyword_191390)
    # Assigning a type to the variable 'if_condition_191391' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 9), 'if_condition_191391', if_condition_191391)
    # SSA begins for if statement (line 571)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a IfExp to a Name (line 573):
    
    # Assigning a IfExp to a Name (line 573):
    
    
    
    # Obtaining the type of the subscript
    int_191392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 32), 'int')
    # Getting the type of 'bounds' (line 573)
    bounds_191393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 25), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___191394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 25), bounds_191393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 573)
    subscript_call_result_191395 = invoke(stypy.reporting.localization.Localization(__file__, 573, 25), getitem___191394, int_191392)
    
    # Getting the type of 'None' (line 573)
    None_191396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 42), 'None')
    # Applying the binary operator 'isnot' (line 573)
    result_is_not_191397 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 25), 'isnot', subscript_call_result_191395, None_191396)
    
    # Testing the type of an if expression (line 573)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 12), result_is_not_191397)
    # SSA begins for if expression (line 573)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_191398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 19), 'int')
    # Getting the type of 'bounds' (line 573)
    bounds_191399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___191400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 12), bounds_191399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 573)
    subscript_call_result_191401 = invoke(stypy.reporting.localization.Localization(__file__, 573, 12), getitem___191400, int_191398)
    
    # SSA branch for the else part of an if expression (line 573)
    module_type_store.open_ssa_branch('if expression else')
    
    # Getting the type of 'np' (line 573)
    np_191402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 53), 'np')
    # Obtaining the member 'inf' of a type (line 573)
    inf_191403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 53), np_191402, 'inf')
    # Applying the 'usub' unary operator (line 573)
    result___neg___191404 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 52), 'usub', inf_191403)
    
    # SSA join for if expression (line 573)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_191405 = union_type.UnionType.add(subscript_call_result_191401, result___neg___191404)
    
    # Assigning a type to the variable 'a' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'a', if_exp_191405)
    
    # Assigning a IfExp to a Name (line 574):
    
    # Assigning a IfExp to a Name (line 574):
    
    
    
    # Obtaining the type of the subscript
    int_191406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 32), 'int')
    # Getting the type of 'bounds' (line 574)
    bounds_191407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 25), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___191408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 25), bounds_191407, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_191409 = invoke(stypy.reporting.localization.Localization(__file__, 574, 25), getitem___191408, int_191406)
    
    # Getting the type of 'None' (line 574)
    None_191410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'None')
    # Applying the binary operator 'isnot' (line 574)
    result_is_not_191411 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 25), 'isnot', subscript_call_result_191409, None_191410)
    
    # Testing the type of an if expression (line 574)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 12), result_is_not_191411)
    # SSA begins for if expression (line 574)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_191412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 19), 'int')
    # Getting the type of 'bounds' (line 574)
    bounds_191413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___191414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 12), bounds_191413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_191415 = invoke(stypy.reporting.localization.Localization(__file__, 574, 12), getitem___191414, int_191412)
    
    # SSA branch for the else part of an if expression (line 574)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'np' (line 574)
    np_191416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 52), 'np')
    # Obtaining the member 'inf' of a type (line 574)
    inf_191417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 52), np_191416, 'inf')
    # SSA join for if expression (line 574)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_191418 = union_type.UnionType.add(subscript_call_result_191415, inf_191417)
    
    # Assigning a type to the variable 'b' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'b', if_exp_191418)
    
    # Assigning a Call to a Name (line 575):
    
    # Assigning a Call to a Name (line 575):
    
    # Call to asarray(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'n' (line 575)
    n_191421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 'n', False)
    
    # Obtaining an instance of the builtin type 'list' (line 575)
    list_191422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 575)
    # Adding element type (line 575)
    # Getting the type of 'a' (line 575)
    a_191423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 26), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 25), list_191422, a_191423)
    
    # Applying the binary operator '*' (line 575)
    result_mul_191424 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 23), '*', n_191421, list_191422)
    
    # Processing the call keyword arguments (line 575)
    # Getting the type of 'np' (line 575)
    np_191425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 36), 'np', False)
    # Obtaining the member 'float64' of a type (line 575)
    float64_191426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 36), np_191425, 'float64')
    keyword_191427 = float64_191426
    kwargs_191428 = {'dtype': keyword_191427}
    # Getting the type of 'np' (line 575)
    np_191419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 575)
    asarray_191420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), np_191419, 'asarray')
    # Calling asarray(args, kwargs) (line 575)
    asarray_call_result_191429 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), asarray_191420, *[result_mul_191424], **kwargs_191428)
    
    # Assigning a type to the variable 'L' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'L', asarray_call_result_191429)
    
    # Assigning a Call to a Name (line 576):
    
    # Assigning a Call to a Name (line 576):
    
    # Call to asarray(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'n' (line 576)
    n_191432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 23), 'n', False)
    
    # Obtaining an instance of the builtin type 'list' (line 576)
    list_191433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 576)
    # Adding element type (line 576)
    # Getting the type of 'b' (line 576)
    b_191434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 26), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 25), list_191433, b_191434)
    
    # Applying the binary operator '*' (line 576)
    result_mul_191435 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 23), '*', n_191432, list_191433)
    
    # Processing the call keyword arguments (line 576)
    # Getting the type of 'np' (line 576)
    np_191436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 36), 'np', False)
    # Obtaining the member 'float64' of a type (line 576)
    float64_191437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 36), np_191436, 'float64')
    keyword_191438 = float64_191437
    kwargs_191439 = {'dtype': keyword_191438}
    # Getting the type of 'np' (line 576)
    np_191430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 576)
    asarray_191431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 12), np_191430, 'asarray')
    # Calling asarray(args, kwargs) (line 576)
    asarray_call_result_191440 = invoke(stypy.reporting.localization.Localization(__file__, 576, 12), asarray_191431, *[result_mul_191435], **kwargs_191439)
    
    # Assigning a type to the variable 'U' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'U', asarray_call_result_191440)
    # SSA branch for the else part of an if statement (line 571)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'bounds' (line 578)
    bounds_191442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 15), 'bounds', False)
    # Processing the call keyword arguments (line 578)
    kwargs_191443 = {}
    # Getting the type of 'len' (line 578)
    len_191441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 11), 'len', False)
    # Calling len(args, kwargs) (line 578)
    len_call_result_191444 = invoke(stypy.reporting.localization.Localization(__file__, 578, 11), len_191441, *[bounds_191442], **kwargs_191443)
    
    # Getting the type of 'n' (line 578)
    n_191445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 26), 'n')
    # Applying the binary operator '!=' (line 578)
    result_ne_191446 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 11), '!=', len_call_result_191444, n_191445)
    
    # Testing the type of an if condition (line 578)
    if_condition_191447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 578, 8), result_ne_191446)
    # Assigning a type to the variable 'if_condition_191447' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'if_condition_191447', if_condition_191447)
    # SSA begins for if statement (line 578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 579):
    
    # Assigning a Num to a Name (line 579):
    int_191448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 21), 'int')
    # Assigning a type to the variable 'status' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'status', int_191448)
    
    # Assigning a Str to a Name (line 580):
    
    # Assigning a Str to a Name (line 580):
    str_191449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 23), 'str', "Invalid input for linprog with method = 'simplex'.  Length of bounds is inconsistent with the length of c")
    # Assigning a type to the variable 'message' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'message', str_191449)
    # SSA branch for the else part of an if statement (line 578)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 583)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Call to range(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'n' (line 584)
    n_191451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 31), 'n', False)
    # Processing the call keyword arguments (line 584)
    kwargs_191452 = {}
    # Getting the type of 'range' (line 584)
    range_191450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 25), 'range', False)
    # Calling range(args, kwargs) (line 584)
    range_call_result_191453 = invoke(stypy.reporting.localization.Localization(__file__, 584, 25), range_191450, *[n_191451], **kwargs_191452)
    
    # Testing the type of a for loop iterable (line 584)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 584, 16), range_call_result_191453)
    # Getting the type of the for loop variable (line 584)
    for_loop_var_191454 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 584, 16), range_call_result_191453)
    # Assigning a type to the variable 'i' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'i', for_loop_var_191454)
    # SSA begins for a for statement (line 584)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to len(...): (line 585)
    # Processing the call arguments (line 585)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 585)
    i_191456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 34), 'i', False)
    # Getting the type of 'bounds' (line 585)
    bounds_191457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 27), 'bounds', False)
    # Obtaining the member '__getitem__' of a type (line 585)
    getitem___191458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 27), bounds_191457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 585)
    subscript_call_result_191459 = invoke(stypy.reporting.localization.Localization(__file__, 585, 27), getitem___191458, i_191456)
    
    # Processing the call keyword arguments (line 585)
    kwargs_191460 = {}
    # Getting the type of 'len' (line 585)
    len_191455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 23), 'len', False)
    # Calling len(args, kwargs) (line 585)
    len_call_result_191461 = invoke(stypy.reporting.localization.Localization(__file__, 585, 23), len_191455, *[subscript_call_result_191459], **kwargs_191460)
    
    int_191462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 41), 'int')
    # Applying the binary operator '!=' (line 585)
    result_ne_191463 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 23), '!=', len_call_result_191461, int_191462)
    
    # Testing the type of an if condition (line 585)
    if_condition_191464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 20), result_ne_191463)
    # Assigning a type to the variable 'if_condition_191464' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'if_condition_191464', if_condition_191464)
    # SSA begins for if statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to IndexError(...): (line 586)
    # Processing the call keyword arguments (line 586)
    kwargs_191466 = {}
    # Getting the type of 'IndexError' (line 586)
    IndexError_191465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 30), 'IndexError', False)
    # Calling IndexError(args, kwargs) (line 586)
    IndexError_call_result_191467 = invoke(stypy.reporting.localization.Localization(__file__, 586, 30), IndexError_191465, *[], **kwargs_191466)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 586, 24), IndexError_call_result_191467, 'raise parameter', BaseException)
    # SSA join for if statement (line 585)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a IfExp to a Subscript (line 587):
    
    # Assigning a IfExp to a Subscript (line 587):
    
    
    
    # Obtaining the type of the subscript
    int_191468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 53), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 587)
    i_191469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 50), 'i')
    # Getting the type of 'bounds' (line 587)
    bounds_191470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 43), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___191471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 43), bounds_191470, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_191472 = invoke(stypy.reporting.localization.Localization(__file__, 587, 43), getitem___191471, i_191469)
    
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___191473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 43), subscript_call_result_191472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_191474 = invoke(stypy.reporting.localization.Localization(__file__, 587, 43), getitem___191473, int_191468)
    
    # Getting the type of 'None' (line 587)
    None_191475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 63), 'None')
    # Applying the binary operator 'isnot' (line 587)
    result_is_not_191476 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 43), 'isnot', subscript_call_result_191474, None_191475)
    
    # Testing the type of an if expression (line 587)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 27), result_is_not_191476)
    # SSA begins for if expression (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_191477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 37), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 587)
    i_191478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 34), 'i')
    # Getting the type of 'bounds' (line 587)
    bounds_191479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 27), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___191480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 27), bounds_191479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_191481 = invoke(stypy.reporting.localization.Localization(__file__, 587, 27), getitem___191480, i_191478)
    
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___191482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 27), subscript_call_result_191481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_191483 = invoke(stypy.reporting.localization.Localization(__file__, 587, 27), getitem___191482, int_191477)
    
    # SSA branch for the else part of an if expression (line 587)
    module_type_store.open_ssa_branch('if expression else')
    
    # Getting the type of 'np' (line 587)
    np_191484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 74), 'np')
    # Obtaining the member 'inf' of a type (line 587)
    inf_191485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 74), np_191484, 'inf')
    # Applying the 'usub' unary operator (line 587)
    result___neg___191486 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 73), 'usub', inf_191485)
    
    # SSA join for if expression (line 587)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_191487 = union_type.UnionType.add(subscript_call_result_191483, result___neg___191486)
    
    # Getting the type of 'L' (line 587)
    L_191488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'L')
    # Getting the type of 'i' (line 587)
    i_191489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 22), 'i')
    # Storing an element on a container (line 587)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 20), L_191488, (i_191489, if_exp_191487))
    
    # Assigning a IfExp to a Subscript (line 588):
    
    # Assigning a IfExp to a Subscript (line 588):
    
    
    
    # Obtaining the type of the subscript
    int_191490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 53), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 588)
    i_191491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 50), 'i')
    # Getting the type of 'bounds' (line 588)
    bounds_191492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 43), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___191493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 43), bounds_191492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_191494 = invoke(stypy.reporting.localization.Localization(__file__, 588, 43), getitem___191493, i_191491)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___191495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 43), subscript_call_result_191494, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_191496 = invoke(stypy.reporting.localization.Localization(__file__, 588, 43), getitem___191495, int_191490)
    
    # Getting the type of 'None' (line 588)
    None_191497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 63), 'None')
    # Applying the binary operator 'isnot' (line 588)
    result_is_not_191498 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 43), 'isnot', subscript_call_result_191496, None_191497)
    
    # Testing the type of an if expression (line 588)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 588, 27), result_is_not_191498)
    # SSA begins for if expression (line 588)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_191499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 37), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 588)
    i_191500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 34), 'i')
    # Getting the type of 'bounds' (line 588)
    bounds_191501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 27), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___191502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 27), bounds_191501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_191503 = invoke(stypy.reporting.localization.Localization(__file__, 588, 27), getitem___191502, i_191500)
    
    # Obtaining the member '__getitem__' of a type (line 588)
    getitem___191504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 27), subscript_call_result_191503, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 588)
    subscript_call_result_191505 = invoke(stypy.reporting.localization.Localization(__file__, 588, 27), getitem___191504, int_191499)
    
    # SSA branch for the else part of an if expression (line 588)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'np' (line 588)
    np_191506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 73), 'np')
    # Obtaining the member 'inf' of a type (line 588)
    inf_191507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 73), np_191506, 'inf')
    # SSA join for if expression (line 588)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_191508 = union_type.UnionType.add(subscript_call_result_191505, inf_191507)
    
    # Getting the type of 'U' (line 588)
    U_191509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'U')
    # Getting the type of 'i' (line 588)
    i_191510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 22), 'i')
    # Storing an element on a container (line 588)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 20), U_191509, (i_191510, if_exp_191508))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 583)
    # SSA branch for the except 'IndexError' branch of a try statement (line 583)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 590):
    
    # Assigning a Num to a Name (line 590):
    int_191511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 25), 'int')
    # Assigning a type to the variable 'status' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 16), 'status', int_191511)
    
    # Assigning a Str to a Name (line 591):
    
    # Assigning a Str to a Name (line 591):
    str_191512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 27), 'str', "Invalid input for linprog with method = 'simplex'.  bounds must be a n x 2 sequence/array where n = len(c).")
    # Assigning a type to the variable 'message' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'message', str_191512)
    # SSA join for try-except statement (line 583)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 578)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 571)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 569)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 595)
    # Processing the call arguments (line 595)
    
    # Getting the type of 'L' (line 595)
    L_191515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 14), 'L', False)
    
    # Getting the type of 'np' (line 595)
    np_191516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 20), 'np', False)
    # Obtaining the member 'inf' of a type (line 595)
    inf_191517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 20), np_191516, 'inf')
    # Applying the 'usub' unary operator (line 595)
    result___neg___191518 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 19), 'usub', inf_191517)
    
    # Applying the binary operator '==' (line 595)
    result_eq_191519 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 14), '==', L_191515, result___neg___191518)
    
    # Processing the call keyword arguments (line 595)
    kwargs_191520 = {}
    # Getting the type of 'np' (line 595)
    np_191513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 595)
    any_191514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 7), np_191513, 'any')
    # Calling any(args, kwargs) (line 595)
    any_call_result_191521 = invoke(stypy.reporting.localization.Localization(__file__, 595, 7), any_191514, *[result_eq_191519], **kwargs_191520)
    
    # Testing the type of an if condition (line 595)
    if_condition_191522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 4), any_call_result_191521)
    # Assigning a type to the variable 'if_condition_191522' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'if_condition_191522', if_condition_191522)
    # SSA begins for if statement (line 595)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 599):
    
    # Assigning a BinOp to a Name (line 599):
    # Getting the type of 'n' (line 599)
    n_191523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'n')
    int_191524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 16), 'int')
    # Applying the binary operator '+' (line 599)
    result_add_191525 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 12), '+', n_191523, int_191524)
    
    # Assigning a type to the variable 'n' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'n', result_add_191525)
    
    # Assigning a Call to a Name (line 600):
    
    # Assigning a Call to a Name (line 600):
    
    # Call to concatenate(...): (line 600)
    # Processing the call arguments (line 600)
    
    # Obtaining an instance of the builtin type 'list' (line 600)
    list_191528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 600)
    # Adding element type (line 600)
    
    # Call to array(...): (line 600)
    # Processing the call arguments (line 600)
    
    # Obtaining an instance of the builtin type 'list' (line 600)
    list_191531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 600)
    # Adding element type (line 600)
    int_191532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 37), list_191531, int_191532)
    
    # Processing the call keyword arguments (line 600)
    kwargs_191533 = {}
    # Getting the type of 'np' (line 600)
    np_191529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 28), 'np', False)
    # Obtaining the member 'array' of a type (line 600)
    array_191530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 28), np_191529, 'array')
    # Calling array(args, kwargs) (line 600)
    array_call_result_191534 = invoke(stypy.reporting.localization.Localization(__file__, 600, 28), array_191530, *[list_191531], **kwargs_191533)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 27), list_191528, array_call_result_191534)
    # Adding element type (line 600)
    # Getting the type of 'L' (line 600)
    L_191535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 43), 'L', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 27), list_191528, L_191535)
    
    # Processing the call keyword arguments (line 600)
    kwargs_191536 = {}
    # Getting the type of 'np' (line 600)
    np_191526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 600)
    concatenate_191527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 12), np_191526, 'concatenate')
    # Calling concatenate(args, kwargs) (line 600)
    concatenate_call_result_191537 = invoke(stypy.reporting.localization.Localization(__file__, 600, 12), concatenate_191527, *[list_191528], **kwargs_191536)
    
    # Assigning a type to the variable 'L' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'L', concatenate_call_result_191537)
    
    # Assigning a Call to a Name (line 601):
    
    # Assigning a Call to a Name (line 601):
    
    # Call to concatenate(...): (line 601)
    # Processing the call arguments (line 601)
    
    # Obtaining an instance of the builtin type 'list' (line 601)
    list_191540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 601)
    # Adding element type (line 601)
    
    # Call to array(...): (line 601)
    # Processing the call arguments (line 601)
    
    # Obtaining an instance of the builtin type 'list' (line 601)
    list_191543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 601)
    # Adding element type (line 601)
    # Getting the type of 'np' (line 601)
    np_191544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 38), 'np', False)
    # Obtaining the member 'inf' of a type (line 601)
    inf_191545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 38), np_191544, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 37), list_191543, inf_191545)
    
    # Processing the call keyword arguments (line 601)
    kwargs_191546 = {}
    # Getting the type of 'np' (line 601)
    np_191541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 28), 'np', False)
    # Obtaining the member 'array' of a type (line 601)
    array_191542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 28), np_191541, 'array')
    # Calling array(args, kwargs) (line 601)
    array_call_result_191547 = invoke(stypy.reporting.localization.Localization(__file__, 601, 28), array_191542, *[list_191543], **kwargs_191546)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 27), list_191540, array_call_result_191547)
    # Adding element type (line 601)
    # Getting the type of 'U' (line 601)
    U_191548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 48), 'U', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 27), list_191540, U_191548)
    
    # Processing the call keyword arguments (line 601)
    kwargs_191549 = {}
    # Getting the type of 'np' (line 601)
    np_191538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 601)
    concatenate_191539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), np_191538, 'concatenate')
    # Calling concatenate(args, kwargs) (line 601)
    concatenate_call_result_191550 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), concatenate_191539, *[list_191540], **kwargs_191549)
    
    # Assigning a type to the variable 'U' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'U', concatenate_call_result_191550)
    
    # Assigning a Call to a Name (line 602):
    
    # Assigning a Call to a Name (line 602):
    
    # Call to concatenate(...): (line 602)
    # Processing the call arguments (line 602)
    
    # Obtaining an instance of the builtin type 'list' (line 602)
    list_191553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 602)
    # Adding element type (line 602)
    
    # Call to array(...): (line 602)
    # Processing the call arguments (line 602)
    
    # Obtaining an instance of the builtin type 'list' (line 602)
    list_191556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 602)
    # Adding element type (line 602)
    int_191557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 38), list_191556, int_191557)
    
    # Processing the call keyword arguments (line 602)
    kwargs_191558 = {}
    # Getting the type of 'np' (line 602)
    np_191554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 29), 'np', False)
    # Obtaining the member 'array' of a type (line 602)
    array_191555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 29), np_191554, 'array')
    # Calling array(args, kwargs) (line 602)
    array_call_result_191559 = invoke(stypy.reporting.localization.Localization(__file__, 602, 29), array_191555, *[list_191556], **kwargs_191558)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 28), list_191553, array_call_result_191559)
    # Adding element type (line 602)
    # Getting the type of 'cc' (line 602)
    cc_191560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 44), 'cc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 28), list_191553, cc_191560)
    
    # Processing the call keyword arguments (line 602)
    kwargs_191561 = {}
    # Getting the type of 'np' (line 602)
    np_191551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 13), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 602)
    concatenate_191552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 13), np_191551, 'concatenate')
    # Calling concatenate(args, kwargs) (line 602)
    concatenate_call_result_191562 = invoke(stypy.reporting.localization.Localization(__file__, 602, 13), concatenate_191552, *[list_191553], **kwargs_191561)
    
    # Assigning a type to the variable 'cc' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'cc', concatenate_call_result_191562)
    
    # Assigning a Call to a Name (line 603):
    
    # Assigning a Call to a Name (line 603):
    
    # Call to hstack(...): (line 603)
    # Processing the call arguments (line 603)
    
    # Obtaining an instance of the builtin type 'list' (line 603)
    list_191565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 603)
    # Adding element type (line 603)
    
    # Call to zeros(...): (line 603)
    # Processing the call arguments (line 603)
    
    # Obtaining an instance of the builtin type 'list' (line 603)
    list_191568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 603)
    # Adding element type (line 603)
    
    # Obtaining the type of the subscript
    int_191569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 45), 'int')
    # Getting the type of 'Aeq' (line 603)
    Aeq_191570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 35), 'Aeq', False)
    # Obtaining the member 'shape' of a type (line 603)
    shape_191571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 35), Aeq_191570, 'shape')
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___191572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 35), shape_191571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_191573 = invoke(stypy.reporting.localization.Localization(__file__, 603, 35), getitem___191572, int_191569)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 34), list_191568, subscript_call_result_191573)
    # Adding element type (line 603)
    int_191574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 34), list_191568, int_191574)
    
    # Processing the call keyword arguments (line 603)
    kwargs_191575 = {}
    # Getting the type of 'np' (line 603)
    np_191566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 25), 'np', False)
    # Obtaining the member 'zeros' of a type (line 603)
    zeros_191567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 25), np_191566, 'zeros')
    # Calling zeros(args, kwargs) (line 603)
    zeros_call_result_191576 = invoke(stypy.reporting.localization.Localization(__file__, 603, 25), zeros_191567, *[list_191568], **kwargs_191575)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 24), list_191565, zeros_call_result_191576)
    # Adding element type (line 603)
    # Getting the type of 'Aeq' (line 603)
    Aeq_191577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 54), 'Aeq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 24), list_191565, Aeq_191577)
    
    # Processing the call keyword arguments (line 603)
    kwargs_191578 = {}
    # Getting the type of 'np' (line 603)
    np_191563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 14), 'np', False)
    # Obtaining the member 'hstack' of a type (line 603)
    hstack_191564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 14), np_191563, 'hstack')
    # Calling hstack(args, kwargs) (line 603)
    hstack_call_result_191579 = invoke(stypy.reporting.localization.Localization(__file__, 603, 14), hstack_191564, *[list_191565], **kwargs_191578)
    
    # Assigning a type to the variable 'Aeq' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'Aeq', hstack_call_result_191579)
    
    # Assigning a Call to a Name (line 604):
    
    # Assigning a Call to a Name (line 604):
    
    # Call to hstack(...): (line 604)
    # Processing the call arguments (line 604)
    
    # Obtaining an instance of the builtin type 'list' (line 604)
    list_191582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 604)
    # Adding element type (line 604)
    
    # Call to zeros(...): (line 604)
    # Processing the call arguments (line 604)
    
    # Obtaining an instance of the builtin type 'list' (line 604)
    list_191585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 604)
    # Adding element type (line 604)
    
    # Obtaining the type of the subscript
    int_191586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 45), 'int')
    # Getting the type of 'Aub' (line 604)
    Aub_191587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 35), 'Aub', False)
    # Obtaining the member 'shape' of a type (line 604)
    shape_191588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 35), Aub_191587, 'shape')
    # Obtaining the member '__getitem__' of a type (line 604)
    getitem___191589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 35), shape_191588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 604)
    subscript_call_result_191590 = invoke(stypy.reporting.localization.Localization(__file__, 604, 35), getitem___191589, int_191586)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 34), list_191585, subscript_call_result_191590)
    # Adding element type (line 604)
    int_191591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 34), list_191585, int_191591)
    
    # Processing the call keyword arguments (line 604)
    kwargs_191592 = {}
    # Getting the type of 'np' (line 604)
    np_191583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 25), 'np', False)
    # Obtaining the member 'zeros' of a type (line 604)
    zeros_191584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 25), np_191583, 'zeros')
    # Calling zeros(args, kwargs) (line 604)
    zeros_call_result_191593 = invoke(stypy.reporting.localization.Localization(__file__, 604, 25), zeros_191584, *[list_191585], **kwargs_191592)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 24), list_191582, zeros_call_result_191593)
    # Adding element type (line 604)
    # Getting the type of 'Aub' (line 604)
    Aub_191594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 54), 'Aub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 24), list_191582, Aub_191594)
    
    # Processing the call keyword arguments (line 604)
    kwargs_191595 = {}
    # Getting the type of 'np' (line 604)
    np_191580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 14), 'np', False)
    # Obtaining the member 'hstack' of a type (line 604)
    hstack_191581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 14), np_191580, 'hstack')
    # Calling hstack(args, kwargs) (line 604)
    hstack_call_result_191596 = invoke(stypy.reporting.localization.Localization(__file__, 604, 14), hstack_191581, *[list_191582], **kwargs_191595)
    
    # Assigning a type to the variable 'Aub' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'Aub', hstack_call_result_191596)
    
    # Assigning a Name to a Name (line 605):
    
    # Assigning a Name to a Name (line 605):
    # Getting the type of 'True' (line 605)
    True_191597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 30), 'True')
    # Assigning a type to the variable 'have_floor_variable' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'have_floor_variable', True_191597)
    # SSA join for if statement (line 595)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'n' (line 610)
    n_191599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 19), 'n', False)
    # Processing the call keyword arguments (line 610)
    kwargs_191600 = {}
    # Getting the type of 'range' (line 610)
    range_191598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 13), 'range', False)
    # Calling range(args, kwargs) (line 610)
    range_call_result_191601 = invoke(stypy.reporting.localization.Localization(__file__, 610, 13), range_191598, *[n_191599], **kwargs_191600)
    
    # Testing the type of a for loop iterable (line 610)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 610, 4), range_call_result_191601)
    # Getting the type of the for loop variable (line 610)
    for_loop_var_191602 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 610, 4), range_call_result_191601)
    # Assigning a type to the variable 'i' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'i', for_loop_var_191602)
    # SSA begins for a for statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 611)
    i_191603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 13), 'i')
    # Getting the type of 'L' (line 611)
    L_191604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 11), 'L')
    # Obtaining the member '__getitem__' of a type (line 611)
    getitem___191605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 11), L_191604, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 611)
    subscript_call_result_191606 = invoke(stypy.reporting.localization.Localization(__file__, 611, 11), getitem___191605, i_191603)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 611)
    i_191607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 20), 'i')
    # Getting the type of 'U' (line 611)
    U_191608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 18), 'U')
    # Obtaining the member '__getitem__' of a type (line 611)
    getitem___191609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 18), U_191608, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 611)
    subscript_call_result_191610 = invoke(stypy.reporting.localization.Localization(__file__, 611, 18), getitem___191609, i_191607)
    
    # Applying the binary operator '>' (line 611)
    result_gt_191611 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 11), '>', subscript_call_result_191606, subscript_call_result_191610)
    
    # Testing the type of an if condition (line 611)
    if_condition_191612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 8), result_gt_191611)
    # Assigning a type to the variable 'if_condition_191612' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'if_condition_191612', if_condition_191612)
    # SSA begins for if statement (line 611)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 612):
    
    # Assigning a Num to a Name (line 612):
    int_191613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 21), 'int')
    # Assigning a type to the variable 'status' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'status', int_191613)
    
    # Assigning a BinOp to a Name (line 613):
    
    # Assigning a BinOp to a Name (line 613):
    str_191614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 23), 'str', "Invalid input for linprog with method = 'simplex'.  Lower bound %d is greater than upper bound%d")
    
    # Obtaining an instance of the builtin type 'tuple' (line 614)
    tuple_191615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 73), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 614)
    # Adding element type (line 614)
    # Getting the type of 'i' (line 614)
    i_191616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 73), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 73), tuple_191615, i_191616)
    # Adding element type (line 614)
    # Getting the type of 'i' (line 614)
    i_191617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 76), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 73), tuple_191615, i_191617)
    
    # Applying the binary operator '%' (line 613)
    result_mod_191618 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 23), '%', str_191614, tuple_191615)
    
    # Assigning a type to the variable 'message' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'message', result_mod_191618)
    # SSA join for if statement (line 611)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinf(...): (line 616)
    # Processing the call arguments (line 616)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 616)
    i_191621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 22), 'i', False)
    # Getting the type of 'L' (line 616)
    L_191622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 20), 'L', False)
    # Obtaining the member '__getitem__' of a type (line 616)
    getitem___191623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 20), L_191622, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 616)
    subscript_call_result_191624 = invoke(stypy.reporting.localization.Localization(__file__, 616, 20), getitem___191623, i_191621)
    
    # Processing the call keyword arguments (line 616)
    kwargs_191625 = {}
    # Getting the type of 'np' (line 616)
    np_191619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 11), 'np', False)
    # Obtaining the member 'isinf' of a type (line 616)
    isinf_191620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 11), np_191619, 'isinf')
    # Calling isinf(args, kwargs) (line 616)
    isinf_call_result_191626 = invoke(stypy.reporting.localization.Localization(__file__, 616, 11), isinf_191620, *[subscript_call_result_191624], **kwargs_191625)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 616)
    i_191627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 32), 'i')
    # Getting the type of 'L' (line 616)
    L_191628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 30), 'L')
    # Obtaining the member '__getitem__' of a type (line 616)
    getitem___191629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 30), L_191628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 616)
    subscript_call_result_191630 = invoke(stypy.reporting.localization.Localization(__file__, 616, 30), getitem___191629, i_191627)
    
    int_191631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 37), 'int')
    # Applying the binary operator '>' (line 616)
    result_gt_191632 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 30), '>', subscript_call_result_191630, int_191631)
    
    # Applying the binary operator 'and' (line 616)
    result_and_keyword_191633 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 11), 'and', isinf_call_result_191626, result_gt_191632)
    
    # Testing the type of an if condition (line 616)
    if_condition_191634 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 8), result_and_keyword_191633)
    # Assigning a type to the variable 'if_condition_191634' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'if_condition_191634', if_condition_191634)
    # SSA begins for if statement (line 616)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 617):
    
    # Assigning a Num to a Name (line 617):
    int_191635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 21), 'int')
    # Assigning a type to the variable 'status' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'status', int_191635)
    
    # Assigning a Str to a Name (line 618):
    
    # Assigning a Str to a Name (line 618):
    str_191636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 23), 'str', "Invalid input for linprog with method = 'simplex'.  Lower bound may not be +infinity")
    # Assigning a type to the variable 'message' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'message', str_191636)
    # SSA join for if statement (line 616)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinf(...): (line 621)
    # Processing the call arguments (line 621)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 621)
    i_191639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 22), 'i', False)
    # Getting the type of 'U' (line 621)
    U_191640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 621)
    getitem___191641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 20), U_191640, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 621)
    subscript_call_result_191642 = invoke(stypy.reporting.localization.Localization(__file__, 621, 20), getitem___191641, i_191639)
    
    # Processing the call keyword arguments (line 621)
    kwargs_191643 = {}
    # Getting the type of 'np' (line 621)
    np_191637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 11), 'np', False)
    # Obtaining the member 'isinf' of a type (line 621)
    isinf_191638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 11), np_191637, 'isinf')
    # Calling isinf(args, kwargs) (line 621)
    isinf_call_result_191644 = invoke(stypy.reporting.localization.Localization(__file__, 621, 11), isinf_191638, *[subscript_call_result_191642], **kwargs_191643)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 621)
    i_191645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 32), 'i')
    # Getting the type of 'U' (line 621)
    U_191646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 30), 'U')
    # Obtaining the member '__getitem__' of a type (line 621)
    getitem___191647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 30), U_191646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 621)
    subscript_call_result_191648 = invoke(stypy.reporting.localization.Localization(__file__, 621, 30), getitem___191647, i_191645)
    
    int_191649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 37), 'int')
    # Applying the binary operator '<' (line 621)
    result_lt_191650 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 30), '<', subscript_call_result_191648, int_191649)
    
    # Applying the binary operator 'and' (line 621)
    result_and_keyword_191651 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 11), 'and', isinf_call_result_191644, result_lt_191650)
    
    # Testing the type of an if condition (line 621)
    if_condition_191652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 621, 8), result_and_keyword_191651)
    # Assigning a type to the variable 'if_condition_191652' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'if_condition_191652', if_condition_191652)
    # SSA begins for if statement (line 621)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 622):
    
    # Assigning a Num to a Name (line 622):
    int_191653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 21), 'int')
    # Assigning a type to the variable 'status' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'status', int_191653)
    
    # Assigning a Str to a Name (line 623):
    
    # Assigning a Str to a Name (line 623):
    str_191654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 23), 'str', "Invalid input for linprog with method = 'simplex'.  Upper bound may not be -infinity")
    # Assigning a type to the variable 'message' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'message', str_191654)
    # SSA join for if statement (line 621)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isfinite(...): (line 626)
    # Processing the call arguments (line 626)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 626)
    i_191657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 25), 'i', False)
    # Getting the type of 'L' (line 626)
    L_191658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 23), 'L', False)
    # Obtaining the member '__getitem__' of a type (line 626)
    getitem___191659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 23), L_191658, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 626)
    subscript_call_result_191660 = invoke(stypy.reporting.localization.Localization(__file__, 626, 23), getitem___191659, i_191657)
    
    # Processing the call keyword arguments (line 626)
    kwargs_191661 = {}
    # Getting the type of 'np' (line 626)
    np_191655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 11), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 626)
    isfinite_191656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 11), np_191655, 'isfinite')
    # Calling isfinite(args, kwargs) (line 626)
    isfinite_call_result_191662 = invoke(stypy.reporting.localization.Localization(__file__, 626, 11), isfinite_191656, *[subscript_call_result_191660], **kwargs_191661)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 626)
    i_191663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 35), 'i')
    # Getting the type of 'L' (line 626)
    L_191664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 33), 'L')
    # Obtaining the member '__getitem__' of a type (line 626)
    getitem___191665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 33), L_191664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 626)
    subscript_call_result_191666 = invoke(stypy.reporting.localization.Localization(__file__, 626, 33), getitem___191665, i_191663)
    
    int_191667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 40), 'int')
    # Applying the binary operator '>' (line 626)
    result_gt_191668 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 33), '>', subscript_call_result_191666, int_191667)
    
    # Applying the binary operator 'and' (line 626)
    result_and_keyword_191669 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 11), 'and', isfinite_call_result_191662, result_gt_191668)
    
    # Testing the type of an if condition (line 626)
    if_condition_191670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 8), result_and_keyword_191669)
    # Assigning a type to the variable 'if_condition_191670' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'if_condition_191670', if_condition_191670)
    # SSA begins for if statement (line 626)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 628):
    
    # Assigning a Call to a Name (line 628):
    
    # Call to vstack(...): (line 628)
    # Processing the call arguments (line 628)
    
    # Obtaining an instance of the builtin type 'list' (line 628)
    list_191673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 628)
    # Adding element type (line 628)
    # Getting the type of 'Aub' (line 628)
    Aub_191674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 29), 'Aub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 28), list_191673, Aub_191674)
    # Adding element type (line 628)
    
    # Call to zeros(...): (line 628)
    # Processing the call arguments (line 628)
    # Getting the type of 'n' (line 628)
    n_191677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 43), 'n', False)
    # Processing the call keyword arguments (line 628)
    kwargs_191678 = {}
    # Getting the type of 'np' (line 628)
    np_191675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 34), 'np', False)
    # Obtaining the member 'zeros' of a type (line 628)
    zeros_191676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 34), np_191675, 'zeros')
    # Calling zeros(args, kwargs) (line 628)
    zeros_call_result_191679 = invoke(stypy.reporting.localization.Localization(__file__, 628, 34), zeros_191676, *[n_191677], **kwargs_191678)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 28), list_191673, zeros_call_result_191679)
    
    # Processing the call keyword arguments (line 628)
    kwargs_191680 = {}
    # Getting the type of 'np' (line 628)
    np_191671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 18), 'np', False)
    # Obtaining the member 'vstack' of a type (line 628)
    vstack_191672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 18), np_191671, 'vstack')
    # Calling vstack(args, kwargs) (line 628)
    vstack_call_result_191681 = invoke(stypy.reporting.localization.Localization(__file__, 628, 18), vstack_191672, *[list_191673], **kwargs_191680)
    
    # Assigning a type to the variable 'Aub' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 12), 'Aub', vstack_call_result_191681)
    
    # Assigning a Num to a Subscript (line 629):
    
    # Assigning a Num to a Subscript (line 629):
    int_191682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 25), 'int')
    # Getting the type of 'Aub' (line 629)
    Aub_191683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 12), 'Aub')
    
    # Obtaining an instance of the builtin type 'tuple' (line 629)
    tuple_191684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 629)
    # Adding element type (line 629)
    int_191685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 629, 16), tuple_191684, int_191685)
    # Adding element type (line 629)
    # Getting the type of 'i' (line 629)
    i_191686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 20), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 629, 16), tuple_191684, i_191686)
    
    # Storing an element on a container (line 629)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 629, 12), Aub_191683, (tuple_191684, int_191682))
    
    # Assigning a Call to a Name (line 630):
    
    # Assigning a Call to a Name (line 630):
    
    # Call to concatenate(...): (line 630)
    # Processing the call arguments (line 630)
    
    # Obtaining an instance of the builtin type 'list' (line 630)
    list_191689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 630)
    # Adding element type (line 630)
    # Getting the type of 'bub' (line 630)
    bub_191690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 34), 'bub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 33), list_191689, bub_191690)
    # Adding element type (line 630)
    
    # Call to array(...): (line 630)
    # Processing the call arguments (line 630)
    
    # Obtaining an instance of the builtin type 'list' (line 630)
    list_191693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 630)
    # Adding element type (line 630)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 630)
    i_191694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 52), 'i', False)
    # Getting the type of 'L' (line 630)
    L_191695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 50), 'L', False)
    # Obtaining the member '__getitem__' of a type (line 630)
    getitem___191696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 50), L_191695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 630)
    subscript_call_result_191697 = invoke(stypy.reporting.localization.Localization(__file__, 630, 50), getitem___191696, i_191694)
    
    # Applying the 'usub' unary operator (line 630)
    result___neg___191698 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 49), 'usub', subscript_call_result_191697)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 48), list_191693, result___neg___191698)
    
    # Processing the call keyword arguments (line 630)
    kwargs_191699 = {}
    # Getting the type of 'np' (line 630)
    np_191691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 39), 'np', False)
    # Obtaining the member 'array' of a type (line 630)
    array_191692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 39), np_191691, 'array')
    # Calling array(args, kwargs) (line 630)
    array_call_result_191700 = invoke(stypy.reporting.localization.Localization(__file__, 630, 39), array_191692, *[list_191693], **kwargs_191699)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 33), list_191689, array_call_result_191700)
    
    # Processing the call keyword arguments (line 630)
    kwargs_191701 = {}
    # Getting the type of 'np' (line 630)
    np_191687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 18), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 630)
    concatenate_191688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 18), np_191687, 'concatenate')
    # Calling concatenate(args, kwargs) (line 630)
    concatenate_call_result_191702 = invoke(stypy.reporting.localization.Localization(__file__, 630, 18), concatenate_191688, *[list_191689], **kwargs_191701)
    
    # Assigning a type to the variable 'bub' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'bub', concatenate_call_result_191702)
    
    # Assigning a Num to a Subscript (line 631):
    
    # Assigning a Num to a Subscript (line 631):
    int_191703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 19), 'int')
    # Getting the type of 'L' (line 631)
    L_191704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'L')
    # Getting the type of 'i' (line 631)
    i_191705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 14), 'i')
    # Storing an element on a container (line 631)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 631, 12), L_191704, (i_191705, int_191703))
    # SSA join for if statement (line 626)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isfinite(...): (line 633)
    # Processing the call arguments (line 633)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 633)
    i_191708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 25), 'i', False)
    # Getting the type of 'U' (line 633)
    U_191709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 23), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 633)
    getitem___191710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 23), U_191709, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 633)
    subscript_call_result_191711 = invoke(stypy.reporting.localization.Localization(__file__, 633, 23), getitem___191710, i_191708)
    
    # Processing the call keyword arguments (line 633)
    kwargs_191712 = {}
    # Getting the type of 'np' (line 633)
    np_191706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 11), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 633)
    isfinite_191707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 11), np_191706, 'isfinite')
    # Calling isfinite(args, kwargs) (line 633)
    isfinite_call_result_191713 = invoke(stypy.reporting.localization.Localization(__file__, 633, 11), isfinite_191707, *[subscript_call_result_191711], **kwargs_191712)
    
    # Testing the type of an if condition (line 633)
    if_condition_191714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 633, 8), isfinite_call_result_191713)
    # Assigning a type to the variable 'if_condition_191714' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'if_condition_191714', if_condition_191714)
    # SSA begins for if statement (line 633)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 635):
    
    # Assigning a Call to a Name (line 635):
    
    # Call to vstack(...): (line 635)
    # Processing the call arguments (line 635)
    
    # Obtaining an instance of the builtin type 'list' (line 635)
    list_191717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 635)
    # Adding element type (line 635)
    # Getting the type of 'Aub' (line 635)
    Aub_191718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 29), 'Aub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 28), list_191717, Aub_191718)
    # Adding element type (line 635)
    
    # Call to zeros(...): (line 635)
    # Processing the call arguments (line 635)
    # Getting the type of 'n' (line 635)
    n_191721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 43), 'n', False)
    # Processing the call keyword arguments (line 635)
    kwargs_191722 = {}
    # Getting the type of 'np' (line 635)
    np_191719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 34), 'np', False)
    # Obtaining the member 'zeros' of a type (line 635)
    zeros_191720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 34), np_191719, 'zeros')
    # Calling zeros(args, kwargs) (line 635)
    zeros_call_result_191723 = invoke(stypy.reporting.localization.Localization(__file__, 635, 34), zeros_191720, *[n_191721], **kwargs_191722)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 28), list_191717, zeros_call_result_191723)
    
    # Processing the call keyword arguments (line 635)
    kwargs_191724 = {}
    # Getting the type of 'np' (line 635)
    np_191715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 18), 'np', False)
    # Obtaining the member 'vstack' of a type (line 635)
    vstack_191716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 18), np_191715, 'vstack')
    # Calling vstack(args, kwargs) (line 635)
    vstack_call_result_191725 = invoke(stypy.reporting.localization.Localization(__file__, 635, 18), vstack_191716, *[list_191717], **kwargs_191724)
    
    # Assigning a type to the variable 'Aub' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'Aub', vstack_call_result_191725)
    
    # Assigning a Num to a Subscript (line 636):
    
    # Assigning a Num to a Subscript (line 636):
    int_191726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 25), 'int')
    # Getting the type of 'Aub' (line 636)
    Aub_191727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'Aub')
    
    # Obtaining an instance of the builtin type 'tuple' (line 636)
    tuple_191728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 636)
    # Adding element type (line 636)
    int_191729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 16), tuple_191728, int_191729)
    # Adding element type (line 636)
    # Getting the type of 'i' (line 636)
    i_191730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 20), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 16), tuple_191728, i_191730)
    
    # Storing an element on a container (line 636)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 12), Aub_191727, (tuple_191728, int_191726))
    
    # Assigning a Call to a Name (line 637):
    
    # Assigning a Call to a Name (line 637):
    
    # Call to concatenate(...): (line 637)
    # Processing the call arguments (line 637)
    
    # Obtaining an instance of the builtin type 'list' (line 637)
    list_191733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 637)
    # Adding element type (line 637)
    # Getting the type of 'bub' (line 637)
    bub_191734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 34), 'bub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 33), list_191733, bub_191734)
    # Adding element type (line 637)
    
    # Call to array(...): (line 637)
    # Processing the call arguments (line 637)
    
    # Obtaining an instance of the builtin type 'list' (line 637)
    list_191737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 637)
    # Adding element type (line 637)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 637)
    i_191738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 51), 'i', False)
    # Getting the type of 'U' (line 637)
    U_191739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 49), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 637)
    getitem___191740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 49), U_191739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 637)
    subscript_call_result_191741 = invoke(stypy.reporting.localization.Localization(__file__, 637, 49), getitem___191740, i_191738)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 48), list_191737, subscript_call_result_191741)
    
    # Processing the call keyword arguments (line 637)
    kwargs_191742 = {}
    # Getting the type of 'np' (line 637)
    np_191735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 39), 'np', False)
    # Obtaining the member 'array' of a type (line 637)
    array_191736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 39), np_191735, 'array')
    # Calling array(args, kwargs) (line 637)
    array_call_result_191743 = invoke(stypy.reporting.localization.Localization(__file__, 637, 39), array_191736, *[list_191737], **kwargs_191742)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 33), list_191733, array_call_result_191743)
    
    # Processing the call keyword arguments (line 637)
    kwargs_191744 = {}
    # Getting the type of 'np' (line 637)
    np_191731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 18), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 637)
    concatenate_191732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 18), np_191731, 'concatenate')
    # Calling concatenate(args, kwargs) (line 637)
    concatenate_call_result_191745 = invoke(stypy.reporting.localization.Localization(__file__, 637, 18), concatenate_191732, *[list_191733], **kwargs_191744)
    
    # Assigning a type to the variable 'bub' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'bub', concatenate_call_result_191745)
    
    # Assigning a Attribute to a Subscript (line 638):
    
    # Assigning a Attribute to a Subscript (line 638):
    # Getting the type of 'np' (line 638)
    np_191746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 19), 'np')
    # Obtaining the member 'inf' of a type (line 638)
    inf_191747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 19), np_191746, 'inf')
    # Getting the type of 'U' (line 638)
    U_191748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), 'U')
    # Getting the type of 'i' (line 638)
    i_191749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 14), 'i')
    # Storing an element on a container (line 638)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 12), U_191748, (i_191749, inf_191747))
    # SSA join for if statement (line 633)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 642)
    # Processing the call arguments (line 642)
    int_191751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 19), 'int')
    # Getting the type of 'n' (line 642)
    n_191752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 22), 'n', False)
    # Processing the call keyword arguments (line 642)
    kwargs_191753 = {}
    # Getting the type of 'range' (line 642)
    range_191750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 13), 'range', False)
    # Calling range(args, kwargs) (line 642)
    range_call_result_191754 = invoke(stypy.reporting.localization.Localization(__file__, 642, 13), range_191750, *[int_191751, n_191752], **kwargs_191753)
    
    # Testing the type of a for loop iterable (line 642)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 642, 4), range_call_result_191754)
    # Getting the type of the for loop variable (line 642)
    for_loop_var_191755 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 642, 4), range_call_result_191754)
    # Assigning a type to the variable 'i' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'i', for_loop_var_191755)
    # SSA begins for a for statement (line 642)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 643)
    i_191756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 13), 'i')
    # Getting the type of 'L' (line 643)
    L_191757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 11), 'L')
    # Obtaining the member '__getitem__' of a type (line 643)
    getitem___191758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 11), L_191757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 643)
    subscript_call_result_191759 = invoke(stypy.reporting.localization.Localization(__file__, 643, 11), getitem___191758, i_191756)
    
    int_191760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 18), 'int')
    # Applying the binary operator '<' (line 643)
    result_lt_191761 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 11), '<', subscript_call_result_191759, int_191760)
    
    # Testing the type of an if condition (line 643)
    if_condition_191762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 643, 8), result_lt_191761)
    # Assigning a type to the variable 'if_condition_191762' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'if_condition_191762', if_condition_191762)
    # SSA begins for if statement (line 643)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Call to isfinite(...): (line 644)
    # Processing the call arguments (line 644)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 644)
    i_191765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 29), 'i', False)
    # Getting the type of 'L' (line 644)
    L_191766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 27), 'L', False)
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___191767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 27), L_191766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_191768 = invoke(stypy.reporting.localization.Localization(__file__, 644, 27), getitem___191767, i_191765)
    
    # Processing the call keyword arguments (line 644)
    kwargs_191769 = {}
    # Getting the type of 'np' (line 644)
    np_191763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 15), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 644)
    isfinite_191764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 15), np_191763, 'isfinite')
    # Calling isfinite(args, kwargs) (line 644)
    isfinite_call_result_191770 = invoke(stypy.reporting.localization.Localization(__file__, 644, 15), isfinite_191764, *[subscript_call_result_191768], **kwargs_191769)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 644)
    i_191771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 39), 'i')
    # Getting the type of 'L' (line 644)
    L_191772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 37), 'L')
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___191773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 37), L_191772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_191774 = invoke(stypy.reporting.localization.Localization(__file__, 644, 37), getitem___191773, i_191771)
    
    int_191775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 44), 'int')
    # Applying the binary operator '<' (line 644)
    result_lt_191776 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 37), '<', subscript_call_result_191774, int_191775)
    
    # Applying the binary operator 'and' (line 644)
    result_and_keyword_191777 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 15), 'and', isfinite_call_result_191770, result_lt_191776)
    
    # Testing the type of an if condition (line 644)
    if_condition_191778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 644, 12), result_and_keyword_191777)
    # Assigning a type to the variable 'if_condition_191778' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'if_condition_191778', if_condition_191778)
    # SSA begins for if statement (line 644)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 649):
    
    # Assigning a BinOp to a Name (line 649):
    # Getting the type of 'beq' (line 649)
    beq_191779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 22), 'beq')
    
    # Obtaining the type of the subscript
    slice_191780 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 649, 28), None, None, None)
    # Getting the type of 'i' (line 649)
    i_191781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 35), 'i')
    # Getting the type of 'Aeq' (line 649)
    Aeq_191782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 28), 'Aeq')
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___191783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 28), Aeq_191782, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_191784 = invoke(stypy.reporting.localization.Localization(__file__, 649, 28), getitem___191783, (slice_191780, i_191781))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 649)
    i_191785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 42), 'i')
    # Getting the type of 'L' (line 649)
    L_191786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 40), 'L')
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___191787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 40), L_191786, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_191788 = invoke(stypy.reporting.localization.Localization(__file__, 649, 40), getitem___191787, i_191785)
    
    # Applying the binary operator '*' (line 649)
    result_mul_191789 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 28), '*', subscript_call_result_191784, subscript_call_result_191788)
    
    # Applying the binary operator '-' (line 649)
    result_sub_191790 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 22), '-', beq_191779, result_mul_191789)
    
    # Assigning a type to the variable 'beq' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 16), 'beq', result_sub_191790)
    
    # Assigning a BinOp to a Name (line 650):
    
    # Assigning a BinOp to a Name (line 650):
    # Getting the type of 'bub' (line 650)
    bub_191791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 22), 'bub')
    
    # Obtaining the type of the subscript
    slice_191792 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 650, 28), None, None, None)
    # Getting the type of 'i' (line 650)
    i_191793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 35), 'i')
    # Getting the type of 'Aub' (line 650)
    Aub_191794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 28), 'Aub')
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___191795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 28), Aub_191794, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_191796 = invoke(stypy.reporting.localization.Localization(__file__, 650, 28), getitem___191795, (slice_191792, i_191793))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 650)
    i_191797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 42), 'i')
    # Getting the type of 'L' (line 650)
    L_191798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 40), 'L')
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___191799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 40), L_191798, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_191800 = invoke(stypy.reporting.localization.Localization(__file__, 650, 40), getitem___191799, i_191797)
    
    # Applying the binary operator '*' (line 650)
    result_mul_191801 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 28), '*', subscript_call_result_191796, subscript_call_result_191800)
    
    # Applying the binary operator '-' (line 650)
    result_sub_191802 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 22), '-', bub_191791, result_mul_191801)
    
    # Assigning a type to the variable 'bub' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'bub', result_sub_191802)
    
    # Assigning a BinOp to a Name (line 653):
    
    # Assigning a BinOp to a Name (line 653):
    # Getting the type of 'f0' (line 653)
    f0_191803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 21), 'f0')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 653)
    i_191804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 29), 'i')
    # Getting the type of 'cc' (line 653)
    cc_191805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 26), 'cc')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___191806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 26), cc_191805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_191807 = invoke(stypy.reporting.localization.Localization(__file__, 653, 26), getitem___191806, i_191804)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 653)
    i_191808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 36), 'i')
    # Getting the type of 'L' (line 653)
    L_191809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 34), 'L')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___191810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 34), L_191809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_191811 = invoke(stypy.reporting.localization.Localization(__file__, 653, 34), getitem___191810, i_191808)
    
    # Applying the binary operator '*' (line 653)
    result_mul_191812 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 26), '*', subscript_call_result_191807, subscript_call_result_191811)
    
    # Applying the binary operator '-' (line 653)
    result_sub_191813 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 21), '-', f0_191803, result_mul_191812)
    
    # Assigning a type to the variable 'f0' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 16), 'f0', result_sub_191813)
    # SSA branch for the else part of an if statement (line 644)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Subscript (line 657):
    
    # Assigning a BinOp to a Subscript (line 657):
    
    # Obtaining the type of the subscript
    slice_191814 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 657, 28), None, None, None)
    int_191815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 35), 'int')
    # Getting the type of 'Aeq' (line 657)
    Aeq_191816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 28), 'Aeq')
    # Obtaining the member '__getitem__' of a type (line 657)
    getitem___191817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 28), Aeq_191816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 657)
    subscript_call_result_191818 = invoke(stypy.reporting.localization.Localization(__file__, 657, 28), getitem___191817, (slice_191814, int_191815))
    
    
    # Obtaining the type of the subscript
    slice_191819 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 657, 40), None, None, None)
    # Getting the type of 'i' (line 657)
    i_191820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 47), 'i')
    # Getting the type of 'Aeq' (line 657)
    Aeq_191821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 40), 'Aeq')
    # Obtaining the member '__getitem__' of a type (line 657)
    getitem___191822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 40), Aeq_191821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 657)
    subscript_call_result_191823 = invoke(stypy.reporting.localization.Localization(__file__, 657, 40), getitem___191822, (slice_191819, i_191820))
    
    # Applying the binary operator '-' (line 657)
    result_sub_191824 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 28), '-', subscript_call_result_191818, subscript_call_result_191823)
    
    # Getting the type of 'Aeq' (line 657)
    Aeq_191825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'Aeq')
    slice_191826 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 657, 16), None, None, None)
    int_191827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 23), 'int')
    # Storing an element on a container (line 657)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 16), Aeq_191825, ((slice_191826, int_191827), result_sub_191824))
    
    # Assigning a BinOp to a Subscript (line 658):
    
    # Assigning a BinOp to a Subscript (line 658):
    
    # Obtaining the type of the subscript
    slice_191828 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 658, 28), None, None, None)
    int_191829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 35), 'int')
    # Getting the type of 'Aub' (line 658)
    Aub_191830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 28), 'Aub')
    # Obtaining the member '__getitem__' of a type (line 658)
    getitem___191831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 28), Aub_191830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 658)
    subscript_call_result_191832 = invoke(stypy.reporting.localization.Localization(__file__, 658, 28), getitem___191831, (slice_191828, int_191829))
    
    
    # Obtaining the type of the subscript
    slice_191833 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 658, 40), None, None, None)
    # Getting the type of 'i' (line 658)
    i_191834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 47), 'i')
    # Getting the type of 'Aub' (line 658)
    Aub_191835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 40), 'Aub')
    # Obtaining the member '__getitem__' of a type (line 658)
    getitem___191836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 40), Aub_191835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 658)
    subscript_call_result_191837 = invoke(stypy.reporting.localization.Localization(__file__, 658, 40), getitem___191836, (slice_191833, i_191834))
    
    # Applying the binary operator '-' (line 658)
    result_sub_191838 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 28), '-', subscript_call_result_191832, subscript_call_result_191837)
    
    # Getting the type of 'Aub' (line 658)
    Aub_191839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 'Aub')
    slice_191840 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 658, 16), None, None, None)
    int_191841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 23), 'int')
    # Storing an element on a container (line 658)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 16), Aub_191839, ((slice_191840, int_191841), result_sub_191838))
    
    # Assigning a BinOp to a Subscript (line 659):
    
    # Assigning a BinOp to a Subscript (line 659):
    
    # Obtaining the type of the subscript
    int_191842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 27), 'int')
    # Getting the type of 'cc' (line 659)
    cc_191843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 24), 'cc')
    # Obtaining the member '__getitem__' of a type (line 659)
    getitem___191844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 24), cc_191843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 659)
    subscript_call_result_191845 = invoke(stypy.reporting.localization.Localization(__file__, 659, 24), getitem___191844, int_191842)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 659)
    i_191846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 35), 'i')
    # Getting the type of 'cc' (line 659)
    cc_191847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 32), 'cc')
    # Obtaining the member '__getitem__' of a type (line 659)
    getitem___191848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 32), cc_191847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 659)
    subscript_call_result_191849 = invoke(stypy.reporting.localization.Localization(__file__, 659, 32), getitem___191848, i_191846)
    
    # Applying the binary operator '-' (line 659)
    result_sub_191850 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 24), '-', subscript_call_result_191845, subscript_call_result_191849)
    
    # Getting the type of 'cc' (line 659)
    cc_191851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'cc')
    int_191852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 19), 'int')
    # Storing an element on a container (line 659)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 16), cc_191851, (int_191852, result_sub_191850))
    # SSA join for if statement (line 644)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 643)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinf(...): (line 661)
    # Processing the call arguments (line 661)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 661)
    i_191855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 22), 'i', False)
    # Getting the type of 'U' (line 661)
    U_191856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 20), 'U', False)
    # Obtaining the member '__getitem__' of a type (line 661)
    getitem___191857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 20), U_191856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 661)
    subscript_call_result_191858 = invoke(stypy.reporting.localization.Localization(__file__, 661, 20), getitem___191857, i_191855)
    
    # Processing the call keyword arguments (line 661)
    kwargs_191859 = {}
    # Getting the type of 'np' (line 661)
    np_191853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 11), 'np', False)
    # Obtaining the member 'isinf' of a type (line 661)
    isinf_191854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 11), np_191853, 'isinf')
    # Calling isinf(args, kwargs) (line 661)
    isinf_call_result_191860 = invoke(stypy.reporting.localization.Localization(__file__, 661, 11), isinf_191854, *[subscript_call_result_191858], **kwargs_191859)
    
    # Testing the type of an if condition (line 661)
    if_condition_191861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 8), isinf_call_result_191860)
    # Assigning a type to the variable 'if_condition_191861' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'if_condition_191861', if_condition_191861)
    # SSA begins for if statement (line 661)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 662)
    i_191862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 17), 'i')
    # Getting the type of 'U' (line 662)
    U_191863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 15), 'U')
    # Obtaining the member '__getitem__' of a type (line 662)
    getitem___191864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 15), U_191863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 662)
    subscript_call_result_191865 = invoke(stypy.reporting.localization.Localization(__file__, 662, 15), getitem___191864, i_191862)
    
    int_191866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 22), 'int')
    # Applying the binary operator '<' (line 662)
    result_lt_191867 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 15), '<', subscript_call_result_191865, int_191866)
    
    # Testing the type of an if condition (line 662)
    if_condition_191868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 12), result_lt_191867)
    # Assigning a type to the variable 'if_condition_191868' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'if_condition_191868', if_condition_191868)
    # SSA begins for if statement (line 662)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 663):
    
    # Assigning a Num to a Name (line 663):
    int_191869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 25), 'int')
    # Assigning a type to the variable 'status' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'status', int_191869)
    
    # Assigning a Str to a Name (line 664):
    
    # Assigning a Str to a Name (line 664):
    str_191870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 27), 'str', "Invalid input for linprog with method = 'simplex'.  Upper bound may not be -inf.")
    # Assigning a type to the variable 'message' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 16), 'message', str_191870)
    # SSA join for if statement (line 662)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 661)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 668):
    
    # Assigning a Call to a Name (line 668):
    
    # Call to len(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'bub' (line 668)
    bub_191872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 14), 'bub', False)
    # Processing the call keyword arguments (line 668)
    kwargs_191873 = {}
    # Getting the type of 'len' (line 668)
    len_191871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 10), 'len', False)
    # Calling len(args, kwargs) (line 668)
    len_call_result_191874 = invoke(stypy.reporting.localization.Localization(__file__, 668, 10), len_191871, *[bub_191872], **kwargs_191873)
    
    # Assigning a type to the variable 'mub' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'mub', len_call_result_191874)
    
    # Assigning a Call to a Name (line 671):
    
    # Assigning a Call to a Name (line 671):
    
    # Call to len(...): (line 671)
    # Processing the call arguments (line 671)
    # Getting the type of 'beq' (line 671)
    beq_191876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 14), 'beq', False)
    # Processing the call keyword arguments (line 671)
    kwargs_191877 = {}
    # Getting the type of 'len' (line 671)
    len_191875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 10), 'len', False)
    # Calling len(args, kwargs) (line 671)
    len_call_result_191878 = invoke(stypy.reporting.localization.Localization(__file__, 671, 10), len_191875, *[beq_191876], **kwargs_191877)
    
    # Assigning a type to the variable 'meq' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'meq', len_call_result_191878)
    
    # Assigning a BinOp to a Name (line 674):
    
    # Assigning a BinOp to a Name (line 674):
    # Getting the type of 'mub' (line 674)
    mub_191879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'mub')
    # Getting the type of 'meq' (line 674)
    meq_191880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'meq')
    # Applying the binary operator '+' (line 674)
    result_add_191881 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 8), '+', mub_191879, meq_191880)
    
    # Assigning a type to the variable 'm' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'm', result_add_191881)
    
    # Assigning a Name to a Name (line 677):
    
    # Assigning a Name to a Name (line 677):
    # Getting the type of 'mub' (line 677)
    mub_191882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 14), 'mub')
    # Assigning a type to the variable 'n_slack' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'n_slack', mub_191882)
    
    # Assigning a BinOp to a Name (line 681):
    
    # Assigning a BinOp to a Name (line 681):
    # Getting the type of 'meq' (line 681)
    meq_191883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 19), 'meq')
    
    # Call to count_nonzero(...): (line 681)
    # Processing the call arguments (line 681)
    
    # Getting the type of 'bub' (line 681)
    bub_191886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 42), 'bub', False)
    int_191887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 48), 'int')
    # Applying the binary operator '<' (line 681)
    result_lt_191888 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 42), '<', bub_191886, int_191887)
    
    # Processing the call keyword arguments (line 681)
    kwargs_191889 = {}
    # Getting the type of 'np' (line 681)
    np_191884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 25), 'np', False)
    # Obtaining the member 'count_nonzero' of a type (line 681)
    count_nonzero_191885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 25), np_191884, 'count_nonzero')
    # Calling count_nonzero(args, kwargs) (line 681)
    count_nonzero_call_result_191890 = invoke(stypy.reporting.localization.Localization(__file__, 681, 25), count_nonzero_191885, *[result_lt_191888], **kwargs_191889)
    
    # Applying the binary operator '+' (line 681)
    result_add_191891 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 19), '+', meq_191883, count_nonzero_call_result_191890)
    
    # Assigning a type to the variable 'n_artificial' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'n_artificial', result_add_191891)
    
    
    # SSA begins for try-except statement (line 683)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Tuple (line 684):
    
    # Assigning a Subscript to a Name (line 684):
    
    # Obtaining the type of the subscript
    int_191892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 8), 'int')
    # Getting the type of 'Aub' (line 684)
    Aub_191893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 29), 'Aub')
    # Obtaining the member 'shape' of a type (line 684)
    shape_191894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 29), Aub_191893, 'shape')
    # Obtaining the member '__getitem__' of a type (line 684)
    getitem___191895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 8), shape_191894, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 684)
    subscript_call_result_191896 = invoke(stypy.reporting.localization.Localization(__file__, 684, 8), getitem___191895, int_191892)
    
    # Assigning a type to the variable 'tuple_var_assignment_190479' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'tuple_var_assignment_190479', subscript_call_result_191896)
    
    # Assigning a Subscript to a Name (line 684):
    
    # Obtaining the type of the subscript
    int_191897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 8), 'int')
    # Getting the type of 'Aub' (line 684)
    Aub_191898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 29), 'Aub')
    # Obtaining the member 'shape' of a type (line 684)
    shape_191899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 29), Aub_191898, 'shape')
    # Obtaining the member '__getitem__' of a type (line 684)
    getitem___191900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 8), shape_191899, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 684)
    subscript_call_result_191901 = invoke(stypy.reporting.localization.Localization(__file__, 684, 8), getitem___191900, int_191897)
    
    # Assigning a type to the variable 'tuple_var_assignment_190480' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'tuple_var_assignment_190480', subscript_call_result_191901)
    
    # Assigning a Name to a Name (line 684):
    # Getting the type of 'tuple_var_assignment_190479' (line 684)
    tuple_var_assignment_190479_191902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'tuple_var_assignment_190479')
    # Assigning a type to the variable 'Aub_rows' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'Aub_rows', tuple_var_assignment_190479_191902)
    
    # Assigning a Name to a Name (line 684):
    # Getting the type of 'tuple_var_assignment_190480' (line 684)
    tuple_var_assignment_190480_191903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'tuple_var_assignment_190480')
    # Assigning a type to the variable 'Aub_cols' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 18), 'Aub_cols', tuple_var_assignment_190480_191903)
    # SSA branch for the except part of a try statement (line 683)
    # SSA branch for the except 'ValueError' branch of a try statement (line 683)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 686)
    # Processing the call arguments (line 686)
    str_191905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 25), 'str', 'Invalid input.  A_ub must be two-dimensional')
    # Processing the call keyword arguments (line 686)
    kwargs_191906 = {}
    # Getting the type of 'ValueError' (line 686)
    ValueError_191904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 686)
    ValueError_call_result_191907 = invoke(stypy.reporting.localization.Localization(__file__, 686, 14), ValueError_191904, *[str_191905], **kwargs_191906)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 686, 8), ValueError_call_result_191907, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 683)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 688)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Tuple (line 689):
    
    # Assigning a Subscript to a Name (line 689):
    
    # Obtaining the type of the subscript
    int_191908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 8), 'int')
    # Getting the type of 'Aeq' (line 689)
    Aeq_191909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 29), 'Aeq')
    # Obtaining the member 'shape' of a type (line 689)
    shape_191910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 29), Aeq_191909, 'shape')
    # Obtaining the member '__getitem__' of a type (line 689)
    getitem___191911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), shape_191910, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 689)
    subscript_call_result_191912 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), getitem___191911, int_191908)
    
    # Assigning a type to the variable 'tuple_var_assignment_190481' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'tuple_var_assignment_190481', subscript_call_result_191912)
    
    # Assigning a Subscript to a Name (line 689):
    
    # Obtaining the type of the subscript
    int_191913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 8), 'int')
    # Getting the type of 'Aeq' (line 689)
    Aeq_191914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 29), 'Aeq')
    # Obtaining the member 'shape' of a type (line 689)
    shape_191915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 29), Aeq_191914, 'shape')
    # Obtaining the member '__getitem__' of a type (line 689)
    getitem___191916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), shape_191915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 689)
    subscript_call_result_191917 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), getitem___191916, int_191913)
    
    # Assigning a type to the variable 'tuple_var_assignment_190482' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'tuple_var_assignment_190482', subscript_call_result_191917)
    
    # Assigning a Name to a Name (line 689):
    # Getting the type of 'tuple_var_assignment_190481' (line 689)
    tuple_var_assignment_190481_191918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'tuple_var_assignment_190481')
    # Assigning a type to the variable 'Aeq_rows' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'Aeq_rows', tuple_var_assignment_190481_191918)
    
    # Assigning a Name to a Name (line 689):
    # Getting the type of 'tuple_var_assignment_190482' (line 689)
    tuple_var_assignment_190482_191919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'tuple_var_assignment_190482')
    # Assigning a type to the variable 'Aeq_cols' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 18), 'Aeq_cols', tuple_var_assignment_190482_191919)
    # SSA branch for the except part of a try statement (line 688)
    # SSA branch for the except 'ValueError' branch of a try statement (line 688)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 691)
    # Processing the call arguments (line 691)
    str_191921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 25), 'str', 'Invalid input.  A_eq must be two-dimensional')
    # Processing the call keyword arguments (line 691)
    kwargs_191922 = {}
    # Getting the type of 'ValueError' (line 691)
    ValueError_191920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 691)
    ValueError_call_result_191923 = invoke(stypy.reporting.localization.Localization(__file__, 691, 14), ValueError_191920, *[str_191921], **kwargs_191922)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 691, 8), ValueError_call_result_191923, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 688)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'Aeq_rows' (line 693)
    Aeq_rows_191924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 7), 'Aeq_rows')
    # Getting the type of 'meq' (line 693)
    meq_191925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 19), 'meq')
    # Applying the binary operator '!=' (line 693)
    result_ne_191926 = python_operator(stypy.reporting.localization.Localization(__file__, 693, 7), '!=', Aeq_rows_191924, meq_191925)
    
    # Testing the type of an if condition (line 693)
    if_condition_191927 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 693, 4), result_ne_191926)
    # Assigning a type to the variable 'if_condition_191927' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'if_condition_191927', if_condition_191927)
    # SSA begins for if statement (line 693)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 694):
    
    # Assigning a Num to a Name (line 694):
    int_191928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 17), 'int')
    # Assigning a type to the variable 'status' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'status', int_191928)
    
    # Assigning a Str to a Name (line 695):
    
    # Assigning a Str to a Name (line 695):
    str_191929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 19), 'str', "Invalid input for linprog with method = 'simplex'.  The number of rows in A_eq must be equal to the number of values in b_eq")
    # Assigning a type to the variable 'message' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'message', str_191929)
    # SSA join for if statement (line 693)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'Aub_rows' (line 699)
    Aub_rows_191930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 7), 'Aub_rows')
    # Getting the type of 'mub' (line 699)
    mub_191931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 19), 'mub')
    # Applying the binary operator '!=' (line 699)
    result_ne_191932 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 7), '!=', Aub_rows_191930, mub_191931)
    
    # Testing the type of an if condition (line 699)
    if_condition_191933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 699, 4), result_ne_191932)
    # Assigning a type to the variable 'if_condition_191933' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'if_condition_191933', if_condition_191933)
    # SSA begins for if statement (line 699)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 700):
    
    # Assigning a Num to a Name (line 700):
    int_191934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 17), 'int')
    # Assigning a type to the variable 'status' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'status', int_191934)
    
    # Assigning a Str to a Name (line 701):
    
    # Assigning a Str to a Name (line 701):
    str_191935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 19), 'str', "Invalid input for linprog with method = 'simplex'.  The number of rows in A_ub must be equal to the number of values in b_ub")
    # Assigning a type to the variable 'message' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'message', str_191935)
    # SSA join for if statement (line 699)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'Aeq_cols' (line 705)
    Aeq_cols_191936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 7), 'Aeq_cols')
    int_191937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 18), 'int')
    # Applying the binary operator '>' (line 705)
    result_gt_191938 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 7), '>', Aeq_cols_191936, int_191937)
    
    
    # Getting the type of 'Aeq_cols' (line 705)
    Aeq_cols_191939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 24), 'Aeq_cols')
    # Getting the type of 'n' (line 705)
    n_191940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 36), 'n')
    # Applying the binary operator '!=' (line 705)
    result_ne_191941 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 24), '!=', Aeq_cols_191939, n_191940)
    
    # Applying the binary operator 'and' (line 705)
    result_and_keyword_191942 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 7), 'and', result_gt_191938, result_ne_191941)
    
    # Testing the type of an if condition (line 705)
    if_condition_191943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 705, 4), result_and_keyword_191942)
    # Assigning a type to the variable 'if_condition_191943' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'if_condition_191943', if_condition_191943)
    # SSA begins for if statement (line 705)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 706):
    
    # Assigning a Num to a Name (line 706):
    int_191944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 17), 'int')
    # Assigning a type to the variable 'status' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'status', int_191944)
    
    # Assigning a Str to a Name (line 707):
    
    # Assigning a Str to a Name (line 707):
    str_191945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 19), 'str', "Invalid input for linprog with method = 'simplex'.  Number of columns in A_eq must be equal to the size of c")
    # Assigning a type to the variable 'message' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'message', str_191945)
    # SSA join for if statement (line 705)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'Aub_cols' (line 711)
    Aub_cols_191946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 7), 'Aub_cols')
    int_191947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 18), 'int')
    # Applying the binary operator '>' (line 711)
    result_gt_191948 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 7), '>', Aub_cols_191946, int_191947)
    
    
    # Getting the type of 'Aub_cols' (line 711)
    Aub_cols_191949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 24), 'Aub_cols')
    # Getting the type of 'n' (line 711)
    n_191950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 36), 'n')
    # Applying the binary operator '!=' (line 711)
    result_ne_191951 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 24), '!=', Aub_cols_191949, n_191950)
    
    # Applying the binary operator 'and' (line 711)
    result_and_keyword_191952 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 7), 'and', result_gt_191948, result_ne_191951)
    
    # Testing the type of an if condition (line 711)
    if_condition_191953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 711, 4), result_and_keyword_191952)
    # Assigning a type to the variable 'if_condition_191953' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'if_condition_191953', if_condition_191953)
    # SSA begins for if statement (line 711)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 712):
    
    # Assigning a Num to a Name (line 712):
    int_191954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 17), 'int')
    # Assigning a type to the variable 'status' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'status', int_191954)
    
    # Assigning a Str to a Name (line 713):
    
    # Assigning a Str to a Name (line 713):
    str_191955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 19), 'str', "Invalid input for linprog with method = 'simplex'.  Number of columns in A_ub must be equal to the size of c")
    # Assigning a type to the variable 'message' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'message', str_191955)
    # SSA join for if statement (line 711)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'status' (line 716)
    status_191956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 7), 'status')
    int_191957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 17), 'int')
    # Applying the binary operator '!=' (line 716)
    result_ne_191958 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 7), '!=', status_191956, int_191957)
    
    # Testing the type of an if condition (line 716)
    if_condition_191959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 716, 4), result_ne_191958)
    # Assigning a type to the variable 'if_condition_191959' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 4), 'if_condition_191959', if_condition_191959)
    # SSA begins for if statement (line 716)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 718)
    # Processing the call arguments (line 718)
    # Getting the type of 'message' (line 718)
    message_191961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 25), 'message', False)
    # Processing the call keyword arguments (line 718)
    kwargs_191962 = {}
    # Getting the type of 'ValueError' (line 718)
    ValueError_191960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 718)
    ValueError_call_result_191963 = invoke(stypy.reporting.localization.Localization(__file__, 718, 14), ValueError_191960, *[message_191961], **kwargs_191962)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 718, 8), ValueError_call_result_191963, 'raise parameter', BaseException)
    # SSA join for if statement (line 716)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 721):
    
    # Assigning a Call to a Name (line 721):
    
    # Call to zeros(...): (line 721)
    # Processing the call arguments (line 721)
    
    # Obtaining an instance of the builtin type 'list' (line 721)
    list_191966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 721)
    # Adding element type (line 721)
    # Getting the type of 'm' (line 721)
    m_191967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 18), 'm', False)
    int_191968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 20), 'int')
    # Applying the binary operator '+' (line 721)
    result_add_191969 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 18), '+', m_191967, int_191968)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 721, 17), list_191966, result_add_191969)
    # Adding element type (line 721)
    # Getting the type of 'n' (line 721)
    n_191970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 23), 'n', False)
    # Getting the type of 'n_slack' (line 721)
    n_slack_191971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 25), 'n_slack', False)
    # Applying the binary operator '+' (line 721)
    result_add_191972 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 23), '+', n_191970, n_slack_191971)
    
    # Getting the type of 'n_artificial' (line 721)
    n_artificial_191973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 33), 'n_artificial', False)
    # Applying the binary operator '+' (line 721)
    result_add_191974 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 32), '+', result_add_191972, n_artificial_191973)
    
    int_191975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 46), 'int')
    # Applying the binary operator '+' (line 721)
    result_add_191976 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 45), '+', result_add_191974, int_191975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 721, 17), list_191966, result_add_191976)
    
    # Processing the call keyword arguments (line 721)
    kwargs_191977 = {}
    # Getting the type of 'np' (line 721)
    np_191964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 721)
    zeros_191965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 8), np_191964, 'zeros')
    # Calling zeros(args, kwargs) (line 721)
    zeros_call_result_191978 = invoke(stypy.reporting.localization.Localization(__file__, 721, 8), zeros_191965, *[list_191966], **kwargs_191977)
    
    # Assigning a type to the variable 'T' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'T', zeros_call_result_191978)
    
    # Assigning a Name to a Subscript (line 724):
    
    # Assigning a Name to a Subscript (line 724):
    # Getting the type of 'cc' (line 724)
    cc_191979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 16), 'cc')
    # Getting the type of 'T' (line 724)
    T_191980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'T')
    int_191981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 6), 'int')
    # Getting the type of 'n' (line 724)
    n_191982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 11), 'n')
    slice_191983 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 724, 4), None, n_191982, None)
    # Storing an element on a container (line 724)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 724, 4), T_191980, ((int_191981, slice_191983), cc_191979))
    
    # Assigning a Name to a Subscript (line 725):
    
    # Assigning a Name to a Subscript (line 725):
    # Getting the type of 'f0' (line 725)
    f0_191984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 16), 'f0')
    # Getting the type of 'T' (line 725)
    T_191985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'T')
    
    # Obtaining an instance of the builtin type 'tuple' (line 725)
    tuple_191986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 725)
    # Adding element type (line 725)
    int_191987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 6), tuple_191986, int_191987)
    # Adding element type (line 725)
    int_191988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 6), tuple_191986, int_191988)
    
    # Storing an element on a container (line 725)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 4), T_191985, (tuple_191986, f0_191984))
    
    # Assigning a Subscript to a Name (line 727):
    
    # Assigning a Subscript to a Name (line 727):
    
    # Obtaining the type of the subscript
    int_191989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 11), 'int')
    slice_191990 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 727, 8), None, int_191989, None)
    int_191991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 15), 'int')
    # Getting the type of 'T' (line 727)
    T_191992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'T')
    # Obtaining the member '__getitem__' of a type (line 727)
    getitem___191993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 8), T_191992, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 727)
    subscript_call_result_191994 = invoke(stypy.reporting.localization.Localization(__file__, 727, 8), getitem___191993, (slice_191990, int_191991))
    
    # Assigning a type to the variable 'b' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'b', subscript_call_result_191994)
    
    
    # Getting the type of 'meq' (line 729)
    meq_191995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 7), 'meq')
    int_191996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 13), 'int')
    # Applying the binary operator '>' (line 729)
    result_gt_191997 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 7), '>', meq_191995, int_191996)
    
    # Testing the type of an if condition (line 729)
    if_condition_191998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 4), result_gt_191997)
    # Assigning a type to the variable 'if_condition_191998' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'if_condition_191998', if_condition_191998)
    # SSA begins for if statement (line 729)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 731):
    
    # Assigning a Name to a Subscript (line 731):
    # Getting the type of 'Aeq' (line 731)
    Aeq_191999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 22), 'Aeq')
    # Getting the type of 'T' (line 731)
    T_192000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'T')
    # Getting the type of 'meq' (line 731)
    meq_192001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 11), 'meq')
    slice_192002 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 731, 8), None, meq_192001, None)
    # Getting the type of 'n' (line 731)
    n_192003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 17), 'n')
    slice_192004 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 731, 8), None, n_192003, None)
    # Storing an element on a container (line 731)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 8), T_192000, ((slice_192002, slice_192004), Aeq_191999))
    
    # Assigning a Name to a Subscript (line 733):
    
    # Assigning a Name to a Subscript (line 733):
    # Getting the type of 'beq' (line 733)
    beq_192005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 18), 'beq')
    # Getting the type of 'b' (line 733)
    b_192006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'b')
    # Getting the type of 'meq' (line 733)
    meq_192007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 11), 'meq')
    slice_192008 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 733, 8), None, meq_192007, None)
    # Storing an element on a container (line 733)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 8), b_192006, (slice_192008, beq_192005))
    # SSA join for if statement (line 729)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mub' (line 734)
    mub_192009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 7), 'mub')
    int_192010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 13), 'int')
    # Applying the binary operator '>' (line 734)
    result_gt_192011 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 7), '>', mub_192009, int_192010)
    
    # Testing the type of an if condition (line 734)
    if_condition_192012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 4), result_gt_192011)
    # Assigning a type to the variable 'if_condition_192012' (line 734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 4), 'if_condition_192012', if_condition_192012)
    # SSA begins for if statement (line 734)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 736):
    
    # Assigning a Name to a Subscript (line 736):
    # Getting the type of 'Aub' (line 736)
    Aub_192013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 29), 'Aub')
    # Getting the type of 'T' (line 736)
    T_192014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'T')
    # Getting the type of 'meq' (line 736)
    meq_192015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 10), 'meq')
    # Getting the type of 'meq' (line 736)
    meq_192016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 14), 'meq')
    # Getting the type of 'mub' (line 736)
    mub_192017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 18), 'mub')
    # Applying the binary operator '+' (line 736)
    result_add_192018 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 14), '+', meq_192016, mub_192017)
    
    slice_192019 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 736, 8), meq_192015, result_add_192018, None)
    # Getting the type of 'n' (line 736)
    n_192020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 24), 'n')
    slice_192021 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 736, 8), None, n_192020, None)
    # Storing an element on a container (line 736)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 8), T_192014, ((slice_192019, slice_192021), Aub_192013))
    
    # Assigning a Name to a Subscript (line 738):
    
    # Assigning a Name to a Subscript (line 738):
    # Getting the type of 'bub' (line 738)
    bub_192022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 25), 'bub')
    # Getting the type of 'b' (line 738)
    b_192023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'b')
    # Getting the type of 'meq' (line 738)
    meq_192024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 10), 'meq')
    # Getting the type of 'meq' (line 738)
    meq_192025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 14), 'meq')
    # Getting the type of 'mub' (line 738)
    mub_192026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 18), 'mub')
    # Applying the binary operator '+' (line 738)
    result_add_192027 = python_operator(stypy.reporting.localization.Localization(__file__, 738, 14), '+', meq_192025, mub_192026)
    
    slice_192028 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 738, 8), meq_192024, result_add_192027, None)
    # Storing an element on a container (line 738)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 8), b_192023, (slice_192028, bub_192022))
    
    # Call to fill_diagonal(...): (line 740)
    # Processing the call arguments (line 740)
    
    # Obtaining the type of the subscript
    # Getting the type of 'meq' (line 740)
    meq_192031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 27), 'meq', False)
    # Getting the type of 'm' (line 740)
    m_192032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 31), 'm', False)
    slice_192033 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 740, 25), meq_192031, m_192032, None)
    # Getting the type of 'n' (line 740)
    n_192034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 34), 'n', False)
    # Getting the type of 'n' (line 740)
    n_192035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 36), 'n', False)
    # Getting the type of 'n_slack' (line 740)
    n_slack_192036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 38), 'n_slack', False)
    # Applying the binary operator '+' (line 740)
    result_add_192037 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 36), '+', n_192035, n_slack_192036)
    
    slice_192038 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 740, 25), n_192034, result_add_192037, None)
    # Getting the type of 'T' (line 740)
    T_192039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 25), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 740)
    getitem___192040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 25), T_192039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 740)
    subscript_call_result_192041 = invoke(stypy.reporting.localization.Localization(__file__, 740, 25), getitem___192040, (slice_192033, slice_192038))
    
    int_192042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 48), 'int')
    # Processing the call keyword arguments (line 740)
    kwargs_192043 = {}
    # Getting the type of 'np' (line 740)
    np_192029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 8), 'np', False)
    # Obtaining the member 'fill_diagonal' of a type (line 740)
    fill_diagonal_192030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 8), np_192029, 'fill_diagonal')
    # Calling fill_diagonal(args, kwargs) (line 740)
    fill_diagonal_call_result_192044 = invoke(stypy.reporting.localization.Localization(__file__, 740, 8), fill_diagonal_192030, *[subscript_call_result_192041, int_192042], **kwargs_192043)
    
    # SSA join for if statement (line 734)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 746):
    
    # Assigning a Num to a Name (line 746):
    int_192045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 14), 'int')
    # Assigning a type to the variable 'slcount' (line 746)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 4), 'slcount', int_192045)
    
    # Assigning a Num to a Name (line 747):
    
    # Assigning a Num to a Name (line 747):
    int_192046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 14), 'int')
    # Assigning a type to the variable 'avcount' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'avcount', int_192046)
    
    # Assigning a Call to a Name (line 748):
    
    # Assigning a Call to a Name (line 748):
    
    # Call to zeros(...): (line 748)
    # Processing the call arguments (line 748)
    # Getting the type of 'm' (line 748)
    m_192049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 21), 'm', False)
    # Processing the call keyword arguments (line 748)
    # Getting the type of 'int' (line 748)
    int_192050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 30), 'int', False)
    keyword_192051 = int_192050
    kwargs_192052 = {'dtype': keyword_192051}
    # Getting the type of 'np' (line 748)
    np_192047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 748)
    zeros_192048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 12), np_192047, 'zeros')
    # Calling zeros(args, kwargs) (line 748)
    zeros_call_result_192053 = invoke(stypy.reporting.localization.Localization(__file__, 748, 12), zeros_192048, *[m_192049], **kwargs_192052)
    
    # Assigning a type to the variable 'basis' (line 748)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'basis', zeros_call_result_192053)
    
    # Assigning a Call to a Name (line 749):
    
    # Assigning a Call to a Name (line 749):
    
    # Call to zeros(...): (line 749)
    # Processing the call arguments (line 749)
    # Getting the type of 'n_artificial' (line 749)
    n_artificial_192056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 28), 'n_artificial', False)
    # Processing the call keyword arguments (line 749)
    # Getting the type of 'int' (line 749)
    int_192057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 48), 'int', False)
    keyword_192058 = int_192057
    kwargs_192059 = {'dtype': keyword_192058}
    # Getting the type of 'np' (line 749)
    np_192054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 749)
    zeros_192055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 19), np_192054, 'zeros')
    # Calling zeros(args, kwargs) (line 749)
    zeros_call_result_192060 = invoke(stypy.reporting.localization.Localization(__file__, 749, 19), zeros_192055, *[n_artificial_192056], **kwargs_192059)
    
    # Assigning a type to the variable 'r_artificial' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'r_artificial', zeros_call_result_192060)
    
    
    # Call to range(...): (line 750)
    # Processing the call arguments (line 750)
    # Getting the type of 'm' (line 750)
    m_192062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 19), 'm', False)
    # Processing the call keyword arguments (line 750)
    kwargs_192063 = {}
    # Getting the type of 'range' (line 750)
    range_192061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 13), 'range', False)
    # Calling range(args, kwargs) (line 750)
    range_call_result_192064 = invoke(stypy.reporting.localization.Localization(__file__, 750, 13), range_192061, *[m_192062], **kwargs_192063)
    
    # Testing the type of a for loop iterable (line 750)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 750, 4), range_call_result_192064)
    # Getting the type of the for loop variable (line 750)
    for_loop_var_192065 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 750, 4), range_call_result_192064)
    # Assigning a type to the variable 'i' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'i', for_loop_var_192065)
    # SSA begins for a for statement (line 750)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'i' (line 751)
    i_192066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 11), 'i')
    # Getting the type of 'meq' (line 751)
    meq_192067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 15), 'meq')
    # Applying the binary operator '<' (line 751)
    result_lt_192068 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 11), '<', i_192066, meq_192067)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 751)
    i_192069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 24), 'i')
    # Getting the type of 'b' (line 751)
    b_192070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 22), 'b')
    # Obtaining the member '__getitem__' of a type (line 751)
    getitem___192071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 22), b_192070, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 751)
    subscript_call_result_192072 = invoke(stypy.reporting.localization.Localization(__file__, 751, 22), getitem___192071, i_192069)
    
    int_192073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 29), 'int')
    # Applying the binary operator '<' (line 751)
    result_lt_192074 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 22), '<', subscript_call_result_192072, int_192073)
    
    # Applying the binary operator 'or' (line 751)
    result_or_keyword_192075 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 11), 'or', result_lt_192068, result_lt_192074)
    
    # Testing the type of an if condition (line 751)
    if_condition_192076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 751, 8), result_or_keyword_192075)
    # Assigning a type to the variable 'if_condition_192076' (line 751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'if_condition_192076', if_condition_192076)
    # SSA begins for if statement (line 751)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 753):
    
    # Assigning a BinOp to a Subscript (line 753):
    # Getting the type of 'n' (line 753)
    n_192077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 23), 'n')
    # Getting the type of 'n_slack' (line 753)
    n_slack_192078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 25), 'n_slack')
    # Applying the binary operator '+' (line 753)
    result_add_192079 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 23), '+', n_192077, n_slack_192078)
    
    # Getting the type of 'avcount' (line 753)
    avcount_192080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 33), 'avcount')
    # Applying the binary operator '+' (line 753)
    result_add_192081 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 32), '+', result_add_192079, avcount_192080)
    
    # Getting the type of 'basis' (line 753)
    basis_192082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 12), 'basis')
    # Getting the type of 'i' (line 753)
    i_192083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 18), 'i')
    # Storing an element on a container (line 753)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 753, 12), basis_192082, (i_192083, result_add_192081))
    
    # Assigning a Name to a Subscript (line 754):
    
    # Assigning a Name to a Subscript (line 754):
    # Getting the type of 'i' (line 754)
    i_192084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 36), 'i')
    # Getting the type of 'r_artificial' (line 754)
    r_artificial_192085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 12), 'r_artificial')
    # Getting the type of 'avcount' (line 754)
    avcount_192086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 25), 'avcount')
    # Storing an element on a container (line 754)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 754, 12), r_artificial_192085, (avcount_192086, i_192084))
    
    # Getting the type of 'avcount' (line 755)
    avcount_192087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'avcount')
    int_192088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 23), 'int')
    # Applying the binary operator '+=' (line 755)
    result_iadd_192089 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 12), '+=', avcount_192087, int_192088)
    # Assigning a type to the variable 'avcount' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'avcount', result_iadd_192089)
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 756)
    i_192090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 17), 'i')
    # Getting the type of 'b' (line 756)
    b_192091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 15), 'b')
    # Obtaining the member '__getitem__' of a type (line 756)
    getitem___192092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 15), b_192091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 756)
    subscript_call_result_192093 = invoke(stypy.reporting.localization.Localization(__file__, 756, 15), getitem___192092, i_192090)
    
    int_192094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 22), 'int')
    # Applying the binary operator '<' (line 756)
    result_lt_192095 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 15), '<', subscript_call_result_192093, int_192094)
    
    # Testing the type of an if condition (line 756)
    if_condition_192096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 756, 12), result_lt_192095)
    # Assigning a type to the variable 'if_condition_192096' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'if_condition_192096', if_condition_192096)
    # SSA begins for if statement (line 756)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'b' (line 757)
    b_192097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 16), 'b')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 757)
    i_192098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 18), 'i')
    # Getting the type of 'b' (line 757)
    b_192099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 16), 'b')
    # Obtaining the member '__getitem__' of a type (line 757)
    getitem___192100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 16), b_192099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 757)
    subscript_call_result_192101 = invoke(stypy.reporting.localization.Localization(__file__, 757, 16), getitem___192100, i_192098)
    
    int_192102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 24), 'int')
    # Applying the binary operator '*=' (line 757)
    result_imul_192103 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 16), '*=', subscript_call_result_192101, int_192102)
    # Getting the type of 'b' (line 757)
    b_192104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 16), 'b')
    # Getting the type of 'i' (line 757)
    i_192105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 18), 'i')
    # Storing an element on a container (line 757)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 757, 16), b_192104, (i_192105, result_imul_192103))
    
    
    # Getting the type of 'T' (line 758)
    T_192106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'T')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 758)
    i_192107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 18), 'i')
    int_192108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 22), 'int')
    slice_192109 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 758, 16), None, int_192108, None)
    # Getting the type of 'T' (line 758)
    T_192110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'T')
    # Obtaining the member '__getitem__' of a type (line 758)
    getitem___192111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 16), T_192110, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 758)
    subscript_call_result_192112 = invoke(stypy.reporting.localization.Localization(__file__, 758, 16), getitem___192111, (i_192107, slice_192109))
    
    int_192113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 29), 'int')
    # Applying the binary operator '*=' (line 758)
    result_imul_192114 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 16), '*=', subscript_call_result_192112, int_192113)
    # Getting the type of 'T' (line 758)
    T_192115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'T')
    # Getting the type of 'i' (line 758)
    i_192116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 18), 'i')
    int_192117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 22), 'int')
    slice_192118 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 758, 16), None, int_192117, None)
    # Storing an element on a container (line 758)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 16), T_192115, ((i_192116, slice_192118), result_imul_192114))
    
    # SSA join for if statement (line 756)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 759):
    
    # Assigning a Num to a Subscript (line 759):
    int_192119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 29), 'int')
    # Getting the type of 'T' (line 759)
    T_192120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 12), 'T')
    
    # Obtaining an instance of the builtin type 'tuple' (line 759)
    tuple_192121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 759)
    # Adding element type (line 759)
    # Getting the type of 'i' (line 759)
    i_192122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 14), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 759, 14), tuple_192121, i_192122)
    # Adding element type (line 759)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 759)
    i_192123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 23), 'i')
    # Getting the type of 'basis' (line 759)
    basis_192124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 17), 'basis')
    # Obtaining the member '__getitem__' of a type (line 759)
    getitem___192125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 17), basis_192124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 759)
    subscript_call_result_192126 = invoke(stypy.reporting.localization.Localization(__file__, 759, 17), getitem___192125, i_192123)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 759, 14), tuple_192121, subscript_call_result_192126)
    
    # Storing an element on a container (line 759)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 759, 12), T_192120, (tuple_192121, int_192119))
    
    # Assigning a Num to a Subscript (line 760):
    
    # Assigning a Num to a Subscript (line 760):
    int_192127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 30), 'int')
    # Getting the type of 'T' (line 760)
    T_192128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'T')
    
    # Obtaining an instance of the builtin type 'tuple' (line 760)
    tuple_192129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 760)
    # Adding element type (line 760)
    int_192130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 14), tuple_192129, int_192130)
    # Adding element type (line 760)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 760)
    i_192131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 24), 'i')
    # Getting the type of 'basis' (line 760)
    basis_192132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 18), 'basis')
    # Obtaining the member '__getitem__' of a type (line 760)
    getitem___192133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 18), basis_192132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 760)
    subscript_call_result_192134 = invoke(stypy.reporting.localization.Localization(__file__, 760, 18), getitem___192133, i_192131)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 14), tuple_192129, subscript_call_result_192134)
    
    # Storing an element on a container (line 760)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 12), T_192128, (tuple_192129, int_192127))
    # SSA branch for the else part of an if statement (line 751)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Subscript (line 763):
    
    # Assigning a BinOp to a Subscript (line 763):
    # Getting the type of 'n' (line 763)
    n_192135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 23), 'n')
    # Getting the type of 'slcount' (line 763)
    slcount_192136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 25), 'slcount')
    # Applying the binary operator '+' (line 763)
    result_add_192137 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 23), '+', n_192135, slcount_192136)
    
    # Getting the type of 'basis' (line 763)
    basis_192138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'basis')
    # Getting the type of 'i' (line 763)
    i_192139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 18), 'i')
    # Storing an element on a container (line 763)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 12), basis_192138, (i_192139, result_add_192137))
    
    # Getting the type of 'slcount' (line 764)
    slcount_192140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'slcount')
    int_192141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 23), 'int')
    # Applying the binary operator '+=' (line 764)
    result_iadd_192142 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 12), '+=', slcount_192140, int_192141)
    # Assigning a type to the variable 'slcount' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'slcount', result_iadd_192142)
    
    # SSA join for if statement (line 751)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'r_artificial' (line 768)
    r_artificial_192143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 13), 'r_artificial')
    # Testing the type of a for loop iterable (line 768)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 768, 4), r_artificial_192143)
    # Getting the type of the for loop variable (line 768)
    for_loop_var_192144 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 768, 4), r_artificial_192143)
    # Assigning a type to the variable 'r' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'r', for_loop_var_192144)
    # SSA begins for a for statement (line 768)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 769):
    
    # Assigning a BinOp to a Subscript (line 769):
    
    # Obtaining the type of the subscript
    int_192145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 21), 'int')
    slice_192146 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 769, 19), None, None, None)
    # Getting the type of 'T' (line 769)
    T_192147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 19), 'T')
    # Obtaining the member '__getitem__' of a type (line 769)
    getitem___192148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 19), T_192147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 769)
    subscript_call_result_192149 = invoke(stypy.reporting.localization.Localization(__file__, 769, 19), getitem___192148, (int_192145, slice_192146))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 769)
    r_192150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 32), 'r')
    slice_192151 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 769, 30), None, None, None)
    # Getting the type of 'T' (line 769)
    T_192152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 30), 'T')
    # Obtaining the member '__getitem__' of a type (line 769)
    getitem___192153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 30), T_192152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 769)
    subscript_call_result_192154 = invoke(stypy.reporting.localization.Localization(__file__, 769, 30), getitem___192153, (r_192150, slice_192151))
    
    # Applying the binary operator '-' (line 769)
    result_sub_192155 = python_operator(stypy.reporting.localization.Localization(__file__, 769, 19), '-', subscript_call_result_192149, subscript_call_result_192154)
    
    # Getting the type of 'T' (line 769)
    T_192156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 8), 'T')
    int_192157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 10), 'int')
    slice_192158 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 769, 8), None, None, None)
    # Storing an element on a container (line 769)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 8), T_192156, ((int_192157, slice_192158), result_sub_192155))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 771):
    
    # Assigning a Subscript to a Name (line 771):
    
    # Obtaining the type of the subscript
    int_192159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 4), 'int')
    
    # Call to _solve_simplex(...): (line 771)
    # Processing the call arguments (line 771)
    # Getting the type of 'T' (line 771)
    T_192161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 34), 'T', False)
    # Getting the type of 'n' (line 771)
    n_192162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 37), 'n', False)
    # Getting the type of 'basis' (line 771)
    basis_192163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 40), 'basis', False)
    # Processing the call keyword arguments (line 771)
    int_192164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 53), 'int')
    keyword_192165 = int_192164
    # Getting the type of 'callback' (line 771)
    callback_192166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 65), 'callback', False)
    keyword_192167 = callback_192166
    # Getting the type of 'maxiter' (line 772)
    maxiter_192168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 42), 'maxiter', False)
    keyword_192169 = maxiter_192168
    # Getting the type of 'tol' (line 772)
    tol_192170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 55), 'tol', False)
    keyword_192171 = tol_192170
    # Getting the type of 'bland' (line 772)
    bland_192172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 66), 'bland', False)
    keyword_192173 = bland_192172
    kwargs_192174 = {'phase': keyword_192165, 'callback': keyword_192167, 'bland': keyword_192173, 'tol': keyword_192171, 'maxiter': keyword_192169}
    # Getting the type of '_solve_simplex' (line 771)
    _solve_simplex_192160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 19), '_solve_simplex', False)
    # Calling _solve_simplex(args, kwargs) (line 771)
    _solve_simplex_call_result_192175 = invoke(stypy.reporting.localization.Localization(__file__, 771, 19), _solve_simplex_192160, *[T_192161, n_192162, basis_192163], **kwargs_192174)
    
    # Obtaining the member '__getitem__' of a type (line 771)
    getitem___192176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 4), _solve_simplex_call_result_192175, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 771)
    subscript_call_result_192177 = invoke(stypy.reporting.localization.Localization(__file__, 771, 4), getitem___192176, int_192159)
    
    # Assigning a type to the variable 'tuple_var_assignment_190483' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'tuple_var_assignment_190483', subscript_call_result_192177)
    
    # Assigning a Subscript to a Name (line 771):
    
    # Obtaining the type of the subscript
    int_192178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 4), 'int')
    
    # Call to _solve_simplex(...): (line 771)
    # Processing the call arguments (line 771)
    # Getting the type of 'T' (line 771)
    T_192180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 34), 'T', False)
    # Getting the type of 'n' (line 771)
    n_192181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 37), 'n', False)
    # Getting the type of 'basis' (line 771)
    basis_192182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 40), 'basis', False)
    # Processing the call keyword arguments (line 771)
    int_192183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 53), 'int')
    keyword_192184 = int_192183
    # Getting the type of 'callback' (line 771)
    callback_192185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 65), 'callback', False)
    keyword_192186 = callback_192185
    # Getting the type of 'maxiter' (line 772)
    maxiter_192187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 42), 'maxiter', False)
    keyword_192188 = maxiter_192187
    # Getting the type of 'tol' (line 772)
    tol_192189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 55), 'tol', False)
    keyword_192190 = tol_192189
    # Getting the type of 'bland' (line 772)
    bland_192191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 66), 'bland', False)
    keyword_192192 = bland_192191
    kwargs_192193 = {'phase': keyword_192184, 'callback': keyword_192186, 'bland': keyword_192192, 'tol': keyword_192190, 'maxiter': keyword_192188}
    # Getting the type of '_solve_simplex' (line 771)
    _solve_simplex_192179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 19), '_solve_simplex', False)
    # Calling _solve_simplex(args, kwargs) (line 771)
    _solve_simplex_call_result_192194 = invoke(stypy.reporting.localization.Localization(__file__, 771, 19), _solve_simplex_192179, *[T_192180, n_192181, basis_192182], **kwargs_192193)
    
    # Obtaining the member '__getitem__' of a type (line 771)
    getitem___192195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 4), _solve_simplex_call_result_192194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 771)
    subscript_call_result_192196 = invoke(stypy.reporting.localization.Localization(__file__, 771, 4), getitem___192195, int_192178)
    
    # Assigning a type to the variable 'tuple_var_assignment_190484' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'tuple_var_assignment_190484', subscript_call_result_192196)
    
    # Assigning a Name to a Name (line 771):
    # Getting the type of 'tuple_var_assignment_190483' (line 771)
    tuple_var_assignment_190483_192197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'tuple_var_assignment_190483')
    # Assigning a type to the variable 'nit1' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'nit1', tuple_var_assignment_190483_192197)
    
    # Assigning a Name to a Name (line 771):
    # Getting the type of 'tuple_var_assignment_190484' (line 771)
    tuple_var_assignment_190484_192198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'tuple_var_assignment_190484')
    # Assigning a type to the variable 'status' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 10), 'status', tuple_var_assignment_190484_192198)
    
    
    
    # Call to abs(...): (line 776)
    # Processing the call arguments (line 776)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 776)
    tuple_192200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 776)
    # Adding element type (line 776)
    int_192201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 776, 13), tuple_192200, int_192201)
    # Adding element type (line 776)
    int_192202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 776, 13), tuple_192200, int_192202)
    
    # Getting the type of 'T' (line 776)
    T_192203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 11), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 776)
    getitem___192204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 11), T_192203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 776)
    subscript_call_result_192205 = invoke(stypy.reporting.localization.Localization(__file__, 776, 11), getitem___192204, tuple_192200)
    
    # Processing the call keyword arguments (line 776)
    kwargs_192206 = {}
    # Getting the type of 'abs' (line 776)
    abs_192199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 7), 'abs', False)
    # Calling abs(args, kwargs) (line 776)
    abs_call_result_192207 = invoke(stypy.reporting.localization.Localization(__file__, 776, 7), abs_192199, *[subscript_call_result_192205], **kwargs_192206)
    
    # Getting the type of 'tol' (line 776)
    tol_192208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 24), 'tol')
    # Applying the binary operator '<' (line 776)
    result_lt_192209 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 7), '<', abs_call_result_192207, tol_192208)
    
    # Testing the type of an if condition (line 776)
    if_condition_192210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 776, 4), result_lt_192209)
    # Assigning a type to the variable 'if_condition_192210' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'if_condition_192210', if_condition_192210)
    # SSA begins for if statement (line 776)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 778):
    
    # Assigning a Subscript to a Name (line 778):
    
    # Obtaining the type of the subscript
    int_192211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 15), 'int')
    slice_192212 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 778, 12), None, int_192211, None)
    slice_192213 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 778, 12), None, None, None)
    # Getting the type of 'T' (line 778)
    T_192214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 12), 'T')
    # Obtaining the member '__getitem__' of a type (line 778)
    getitem___192215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 12), T_192214, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 778)
    subscript_call_result_192216 = invoke(stypy.reporting.localization.Localization(__file__, 778, 12), getitem___192215, (slice_192212, slice_192213))
    
    # Assigning a type to the variable 'T' (line 778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 8), 'T', subscript_call_result_192216)
    
    # Assigning a Call to a Name (line 780):
    
    # Assigning a Call to a Name (line 780):
    
    # Call to delete(...): (line 780)
    # Processing the call arguments (line 780)
    # Getting the type of 'T' (line 780)
    T_192219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 22), 'T', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 780)
    n_192220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 31), 'n', False)
    # Getting the type of 'n_slack' (line 780)
    n_slack_192221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 33), 'n_slack', False)
    # Applying the binary operator '+' (line 780)
    result_add_192222 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 31), '+', n_192220, n_slack_192221)
    
    # Getting the type of 'n' (line 780)
    n_192223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 41), 'n', False)
    # Getting the type of 'n_slack' (line 780)
    n_slack_192224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 43), 'n_slack', False)
    # Applying the binary operator '+' (line 780)
    result_add_192225 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 41), '+', n_192223, n_slack_192224)
    
    # Getting the type of 'n_artificial' (line 780)
    n_artificial_192226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 51), 'n_artificial', False)
    # Applying the binary operator '+' (line 780)
    result_add_192227 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 50), '+', result_add_192225, n_artificial_192226)
    
    slice_192228 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 780, 25), result_add_192222, result_add_192227, None)
    # Getting the type of 'np' (line 780)
    np_192229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 25), 'np', False)
    # Obtaining the member 's_' of a type (line 780)
    s__192230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 25), np_192229, 's_')
    # Obtaining the member '__getitem__' of a type (line 780)
    getitem___192231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 25), s__192230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 780)
    subscript_call_result_192232 = invoke(stypy.reporting.localization.Localization(__file__, 780, 25), getitem___192231, slice_192228)
    
    int_192233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 66), 'int')
    # Processing the call keyword arguments (line 780)
    kwargs_192234 = {}
    # Getting the type of 'np' (line 780)
    np_192217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 12), 'np', False)
    # Obtaining the member 'delete' of a type (line 780)
    delete_192218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 12), np_192217, 'delete')
    # Calling delete(args, kwargs) (line 780)
    delete_call_result_192235 = invoke(stypy.reporting.localization.Localization(__file__, 780, 12), delete_192218, *[T_192219, subscript_call_result_192232, int_192233], **kwargs_192234)
    
    # Assigning a type to the variable 'T' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'T', delete_call_result_192235)
    # SSA branch for the else part of an if statement (line 776)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 783):
    
    # Assigning a Num to a Name (line 783):
    int_192236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 17), 'int')
    # Assigning a type to the variable 'status' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'status', int_192236)
    # SSA join for if statement (line 776)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'status' (line 785)
    status_192237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 7), 'status')
    int_192238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 17), 'int')
    # Applying the binary operator '!=' (line 785)
    result_ne_192239 = python_operator(stypy.reporting.localization.Localization(__file__, 785, 7), '!=', status_192237, int_192238)
    
    # Testing the type of an if condition (line 785)
    if_condition_192240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 785, 4), result_ne_192239)
    # Assigning a type to the variable 'if_condition_192240' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'if_condition_192240', if_condition_192240)
    # SSA begins for if statement (line 785)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 786):
    
    # Assigning a Subscript to a Name (line 786):
    
    # Obtaining the type of the subscript
    # Getting the type of 'status' (line 786)
    status_192241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 27), 'status')
    # Getting the type of 'messages' (line 786)
    messages_192242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 18), 'messages')
    # Obtaining the member '__getitem__' of a type (line 786)
    getitem___192243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 18), messages_192242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 786)
    subscript_call_result_192244 = invoke(stypy.reporting.localization.Localization(__file__, 786, 18), getitem___192243, status_192241)
    
    # Assigning a type to the variable 'message' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'message', subscript_call_result_192244)
    
    # Getting the type of 'disp' (line 787)
    disp_192245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 11), 'disp')
    # Testing the type of an if condition (line 787)
    if_condition_192246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 787, 8), disp_192245)
    # Assigning a type to the variable 'if_condition_192246' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'if_condition_192246', if_condition_192246)
    # SSA begins for if statement (line 787)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 788)
    # Processing the call arguments (line 788)
    # Getting the type of 'message' (line 788)
    message_192248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 18), 'message', False)
    # Processing the call keyword arguments (line 788)
    kwargs_192249 = {}
    # Getting the type of 'print' (line 788)
    print_192247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 12), 'print', False)
    # Calling print(args, kwargs) (line 788)
    print_call_result_192250 = invoke(stypy.reporting.localization.Localization(__file__, 788, 12), print_192247, *[message_192248], **kwargs_192249)
    
    # SSA join for if statement (line 787)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to OptimizeResult(...): (line 789)
    # Processing the call keyword arguments (line 789)
    # Getting the type of 'np' (line 789)
    np_192252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 32), 'np', False)
    # Obtaining the member 'nan' of a type (line 789)
    nan_192253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 32), np_192252, 'nan')
    keyword_192254 = nan_192253
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 789)
    tuple_192255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 789)
    # Adding element type (line 789)
    int_192256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 789, 47), tuple_192255, int_192256)
    # Adding element type (line 789)
    int_192257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 789, 47), tuple_192255, int_192257)
    
    # Getting the type of 'T' (line 789)
    T_192258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 45), 'T', False)
    # Obtaining the member '__getitem__' of a type (line 789)
    getitem___192259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 45), T_192258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 789)
    subscript_call_result_192260 = invoke(stypy.reporting.localization.Localization(__file__, 789, 45), getitem___192259, tuple_192255)
    
    # Applying the 'usub' unary operator (line 789)
    result___neg___192261 = python_operator(stypy.reporting.localization.Localization(__file__, 789, 44), 'usub', subscript_call_result_192260)
    
    keyword_192262 = result___neg___192261
    # Getting the type of 'nit1' (line 789)
    nit1_192263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 60), 'nit1', False)
    keyword_192264 = nit1_192263
    # Getting the type of 'status' (line 790)
    status_192265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 37), 'status', False)
    keyword_192266 = status_192265
    # Getting the type of 'message' (line 790)
    message_192267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 53), 'message', False)
    keyword_192268 = message_192267
    # Getting the type of 'False' (line 790)
    False_192269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 70), 'False', False)
    keyword_192270 = False_192269
    kwargs_192271 = {'status': keyword_192266, 'success': keyword_192270, 'fun': keyword_192262, 'x': keyword_192254, 'message': keyword_192268, 'nit': keyword_192264}
    # Getting the type of 'OptimizeResult' (line 789)
    OptimizeResult_192251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 15), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 789)
    OptimizeResult_call_result_192272 = invoke(stypy.reporting.localization.Localization(__file__, 789, 15), OptimizeResult_192251, *[], **kwargs_192271)
    
    # Assigning a type to the variable 'stypy_return_type' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'stypy_return_type', OptimizeResult_call_result_192272)
    # SSA join for if statement (line 785)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 793):
    
    # Assigning a Subscript to a Name (line 793):
    
    # Obtaining the type of the subscript
    int_192273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 4), 'int')
    
    # Call to _solve_simplex(...): (line 793)
    # Processing the call arguments (line 793)
    # Getting the type of 'T' (line 793)
    T_192275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 34), 'T', False)
    # Getting the type of 'n' (line 793)
    n_192276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 37), 'n', False)
    # Getting the type of 'basis' (line 793)
    basis_192277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 40), 'basis', False)
    # Processing the call keyword arguments (line 793)
    # Getting the type of 'maxiter' (line 793)
    maxiter_192278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 55), 'maxiter', False)
    # Getting the type of 'nit1' (line 793)
    nit1_192279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 63), 'nit1', False)
    # Applying the binary operator '-' (line 793)
    result_sub_192280 = python_operator(stypy.reporting.localization.Localization(__file__, 793, 55), '-', maxiter_192278, nit1_192279)
    
    keyword_192281 = result_sub_192280
    int_192282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 75), 'int')
    keyword_192283 = int_192282
    # Getting the type of 'callback' (line 794)
    callback_192284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 43), 'callback', False)
    keyword_192285 = callback_192284
    # Getting the type of 'tol' (line 794)
    tol_192286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 57), 'tol', False)
    keyword_192287 = tol_192286
    # Getting the type of 'nit1' (line 794)
    nit1_192288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 67), 'nit1', False)
    keyword_192289 = nit1_192288
    # Getting the type of 'bland' (line 795)
    bland_192290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 40), 'bland', False)
    keyword_192291 = bland_192290
    kwargs_192292 = {'nit0': keyword_192289, 'bland': keyword_192291, 'callback': keyword_192285, 'tol': keyword_192287, 'maxiter': keyword_192281, 'phase': keyword_192283}
    # Getting the type of '_solve_simplex' (line 793)
    _solve_simplex_192274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 19), '_solve_simplex', False)
    # Calling _solve_simplex(args, kwargs) (line 793)
    _solve_simplex_call_result_192293 = invoke(stypy.reporting.localization.Localization(__file__, 793, 19), _solve_simplex_192274, *[T_192275, n_192276, basis_192277], **kwargs_192292)
    
    # Obtaining the member '__getitem__' of a type (line 793)
    getitem___192294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 4), _solve_simplex_call_result_192293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 793)
    subscript_call_result_192295 = invoke(stypy.reporting.localization.Localization(__file__, 793, 4), getitem___192294, int_192273)
    
    # Assigning a type to the variable 'tuple_var_assignment_190485' (line 793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'tuple_var_assignment_190485', subscript_call_result_192295)
    
    # Assigning a Subscript to a Name (line 793):
    
    # Obtaining the type of the subscript
    int_192296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 4), 'int')
    
    # Call to _solve_simplex(...): (line 793)
    # Processing the call arguments (line 793)
    # Getting the type of 'T' (line 793)
    T_192298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 34), 'T', False)
    # Getting the type of 'n' (line 793)
    n_192299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 37), 'n', False)
    # Getting the type of 'basis' (line 793)
    basis_192300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 40), 'basis', False)
    # Processing the call keyword arguments (line 793)
    # Getting the type of 'maxiter' (line 793)
    maxiter_192301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 55), 'maxiter', False)
    # Getting the type of 'nit1' (line 793)
    nit1_192302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 63), 'nit1', False)
    # Applying the binary operator '-' (line 793)
    result_sub_192303 = python_operator(stypy.reporting.localization.Localization(__file__, 793, 55), '-', maxiter_192301, nit1_192302)
    
    keyword_192304 = result_sub_192303
    int_192305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 75), 'int')
    keyword_192306 = int_192305
    # Getting the type of 'callback' (line 794)
    callback_192307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 43), 'callback', False)
    keyword_192308 = callback_192307
    # Getting the type of 'tol' (line 794)
    tol_192309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 57), 'tol', False)
    keyword_192310 = tol_192309
    # Getting the type of 'nit1' (line 794)
    nit1_192311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 67), 'nit1', False)
    keyword_192312 = nit1_192311
    # Getting the type of 'bland' (line 795)
    bland_192313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 40), 'bland', False)
    keyword_192314 = bland_192313
    kwargs_192315 = {'nit0': keyword_192312, 'bland': keyword_192314, 'callback': keyword_192308, 'tol': keyword_192310, 'maxiter': keyword_192304, 'phase': keyword_192306}
    # Getting the type of '_solve_simplex' (line 793)
    _solve_simplex_192297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 19), '_solve_simplex', False)
    # Calling _solve_simplex(args, kwargs) (line 793)
    _solve_simplex_call_result_192316 = invoke(stypy.reporting.localization.Localization(__file__, 793, 19), _solve_simplex_192297, *[T_192298, n_192299, basis_192300], **kwargs_192315)
    
    # Obtaining the member '__getitem__' of a type (line 793)
    getitem___192317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 4), _solve_simplex_call_result_192316, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 793)
    subscript_call_result_192318 = invoke(stypy.reporting.localization.Localization(__file__, 793, 4), getitem___192317, int_192296)
    
    # Assigning a type to the variable 'tuple_var_assignment_190486' (line 793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'tuple_var_assignment_190486', subscript_call_result_192318)
    
    # Assigning a Name to a Name (line 793):
    # Getting the type of 'tuple_var_assignment_190485' (line 793)
    tuple_var_assignment_190485_192319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'tuple_var_assignment_190485')
    # Assigning a type to the variable 'nit2' (line 793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'nit2', tuple_var_assignment_190485_192319)
    
    # Assigning a Name to a Name (line 793):
    # Getting the type of 'tuple_var_assignment_190486' (line 793)
    tuple_var_assignment_190486_192320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'tuple_var_assignment_190486')
    # Assigning a type to the variable 'status' (line 793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 10), 'status', tuple_var_assignment_190486_192320)
    
    # Assigning a Call to a Name (line 797):
    
    # Assigning a Call to a Name (line 797):
    
    # Call to zeros(...): (line 797)
    # Processing the call arguments (line 797)
    # Getting the type of 'n' (line 797)
    n_192323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 24), 'n', False)
    # Getting the type of 'n_slack' (line 797)
    n_slack_192324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 26), 'n_slack', False)
    # Applying the binary operator '+' (line 797)
    result_add_192325 = python_operator(stypy.reporting.localization.Localization(__file__, 797, 24), '+', n_192323, n_slack_192324)
    
    # Getting the type of 'n_artificial' (line 797)
    n_artificial_192326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 34), 'n_artificial', False)
    # Applying the binary operator '+' (line 797)
    result_add_192327 = python_operator(stypy.reporting.localization.Localization(__file__, 797, 33), '+', result_add_192325, n_artificial_192326)
    
    # Processing the call keyword arguments (line 797)
    kwargs_192328 = {}
    # Getting the type of 'np' (line 797)
    np_192321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 797)
    zeros_192322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 15), np_192321, 'zeros')
    # Calling zeros(args, kwargs) (line 797)
    zeros_call_result_192329 = invoke(stypy.reporting.localization.Localization(__file__, 797, 15), zeros_192322, *[result_add_192327], **kwargs_192328)
    
    # Assigning a type to the variable 'solution' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'solution', zeros_call_result_192329)
    
    # Assigning a Subscript to a Subscript (line 798):
    
    # Assigning a Subscript to a Subscript (line 798):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 798)
    m_192330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 29), 'm')
    slice_192331 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 798, 26), None, m_192330, None)
    int_192332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 32), 'int')
    # Getting the type of 'T' (line 798)
    T_192333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 26), 'T')
    # Obtaining the member '__getitem__' of a type (line 798)
    getitem___192334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 26), T_192333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 798)
    subscript_call_result_192335 = invoke(stypy.reporting.localization.Localization(__file__, 798, 26), getitem___192334, (slice_192331, int_192332))
    
    # Getting the type of 'solution' (line 798)
    solution_192336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'solution')
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 798)
    m_192337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 20), 'm')
    slice_192338 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 798, 13), None, m_192337, None)
    # Getting the type of 'basis' (line 798)
    basis_192339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 13), 'basis')
    # Obtaining the member '__getitem__' of a type (line 798)
    getitem___192340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 13), basis_192339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 798)
    subscript_call_result_192341 = invoke(stypy.reporting.localization.Localization(__file__, 798, 13), getitem___192340, slice_192338)
    
    # Storing an element on a container (line 798)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 798, 4), solution_192336, (subscript_call_result_192341, subscript_call_result_192335))
    
    # Assigning a Subscript to a Name (line 799):
    
    # Assigning a Subscript to a Name (line 799):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 799)
    n_192342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 18), 'n')
    slice_192343 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 799, 8), None, n_192342, None)
    # Getting the type of 'solution' (line 799)
    solution_192344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'solution')
    # Obtaining the member '__getitem__' of a type (line 799)
    getitem___192345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 8), solution_192344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 799)
    subscript_call_result_192346 = invoke(stypy.reporting.localization.Localization(__file__, 799, 8), getitem___192345, slice_192343)
    
    # Assigning a type to the variable 'x' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'x', subscript_call_result_192346)
    
    # Assigning a Subscript to a Name (line 800):
    
    # Assigning a Subscript to a Name (line 800):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 800)
    n_192347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 21), 'n')
    # Getting the type of 'n' (line 800)
    n_192348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 23), 'n')
    # Getting the type of 'n_slack' (line 800)
    n_slack_192349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 25), 'n_slack')
    # Applying the binary operator '+' (line 800)
    result_add_192350 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 23), '+', n_192348, n_slack_192349)
    
    slice_192351 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 800, 12), n_192347, result_add_192350, None)
    # Getting the type of 'solution' (line 800)
    solution_192352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 12), 'solution')
    # Obtaining the member '__getitem__' of a type (line 800)
    getitem___192353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 12), solution_192352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 800)
    subscript_call_result_192354 = invoke(stypy.reporting.localization.Localization(__file__, 800, 12), getitem___192353, slice_192351)
    
    # Assigning a type to the variable 'slack' (line 800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 4), 'slack', subscript_call_result_192354)
    
    # Assigning a Call to a Name (line 804):
    
    # Assigning a Call to a Name (line 804):
    
    # Call to filled(...): (line 804)
    # Processing the call keyword arguments (line 804)
    kwargs_192370 = {}
    
    # Call to array(...): (line 804)
    # Processing the call arguments (line 804)
    # Getting the type of 'L' (line 804)
    L_192358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 27), 'L', False)
    # Processing the call keyword arguments (line 804)
    
    # Call to isinf(...): (line 804)
    # Processing the call arguments (line 804)
    # Getting the type of 'L' (line 804)
    L_192361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 44), 'L', False)
    # Processing the call keyword arguments (line 804)
    kwargs_192362 = {}
    # Getting the type of 'np' (line 804)
    np_192359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 35), 'np', False)
    # Obtaining the member 'isinf' of a type (line 804)
    isinf_192360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 35), np_192359, 'isinf')
    # Calling isinf(args, kwargs) (line 804)
    isinf_call_result_192363 = invoke(stypy.reporting.localization.Localization(__file__, 804, 35), isinf_192360, *[L_192361], **kwargs_192362)
    
    keyword_192364 = isinf_call_result_192363
    float_192365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 59), 'float')
    keyword_192366 = float_192365
    kwargs_192367 = {'fill_value': keyword_192366, 'mask': keyword_192364}
    # Getting the type of 'np' (line 804)
    np_192355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 15), 'np', False)
    # Obtaining the member 'ma' of a type (line 804)
    ma_192356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 15), np_192355, 'ma')
    # Obtaining the member 'array' of a type (line 804)
    array_192357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 15), ma_192356, 'array')
    # Calling array(args, kwargs) (line 804)
    array_call_result_192368 = invoke(stypy.reporting.localization.Localization(__file__, 804, 15), array_192357, *[L_192358], **kwargs_192367)
    
    # Obtaining the member 'filled' of a type (line 804)
    filled_192369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 15), array_call_result_192368, 'filled')
    # Calling filled(args, kwargs) (line 804)
    filled_call_result_192371 = invoke(stypy.reporting.localization.Localization(__file__, 804, 15), filled_192369, *[], **kwargs_192370)
    
    # Assigning a type to the variable 'masked_L' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'masked_L', filled_call_result_192371)
    
    # Assigning a BinOp to a Name (line 805):
    
    # Assigning a BinOp to a Name (line 805):
    # Getting the type of 'x' (line 805)
    x_192372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'x')
    # Getting the type of 'masked_L' (line 805)
    masked_L_192373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 12), 'masked_L')
    # Applying the binary operator '+' (line 805)
    result_add_192374 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 8), '+', x_192372, masked_L_192373)
    
    # Assigning a type to the variable 'x' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'x', result_add_192374)
    
    # Getting the type of 'have_floor_variable' (line 809)
    have_floor_variable_192375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 7), 'have_floor_variable')
    # Testing the type of an if condition (line 809)
    if_condition_192376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 809, 4), have_floor_variable_192375)
    # Assigning a type to the variable 'if_condition_192376' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'if_condition_192376', if_condition_192376)
    # SSA begins for if statement (line 809)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 810)
    # Processing the call arguments (line 810)
    int_192378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 23), 'int')
    # Getting the type of 'n' (line 810)
    n_192379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 26), 'n', False)
    # Processing the call keyword arguments (line 810)
    kwargs_192380 = {}
    # Getting the type of 'range' (line 810)
    range_192377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 17), 'range', False)
    # Calling range(args, kwargs) (line 810)
    range_call_result_192381 = invoke(stypy.reporting.localization.Localization(__file__, 810, 17), range_192377, *[int_192378, n_192379], **kwargs_192380)
    
    # Testing the type of a for loop iterable (line 810)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 810, 8), range_call_result_192381)
    # Getting the type of the for loop variable (line 810)
    for_loop_var_192382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 810, 8), range_call_result_192381)
    # Assigning a type to the variable 'i' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'i', for_loop_var_192382)
    # SSA begins for a for statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isinf(...): (line 811)
    # Processing the call arguments (line 811)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 811)
    i_192385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 26), 'i', False)
    # Getting the type of 'L' (line 811)
    L_192386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 24), 'L', False)
    # Obtaining the member '__getitem__' of a type (line 811)
    getitem___192387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 24), L_192386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 811)
    subscript_call_result_192388 = invoke(stypy.reporting.localization.Localization(__file__, 811, 24), getitem___192387, i_192385)
    
    # Processing the call keyword arguments (line 811)
    kwargs_192389 = {}
    # Getting the type of 'np' (line 811)
    np_192383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 15), 'np', False)
    # Obtaining the member 'isinf' of a type (line 811)
    isinf_192384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 15), np_192383, 'isinf')
    # Calling isinf(args, kwargs) (line 811)
    isinf_call_result_192390 = invoke(stypy.reporting.localization.Localization(__file__, 811, 15), isinf_192384, *[subscript_call_result_192388], **kwargs_192389)
    
    # Testing the type of an if condition (line 811)
    if_condition_192391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 811, 12), isinf_call_result_192390)
    # Assigning a type to the variable 'if_condition_192391' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'if_condition_192391', if_condition_192391)
    # SSA begins for if statement (line 811)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'x' (line 812)
    x_192392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 16), 'x')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 812)
    i_192393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 18), 'i')
    # Getting the type of 'x' (line 812)
    x_192394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 16), 'x')
    # Obtaining the member '__getitem__' of a type (line 812)
    getitem___192395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 16), x_192394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 812)
    subscript_call_result_192396 = invoke(stypy.reporting.localization.Localization(__file__, 812, 16), getitem___192395, i_192393)
    
    
    # Obtaining the type of the subscript
    int_192397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 26), 'int')
    # Getting the type of 'x' (line 812)
    x_192398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 24), 'x')
    # Obtaining the member '__getitem__' of a type (line 812)
    getitem___192399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 24), x_192398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 812)
    subscript_call_result_192400 = invoke(stypy.reporting.localization.Localization(__file__, 812, 24), getitem___192399, int_192397)
    
    # Applying the binary operator '-=' (line 812)
    result_isub_192401 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 16), '-=', subscript_call_result_192396, subscript_call_result_192400)
    # Getting the type of 'x' (line 812)
    x_192402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 16), 'x')
    # Getting the type of 'i' (line 812)
    i_192403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 18), 'i')
    # Storing an element on a container (line 812)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 16), x_192402, (i_192403, result_isub_192401))
    
    # SSA join for if statement (line 811)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 813):
    
    # Assigning a Subscript to a Name (line 813):
    
    # Obtaining the type of the subscript
    int_192404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 14), 'int')
    slice_192405 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 813, 12), int_192404, None, None)
    # Getting the type of 'x' (line 813)
    x_192406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 813)
    getitem___192407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 12), x_192406, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 813)
    subscript_call_result_192408 = invoke(stypy.reporting.localization.Localization(__file__, 813, 12), getitem___192407, slice_192405)
    
    # Assigning a type to the variable 'x' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'x', subscript_call_result_192408)
    # SSA join for if statement (line 809)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a UnaryOp to a Name (line 816):
    
    # Assigning a UnaryOp to a Name (line 816):
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 816)
    tuple_192409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 816)
    # Adding element type (line 816)
    int_192410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 13), tuple_192409, int_192410)
    # Adding element type (line 816)
    int_192411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 13), tuple_192409, int_192411)
    
    # Getting the type of 'T' (line 816)
    T_192412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 11), 'T')
    # Obtaining the member '__getitem__' of a type (line 816)
    getitem___192413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 11), T_192412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 816)
    subscript_call_result_192414 = invoke(stypy.reporting.localization.Localization(__file__, 816, 11), getitem___192413, tuple_192409)
    
    # Applying the 'usub' unary operator (line 816)
    result___neg___192415 = python_operator(stypy.reporting.localization.Localization(__file__, 816, 10), 'usub', subscript_call_result_192414)
    
    # Assigning a type to the variable 'obj' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'obj', result___neg___192415)
    
    
    # Getting the type of 'status' (line 818)
    status_192416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 7), 'status')
    
    # Obtaining an instance of the builtin type 'tuple' (line 818)
    tuple_192417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 818)
    # Adding element type (line 818)
    int_192418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 18), tuple_192417, int_192418)
    # Adding element type (line 818)
    int_192419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 18), tuple_192417, int_192419)
    
    # Applying the binary operator 'in' (line 818)
    result_contains_192420 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 7), 'in', status_192416, tuple_192417)
    
    # Testing the type of an if condition (line 818)
    if_condition_192421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 818, 4), result_contains_192420)
    # Assigning a type to the variable 'if_condition_192421' (line 818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 4), 'if_condition_192421', if_condition_192421)
    # SSA begins for if statement (line 818)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'disp' (line 819)
    disp_192422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 11), 'disp')
    # Testing the type of an if condition (line 819)
    if_condition_192423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 819, 8), disp_192422)
    # Assigning a type to the variable 'if_condition_192423' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 8), 'if_condition_192423', if_condition_192423)
    # SSA begins for if statement (line 819)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 820)
    # Processing the call arguments (line 820)
    
    # Obtaining the type of the subscript
    # Getting the type of 'status' (line 820)
    status_192425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 27), 'status', False)
    # Getting the type of 'messages' (line 820)
    messages_192426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 18), 'messages', False)
    # Obtaining the member '__getitem__' of a type (line 820)
    getitem___192427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 18), messages_192426, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 820)
    subscript_call_result_192428 = invoke(stypy.reporting.localization.Localization(__file__, 820, 18), getitem___192427, status_192425)
    
    # Processing the call keyword arguments (line 820)
    kwargs_192429 = {}
    # Getting the type of 'print' (line 820)
    print_192424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 12), 'print', False)
    # Calling print(args, kwargs) (line 820)
    print_call_result_192430 = invoke(stypy.reporting.localization.Localization(__file__, 820, 12), print_192424, *[subscript_call_result_192428], **kwargs_192429)
    
    
    # Call to print(...): (line 821)
    # Processing the call arguments (line 821)
    
    # Call to format(...): (line 821)
    # Processing the call arguments (line 821)
    # Getting the type of 'obj' (line 821)
    obj_192434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 72), 'obj', False)
    # Processing the call keyword arguments (line 821)
    kwargs_192435 = {}
    str_192432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 18), 'str', '         Current function value: {0: <12.6f}')
    # Obtaining the member 'format' of a type (line 821)
    format_192433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 18), str_192432, 'format')
    # Calling format(args, kwargs) (line 821)
    format_call_result_192436 = invoke(stypy.reporting.localization.Localization(__file__, 821, 18), format_192433, *[obj_192434], **kwargs_192435)
    
    # Processing the call keyword arguments (line 821)
    kwargs_192437 = {}
    # Getting the type of 'print' (line 821)
    print_192431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 12), 'print', False)
    # Calling print(args, kwargs) (line 821)
    print_call_result_192438 = invoke(stypy.reporting.localization.Localization(__file__, 821, 12), print_192431, *[format_call_result_192436], **kwargs_192437)
    
    
    # Call to print(...): (line 822)
    # Processing the call arguments (line 822)
    
    # Call to format(...): (line 822)
    # Processing the call arguments (line 822)
    # Getting the type of 'nit2' (line 822)
    nit2_192442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 54), 'nit2', False)
    # Processing the call keyword arguments (line 822)
    kwargs_192443 = {}
    str_192440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 18), 'str', '         Iterations: {0:d}')
    # Obtaining the member 'format' of a type (line 822)
    format_192441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 18), str_192440, 'format')
    # Calling format(args, kwargs) (line 822)
    format_call_result_192444 = invoke(stypy.reporting.localization.Localization(__file__, 822, 18), format_192441, *[nit2_192442], **kwargs_192443)
    
    # Processing the call keyword arguments (line 822)
    kwargs_192445 = {}
    # Getting the type of 'print' (line 822)
    print_192439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 12), 'print', False)
    # Calling print(args, kwargs) (line 822)
    print_call_result_192446 = invoke(stypy.reporting.localization.Localization(__file__, 822, 12), print_192439, *[format_call_result_192444], **kwargs_192445)
    
    # SSA join for if statement (line 819)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 818)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'disp' (line 824)
    disp_192447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 11), 'disp')
    # Testing the type of an if condition (line 824)
    if_condition_192448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 824, 8), disp_192447)
    # Assigning a type to the variable 'if_condition_192448' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 8), 'if_condition_192448', if_condition_192448)
    # SSA begins for if statement (line 824)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 825)
    # Processing the call arguments (line 825)
    
    # Obtaining the type of the subscript
    # Getting the type of 'status' (line 825)
    status_192450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 27), 'status', False)
    # Getting the type of 'messages' (line 825)
    messages_192451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 18), 'messages', False)
    # Obtaining the member '__getitem__' of a type (line 825)
    getitem___192452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 18), messages_192451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 825)
    subscript_call_result_192453 = invoke(stypy.reporting.localization.Localization(__file__, 825, 18), getitem___192452, status_192450)
    
    # Processing the call keyword arguments (line 825)
    kwargs_192454 = {}
    # Getting the type of 'print' (line 825)
    print_192449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 12), 'print', False)
    # Calling print(args, kwargs) (line 825)
    print_call_result_192455 = invoke(stypy.reporting.localization.Localization(__file__, 825, 12), print_192449, *[subscript_call_result_192453], **kwargs_192454)
    
    
    # Call to print(...): (line 826)
    # Processing the call arguments (line 826)
    
    # Call to format(...): (line 826)
    # Processing the call arguments (line 826)
    # Getting the type of 'nit2' (line 826)
    nit2_192459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 54), 'nit2', False)
    # Processing the call keyword arguments (line 826)
    kwargs_192460 = {}
    str_192457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 18), 'str', '         Iterations: {0:d}')
    # Obtaining the member 'format' of a type (line 826)
    format_192458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 18), str_192457, 'format')
    # Calling format(args, kwargs) (line 826)
    format_call_result_192461 = invoke(stypy.reporting.localization.Localization(__file__, 826, 18), format_192458, *[nit2_192459], **kwargs_192460)
    
    # Processing the call keyword arguments (line 826)
    kwargs_192462 = {}
    # Getting the type of 'print' (line 826)
    print_192456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 12), 'print', False)
    # Calling print(args, kwargs) (line 826)
    print_call_result_192463 = invoke(stypy.reporting.localization.Localization(__file__, 826, 12), print_192456, *[format_call_result_192461], **kwargs_192462)
    
    # SSA join for if statement (line 824)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 818)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to OptimizeResult(...): (line 828)
    # Processing the call keyword arguments (line 828)
    # Getting the type of 'x' (line 828)
    x_192465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 28), 'x', False)
    keyword_192466 = x_192465
    # Getting the type of 'obj' (line 828)
    obj_192467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 35), 'obj', False)
    keyword_192468 = obj_192467
    
    # Call to int(...): (line 828)
    # Processing the call arguments (line 828)
    # Getting the type of 'nit2' (line 828)
    nit2_192470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 48), 'nit2', False)
    # Processing the call keyword arguments (line 828)
    kwargs_192471 = {}
    # Getting the type of 'int' (line 828)
    int_192469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 44), 'int', False)
    # Calling int(args, kwargs) (line 828)
    int_call_result_192472 = invoke(stypy.reporting.localization.Localization(__file__, 828, 44), int_192469, *[nit2_192470], **kwargs_192471)
    
    keyword_192473 = int_call_result_192472
    # Getting the type of 'status' (line 828)
    status_192474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 62), 'status', False)
    keyword_192475 = status_192474
    # Getting the type of 'slack' (line 829)
    slack_192476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 32), 'slack', False)
    keyword_192477 = slack_192476
    
    # Obtaining the type of the subscript
    # Getting the type of 'status' (line 829)
    status_192478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 56), 'status', False)
    # Getting the type of 'messages' (line 829)
    messages_192479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 47), 'messages', False)
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___192480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 47), messages_192479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_192481 = invoke(stypy.reporting.localization.Localization(__file__, 829, 47), getitem___192480, status_192478)
    
    keyword_192482 = subscript_call_result_192481
    
    # Getting the type of 'status' (line 830)
    status_192483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 35), 'status', False)
    int_192484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 45), 'int')
    # Applying the binary operator '==' (line 830)
    result_eq_192485 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 35), '==', status_192483, int_192484)
    
    keyword_192486 = result_eq_192485
    kwargs_192487 = {'status': keyword_192475, 'slack': keyword_192477, 'success': keyword_192486, 'fun': keyword_192468, 'x': keyword_192466, 'message': keyword_192482, 'nit': keyword_192473}
    # Getting the type of 'OptimizeResult' (line 828)
    OptimizeResult_192464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 828)
    OptimizeResult_call_result_192488 = invoke(stypy.reporting.localization.Localization(__file__, 828, 11), OptimizeResult_192464, *[], **kwargs_192487)
    
    # Assigning a type to the variable 'stypy_return_type' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 4), 'stypy_return_type', OptimizeResult_call_result_192488)
    
    # ################# End of '_linprog_simplex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_linprog_simplex' in the type store
    # Getting the type of 'stypy_return_type' (line 392)
    stypy_return_type_192489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192489)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_linprog_simplex'
    return stypy_return_type_192489

# Assigning a type to the variable '_linprog_simplex' (line 392)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 0), '_linprog_simplex', _linprog_simplex)

@norecursion
def linprog(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 833)
    None_192490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 20), 'None')
    # Getting the type of 'None' (line 833)
    None_192491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 31), 'None')
    # Getting the type of 'None' (line 833)
    None_192492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 42), 'None')
    # Getting the type of 'None' (line 833)
    None_192493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 53), 'None')
    # Getting the type of 'None' (line 834)
    None_192494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 19), 'None')
    str_192495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 32), 'str', 'simplex')
    # Getting the type of 'None' (line 834)
    None_192496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 52), 'None')
    # Getting the type of 'None' (line 835)
    None_192497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 20), 'None')
    defaults = [None_192490, None_192491, None_192492, None_192493, None_192494, str_192495, None_192496, None_192497]
    # Create a new context for function 'linprog'
    module_type_store = module_type_store.open_function_context('linprog', 833, 0, False)
    
    # Passed parameters checking function
    linprog.stypy_localization = localization
    linprog.stypy_type_of_self = None
    linprog.stypy_type_store = module_type_store
    linprog.stypy_function_name = 'linprog'
    linprog.stypy_param_names_list = ['c', 'A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds', 'method', 'callback', 'options']
    linprog.stypy_varargs_param_name = None
    linprog.stypy_kwargs_param_name = None
    linprog.stypy_call_defaults = defaults
    linprog.stypy_call_varargs = varargs
    linprog.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'linprog', ['c', 'A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds', 'method', 'callback', 'options'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'linprog', localization, ['c', 'A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds', 'method', 'callback', 'options'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'linprog(...)' code ##################

    str_192498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, (-1)), 'str', '\n    Minimize a linear objective function subject to linear\n    equality and inequality constraints.\n\n    Linear Programming is intended to solve the following problem form::\n\n        Minimize:     c^T * x\n\n        Subject to:   A_ub * x <= b_ub\n                      A_eq * x == b_eq\n\n    Parameters\n    ----------\n    c : array_like\n        Coefficients of the linear objective function to be minimized.\n    A_ub : array_like, optional\n        2-D array which, when matrix-multiplied by ``x``, gives the values of\n        the upper-bound inequality constraints at ``x``.\n    b_ub : array_like, optional\n        1-D array of values representing the upper-bound of each inequality\n        constraint (row) in ``A_ub``.\n    A_eq : array_like, optional\n        2-D array which, when matrix-multiplied by ``x``, gives the values of\n        the equality constraints at ``x``.\n    b_eq : array_like, optional\n        1-D array of values representing the RHS of each equality constraint\n        (row) in ``A_eq``.\n    bounds : sequence, optional\n        ``(min, max)`` pairs for each element in ``x``, defining\n        the bounds on that parameter. Use None for one of ``min`` or\n        ``max`` when there is no bound in that direction. By default\n        bounds are ``(0, None)`` (non-negative)\n        If a sequence containing a single tuple is provided, then ``min`` and\n        ``max`` will be applied to all variables in the problem.\n    method : str, optional\n        Type of solver.  :ref:`\'simplex\' <optimize.linprog-simplex>`\n        and :ref:`\'interior-point\' <optimize.linprog-interior-point>`\n        are supported.\n    callback : callable, optional (simplex only)\n        If a callback function is provide, it will be called within each\n        iteration of the simplex algorithm. The callback must have the\n        signature ``callback(xk, **kwargs)`` where ``xk`` is the current\n        solution vector and ``kwargs`` is a dictionary containing the\n        following::\n\n            "tableau" : The current Simplex algorithm tableau\n            "nit" : The current iteration.\n            "pivot" : The pivot (row, column) used for the next iteration.\n            "phase" : Whether the algorithm is in Phase 1 or Phase 2.\n            "basis" : The indices of the columns of the basic variables.\n\n    options : dict, optional\n        A dictionary of solver options. All methods accept the following\n        generic options:\n\n            maxiter : int\n                Maximum number of iterations to perform.\n            disp : bool\n                Set to True to print convergence messages.\n\n        For method-specific options, see :func:`show_options(\'linprog\')`.\n\n    Returns\n    -------\n    A `scipy.optimize.OptimizeResult` consisting of the following fields:\n\n        x : ndarray\n            The independent variable vector which optimizes the linear\n            programming problem.\n        fun : float\n            Value of the objective function.\n        slack : ndarray\n            The values of the slack variables.  Each slack variable corresponds\n            to an inequality constraint.  If the slack is zero, then the\n            corresponding constraint is active.\n        success : bool\n            Returns True if the algorithm succeeded in finding an optimal\n            solution.\n        status : int\n            An integer representing the exit status of the optimization::\n\n                 0 : Optimization terminated successfully\n                 1 : Iteration limit reached\n                 2 : Problem appears to be infeasible\n                 3 : Problem appears to be unbounded\n\n        nit : int\n            The number of iterations performed.\n        message : str\n            A string descriptor of the exit status of the optimization.\n\n    See Also\n    --------\n    show_options : Additional options accepted by the solvers\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    \'method\' parameter. The default method\n    is :ref:`Simplex <optimize.linprog-simplex>`.\n    :ref:`Interior point <optimize.linprog-interior-point>` is also available.\n\n    Method *simplex* uses the simplex algorithm (as it relates to linear\n    programming, NOT the Nelder-Mead simplex) [1]_, [2]_. This algorithm\n    should be reasonably reliable and fast for small problems.\n\n    .. versionadded:: 0.15.0\n\n    Method *interior-point* uses the primal-dual path following algorithm\n    as outlined in [4]_. This algorithm is intended to provide a faster\n    and more reliable alternative to *simplex*, especially for large,\n    sparse problems. Note, however, that the solution returned may be slightly\n    less accurate than that of the simplex method and may not correspond with a\n    vertex of the polytope defined by the constraints.\n\n    References\n    ----------\n    .. [1] Dantzig, George B., Linear programming and extensions. Rand\n           Corporation Research Study Princeton Univ. Press, Princeton, NJ,\n           1963\n    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to\n           Mathematical Programming", McGraw-Hill, Chapter 4.\n    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.\n           Mathematics of Operations Research (2), 1977: pp. 103-107.\n    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point\n           optimizer for linear programming: an implementation of the\n           homogeneous algorithm." High performance optimization. Springer US,\n           2000. 197-232.\n    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in\n           large-scale linear programming." Optimization Methods and Software\n           6.3 (1995): 219-227.\n    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear\n           Programming based on Newton\'s Method." Unpublished Course Notes,\n           March 2004. Available 2/25/2017 at\n           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf\n    .. [7] Fourer, Robert. "Solving Linear Programs by Interior-Point Methods."\n           Unpublished Course Notes, August 26, 2005. Available 2/25/2017 at\n           http://www.4er.org/CourseNotes/Book%20B/B-III.pdf\n    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear\n           programming." Mathematical Programming 71.2 (1995): 221-245.\n    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear\n           programming." Athena Scientific 1 (1997): 997.\n    .. [10] Andersen, Erling D., et al. Implementation of interior point\n            methods for large scale linear programming. HEC/Universite de\n            Geneve, 1996.\n\n    Examples\n    --------\n    Consider the following problem:\n\n    Minimize: f = -1*x[0] + 4*x[1]\n\n    Subject to: -3*x[0] + 1*x[1] <= 6\n                 1*x[0] + 2*x[1] <= 4\n                            x[1] >= -3\n\n    where:  -inf <= x[0] <= inf\n\n    This problem deviates from the standard linear programming problem.\n    In standard form, linear programming problems assume the variables x are\n    non-negative.  Since the variables don\'t have standard bounds where\n    0 <= x <= inf, the bounds of the variables must be explicitly set.\n\n    There are two upper-bound constraints, which can be expressed as\n\n    dot(A_ub, x) <= b_ub\n\n    The input for this problem is as follows:\n\n    >>> c = [-1, 4]\n    >>> A = [[-3, 1], [1, 2]]\n    >>> b = [6, 4]\n    >>> x0_bounds = (None, None)\n    >>> x1_bounds = (-3, None)\n    >>> from scipy.optimize import linprog\n    >>> res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),\n    ...               options={"disp": True})\n    Optimization terminated successfully.\n         Current function value: -22.000000\n         Iterations: 1\n    >>> print(res)\n         fun: -22.0\n     message: \'Optimization terminated successfully.\'\n         nit: 1\n       slack: array([ 39.,   0.])\n      status: 0\n     success: True\n           x: array([ 10.,  -3.])\n\n    Note the actual objective value is 11.428571.  In this case we minimized\n    the negative of the objective function.\n\n    ')
    
    # Assigning a Call to a Name (line 1029):
    
    # Assigning a Call to a Name (line 1029):
    
    # Call to lower(...): (line 1029)
    # Processing the call keyword arguments (line 1029)
    kwargs_192501 = {}
    # Getting the type of 'method' (line 1029)
    method_192499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 11), 'method', False)
    # Obtaining the member 'lower' of a type (line 1029)
    lower_192500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1029, 11), method_192499, 'lower')
    # Calling lower(args, kwargs) (line 1029)
    lower_call_result_192502 = invoke(stypy.reporting.localization.Localization(__file__, 1029, 11), lower_192500, *[], **kwargs_192501)
    
    # Assigning a type to the variable 'meth' (line 1029)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1029, 4), 'meth', lower_call_result_192502)
    
    # Type idiom detected: calculating its left and rigth part (line 1030)
    # Getting the type of 'options' (line 1030)
    options_192503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 7), 'options')
    # Getting the type of 'None' (line 1030)
    None_192504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 18), 'None')
    
    (may_be_192505, more_types_in_union_192506) = may_be_none(options_192503, None_192504)

    if may_be_192505:

        if more_types_in_union_192506:
            # Runtime conditional SSA (line 1030)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 1031):
        
        # Assigning a Dict to a Name (line 1031):
        
        # Obtaining an instance of the builtin type 'dict' (line 1031)
        dict_192507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1031)
        
        # Assigning a type to the variable 'options' (line 1031)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'options', dict_192507)

        if more_types_in_union_192506:
            # SSA join for if statement (line 1030)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'meth' (line 1033)
    meth_192508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 7), 'meth')
    str_192509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 15), 'str', 'simplex')
    # Applying the binary operator '==' (line 1033)
    result_eq_192510 = python_operator(stypy.reporting.localization.Localization(__file__, 1033, 7), '==', meth_192508, str_192509)
    
    # Testing the type of an if condition (line 1033)
    if_condition_192511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1033, 4), result_eq_192510)
    # Assigning a type to the variable 'if_condition_192511' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'if_condition_192511', if_condition_192511)
    # SSA begins for if statement (line 1033)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _linprog_simplex(...): (line 1034)
    # Processing the call arguments (line 1034)
    # Getting the type of 'c' (line 1034)
    c_192513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 32), 'c', False)
    # Processing the call keyword arguments (line 1034)
    # Getting the type of 'A_ub' (line 1034)
    A_ub_192514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 40), 'A_ub', False)
    keyword_192515 = A_ub_192514
    # Getting the type of 'b_ub' (line 1034)
    b_ub_192516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 51), 'b_ub', False)
    keyword_192517 = b_ub_192516
    # Getting the type of 'A_eq' (line 1034)
    A_eq_192518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 62), 'A_eq', False)
    keyword_192519 = A_eq_192518
    # Getting the type of 'b_eq' (line 1034)
    b_eq_192520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 73), 'b_eq', False)
    keyword_192521 = b_eq_192520
    # Getting the type of 'bounds' (line 1035)
    bounds_192522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 39), 'bounds', False)
    keyword_192523 = bounds_192522
    # Getting the type of 'callback' (line 1035)
    callback_192524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 56), 'callback', False)
    keyword_192525 = callback_192524
    # Getting the type of 'options' (line 1035)
    options_192526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 68), 'options', False)
    kwargs_192527 = {'A_ub': keyword_192515, 'A_eq': keyword_192519, 'bounds': keyword_192523, 'callback': keyword_192525, 'options_192526': options_192526, 'b_ub': keyword_192517, 'b_eq': keyword_192521}
    # Getting the type of '_linprog_simplex' (line 1034)
    _linprog_simplex_192512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 15), '_linprog_simplex', False)
    # Calling _linprog_simplex(args, kwargs) (line 1034)
    _linprog_simplex_call_result_192528 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 15), _linprog_simplex_192512, *[c_192513], **kwargs_192527)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'stypy_return_type', _linprog_simplex_call_result_192528)
    # SSA branch for the else part of an if statement (line 1033)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'meth' (line 1036)
    meth_192529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 9), 'meth')
    str_192530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 17), 'str', 'interior-point')
    # Applying the binary operator '==' (line 1036)
    result_eq_192531 = python_operator(stypy.reporting.localization.Localization(__file__, 1036, 9), '==', meth_192529, str_192530)
    
    # Testing the type of an if condition (line 1036)
    if_condition_192532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1036, 9), result_eq_192531)
    # Assigning a type to the variable 'if_condition_192532' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 9), 'if_condition_192532', if_condition_192532)
    # SSA begins for if statement (line 1036)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _linprog_ip(...): (line 1037)
    # Processing the call arguments (line 1037)
    # Getting the type of 'c' (line 1037)
    c_192534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 27), 'c', False)
    # Processing the call keyword arguments (line 1037)
    # Getting the type of 'A_ub' (line 1037)
    A_ub_192535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 35), 'A_ub', False)
    keyword_192536 = A_ub_192535
    # Getting the type of 'b_ub' (line 1037)
    b_ub_192537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 46), 'b_ub', False)
    keyword_192538 = b_ub_192537
    # Getting the type of 'A_eq' (line 1037)
    A_eq_192539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 57), 'A_eq', False)
    keyword_192540 = A_eq_192539
    # Getting the type of 'b_eq' (line 1037)
    b_eq_192541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 68), 'b_eq', False)
    keyword_192542 = b_eq_192541
    # Getting the type of 'bounds' (line 1038)
    bounds_192543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 34), 'bounds', False)
    keyword_192544 = bounds_192543
    # Getting the type of 'callback' (line 1038)
    callback_192545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 51), 'callback', False)
    keyword_192546 = callback_192545
    # Getting the type of 'options' (line 1038)
    options_192547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 63), 'options', False)
    kwargs_192548 = {'A_ub': keyword_192536, 'A_eq': keyword_192540, 'bounds': keyword_192544, 'options_192547': options_192547, 'callback': keyword_192546, 'b_ub': keyword_192538, 'b_eq': keyword_192542}
    # Getting the type of '_linprog_ip' (line 1037)
    _linprog_ip_192533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 15), '_linprog_ip', False)
    # Calling _linprog_ip(args, kwargs) (line 1037)
    _linprog_ip_call_result_192549 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 15), _linprog_ip_192533, *[c_192534], **kwargs_192548)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'stypy_return_type', _linprog_ip_call_result_192549)
    # SSA branch for the else part of an if statement (line 1036)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 1040)
    # Processing the call arguments (line 1040)
    str_192551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 25), 'str', 'Unknown solver %s')
    # Getting the type of 'method' (line 1040)
    method_192552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 47), 'method', False)
    # Applying the binary operator '%' (line 1040)
    result_mod_192553 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 25), '%', str_192551, method_192552)
    
    # Processing the call keyword arguments (line 1040)
    kwargs_192554 = {}
    # Getting the type of 'ValueError' (line 1040)
    ValueError_192550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1040)
    ValueError_call_result_192555 = invoke(stypy.reporting.localization.Localization(__file__, 1040, 14), ValueError_192550, *[result_mod_192553], **kwargs_192554)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1040, 8), ValueError_call_result_192555, 'raise parameter', BaseException)
    # SSA join for if statement (line 1036)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1033)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'linprog(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'linprog' in the type store
    # Getting the type of 'stypy_return_type' (line 833)
    stypy_return_type_192556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192556)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'linprog'
    return stypy_return_type_192556

# Assigning a type to the variable 'linprog' (line 833)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 0), 'linprog', linprog)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
