
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: basinhopping: The basinhopping global optimization algorithm
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy import cos, sin
8: import scipy.optimize
9: import collections
10: from scipy._lib._util import check_random_state
11: 
12: __all__ = ['basinhopping']
13: 
14: 
15: class Storage(object):
16:     '''
17:     Class used to store the lowest energy structure
18:     '''
19:     def __init__(self, minres):
20:         self._add(minres)
21: 
22:     def _add(self, minres):
23:         self.minres = minres
24:         self.minres.x = np.copy(minres.x)
25: 
26:     def update(self, minres):
27:         if minres.fun < self.minres.fun:
28:             self._add(minres)
29:             return True
30:         else:
31:             return False
32: 
33:     def get_lowest(self):
34:         return self.minres
35: 
36: 
37: class BasinHoppingRunner(object):
38:     '''This class implements the core of the basinhopping algorithm.
39: 
40:     x0 : ndarray
41:         The starting coordinates.
42:     minimizer : callable
43:         The local minimizer, with signature ``result = minimizer(x)``.
44:         The return value is an `optimize.OptimizeResult` object.
45:     step_taking : callable
46:         This function displaces the coordinates randomly.  Signature should
47:         be ``x_new = step_taking(x)``.  Note that `x` may be modified in-place.
48:     accept_tests : list of callables
49:         Each test is passed the kwargs `f_new`, `x_new`, `f_old` and
50:         `x_old`.  These tests will be used to judge whether or not to accept
51:         the step.  The acceptable return values are True, False, or ``"force
52:         accept"``.  If any of the tests return False then the step is rejected.
53:         If the latter, then this will override any other tests in order to
54:         accept the step. This can be used, for example, to forcefully escape
55:         from a local minimum that ``basinhopping`` is trapped in.
56:     disp : bool, optional
57:         Display status messages.
58: 
59:     '''
60:     def __init__(self, x0, minimizer, step_taking, accept_tests, disp=False):
61:         self.x = np.copy(x0)
62:         self.minimizer = minimizer
63:         self.step_taking = step_taking
64:         self.accept_tests = accept_tests
65:         self.disp = disp
66: 
67:         self.nstep = 0
68: 
69:         # initialize return object
70:         self.res = scipy.optimize.OptimizeResult()
71:         self.res.minimization_failures = 0
72: 
73:         # do initial minimization
74:         minres = minimizer(self.x)
75:         if not minres.success:
76:             self.res.minimization_failures += 1
77:             if self.disp:
78:                 print("warning: basinhopping: local minimization failure")
79:         self.x = np.copy(minres.x)
80:         self.energy = minres.fun
81:         if self.disp:
82:             print("basinhopping step %d: f %g" % (self.nstep, self.energy))
83: 
84:         # initialize storage class
85:         self.storage = Storage(minres)
86: 
87:         if hasattr(minres, "nfev"):
88:             self.res.nfev = minres.nfev
89:         if hasattr(minres, "njev"):
90:             self.res.njev = minres.njev
91:         if hasattr(minres, "nhev"):
92:             self.res.nhev = minres.nhev
93: 
94:     def _monte_carlo_step(self):
95:         '''Do one monte carlo iteration
96: 
97:         Randomly displace the coordinates, minimize, and decide whether
98:         or not to accept the new coordinates.
99:         '''
100:         # Take a random step.  Make a copy of x because the step_taking
101:         # algorithm might change x in place
102:         x_after_step = np.copy(self.x)
103:         x_after_step = self.step_taking(x_after_step)
104: 
105:         # do a local minimization
106:         minres = self.minimizer(x_after_step)
107:         x_after_quench = minres.x
108:         energy_after_quench = minres.fun
109:         if not minres.success:
110:             self.res.minimization_failures += 1
111:             if self.disp:
112:                 print("warning: basinhopping: local minimization failure")
113: 
114:         if hasattr(minres, "nfev"):
115:             self.res.nfev += minres.nfev
116:         if hasattr(minres, "njev"):
117:             self.res.njev += minres.njev
118:         if hasattr(minres, "nhev"):
119:             self.res.nhev += minres.nhev
120: 
121:         # accept the move based on self.accept_tests. If any test is False,
122:         # than reject the step.  If any test returns the special value, the
123:         # string 'force accept', accept the step regardless.  This can be used
124:         # to forcefully escape from a local minimum if normal basin hopping
125:         # steps are not sufficient.
126:         accept = True
127:         for test in self.accept_tests:
128:             testres = test(f_new=energy_after_quench, x_new=x_after_quench,
129:                            f_old=self.energy, x_old=self.x)
130:             if testres == 'force accept':
131:                 accept = True
132:                 break
133:             elif not testres:
134:                 accept = False
135: 
136:         # Report the result of the acceptance test to the take step class.
137:         # This is for adaptive step taking
138:         if hasattr(self.step_taking, "report"):
139:             self.step_taking.report(accept, f_new=energy_after_quench,
140:                                     x_new=x_after_quench, f_old=self.energy,
141:                                     x_old=self.x)
142: 
143:         return accept, minres
144: 
145:     def one_cycle(self):
146:         '''Do one cycle of the basinhopping algorithm
147:         '''
148:         self.nstep += 1
149:         new_global_min = False
150: 
151:         accept, minres = self._monte_carlo_step()
152: 
153:         if accept:
154:             self.energy = minres.fun
155:             self.x = np.copy(minres.x)
156:             new_global_min = self.storage.update(minres)
157: 
158:         # print some information
159:         if self.disp:
160:             self.print_report(minres.fun, accept)
161:             if new_global_min:
162:                 print("found new global minimum on step %d with function"
163:                       " value %g" % (self.nstep, self.energy))
164: 
165:         # save some variables as BasinHoppingRunner attributes
166:         self.xtrial = minres.x
167:         self.energy_trial = minres.fun
168:         self.accept = accept
169: 
170:         return new_global_min
171: 
172:     def print_report(self, energy_trial, accept):
173:         '''print a status update'''
174:         minres = self.storage.get_lowest()
175:         print("basinhopping step %d: f %g trial_f %g accepted %d "
176:               " lowest_f %g" % (self.nstep, self.energy, energy_trial,
177:                                 accept, minres.fun))
178: 
179: 
180: class AdaptiveStepsize(object):
181:     '''
182:     Class to implement adaptive stepsize.
183: 
184:     This class wraps the step taking class and modifies the stepsize to
185:     ensure the true acceptance rate is as close as possible to the target.
186: 
187:     Parameters
188:     ----------
189:     takestep : callable
190:         The step taking routine.  Must contain modifiable attribute
191:         takestep.stepsize
192:     accept_rate : float, optional
193:         The target step acceptance rate
194:     interval : int, optional
195:         Interval for how often to update the stepsize
196:     factor : float, optional
197:         The step size is multiplied or divided by this factor upon each
198:         update.
199:     verbose : bool, optional
200:         Print information about each update
201: 
202:     '''
203:     def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9,
204:                  verbose=True):
205:         self.takestep = takestep
206:         self.target_accept_rate = accept_rate
207:         self.interval = interval
208:         self.factor = factor
209:         self.verbose = verbose
210: 
211:         self.nstep = 0
212:         self.nstep_tot = 0
213:         self.naccept = 0
214: 
215:     def __call__(self, x):
216:         return self.take_step(x)
217: 
218:     def _adjust_step_size(self):
219:         old_stepsize = self.takestep.stepsize
220:         accept_rate = float(self.naccept) / self.nstep
221:         if accept_rate > self.target_accept_rate:
222:             #We're accepting too many steps.  This generally means we're
223:             #trapped in a basin.  Take bigger steps
224:             self.takestep.stepsize /= self.factor
225:         else:
226:             #We're not accepting enough steps.  Take smaller steps
227:             self.takestep.stepsize *= self.factor
228:         if self.verbose:
229:             print("adaptive stepsize: acceptance rate %f target %f new "
230:                   "stepsize %g old stepsize %g" % (accept_rate,
231:                   self.target_accept_rate, self.takestep.stepsize,
232:                   old_stepsize))
233: 
234:     def take_step(self, x):
235:         self.nstep += 1
236:         self.nstep_tot += 1
237:         if self.nstep % self.interval == 0:
238:             self._adjust_step_size()
239:         return self.takestep(x)
240: 
241:     def report(self, accept, **kwargs):
242:         "called by basinhopping to report the result of the step"
243:         if accept:
244:             self.naccept += 1
245: 
246: 
247: class RandomDisplacement(object):
248:     '''
249:     Add a random displacement of maximum size, stepsize, to the coordinates
250: 
251:     update x inplace
252: 
253:     Parameters
254:     ----------
255:     stepsize : float, optional
256:         stepsize
257:     random_state : None or `np.random.RandomState` instance, optional
258:         The random number generator that generates the displacements
259:     '''
260:     def __init__(self, stepsize=0.5, random_state=None):
261:         self.stepsize = stepsize
262:         self.random_state = check_random_state(random_state)
263: 
264:     def __call__(self, x):
265:         x += self.random_state.uniform(-self.stepsize, self.stepsize, np.shape(x))
266:         return x
267: 
268: 
269: class MinimizerWrapper(object):
270:     '''
271:     wrap a minimizer function as a minimizer class
272:     '''
273:     def __init__(self, minimizer, func=None, **kwargs):
274:         self.minimizer = minimizer
275:         self.func = func
276:         self.kwargs = kwargs
277: 
278:     def __call__(self, x0):
279:         if self.func is None:
280:             return self.minimizer(x0, **self.kwargs)
281:         else:
282:             return self.minimizer(self.func, x0, **self.kwargs)
283: 
284: 
285: class Metropolis(object):
286:     '''
287:     Metropolis acceptance criterion
288: 
289:     Parameters
290:     ----------
291:     random_state : None or `np.random.RandomState` object
292:         Random number generator used for acceptance test
293:     '''
294:     def __init__(self, T, random_state=None):
295:         self.beta = 1.0 / T
296:         self.random_state = check_random_state(random_state)
297: 
298:     def accept_reject(self, energy_new, energy_old):
299:         w = np.exp(min(0, -(energy_new - energy_old) * self.beta))
300:         rand = self.random_state.rand()
301:         return w >= rand
302: 
303:     def __call__(self, **kwargs):
304:         '''
305:         f_new and f_old are mandatory in kwargs
306:         '''
307:         return bool(self.accept_reject(kwargs["f_new"],
308:                     kwargs["f_old"]))
309: 
310: 
311: def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5,
312:                  minimizer_kwargs=None, take_step=None, accept_test=None,
313:                  callback=None, interval=50, disp=False, niter_success=None,
314:                  seed=None):
315:     '''
316:     Find the global minimum of a function using the basin-hopping algorithm
317: 
318:     Parameters
319:     ----------
320:     func : callable ``f(x, *args)``
321:         Function to be optimized.  ``args`` can be passed as an optional item
322:         in the dict ``minimizer_kwargs``
323:     x0 : ndarray
324:         Initial guess.
325:     niter : integer, optional
326:         The number of basin hopping iterations
327:     T : float, optional
328:         The "temperature" parameter for the accept or reject criterion.  Higher
329:         "temperatures" mean that larger jumps in function value will be
330:         accepted.  For best results ``T`` should be comparable to the
331:         separation
332:         (in function value) between local minima.
333:     stepsize : float, optional
334:         initial step size for use in the random displacement.
335:     minimizer_kwargs : dict, optional
336:         Extra keyword arguments to be passed to the minimizer
337:         ``scipy.optimize.minimize()`` Some important options could be:
338: 
339:             method : str
340:                 The minimization method (e.g. ``"L-BFGS-B"``)
341:             args : tuple
342:                 Extra arguments passed to the objective function (``func``) and
343:                 its derivatives (Jacobian, Hessian).
344: 
345:     take_step : callable ``take_step(x)``, optional
346:         Replace the default step taking routine with this routine.  The default
347:         step taking routine is a random displacement of the coordinates, but
348:         other step taking algorithms may be better for some systems.
349:         ``take_step`` can optionally have the attribute ``take_step.stepsize``.
350:         If this attribute exists, then ``basinhopping`` will adjust
351:         ``take_step.stepsize`` in order to try to optimize the global minimum
352:         search.
353:     accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
354:         Define a test which will be used to judge whether or not to accept the
355:         step.  This will be used in addition to the Metropolis test based on
356:         "temperature" ``T``.  The acceptable return values are True,
357:         False, or ``"force accept"``. If any of the tests return False
358:         then the step is rejected. If the latter, then this will override any
359:         other tests in order to accept the step. This can be used, for example,
360:         to forcefully escape from a local minimum that ``basinhopping`` is
361:         trapped in.
362:     callback : callable, ``callback(x, f, accept)``, optional
363:         A callback function which will be called for all minima found.  ``x``
364:         and ``f`` are the coordinates and function value of the trial minimum,
365:         and ``accept`` is whether or not that minimum was accepted.  This can be
366:         used, for example, to save the lowest N minima found.  Also,
367:         ``callback`` can be used to specify a user defined stop criterion by
368:         optionally returning True to stop the ``basinhopping`` routine.
369:     interval : integer, optional
370:         interval for how often to update the ``stepsize``
371:     disp : bool, optional
372:         Set to True to print status messages
373:     niter_success : integer, optional
374:         Stop the run if the global minimum candidate remains the same for this
375:         number of iterations.
376:     seed : int or `np.random.RandomState`, optional
377:         If `seed` is not specified the `np.RandomState` singleton is used.
378:         If `seed` is an int, a new `np.random.RandomState` instance is used,
379:         seeded with seed.
380:         If `seed` is already a `np.random.RandomState instance`, then that
381:         `np.random.RandomState` instance is used.
382:         Specify `seed` for repeatable minimizations. The random numbers
383:         generated with this seed only affect the default Metropolis
384:         `accept_test` and the default `take_step`. If you supply your own
385:         `take_step` and `accept_test`, and these functions use random
386:         number generation, then those functions are responsible for the state
387:         of their random number generator.
388: 
389:     Returns
390:     -------
391:     res : OptimizeResult
392:         The optimization result represented as a ``OptimizeResult`` object.  Important
393:         attributes are: ``x`` the solution array, ``fun`` the value of the
394:         function at the solution, and ``message`` which describes the cause of
395:         the termination. The ``OptimzeResult`` object returned by the selected
396:         minimizer at the lowest minimum is also contained within this object
397:         and can be accessed through the ``lowest_optimization_result`` attribute.
398:         See `OptimizeResult` for a description of other attributes.
399: 
400:     See Also
401:     --------
402:     minimize :
403:         The local minimization function called once for each basinhopping step.
404:         ``minimizer_kwargs`` is passed to this routine.
405: 
406:     Notes
407:     -----
408:     Basin-hopping is a stochastic algorithm which attempts to find the global
409:     minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_
410:     [4]_.  The algorithm in its current form was described by David Wales and
411:     Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.
412: 
413:     The algorithm is iterative with each cycle composed of the following
414:     features
415: 
416:     1) random perturbation of the coordinates
417: 
418:     2) local minimization
419: 
420:     3) accept or reject the new coordinates based on the minimized function
421:        value
422: 
423:     The acceptance test used here is the Metropolis criterion of standard Monte
424:     Carlo algorithms, although there are many other possibilities [3]_.
425: 
426:     This global minimization method has been shown to be extremely efficient
427:     for a wide variety of problems in physics and chemistry.  It is
428:     particularly useful when the function has many minima separated by large
429:     barriers. See the Cambridge Cluster Database
430:     http://www-wales.ch.cam.ac.uk/CCD.html for databases of molecular systems
431:     that have been optimized primarily using basin-hopping.  This database
432:     includes minimization problems exceeding 300 degrees of freedom.
433: 
434:     See the free software program GMIN (http://www-wales.ch.cam.ac.uk/GMIN) for
435:     a Fortran implementation of basin-hopping.  This implementation has many
436:     different variations of the procedure described above, including more
437:     advanced step taking algorithms and alternate acceptance criterion.
438: 
439:     For stochastic global optimization there is no way to determine if the true
440:     global minimum has actually been found. Instead, as a consistency check,
441:     the algorithm can be run from a number of different random starting points
442:     to ensure the lowest minimum found in each example has converged to the
443:     global minimum.  For this reason ``basinhopping`` will by default simply
444:     run for the number of iterations ``niter`` and return the lowest minimum
445:     found.  It is left to the user to ensure that this is in fact the global
446:     minimum.
447: 
448:     Choosing ``stepsize``:  This is a crucial parameter in ``basinhopping`` and
449:     depends on the problem being solved.  Ideally it should be comparable to
450:     the typical separation between local minima of the function being
451:     optimized.  ``basinhopping`` will, by default, adjust ``stepsize`` to find
452:     an optimal value, but this may take many iterations.  You will get quicker
453:     results if you set a sensible value for ``stepsize``.
454: 
455:     Choosing ``T``: The parameter ``T`` is the temperature used in the
456:     metropolis criterion.  Basinhopping steps are accepted with probability
457:     ``1`` if ``func(xnew) < func(xold)``, or otherwise with probability::
458: 
459:         exp( -(func(xnew) - func(xold)) / T )
460: 
461:     So, for best results, ``T`` should to be comparable to the typical
462:     difference in function values between local minima.
463: 
464:     .. versionadded:: 0.12.0
465: 
466:     References
467:     ----------
468:     .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,
469:         Cambridge, UK.
470:     .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and
471:         the Lowest Energy Structures of Lennard-Jones Clusters Containing up to
472:         110 Atoms.  Journal of Physical Chemistry A, 1997, 101, 5111.
473:     .. [3] Li, Z. and Scheraga, H. A., Monte Carlo-minimization approach to the
474:         multiple-minima problem in protein folding, Proc. Natl. Acad. Sci. USA,
475:         1987, 84, 6611.
476:     .. [4] Wales, D. J. and Scheraga, H. A., Global optimization of clusters,
477:         crystals, and biomolecules, Science, 1999, 285, 1368.
478: 
479:     Examples
480:     --------
481:     The following example is a one-dimensional minimization problem,  with many
482:     local minima superimposed on a parabola.
483: 
484:     >>> from scipy.optimize import basinhopping
485:     >>> func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
486:     >>> x0=[1.]
487: 
488:     Basinhopping, internally, uses a local minimization algorithm.  We will use
489:     the parameter ``minimizer_kwargs`` to tell basinhopping which algorithm to
490:     use and how to set up that minimizer.  This parameter will be passed to
491:     ``scipy.optimize.minimize()``.
492: 
493:     >>> minimizer_kwargs = {"method": "BFGS"}
494:     >>> ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,
495:     ...                    niter=200)
496:     >>> print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
497:     global minimum: x = -0.1951, f(x0) = -1.0009
498: 
499:     Next consider a two-dimensional minimization problem. Also, this time we
500:     will use gradient information to significantly speed up the search.
501: 
502:     >>> def func2d(x):
503:     ...     f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
504:     ...                                                            0.2) * x[0]
505:     ...     df = np.zeros(2)
506:     ...     df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
507:     ...     df[1] = 2. * x[1] + 0.2
508:     ...     return f, df
509: 
510:     We'll also use a different local minimization algorithm.  Also we must tell
511:     the minimizer that our function returns both energy and gradient (jacobian)
512: 
513:     >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
514:     >>> x0 = [1.0, 1.0]
515:     >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
516:     ...                    niter=200)
517:     >>> print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
518:     ...                                                           ret.x[1],
519:     ...                                                           ret.fun))
520:     global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109
521: 
522: 
523:     Here is an example using a custom step taking routine.  Imagine you want
524:     the first coordinate to take larger steps then the rest of the coordinates.
525:     This can be implemented like so:
526: 
527:     >>> class MyTakeStep(object):
528:     ...    def __init__(self, stepsize=0.5):
529:     ...        self.stepsize = stepsize
530:     ...    def __call__(self, x):
531:     ...        s = self.stepsize
532:     ...        x[0] += np.random.uniform(-2.*s, 2.*s)
533:     ...        x[1:] += np.random.uniform(-s, s, x[1:].shape)
534:     ...        return x
535: 
536:     Since ``MyTakeStep.stepsize`` exists basinhopping will adjust the magnitude
537:     of ``stepsize`` to optimize the search.  We'll use the same 2-D function as
538:     before
539: 
540:     >>> mytakestep = MyTakeStep()
541:     >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
542:     ...                    niter=200, take_step=mytakestep)
543:     >>> print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
544:     ...                                                           ret.x[1],
545:     ...                                                           ret.fun))
546:     global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109
547: 
548: 
549:     Now let's do an example using a custom callback function which prints the
550:     value of every minimum found
551: 
552:     >>> def print_fun(x, f, accepted):
553:     ...         print("at minimum %.4f accepted %d" % (f, int(accepted)))
554: 
555:     We'll run it for only 10 basinhopping steps this time.
556: 
557:     >>> np.random.seed(1)
558:     >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
559:     ...                    niter=10, callback=print_fun)
560:     at minimum 0.4159 accepted 1
561:     at minimum -0.9073 accepted 1
562:     at minimum -0.1021 accepted 1
563:     at minimum -0.1021 accepted 1
564:     at minimum 0.9102 accepted 1
565:     at minimum 0.9102 accepted 1
566:     at minimum 2.2945 accepted 0
567:     at minimum -0.1021 accepted 1
568:     at minimum -1.0109 accepted 1
569:     at minimum -1.0109 accepted 1
570: 
571: 
572:     The minimum at -1.0109 is actually the global minimum, found already on the
573:     8th iteration.
574: 
575:     Now let's implement bounds on the problem using a custom ``accept_test``:
576: 
577:     >>> class MyBounds(object):
578:     ...     def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
579:     ...         self.xmax = np.array(xmax)
580:     ...         self.xmin = np.array(xmin)
581:     ...     def __call__(self, **kwargs):
582:     ...         x = kwargs["x_new"]
583:     ...         tmax = bool(np.all(x <= self.xmax))
584:     ...         tmin = bool(np.all(x >= self.xmin))
585:     ...         return tmax and tmin
586: 
587:     >>> mybounds = MyBounds()
588:     >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
589:     ...                    niter=10, accept_test=mybounds)
590: 
591:     '''
592:     x0 = np.array(x0)
593: 
594:     # set up the np.random.RandomState generator
595:     rng = check_random_state(seed)
596: 
597:     # set up minimizer
598:     if minimizer_kwargs is None:
599:         minimizer_kwargs = dict()
600:     wrapped_minimizer = MinimizerWrapper(scipy.optimize.minimize, func,
601:                                          **minimizer_kwargs)
602: 
603:     # set up step taking algorithm
604:     if take_step is not None:
605:         if not isinstance(take_step, collections.Callable):
606:             raise TypeError("take_step must be callable")
607:         # if take_step.stepsize exists then use AdaptiveStepsize to control
608:         # take_step.stepsize
609:         if hasattr(take_step, "stepsize"):
610:             take_step_wrapped = AdaptiveStepsize(take_step, interval=interval,
611:                                                  verbose=disp)
612:         else:
613:             take_step_wrapped = take_step
614:     else:
615:         # use default
616:         displace = RandomDisplacement(stepsize=stepsize, random_state=rng)
617:         take_step_wrapped = AdaptiveStepsize(displace, interval=interval,
618:                                              verbose=disp)
619: 
620:     # set up accept tests
621:     if accept_test is not None:
622:         if not isinstance(accept_test, collections.Callable):
623:             raise TypeError("accept_test must be callable")
624:         accept_tests = [accept_test]
625:     else:
626:         accept_tests = []
627:     # use default
628:     metropolis = Metropolis(T, random_state=rng)
629:     accept_tests.append(metropolis)
630: 
631:     if niter_success is None:
632:         niter_success = niter + 2
633: 
634:     bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
635:                             accept_tests, disp=disp)
636: 
637:     # start main iteration loop
638:     count, i = 0, 0
639:     message = ["requested number of basinhopping iterations completed"
640:                " successfully"]
641:     for i in range(niter):
642:         new_global_min = bh.one_cycle()
643: 
644:         if isinstance(callback, collections.Callable):
645:             # should we pass a copy of x?
646:             val = callback(bh.xtrial, bh.energy_trial, bh.accept)
647:             if val is not None:
648:                 if val:
649:                     message = ["callback function requested stop early by"
650:                                "returning True"]
651:                     break
652: 
653:         count += 1
654:         if new_global_min:
655:             count = 0
656:         elif count > niter_success:
657:             message = ["success condition satisfied"]
658:             break
659: 
660:     # prepare return object
661:     res = bh.res
662:     res.lowest_optimization_result = bh.storage.get_lowest()
663:     res.x = np.copy(res.lowest_optimization_result.x)
664:     res.fun = res.lowest_optimization_result.fun
665:     res.message = message
666:     res.nit = i + 1
667:     return res
668: 
669: 
670: def _test_func2d_nograd(x):
671:     f = (cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
672:          + 1.010876184442655)
673:     return f
674: 
675: 
676: def _test_func2d(x):
677:     f = (cos(14.5 * x[0] - 0.3) + (x[0] + 0.2) * x[0] + cos(14.5 * x[1] -
678:          0.3) + (x[1] + 0.2) * x[1] + x[0] * x[1] + 1.963879482144252)
679:     df = np.zeros(2)
680:     df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2 + x[1]
681:     df[1] = -14.5 * sin(14.5 * x[1] - 0.3) + 2. * x[1] + 0.2 + x[0]
682:     return f, df
683: 
684: if __name__ == "__main__":
685:     print("\n\nminimize a 2d function without gradient")
686:     # minimum expected at ~[-0.195, -0.1]
687:     kwargs = {"method": "L-BFGS-B"}
688:     x0 = np.array([1.0, 1.])
689:     scipy.optimize.minimize(_test_func2d_nograd, x0, **kwargs)
690:     ret = basinhopping(_test_func2d_nograd, x0, minimizer_kwargs=kwargs,
691:                        niter=200, disp=False)
692:     print("minimum expected at  func([-0.195, -0.1]) = 0.0")
693:     print(ret)
694: 
695:     print("\n\ntry a harder 2d problem")
696:     kwargs = {"method": "L-BFGS-B", "jac": True}
697:     x0 = np.array([1.0, 1.0])
698:     ret = basinhopping(_test_func2d, x0, minimizer_kwargs=kwargs, niter=200,
699:                        disp=False)
700:     print("minimum expected at ~, func([-0.19415263, -0.19415263]) = 0")
701:     print(ret)
702: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_187342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nbasinhopping: The basinhopping global optimization algorithm\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_187343 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_187343) is not StypyTypeError):

    if (import_187343 != 'pyd_module'):
        __import__(import_187343)
        sys_modules_187344 = sys.modules[import_187343]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_187344.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_187343)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy import cos, sin' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_187345 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_187345) is not StypyTypeError):

    if (import_187345 != 'pyd_module'):
        __import__(import_187345)
        sys_modules_187346 = sys.modules[import_187345]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', sys_modules_187346.module_type_store, module_type_store, ['cos', 'sin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_187346, sys_modules_187346.module_type_store, module_type_store)
    else:
        from numpy import cos, sin

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', None, module_type_store, ['cos', 'sin'], [cos, sin])

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_187345)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import scipy.optimize' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_187347 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize')

if (type(import_187347) is not StypyTypeError):

    if (import_187347 != 'pyd_module'):
        __import__(import_187347)
        sys_modules_187348 = sys.modules[import_187347]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', sys_modules_187348.module_type_store, module_type_store)
    else:
        import scipy.optimize

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', scipy.optimize, module_type_store)

else:
    # Assigning a type to the variable 'scipy.optimize' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', import_187347)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import collections' statement (line 9)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'collections', collections, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib._util import check_random_state' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_187349 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util')

if (type(import_187349) is not StypyTypeError):

    if (import_187349 != 'pyd_module'):
        __import__(import_187349)
        sys_modules_187350 = sys.modules[import_187349]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', sys_modules_187350.module_type_store, module_type_store, ['check_random_state'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_187350, sys_modules_187350.module_type_store, module_type_store)
    else:
        from scipy._lib._util import check_random_state

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', None, module_type_store, ['check_random_state'], [check_random_state])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', import_187349)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['basinhopping']
module_type_store.set_exportable_members(['basinhopping'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_187351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_187352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'basinhopping')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_187351, str_187352)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_187351)
# Declaration of the 'Storage' class

class Storage(object, ):
    str_187353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', '\n    Class used to store the lowest energy structure\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Storage.__init__', ['minres'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['minres'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to _add(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'minres' (line 20)
        minres_187356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'minres', False)
        # Processing the call keyword arguments (line 20)
        kwargs_187357 = {}
        # Getting the type of 'self' (line 20)
        self_187354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', False)
        # Obtaining the member '_add' of a type (line 20)
        _add_187355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_187354, '_add')
        # Calling _add(args, kwargs) (line 20)
        _add_call_result_187358 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), _add_187355, *[minres_187356], **kwargs_187357)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add'
        module_type_store = module_type_store.open_function_context('_add', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Storage._add.__dict__.__setitem__('stypy_localization', localization)
        Storage._add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Storage._add.__dict__.__setitem__('stypy_type_store', module_type_store)
        Storage._add.__dict__.__setitem__('stypy_function_name', 'Storage._add')
        Storage._add.__dict__.__setitem__('stypy_param_names_list', ['minres'])
        Storage._add.__dict__.__setitem__('stypy_varargs_param_name', None)
        Storage._add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Storage._add.__dict__.__setitem__('stypy_call_defaults', defaults)
        Storage._add.__dict__.__setitem__('stypy_call_varargs', varargs)
        Storage._add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Storage._add.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Storage._add', ['minres'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add', localization, ['minres'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add(...)' code ##################

        
        # Assigning a Name to a Attribute (line 23):
        
        # Assigning a Name to a Attribute (line 23):
        # Getting the type of 'minres' (line 23)
        minres_187359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'minres')
        # Getting the type of 'self' (line 23)
        self_187360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'minres' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_187360, 'minres', minres_187359)
        
        # Assigning a Call to a Attribute (line 24):
        
        # Assigning a Call to a Attribute (line 24):
        
        # Call to copy(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'minres' (line 24)
        minres_187363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 32), 'minres', False)
        # Obtaining the member 'x' of a type (line 24)
        x_187364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 32), minres_187363, 'x')
        # Processing the call keyword arguments (line 24)
        kwargs_187365 = {}
        # Getting the type of 'np' (line 24)
        np_187361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'np', False)
        # Obtaining the member 'copy' of a type (line 24)
        copy_187362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 24), np_187361, 'copy')
        # Calling copy(args, kwargs) (line 24)
        copy_call_result_187366 = invoke(stypy.reporting.localization.Localization(__file__, 24, 24), copy_187362, *[x_187364], **kwargs_187365)
        
        # Getting the type of 'self' (line 24)
        self_187367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Obtaining the member 'minres' of a type (line 24)
        minres_187368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_187367, 'minres')
        # Setting the type of the member 'x' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), minres_187368, 'x', copy_call_result_187366)
        
        # ################# End of '_add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_187369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add'
        return stypy_return_type_187369


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Storage.update.__dict__.__setitem__('stypy_localization', localization)
        Storage.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Storage.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        Storage.update.__dict__.__setitem__('stypy_function_name', 'Storage.update')
        Storage.update.__dict__.__setitem__('stypy_param_names_list', ['minres'])
        Storage.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        Storage.update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Storage.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        Storage.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        Storage.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Storage.update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Storage.update', ['minres'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['minres'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        
        
        # Getting the type of 'minres' (line 27)
        minres_187370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'minres')
        # Obtaining the member 'fun' of a type (line 27)
        fun_187371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), minres_187370, 'fun')
        # Getting the type of 'self' (line 27)
        self_187372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'self')
        # Obtaining the member 'minres' of a type (line 27)
        minres_187373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 24), self_187372, 'minres')
        # Obtaining the member 'fun' of a type (line 27)
        fun_187374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 24), minres_187373, 'fun')
        # Applying the binary operator '<' (line 27)
        result_lt_187375 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 11), '<', fun_187371, fun_187374)
        
        # Testing the type of an if condition (line 27)
        if_condition_187376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 8), result_lt_187375)
        # Assigning a type to the variable 'if_condition_187376' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_condition_187376', if_condition_187376)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _add(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'minres' (line 28)
        minres_187379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'minres', False)
        # Processing the call keyword arguments (line 28)
        kwargs_187380 = {}
        # Getting the type of 'self' (line 28)
        self_187377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'self', False)
        # Obtaining the member '_add' of a type (line 28)
        _add_187378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), self_187377, '_add')
        # Calling _add(args, kwargs) (line 28)
        _add_call_result_187381 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), _add_187378, *[minres_187379], **kwargs_187380)
        
        # Getting the type of 'True' (line 29)
        True_187382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'stypy_return_type', True_187382)
        # SSA branch for the else part of an if statement (line 27)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'False' (line 31)
        False_187383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'stypy_return_type', False_187383)
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_187384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187384)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_187384


    @norecursion
    def get_lowest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_lowest'
        module_type_store = module_type_store.open_function_context('get_lowest', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Storage.get_lowest.__dict__.__setitem__('stypy_localization', localization)
        Storage.get_lowest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Storage.get_lowest.__dict__.__setitem__('stypy_type_store', module_type_store)
        Storage.get_lowest.__dict__.__setitem__('stypy_function_name', 'Storage.get_lowest')
        Storage.get_lowest.__dict__.__setitem__('stypy_param_names_list', [])
        Storage.get_lowest.__dict__.__setitem__('stypy_varargs_param_name', None)
        Storage.get_lowest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Storage.get_lowest.__dict__.__setitem__('stypy_call_defaults', defaults)
        Storage.get_lowest.__dict__.__setitem__('stypy_call_varargs', varargs)
        Storage.get_lowest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Storage.get_lowest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Storage.get_lowest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_lowest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_lowest(...)' code ##################

        # Getting the type of 'self' (line 34)
        self_187385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'self')
        # Obtaining the member 'minres' of a type (line 34)
        minres_187386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), self_187385, 'minres')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', minres_187386)
        
        # ################# End of 'get_lowest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_lowest' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_187387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_lowest'
        return stypy_return_type_187387


# Assigning a type to the variable 'Storage' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Storage', Storage)
# Declaration of the 'BasinHoppingRunner' class

class BasinHoppingRunner(object, ):
    str_187388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', 'This class implements the core of the basinhopping algorithm.\n\n    x0 : ndarray\n        The starting coordinates.\n    minimizer : callable\n        The local minimizer, with signature ``result = minimizer(x)``.\n        The return value is an `optimize.OptimizeResult` object.\n    step_taking : callable\n        This function displaces the coordinates randomly.  Signature should\n        be ``x_new = step_taking(x)``.  Note that `x` may be modified in-place.\n    accept_tests : list of callables\n        Each test is passed the kwargs `f_new`, `x_new`, `f_old` and\n        `x_old`.  These tests will be used to judge whether or not to accept\n        the step.  The acceptable return values are True, False, or ``"force\n        accept"``.  If any of the tests return False then the step is rejected.\n        If the latter, then this will override any other tests in order to\n        accept the step. This can be used, for example, to forcefully escape\n        from a local minimum that ``basinhopping`` is trapped in.\n    disp : bool, optional\n        Display status messages.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 60)
        False_187389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 70), 'False')
        defaults = [False_187389]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BasinHoppingRunner.__init__', ['x0', 'minimizer', 'step_taking', 'accept_tests', 'disp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x0', 'minimizer', 'step_taking', 'accept_tests', 'disp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 61):
        
        # Assigning a Call to a Attribute (line 61):
        
        # Call to copy(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'x0' (line 61)
        x0_187392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'x0', False)
        # Processing the call keyword arguments (line 61)
        kwargs_187393 = {}
        # Getting the type of 'np' (line 61)
        np_187390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'np', False)
        # Obtaining the member 'copy' of a type (line 61)
        copy_187391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), np_187390, 'copy')
        # Calling copy(args, kwargs) (line 61)
        copy_call_result_187394 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), copy_187391, *[x0_187392], **kwargs_187393)
        
        # Getting the type of 'self' (line 61)
        self_187395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'x' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_187395, 'x', copy_call_result_187394)
        
        # Assigning a Name to a Attribute (line 62):
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'minimizer' (line 62)
        minimizer_187396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'minimizer')
        # Getting the type of 'self' (line 62)
        self_187397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'minimizer' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_187397, 'minimizer', minimizer_187396)
        
        # Assigning a Name to a Attribute (line 63):
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'step_taking' (line 63)
        step_taking_187398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'step_taking')
        # Getting the type of 'self' (line 63)
        self_187399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'step_taking' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_187399, 'step_taking', step_taking_187398)
        
        # Assigning a Name to a Attribute (line 64):
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'accept_tests' (line 64)
        accept_tests_187400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'accept_tests')
        # Getting the type of 'self' (line 64)
        self_187401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'accept_tests' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_187401, 'accept_tests', accept_tests_187400)
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'disp' (line 65)
        disp_187402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'disp')
        # Getting the type of 'self' (line 65)
        self_187403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'disp' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_187403, 'disp', disp_187402)
        
        # Assigning a Num to a Attribute (line 67):
        
        # Assigning a Num to a Attribute (line 67):
        int_187404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'int')
        # Getting the type of 'self' (line 67)
        self_187405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'nstep' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_187405, 'nstep', int_187404)
        
        # Assigning a Call to a Attribute (line 70):
        
        # Assigning a Call to a Attribute (line 70):
        
        # Call to OptimizeResult(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_187409 = {}
        # Getting the type of 'scipy' (line 70)
        scipy_187406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'scipy', False)
        # Obtaining the member 'optimize' of a type (line 70)
        optimize_187407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), scipy_187406, 'optimize')
        # Obtaining the member 'OptimizeResult' of a type (line 70)
        OptimizeResult_187408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), optimize_187407, 'OptimizeResult')
        # Calling OptimizeResult(args, kwargs) (line 70)
        OptimizeResult_call_result_187410 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), OptimizeResult_187408, *[], **kwargs_187409)
        
        # Getting the type of 'self' (line 70)
        self_187411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'res' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_187411, 'res', OptimizeResult_call_result_187410)
        
        # Assigning a Num to a Attribute (line 71):
        
        # Assigning a Num to a Attribute (line 71):
        int_187412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 41), 'int')
        # Getting the type of 'self' (line 71)
        self_187413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Obtaining the member 'res' of a type (line 71)
        res_187414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_187413, 'res')
        # Setting the type of the member 'minimization_failures' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), res_187414, 'minimization_failures', int_187412)
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to minimizer(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'self' (line 74)
        self_187416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'self', False)
        # Obtaining the member 'x' of a type (line 74)
        x_187417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 27), self_187416, 'x')
        # Processing the call keyword arguments (line 74)
        kwargs_187418 = {}
        # Getting the type of 'minimizer' (line 74)
        minimizer_187415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'minimizer', False)
        # Calling minimizer(args, kwargs) (line 74)
        minimizer_call_result_187419 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), minimizer_187415, *[x_187417], **kwargs_187418)
        
        # Assigning a type to the variable 'minres' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'minres', minimizer_call_result_187419)
        
        
        # Getting the type of 'minres' (line 75)
        minres_187420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'minres')
        # Obtaining the member 'success' of a type (line 75)
        success_187421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), minres_187420, 'success')
        # Applying the 'not' unary operator (line 75)
        result_not__187422 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), 'not', success_187421)
        
        # Testing the type of an if condition (line 75)
        if_condition_187423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_not__187422)
        # Assigning a type to the variable 'if_condition_187423' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_187423', if_condition_187423)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 76)
        self_187424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'self')
        # Obtaining the member 'res' of a type (line 76)
        res_187425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), self_187424, 'res')
        # Obtaining the member 'minimization_failures' of a type (line 76)
        minimization_failures_187426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), res_187425, 'minimization_failures')
        int_187427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 46), 'int')
        # Applying the binary operator '+=' (line 76)
        result_iadd_187428 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 12), '+=', minimization_failures_187426, int_187427)
        # Getting the type of 'self' (line 76)
        self_187429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'self')
        # Obtaining the member 'res' of a type (line 76)
        res_187430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), self_187429, 'res')
        # Setting the type of the member 'minimization_failures' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), res_187430, 'minimization_failures', result_iadd_187428)
        
        
        # Getting the type of 'self' (line 77)
        self_187431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'self')
        # Obtaining the member 'disp' of a type (line 77)
        disp_187432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 15), self_187431, 'disp')
        # Testing the type of an if condition (line 77)
        if_condition_187433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 12), disp_187432)
        # Assigning a type to the variable 'if_condition_187433' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'if_condition_187433', if_condition_187433)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 78)
        # Processing the call arguments (line 78)
        str_187435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'str', 'warning: basinhopping: local minimization failure')
        # Processing the call keyword arguments (line 78)
        kwargs_187436 = {}
        # Getting the type of 'print' (line 78)
        print_187434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'print', False)
        # Calling print(args, kwargs) (line 78)
        print_call_result_187437 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), print_187434, *[str_187435], **kwargs_187436)
        
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 79):
        
        # Assigning a Call to a Attribute (line 79):
        
        # Call to copy(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'minres' (line 79)
        minres_187440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'minres', False)
        # Obtaining the member 'x' of a type (line 79)
        x_187441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), minres_187440, 'x')
        # Processing the call keyword arguments (line 79)
        kwargs_187442 = {}
        # Getting the type of 'np' (line 79)
        np_187438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'np', False)
        # Obtaining the member 'copy' of a type (line 79)
        copy_187439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 17), np_187438, 'copy')
        # Calling copy(args, kwargs) (line 79)
        copy_call_result_187443 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), copy_187439, *[x_187441], **kwargs_187442)
        
        # Getting the type of 'self' (line 79)
        self_187444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'x' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_187444, 'x', copy_call_result_187443)
        
        # Assigning a Attribute to a Attribute (line 80):
        
        # Assigning a Attribute to a Attribute (line 80):
        # Getting the type of 'minres' (line 80)
        minres_187445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'minres')
        # Obtaining the member 'fun' of a type (line 80)
        fun_187446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 22), minres_187445, 'fun')
        # Getting the type of 'self' (line 80)
        self_187447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Setting the type of the member 'energy' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_187447, 'energy', fun_187446)
        
        # Getting the type of 'self' (line 81)
        self_187448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'self')
        # Obtaining the member 'disp' of a type (line 81)
        disp_187449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), self_187448, 'disp')
        # Testing the type of an if condition (line 81)
        if_condition_187450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), disp_187449)
        # Assigning a type to the variable 'if_condition_187450' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_187450', if_condition_187450)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 82)
        # Processing the call arguments (line 82)
        str_187452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'str', 'basinhopping step %d: f %g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_187453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'self' (line 82)
        self_187454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 50), 'self', False)
        # Obtaining the member 'nstep' of a type (line 82)
        nstep_187455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 50), self_187454, 'nstep')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 50), tuple_187453, nstep_187455)
        # Adding element type (line 82)
        # Getting the type of 'self' (line 82)
        self_187456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 62), 'self', False)
        # Obtaining the member 'energy' of a type (line 82)
        energy_187457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 62), self_187456, 'energy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 50), tuple_187453, energy_187457)
        
        # Applying the binary operator '%' (line 82)
        result_mod_187458 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 18), '%', str_187452, tuple_187453)
        
        # Processing the call keyword arguments (line 82)
        kwargs_187459 = {}
        # Getting the type of 'print' (line 82)
        print_187451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'print', False)
        # Calling print(args, kwargs) (line 82)
        print_call_result_187460 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), print_187451, *[result_mod_187458], **kwargs_187459)
        
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 85):
        
        # Assigning a Call to a Attribute (line 85):
        
        # Call to Storage(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'minres' (line 85)
        minres_187462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'minres', False)
        # Processing the call keyword arguments (line 85)
        kwargs_187463 = {}
        # Getting the type of 'Storage' (line 85)
        Storage_187461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'Storage', False)
        # Calling Storage(args, kwargs) (line 85)
        Storage_call_result_187464 = invoke(stypy.reporting.localization.Localization(__file__, 85, 23), Storage_187461, *[minres_187462], **kwargs_187463)
        
        # Getting the type of 'self' (line 85)
        self_187465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'storage' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_187465, 'storage', Storage_call_result_187464)
        
        # Type idiom detected: calculating its left and rigth part (line 87)
        str_187466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'str', 'nfev')
        # Getting the type of 'minres' (line 87)
        minres_187467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'minres')
        
        (may_be_187468, more_types_in_union_187469) = may_provide_member(str_187466, minres_187467)

        if may_be_187468:

            if more_types_in_union_187469:
                # Runtime conditional SSA (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'minres' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'minres', remove_not_member_provider_from_union(minres_187467, 'nfev'))
            
            # Assigning a Attribute to a Attribute (line 88):
            
            # Assigning a Attribute to a Attribute (line 88):
            # Getting the type of 'minres' (line 88)
            minres_187470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'minres')
            # Obtaining the member 'nfev' of a type (line 88)
            nfev_187471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 28), minres_187470, 'nfev')
            # Getting the type of 'self' (line 88)
            self_187472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self')
            # Obtaining the member 'res' of a type (line 88)
            res_187473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), self_187472, 'res')
            # Setting the type of the member 'nfev' of a type (line 88)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), res_187473, 'nfev', nfev_187471)

            if more_types_in_union_187469:
                # SSA join for if statement (line 87)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 89)
        str_187474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 27), 'str', 'njev')
        # Getting the type of 'minres' (line 89)
        minres_187475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'minres')
        
        (may_be_187476, more_types_in_union_187477) = may_provide_member(str_187474, minres_187475)

        if may_be_187476:

            if more_types_in_union_187477:
                # Runtime conditional SSA (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'minres' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'minres', remove_not_member_provider_from_union(minres_187475, 'njev'))
            
            # Assigning a Attribute to a Attribute (line 90):
            
            # Assigning a Attribute to a Attribute (line 90):
            # Getting the type of 'minres' (line 90)
            minres_187478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'minres')
            # Obtaining the member 'njev' of a type (line 90)
            njev_187479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), minres_187478, 'njev')
            # Getting the type of 'self' (line 90)
            self_187480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'self')
            # Obtaining the member 'res' of a type (line 90)
            res_187481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), self_187480, 'res')
            # Setting the type of the member 'njev' of a type (line 90)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), res_187481, 'njev', njev_187479)

            if more_types_in_union_187477:
                # SSA join for if statement (line 89)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 91)
        str_187482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'str', 'nhev')
        # Getting the type of 'minres' (line 91)
        minres_187483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'minres')
        
        (may_be_187484, more_types_in_union_187485) = may_provide_member(str_187482, minres_187483)

        if may_be_187484:

            if more_types_in_union_187485:
                # Runtime conditional SSA (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'minres' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'minres', remove_not_member_provider_from_union(minres_187483, 'nhev'))
            
            # Assigning a Attribute to a Attribute (line 92):
            
            # Assigning a Attribute to a Attribute (line 92):
            # Getting the type of 'minres' (line 92)
            minres_187486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'minres')
            # Obtaining the member 'nhev' of a type (line 92)
            nhev_187487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), minres_187486, 'nhev')
            # Getting the type of 'self' (line 92)
            self_187488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self')
            # Obtaining the member 'res' of a type (line 92)
            res_187489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_187488, 'res')
            # Setting the type of the member 'nhev' of a type (line 92)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), res_187489, 'nhev', nhev_187487)

            if more_types_in_union_187485:
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _monte_carlo_step(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_monte_carlo_step'
        module_type_store = module_type_store.open_function_context('_monte_carlo_step', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_localization', localization)
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_type_store', module_type_store)
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_function_name', 'BasinHoppingRunner._monte_carlo_step')
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_param_names_list', [])
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_varargs_param_name', None)
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_call_defaults', defaults)
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_call_varargs', varargs)
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BasinHoppingRunner._monte_carlo_step.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BasinHoppingRunner._monte_carlo_step', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_monte_carlo_step', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_monte_carlo_step(...)' code ##################

        str_187490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'str', 'Do one monte carlo iteration\n\n        Randomly displace the coordinates, minimize, and decide whether\n        or not to accept the new coordinates.\n        ')
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to copy(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'self' (line 102)
        self_187493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'self', False)
        # Obtaining the member 'x' of a type (line 102)
        x_187494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 31), self_187493, 'x')
        # Processing the call keyword arguments (line 102)
        kwargs_187495 = {}
        # Getting the type of 'np' (line 102)
        np_187491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'np', False)
        # Obtaining the member 'copy' of a type (line 102)
        copy_187492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 23), np_187491, 'copy')
        # Calling copy(args, kwargs) (line 102)
        copy_call_result_187496 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), copy_187492, *[x_187494], **kwargs_187495)
        
        # Assigning a type to the variable 'x_after_step' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'x_after_step', copy_call_result_187496)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to step_taking(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'x_after_step' (line 103)
        x_after_step_187499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'x_after_step', False)
        # Processing the call keyword arguments (line 103)
        kwargs_187500 = {}
        # Getting the type of 'self' (line 103)
        self_187497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'self', False)
        # Obtaining the member 'step_taking' of a type (line 103)
        step_taking_187498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 23), self_187497, 'step_taking')
        # Calling step_taking(args, kwargs) (line 103)
        step_taking_call_result_187501 = invoke(stypy.reporting.localization.Localization(__file__, 103, 23), step_taking_187498, *[x_after_step_187499], **kwargs_187500)
        
        # Assigning a type to the variable 'x_after_step' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'x_after_step', step_taking_call_result_187501)
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to minimizer(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'x_after_step' (line 106)
        x_after_step_187504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 32), 'x_after_step', False)
        # Processing the call keyword arguments (line 106)
        kwargs_187505 = {}
        # Getting the type of 'self' (line 106)
        self_187502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'self', False)
        # Obtaining the member 'minimizer' of a type (line 106)
        minimizer_187503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 17), self_187502, 'minimizer')
        # Calling minimizer(args, kwargs) (line 106)
        minimizer_call_result_187506 = invoke(stypy.reporting.localization.Localization(__file__, 106, 17), minimizer_187503, *[x_after_step_187504], **kwargs_187505)
        
        # Assigning a type to the variable 'minres' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'minres', minimizer_call_result_187506)
        
        # Assigning a Attribute to a Name (line 107):
        
        # Assigning a Attribute to a Name (line 107):
        # Getting the type of 'minres' (line 107)
        minres_187507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'minres')
        # Obtaining the member 'x' of a type (line 107)
        x_187508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 25), minres_187507, 'x')
        # Assigning a type to the variable 'x_after_quench' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'x_after_quench', x_187508)
        
        # Assigning a Attribute to a Name (line 108):
        
        # Assigning a Attribute to a Name (line 108):
        # Getting the type of 'minres' (line 108)
        minres_187509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'minres')
        # Obtaining the member 'fun' of a type (line 108)
        fun_187510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 30), minres_187509, 'fun')
        # Assigning a type to the variable 'energy_after_quench' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'energy_after_quench', fun_187510)
        
        
        # Getting the type of 'minres' (line 109)
        minres_187511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'minres')
        # Obtaining the member 'success' of a type (line 109)
        success_187512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), minres_187511, 'success')
        # Applying the 'not' unary operator (line 109)
        result_not__187513 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), 'not', success_187512)
        
        # Testing the type of an if condition (line 109)
        if_condition_187514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_not__187513)
        # Assigning a type to the variable 'if_condition_187514' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_187514', if_condition_187514)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 110)
        self_187515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'self')
        # Obtaining the member 'res' of a type (line 110)
        res_187516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), self_187515, 'res')
        # Obtaining the member 'minimization_failures' of a type (line 110)
        minimization_failures_187517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), res_187516, 'minimization_failures')
        int_187518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 46), 'int')
        # Applying the binary operator '+=' (line 110)
        result_iadd_187519 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 12), '+=', minimization_failures_187517, int_187518)
        # Getting the type of 'self' (line 110)
        self_187520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'self')
        # Obtaining the member 'res' of a type (line 110)
        res_187521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), self_187520, 'res')
        # Setting the type of the member 'minimization_failures' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), res_187521, 'minimization_failures', result_iadd_187519)
        
        
        # Getting the type of 'self' (line 111)
        self_187522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'self')
        # Obtaining the member 'disp' of a type (line 111)
        disp_187523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), self_187522, 'disp')
        # Testing the type of an if condition (line 111)
        if_condition_187524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), disp_187523)
        # Assigning a type to the variable 'if_condition_187524' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_187524', if_condition_187524)
        # SSA begins for if statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 112)
        # Processing the call arguments (line 112)
        str_187526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 22), 'str', 'warning: basinhopping: local minimization failure')
        # Processing the call keyword arguments (line 112)
        kwargs_187527 = {}
        # Getting the type of 'print' (line 112)
        print_187525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'print', False)
        # Calling print(args, kwargs) (line 112)
        print_call_result_187528 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), print_187525, *[str_187526], **kwargs_187527)
        
        # SSA join for if statement (line 111)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 114)
        str_187529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'str', 'nfev')
        # Getting the type of 'minres' (line 114)
        minres_187530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'minres')
        
        (may_be_187531, more_types_in_union_187532) = may_provide_member(str_187529, minres_187530)

        if may_be_187531:

            if more_types_in_union_187532:
                # Runtime conditional SSA (line 114)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'minres' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'minres', remove_not_member_provider_from_union(minres_187530, 'nfev'))
            
            # Getting the type of 'self' (line 115)
            self_187533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self')
            # Obtaining the member 'res' of a type (line 115)
            res_187534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_187533, 'res')
            # Obtaining the member 'nfev' of a type (line 115)
            nfev_187535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), res_187534, 'nfev')
            # Getting the type of 'minres' (line 115)
            minres_187536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'minres')
            # Obtaining the member 'nfev' of a type (line 115)
            nfev_187537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 29), minres_187536, 'nfev')
            # Applying the binary operator '+=' (line 115)
            result_iadd_187538 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 12), '+=', nfev_187535, nfev_187537)
            # Getting the type of 'self' (line 115)
            self_187539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self')
            # Obtaining the member 'res' of a type (line 115)
            res_187540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_187539, 'res')
            # Setting the type of the member 'nfev' of a type (line 115)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), res_187540, 'nfev', result_iadd_187538)
            

            if more_types_in_union_187532:
                # SSA join for if statement (line 114)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 116)
        str_187541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 27), 'str', 'njev')
        # Getting the type of 'minres' (line 116)
        minres_187542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'minres')
        
        (may_be_187543, more_types_in_union_187544) = may_provide_member(str_187541, minres_187542)

        if may_be_187543:

            if more_types_in_union_187544:
                # Runtime conditional SSA (line 116)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'minres' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'minres', remove_not_member_provider_from_union(minres_187542, 'njev'))
            
            # Getting the type of 'self' (line 117)
            self_187545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'self')
            # Obtaining the member 'res' of a type (line 117)
            res_187546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), self_187545, 'res')
            # Obtaining the member 'njev' of a type (line 117)
            njev_187547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), res_187546, 'njev')
            # Getting the type of 'minres' (line 117)
            minres_187548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'minres')
            # Obtaining the member 'njev' of a type (line 117)
            njev_187549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 29), minres_187548, 'njev')
            # Applying the binary operator '+=' (line 117)
            result_iadd_187550 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '+=', njev_187547, njev_187549)
            # Getting the type of 'self' (line 117)
            self_187551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'self')
            # Obtaining the member 'res' of a type (line 117)
            res_187552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), self_187551, 'res')
            # Setting the type of the member 'njev' of a type (line 117)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), res_187552, 'njev', result_iadd_187550)
            

            if more_types_in_union_187544:
                # SSA join for if statement (line 116)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 118)
        str_187553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 27), 'str', 'nhev')
        # Getting the type of 'minres' (line 118)
        minres_187554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'minres')
        
        (may_be_187555, more_types_in_union_187556) = may_provide_member(str_187553, minres_187554)

        if may_be_187555:

            if more_types_in_union_187556:
                # Runtime conditional SSA (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'minres' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'minres', remove_not_member_provider_from_union(minres_187554, 'nhev'))
            
            # Getting the type of 'self' (line 119)
            self_187557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'self')
            # Obtaining the member 'res' of a type (line 119)
            res_187558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), self_187557, 'res')
            # Obtaining the member 'nhev' of a type (line 119)
            nhev_187559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), res_187558, 'nhev')
            # Getting the type of 'minres' (line 119)
            minres_187560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'minres')
            # Obtaining the member 'nhev' of a type (line 119)
            nhev_187561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 29), minres_187560, 'nhev')
            # Applying the binary operator '+=' (line 119)
            result_iadd_187562 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 12), '+=', nhev_187559, nhev_187561)
            # Getting the type of 'self' (line 119)
            self_187563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'self')
            # Obtaining the member 'res' of a type (line 119)
            res_187564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), self_187563, 'res')
            # Setting the type of the member 'nhev' of a type (line 119)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), res_187564, 'nhev', result_iadd_187562)
            

            if more_types_in_union_187556:
                # SSA join for if statement (line 118)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 126):
        
        # Assigning a Name to a Name (line 126):
        # Getting the type of 'True' (line 126)
        True_187565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'True')
        # Assigning a type to the variable 'accept' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'accept', True_187565)
        
        # Getting the type of 'self' (line 127)
        self_187566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'self')
        # Obtaining the member 'accept_tests' of a type (line 127)
        accept_tests_187567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 20), self_187566, 'accept_tests')
        # Testing the type of a for loop iterable (line 127)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 8), accept_tests_187567)
        # Getting the type of the for loop variable (line 127)
        for_loop_var_187568 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 8), accept_tests_187567)
        # Assigning a type to the variable 'test' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'test', for_loop_var_187568)
        # SSA begins for a for statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to test(...): (line 128)
        # Processing the call keyword arguments (line 128)
        # Getting the type of 'energy_after_quench' (line 128)
        energy_after_quench_187570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 33), 'energy_after_quench', False)
        keyword_187571 = energy_after_quench_187570
        # Getting the type of 'x_after_quench' (line 128)
        x_after_quench_187572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 60), 'x_after_quench', False)
        keyword_187573 = x_after_quench_187572
        # Getting the type of 'self' (line 129)
        self_187574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'self', False)
        # Obtaining the member 'energy' of a type (line 129)
        energy_187575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 33), self_187574, 'energy')
        keyword_187576 = energy_187575
        # Getting the type of 'self' (line 129)
        self_187577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 52), 'self', False)
        # Obtaining the member 'x' of a type (line 129)
        x_187578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 52), self_187577, 'x')
        keyword_187579 = x_187578
        kwargs_187580 = {'f_new': keyword_187571, 'f_old': keyword_187576, 'x_new': keyword_187573, 'x_old': keyword_187579}
        # Getting the type of 'test' (line 128)
        test_187569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'test', False)
        # Calling test(args, kwargs) (line 128)
        test_call_result_187581 = invoke(stypy.reporting.localization.Localization(__file__, 128, 22), test_187569, *[], **kwargs_187580)
        
        # Assigning a type to the variable 'testres' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'testres', test_call_result_187581)
        
        
        # Getting the type of 'testres' (line 130)
        testres_187582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'testres')
        str_187583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 26), 'str', 'force accept')
        # Applying the binary operator '==' (line 130)
        result_eq_187584 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 15), '==', testres_187582, str_187583)
        
        # Testing the type of an if condition (line 130)
        if_condition_187585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 12), result_eq_187584)
        # Assigning a type to the variable 'if_condition_187585' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'if_condition_187585', if_condition_187585)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 131):
        
        # Assigning a Name to a Name (line 131):
        # Getting the type of 'True' (line 131)
        True_187586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'True')
        # Assigning a type to the variable 'accept' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'accept', True_187586)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'testres' (line 133)
        testres_187587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 21), 'testres')
        # Applying the 'not' unary operator (line 133)
        result_not__187588 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 17), 'not', testres_187587)
        
        # Testing the type of an if condition (line 133)
        if_condition_187589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 17), result_not__187588)
        # Assigning a type to the variable 'if_condition_187589' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'if_condition_187589', if_condition_187589)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 134):
        
        # Assigning a Name to a Name (line 134):
        # Getting the type of 'False' (line 134)
        False_187590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'False')
        # Assigning a type to the variable 'accept' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'accept', False_187590)
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 138)
        str_187591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 37), 'str', 'report')
        # Getting the type of 'self' (line 138)
        self_187592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'self')
        # Obtaining the member 'step_taking' of a type (line 138)
        step_taking_187593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), self_187592, 'step_taking')
        
        (may_be_187594, more_types_in_union_187595) = may_provide_member(str_187591, step_taking_187593)

        if may_be_187594:

            if more_types_in_union_187595:
                # Runtime conditional SSA (line 138)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 138)
            self_187596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self')
            # Obtaining the member 'step_taking' of a type (line 138)
            step_taking_187597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_187596, 'step_taking')
            # Setting the type of the member 'step_taking' of a type (line 138)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_187596, 'step_taking', remove_not_member_provider_from_union(step_taking_187593, 'report'))
            
            # Call to report(...): (line 139)
            # Processing the call arguments (line 139)
            # Getting the type of 'accept' (line 139)
            accept_187601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 36), 'accept', False)
            # Processing the call keyword arguments (line 139)
            # Getting the type of 'energy_after_quench' (line 139)
            energy_after_quench_187602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'energy_after_quench', False)
            keyword_187603 = energy_after_quench_187602
            # Getting the type of 'x_after_quench' (line 140)
            x_after_quench_187604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 42), 'x_after_quench', False)
            keyword_187605 = x_after_quench_187604
            # Getting the type of 'self' (line 140)
            self_187606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 64), 'self', False)
            # Obtaining the member 'energy' of a type (line 140)
            energy_187607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 64), self_187606, 'energy')
            keyword_187608 = energy_187607
            # Getting the type of 'self' (line 141)
            self_187609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), 'self', False)
            # Obtaining the member 'x' of a type (line 141)
            x_187610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 42), self_187609, 'x')
            keyword_187611 = x_187610
            kwargs_187612 = {'f_new': keyword_187603, 'f_old': keyword_187608, 'x_new': keyword_187605, 'x_old': keyword_187611}
            # Getting the type of 'self' (line 139)
            self_187598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'self', False)
            # Obtaining the member 'step_taking' of a type (line 139)
            step_taking_187599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), self_187598, 'step_taking')
            # Obtaining the member 'report' of a type (line 139)
            report_187600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), step_taking_187599, 'report')
            # Calling report(args, kwargs) (line 139)
            report_call_result_187613 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), report_187600, *[accept_187601], **kwargs_187612)
            

            if more_types_in_union_187595:
                # SSA join for if statement (line 138)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 143)
        tuple_187614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 143)
        # Adding element type (line 143)
        # Getting the type of 'accept' (line 143)
        accept_187615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'accept')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 15), tuple_187614, accept_187615)
        # Adding element type (line 143)
        # Getting the type of 'minres' (line 143)
        minres_187616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'minres')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 15), tuple_187614, minres_187616)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', tuple_187614)
        
        # ################# End of '_monte_carlo_step(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_monte_carlo_step' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_187617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187617)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_monte_carlo_step'
        return stypy_return_type_187617


    @norecursion
    def one_cycle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'one_cycle'
        module_type_store = module_type_store.open_function_context('one_cycle', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_localization', localization)
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_type_store', module_type_store)
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_function_name', 'BasinHoppingRunner.one_cycle')
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_param_names_list', [])
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_varargs_param_name', None)
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_call_defaults', defaults)
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_call_varargs', varargs)
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BasinHoppingRunner.one_cycle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BasinHoppingRunner.one_cycle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'one_cycle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'one_cycle(...)' code ##################

        str_187618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'str', 'Do one cycle of the basinhopping algorithm\n        ')
        
        # Getting the type of 'self' (line 148)
        self_187619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self')
        # Obtaining the member 'nstep' of a type (line 148)
        nstep_187620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_187619, 'nstep')
        int_187621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 22), 'int')
        # Applying the binary operator '+=' (line 148)
        result_iadd_187622 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 8), '+=', nstep_187620, int_187621)
        # Getting the type of 'self' (line 148)
        self_187623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self')
        # Setting the type of the member 'nstep' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_187623, 'nstep', result_iadd_187622)
        
        
        # Assigning a Name to a Name (line 149):
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'False' (line 149)
        False_187624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'False')
        # Assigning a type to the variable 'new_global_min' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'new_global_min', False_187624)
        
        # Assigning a Call to a Tuple (line 151):
        
        # Assigning a Subscript to a Name (line 151):
        
        # Obtaining the type of the subscript
        int_187625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
        
        # Call to _monte_carlo_step(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_187628 = {}
        # Getting the type of 'self' (line 151)
        self_187626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'self', False)
        # Obtaining the member '_monte_carlo_step' of a type (line 151)
        _monte_carlo_step_187627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), self_187626, '_monte_carlo_step')
        # Calling _monte_carlo_step(args, kwargs) (line 151)
        _monte_carlo_step_call_result_187629 = invoke(stypy.reporting.localization.Localization(__file__, 151, 25), _monte_carlo_step_187627, *[], **kwargs_187628)
        
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___187630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), _monte_carlo_step_call_result_187629, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_187631 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___187630, int_187625)
        
        # Assigning a type to the variable 'tuple_var_assignment_187338' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_187338', subscript_call_result_187631)
        
        # Assigning a Subscript to a Name (line 151):
        
        # Obtaining the type of the subscript
        int_187632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
        
        # Call to _monte_carlo_step(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_187635 = {}
        # Getting the type of 'self' (line 151)
        self_187633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'self', False)
        # Obtaining the member '_monte_carlo_step' of a type (line 151)
        _monte_carlo_step_187634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), self_187633, '_monte_carlo_step')
        # Calling _monte_carlo_step(args, kwargs) (line 151)
        _monte_carlo_step_call_result_187636 = invoke(stypy.reporting.localization.Localization(__file__, 151, 25), _monte_carlo_step_187634, *[], **kwargs_187635)
        
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___187637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), _monte_carlo_step_call_result_187636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_187638 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___187637, int_187632)
        
        # Assigning a type to the variable 'tuple_var_assignment_187339' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_187339', subscript_call_result_187638)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'tuple_var_assignment_187338' (line 151)
        tuple_var_assignment_187338_187639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_187338')
        # Assigning a type to the variable 'accept' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'accept', tuple_var_assignment_187338_187639)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'tuple_var_assignment_187339' (line 151)
        tuple_var_assignment_187339_187640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_187339')
        # Assigning a type to the variable 'minres' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'minres', tuple_var_assignment_187339_187640)
        
        # Getting the type of 'accept' (line 153)
        accept_187641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'accept')
        # Testing the type of an if condition (line 153)
        if_condition_187642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), accept_187641)
        # Assigning a type to the variable 'if_condition_187642' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_187642', if_condition_187642)
        # SSA begins for if statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 154):
        
        # Assigning a Attribute to a Attribute (line 154):
        # Getting the type of 'minres' (line 154)
        minres_187643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'minres')
        # Obtaining the member 'fun' of a type (line 154)
        fun_187644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 26), minres_187643, 'fun')
        # Getting the type of 'self' (line 154)
        self_187645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'self')
        # Setting the type of the member 'energy' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), self_187645, 'energy', fun_187644)
        
        # Assigning a Call to a Attribute (line 155):
        
        # Assigning a Call to a Attribute (line 155):
        
        # Call to copy(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'minres' (line 155)
        minres_187648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'minres', False)
        # Obtaining the member 'x' of a type (line 155)
        x_187649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 29), minres_187648, 'x')
        # Processing the call keyword arguments (line 155)
        kwargs_187650 = {}
        # Getting the type of 'np' (line 155)
        np_187646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'np', False)
        # Obtaining the member 'copy' of a type (line 155)
        copy_187647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 21), np_187646, 'copy')
        # Calling copy(args, kwargs) (line 155)
        copy_call_result_187651 = invoke(stypy.reporting.localization.Localization(__file__, 155, 21), copy_187647, *[x_187649], **kwargs_187650)
        
        # Getting the type of 'self' (line 155)
        self_187652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self')
        # Setting the type of the member 'x' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_187652, 'x', copy_call_result_187651)
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to update(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'minres' (line 156)
        minres_187656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 49), 'minres', False)
        # Processing the call keyword arguments (line 156)
        kwargs_187657 = {}
        # Getting the type of 'self' (line 156)
        self_187653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'self', False)
        # Obtaining the member 'storage' of a type (line 156)
        storage_187654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 29), self_187653, 'storage')
        # Obtaining the member 'update' of a type (line 156)
        update_187655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 29), storage_187654, 'update')
        # Calling update(args, kwargs) (line 156)
        update_call_result_187658 = invoke(stypy.reporting.localization.Localization(__file__, 156, 29), update_187655, *[minres_187656], **kwargs_187657)
        
        # Assigning a type to the variable 'new_global_min' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'new_global_min', update_call_result_187658)
        # SSA join for if statement (line 153)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 159)
        self_187659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'self')
        # Obtaining the member 'disp' of a type (line 159)
        disp_187660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), self_187659, 'disp')
        # Testing the type of an if condition (line 159)
        if_condition_187661 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), disp_187660)
        # Assigning a type to the variable 'if_condition_187661' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_187661', if_condition_187661)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print_report(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'minres' (line 160)
        minres_187664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 30), 'minres', False)
        # Obtaining the member 'fun' of a type (line 160)
        fun_187665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 30), minres_187664, 'fun')
        # Getting the type of 'accept' (line 160)
        accept_187666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 42), 'accept', False)
        # Processing the call keyword arguments (line 160)
        kwargs_187667 = {}
        # Getting the type of 'self' (line 160)
        self_187662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'self', False)
        # Obtaining the member 'print_report' of a type (line 160)
        print_report_187663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), self_187662, 'print_report')
        # Calling print_report(args, kwargs) (line 160)
        print_report_call_result_187668 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), print_report_187663, *[fun_187665, accept_187666], **kwargs_187667)
        
        
        # Getting the type of 'new_global_min' (line 161)
        new_global_min_187669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'new_global_min')
        # Testing the type of an if condition (line 161)
        if_condition_187670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), new_global_min_187669)
        # Assigning a type to the variable 'if_condition_187670' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_187670', if_condition_187670)
        # SSA begins for if statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 162)
        # Processing the call arguments (line 162)
        str_187672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 22), 'str', 'found new global minimum on step %d with function value %g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 163)
        tuple_187673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 163)
        # Adding element type (line 163)
        # Getting the type of 'self' (line 163)
        self_187674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'self', False)
        # Obtaining the member 'nstep' of a type (line 163)
        nstep_187675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 37), self_187674, 'nstep')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 37), tuple_187673, nstep_187675)
        # Adding element type (line 163)
        # Getting the type of 'self' (line 163)
        self_187676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 49), 'self', False)
        # Obtaining the member 'energy' of a type (line 163)
        energy_187677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 49), self_187676, 'energy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 37), tuple_187673, energy_187677)
        
        # Applying the binary operator '%' (line 162)
        result_mod_187678 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 22), '%', str_187672, tuple_187673)
        
        # Processing the call keyword arguments (line 162)
        kwargs_187679 = {}
        # Getting the type of 'print' (line 162)
        print_187671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'print', False)
        # Calling print(args, kwargs) (line 162)
        print_call_result_187680 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), print_187671, *[result_mod_187678], **kwargs_187679)
        
        # SSA join for if statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 166):
        
        # Assigning a Attribute to a Attribute (line 166):
        # Getting the type of 'minres' (line 166)
        minres_187681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'minres')
        # Obtaining the member 'x' of a type (line 166)
        x_187682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 22), minres_187681, 'x')
        # Getting the type of 'self' (line 166)
        self_187683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member 'xtrial' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_187683, 'xtrial', x_187682)
        
        # Assigning a Attribute to a Attribute (line 167):
        
        # Assigning a Attribute to a Attribute (line 167):
        # Getting the type of 'minres' (line 167)
        minres_187684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'minres')
        # Obtaining the member 'fun' of a type (line 167)
        fun_187685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 28), minres_187684, 'fun')
        # Getting the type of 'self' (line 167)
        self_187686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'energy_trial' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_187686, 'energy_trial', fun_187685)
        
        # Assigning a Name to a Attribute (line 168):
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'accept' (line 168)
        accept_187687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'accept')
        # Getting the type of 'self' (line 168)
        self_187688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member 'accept' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_187688, 'accept', accept_187687)
        # Getting the type of 'new_global_min' (line 170)
        new_global_min_187689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'new_global_min')
        # Assigning a type to the variable 'stypy_return_type' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stypy_return_type', new_global_min_187689)
        
        # ################# End of 'one_cycle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'one_cycle' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_187690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187690)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'one_cycle'
        return stypy_return_type_187690


    @norecursion
    def print_report(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_report'
        module_type_store = module_type_store.open_function_context('print_report', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_localization', localization)
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_type_store', module_type_store)
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_function_name', 'BasinHoppingRunner.print_report')
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_param_names_list', ['energy_trial', 'accept'])
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_varargs_param_name', None)
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_call_defaults', defaults)
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_call_varargs', varargs)
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BasinHoppingRunner.print_report.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BasinHoppingRunner.print_report', ['energy_trial', 'accept'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_report', localization, ['energy_trial', 'accept'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_report(...)' code ##################

        str_187691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 8), 'str', 'print a status update')
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to get_lowest(...): (line 174)
        # Processing the call keyword arguments (line 174)
        kwargs_187695 = {}
        # Getting the type of 'self' (line 174)
        self_187692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'self', False)
        # Obtaining the member 'storage' of a type (line 174)
        storage_187693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 17), self_187692, 'storage')
        # Obtaining the member 'get_lowest' of a type (line 174)
        get_lowest_187694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 17), storage_187693, 'get_lowest')
        # Calling get_lowest(args, kwargs) (line 174)
        get_lowest_call_result_187696 = invoke(stypy.reporting.localization.Localization(__file__, 174, 17), get_lowest_187694, *[], **kwargs_187695)
        
        # Assigning a type to the variable 'minres' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'minres', get_lowest_call_result_187696)
        
        # Call to print(...): (line 175)
        # Processing the call arguments (line 175)
        str_187698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 14), 'str', 'basinhopping step %d: f %g trial_f %g accepted %d  lowest_f %g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 176)
        tuple_187699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 176)
        # Adding element type (line 176)
        # Getting the type of 'self' (line 176)
        self_187700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'self', False)
        # Obtaining the member 'nstep' of a type (line 176)
        nstep_187701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 32), self_187700, 'nstep')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), tuple_187699, nstep_187701)
        # Adding element type (line 176)
        # Getting the type of 'self' (line 176)
        self_187702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'self', False)
        # Obtaining the member 'energy' of a type (line 176)
        energy_187703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 44), self_187702, 'energy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), tuple_187699, energy_187703)
        # Adding element type (line 176)
        # Getting the type of 'energy_trial' (line 176)
        energy_trial_187704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 57), 'energy_trial', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), tuple_187699, energy_trial_187704)
        # Adding element type (line 176)
        # Getting the type of 'accept' (line 177)
        accept_187705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'accept', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), tuple_187699, accept_187705)
        # Adding element type (line 176)
        # Getting the type of 'minres' (line 177)
        minres_187706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 40), 'minres', False)
        # Obtaining the member 'fun' of a type (line 177)
        fun_187707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 40), minres_187706, 'fun')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), tuple_187699, fun_187707)
        
        # Applying the binary operator '%' (line 175)
        result_mod_187708 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 14), '%', str_187698, tuple_187699)
        
        # Processing the call keyword arguments (line 175)
        kwargs_187709 = {}
        # Getting the type of 'print' (line 175)
        print_187697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'print', False)
        # Calling print(args, kwargs) (line 175)
        print_call_result_187710 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), print_187697, *[result_mod_187708], **kwargs_187709)
        
        
        # ################# End of 'print_report(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_report' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_187711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_report'
        return stypy_return_type_187711


# Assigning a type to the variable 'BasinHoppingRunner' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'BasinHoppingRunner', BasinHoppingRunner)
# Declaration of the 'AdaptiveStepsize' class

class AdaptiveStepsize(object, ):
    str_187712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, (-1)), 'str', '\n    Class to implement adaptive stepsize.\n\n    This class wraps the step taking class and modifies the stepsize to\n    ensure the true acceptance rate is as close as possible to the target.\n\n    Parameters\n    ----------\n    takestep : callable\n        The step taking routine.  Must contain modifiable attribute\n        takestep.stepsize\n    accept_rate : float, optional\n        The target step acceptance rate\n    interval : int, optional\n        Interval for how often to update the stepsize\n    factor : float, optional\n        The step size is multiplied or divided by this factor upon each\n        update.\n    verbose : bool, optional\n        Print information about each update\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_187713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 45), 'float')
        int_187714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 59), 'int')
        float_187715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 70), 'float')
        # Getting the type of 'True' (line 204)
        True_187716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 25), 'True')
        defaults = [float_187713, int_187714, float_187715, True_187716]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AdaptiveStepsize.__init__', ['takestep', 'accept_rate', 'interval', 'factor', 'verbose'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['takestep', 'accept_rate', 'interval', 'factor', 'verbose'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 205):
        
        # Assigning a Name to a Attribute (line 205):
        # Getting the type of 'takestep' (line 205)
        takestep_187717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'takestep')
        # Getting the type of 'self' (line 205)
        self_187718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member 'takestep' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_187718, 'takestep', takestep_187717)
        
        # Assigning a Name to a Attribute (line 206):
        
        # Assigning a Name to a Attribute (line 206):
        # Getting the type of 'accept_rate' (line 206)
        accept_rate_187719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 34), 'accept_rate')
        # Getting the type of 'self' (line 206)
        self_187720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member 'target_accept_rate' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_187720, 'target_accept_rate', accept_rate_187719)
        
        # Assigning a Name to a Attribute (line 207):
        
        # Assigning a Name to a Attribute (line 207):
        # Getting the type of 'interval' (line 207)
        interval_187721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'interval')
        # Getting the type of 'self' (line 207)
        self_187722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        # Setting the type of the member 'interval' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_187722, 'interval', interval_187721)
        
        # Assigning a Name to a Attribute (line 208):
        
        # Assigning a Name to a Attribute (line 208):
        # Getting the type of 'factor' (line 208)
        factor_187723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), 'factor')
        # Getting the type of 'self' (line 208)
        self_187724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'self')
        # Setting the type of the member 'factor' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), self_187724, 'factor', factor_187723)
        
        # Assigning a Name to a Attribute (line 209):
        
        # Assigning a Name to a Attribute (line 209):
        # Getting the type of 'verbose' (line 209)
        verbose_187725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 23), 'verbose')
        # Getting the type of 'self' (line 209)
        self_187726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 209)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), self_187726, 'verbose', verbose_187725)
        
        # Assigning a Num to a Attribute (line 211):
        
        # Assigning a Num to a Attribute (line 211):
        int_187727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 21), 'int')
        # Getting the type of 'self' (line 211)
        self_187728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self')
        # Setting the type of the member 'nstep' of a type (line 211)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_187728, 'nstep', int_187727)
        
        # Assigning a Num to a Attribute (line 212):
        
        # Assigning a Num to a Attribute (line 212):
        int_187729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 25), 'int')
        # Getting the type of 'self' (line 212)
        self_187730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self')
        # Setting the type of the member 'nstep_tot' of a type (line 212)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_187730, 'nstep_tot', int_187729)
        
        # Assigning a Num to a Attribute (line 213):
        
        # Assigning a Num to a Attribute (line 213):
        int_187731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 23), 'int')
        # Getting the type of 'self' (line 213)
        self_187732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'self')
        # Setting the type of the member 'naccept' of a type (line 213)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), self_187732, 'naccept', int_187731)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_localization', localization)
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_function_name', 'AdaptiveStepsize.__call__')
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AdaptiveStepsize.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AdaptiveStepsize.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to take_step(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'x' (line 216)
        x_187735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 30), 'x', False)
        # Processing the call keyword arguments (line 216)
        kwargs_187736 = {}
        # Getting the type of 'self' (line 216)
        self_187733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'self', False)
        # Obtaining the member 'take_step' of a type (line 216)
        take_step_187734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 15), self_187733, 'take_step')
        # Calling take_step(args, kwargs) (line 216)
        take_step_call_result_187737 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), take_step_187734, *[x_187735], **kwargs_187736)
        
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', take_step_call_result_187737)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_187738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_187738


    @norecursion
    def _adjust_step_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjust_step_size'
        module_type_store = module_type_store.open_function_context('_adjust_step_size', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_localization', localization)
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_function_name', 'AdaptiveStepsize._adjust_step_size')
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_param_names_list', [])
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AdaptiveStepsize._adjust_step_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AdaptiveStepsize._adjust_step_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjust_step_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjust_step_size(...)' code ##################

        
        # Assigning a Attribute to a Name (line 219):
        
        # Assigning a Attribute to a Name (line 219):
        # Getting the type of 'self' (line 219)
        self_187739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'self')
        # Obtaining the member 'takestep' of a type (line 219)
        takestep_187740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 23), self_187739, 'takestep')
        # Obtaining the member 'stepsize' of a type (line 219)
        stepsize_187741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 23), takestep_187740, 'stepsize')
        # Assigning a type to the variable 'old_stepsize' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'old_stepsize', stepsize_187741)
        
        # Assigning a BinOp to a Name (line 220):
        
        # Assigning a BinOp to a Name (line 220):
        
        # Call to float(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'self' (line 220)
        self_187743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'self', False)
        # Obtaining the member 'naccept' of a type (line 220)
        naccept_187744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 28), self_187743, 'naccept')
        # Processing the call keyword arguments (line 220)
        kwargs_187745 = {}
        # Getting the type of 'float' (line 220)
        float_187742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'float', False)
        # Calling float(args, kwargs) (line 220)
        float_call_result_187746 = invoke(stypy.reporting.localization.Localization(__file__, 220, 22), float_187742, *[naccept_187744], **kwargs_187745)
        
        # Getting the type of 'self' (line 220)
        self_187747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 44), 'self')
        # Obtaining the member 'nstep' of a type (line 220)
        nstep_187748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 44), self_187747, 'nstep')
        # Applying the binary operator 'div' (line 220)
        result_div_187749 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 22), 'div', float_call_result_187746, nstep_187748)
        
        # Assigning a type to the variable 'accept_rate' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'accept_rate', result_div_187749)
        
        
        # Getting the type of 'accept_rate' (line 221)
        accept_rate_187750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'accept_rate')
        # Getting the type of 'self' (line 221)
        self_187751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'self')
        # Obtaining the member 'target_accept_rate' of a type (line 221)
        target_accept_rate_187752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 25), self_187751, 'target_accept_rate')
        # Applying the binary operator '>' (line 221)
        result_gt_187753 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 11), '>', accept_rate_187750, target_accept_rate_187752)
        
        # Testing the type of an if condition (line 221)
        if_condition_187754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), result_gt_187753)
        # Assigning a type to the variable 'if_condition_187754' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_187754', if_condition_187754)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 224)
        self_187755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'self')
        # Obtaining the member 'takestep' of a type (line 224)
        takestep_187756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), self_187755, 'takestep')
        # Obtaining the member 'stepsize' of a type (line 224)
        stepsize_187757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), takestep_187756, 'stepsize')
        # Getting the type of 'self' (line 224)
        self_187758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 38), 'self')
        # Obtaining the member 'factor' of a type (line 224)
        factor_187759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 38), self_187758, 'factor')
        # Applying the binary operator 'div=' (line 224)
        result_div_187760 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 12), 'div=', stepsize_187757, factor_187759)
        # Getting the type of 'self' (line 224)
        self_187761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'self')
        # Obtaining the member 'takestep' of a type (line 224)
        takestep_187762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), self_187761, 'takestep')
        # Setting the type of the member 'stepsize' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), takestep_187762, 'stepsize', result_div_187760)
        
        # SSA branch for the else part of an if statement (line 221)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 227)
        self_187763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'self')
        # Obtaining the member 'takestep' of a type (line 227)
        takestep_187764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), self_187763, 'takestep')
        # Obtaining the member 'stepsize' of a type (line 227)
        stepsize_187765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), takestep_187764, 'stepsize')
        # Getting the type of 'self' (line 227)
        self_187766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 38), 'self')
        # Obtaining the member 'factor' of a type (line 227)
        factor_187767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 38), self_187766, 'factor')
        # Applying the binary operator '*=' (line 227)
        result_imul_187768 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 12), '*=', stepsize_187765, factor_187767)
        # Getting the type of 'self' (line 227)
        self_187769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'self')
        # Obtaining the member 'takestep' of a type (line 227)
        takestep_187770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), self_187769, 'takestep')
        # Setting the type of the member 'stepsize' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), takestep_187770, 'stepsize', result_imul_187768)
        
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 228)
        self_187771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 228)
        verbose_187772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 11), self_187771, 'verbose')
        # Testing the type of an if condition (line 228)
        if_condition_187773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), verbose_187772)
        # Assigning a type to the variable 'if_condition_187773' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_187773', if_condition_187773)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 229)
        # Processing the call arguments (line 229)
        str_187775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 18), 'str', 'adaptive stepsize: acceptance rate %f target %f new stepsize %g old stepsize %g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 230)
        tuple_187776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 230)
        # Adding element type (line 230)
        # Getting the type of 'accept_rate' (line 230)
        accept_rate_187777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 51), 'accept_rate', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 51), tuple_187776, accept_rate_187777)
        # Adding element type (line 230)
        # Getting the type of 'self' (line 231)
        self_187778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'self', False)
        # Obtaining the member 'target_accept_rate' of a type (line 231)
        target_accept_rate_187779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 18), self_187778, 'target_accept_rate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 51), tuple_187776, target_accept_rate_187779)
        # Adding element type (line 230)
        # Getting the type of 'self' (line 231)
        self_187780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'self', False)
        # Obtaining the member 'takestep' of a type (line 231)
        takestep_187781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 43), self_187780, 'takestep')
        # Obtaining the member 'stepsize' of a type (line 231)
        stepsize_187782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 43), takestep_187781, 'stepsize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 51), tuple_187776, stepsize_187782)
        # Adding element type (line 230)
        # Getting the type of 'old_stepsize' (line 232)
        old_stepsize_187783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), 'old_stepsize', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 51), tuple_187776, old_stepsize_187783)
        
        # Applying the binary operator '%' (line 229)
        result_mod_187784 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 18), '%', str_187775, tuple_187776)
        
        # Processing the call keyword arguments (line 229)
        kwargs_187785 = {}
        # Getting the type of 'print' (line 229)
        print_187774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'print', False)
        # Calling print(args, kwargs) (line 229)
        print_call_result_187786 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), print_187774, *[result_mod_187784], **kwargs_187785)
        
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_adjust_step_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjust_step_size' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_187787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjust_step_size'
        return stypy_return_type_187787


    @norecursion
    def take_step(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'take_step'
        module_type_store = module_type_store.open_function_context('take_step', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_localization', localization)
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_type_store', module_type_store)
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_function_name', 'AdaptiveStepsize.take_step')
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_param_names_list', ['x'])
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_varargs_param_name', None)
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_call_defaults', defaults)
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_call_varargs', varargs)
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AdaptiveStepsize.take_step.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AdaptiveStepsize.take_step', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'take_step', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'take_step(...)' code ##################

        
        # Getting the type of 'self' (line 235)
        self_187788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Obtaining the member 'nstep' of a type (line 235)
        nstep_187789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_187788, 'nstep')
        int_187790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'int')
        # Applying the binary operator '+=' (line 235)
        result_iadd_187791 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 8), '+=', nstep_187789, int_187790)
        # Getting the type of 'self' (line 235)
        self_187792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member 'nstep' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_187792, 'nstep', result_iadd_187791)
        
        
        # Getting the type of 'self' (line 236)
        self_187793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self')
        # Obtaining the member 'nstep_tot' of a type (line 236)
        nstep_tot_187794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_187793, 'nstep_tot')
        int_187795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'int')
        # Applying the binary operator '+=' (line 236)
        result_iadd_187796 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 8), '+=', nstep_tot_187794, int_187795)
        # Getting the type of 'self' (line 236)
        self_187797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self')
        # Setting the type of the member 'nstep_tot' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_187797, 'nstep_tot', result_iadd_187796)
        
        
        
        # Getting the type of 'self' (line 237)
        self_187798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'self')
        # Obtaining the member 'nstep' of a type (line 237)
        nstep_187799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 11), self_187798, 'nstep')
        # Getting the type of 'self' (line 237)
        self_187800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 24), 'self')
        # Obtaining the member 'interval' of a type (line 237)
        interval_187801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 24), self_187800, 'interval')
        # Applying the binary operator '%' (line 237)
        result_mod_187802 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 11), '%', nstep_187799, interval_187801)
        
        int_187803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 41), 'int')
        # Applying the binary operator '==' (line 237)
        result_eq_187804 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 11), '==', result_mod_187802, int_187803)
        
        # Testing the type of an if condition (line 237)
        if_condition_187805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), result_eq_187804)
        # Assigning a type to the variable 'if_condition_187805' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_187805', if_condition_187805)
        # SSA begins for if statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _adjust_step_size(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_187808 = {}
        # Getting the type of 'self' (line 238)
        self_187806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'self', False)
        # Obtaining the member '_adjust_step_size' of a type (line 238)
        _adjust_step_size_187807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), self_187806, '_adjust_step_size')
        # Calling _adjust_step_size(args, kwargs) (line 238)
        _adjust_step_size_call_result_187809 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), _adjust_step_size_187807, *[], **kwargs_187808)
        
        # SSA join for if statement (line 237)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to takestep(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'x' (line 239)
        x_187812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 29), 'x', False)
        # Processing the call keyword arguments (line 239)
        kwargs_187813 = {}
        # Getting the type of 'self' (line 239)
        self_187810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'self', False)
        # Obtaining the member 'takestep' of a type (line 239)
        takestep_187811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 15), self_187810, 'takestep')
        # Calling takestep(args, kwargs) (line 239)
        takestep_call_result_187814 = invoke(stypy.reporting.localization.Localization(__file__, 239, 15), takestep_187811, *[x_187812], **kwargs_187813)
        
        # Assigning a type to the variable 'stypy_return_type' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'stypy_return_type', takestep_call_result_187814)
        
        # ################# End of 'take_step(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'take_step' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_187815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187815)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'take_step'
        return stypy_return_type_187815


    @norecursion
    def report(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'report'
        module_type_store = module_type_store.open_function_context('report', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_localization', localization)
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_type_store', module_type_store)
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_function_name', 'AdaptiveStepsize.report')
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_param_names_list', ['accept'])
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_varargs_param_name', None)
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_call_defaults', defaults)
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_call_varargs', varargs)
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AdaptiveStepsize.report.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AdaptiveStepsize.report', ['accept'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'report', localization, ['accept'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'report(...)' code ##################

        str_187816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 8), 'str', 'called by basinhopping to report the result of the step')
        
        # Getting the type of 'accept' (line 243)
        accept_187817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'accept')
        # Testing the type of an if condition (line 243)
        if_condition_187818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), accept_187817)
        # Assigning a type to the variable 'if_condition_187818' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_187818', if_condition_187818)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 244)
        self_187819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self')
        # Obtaining the member 'naccept' of a type (line 244)
        naccept_187820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), self_187819, 'naccept')
        int_187821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 28), 'int')
        # Applying the binary operator '+=' (line 244)
        result_iadd_187822 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), '+=', naccept_187820, int_187821)
        # Getting the type of 'self' (line 244)
        self_187823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self')
        # Setting the type of the member 'naccept' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), self_187823, 'naccept', result_iadd_187822)
        
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'report(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'report' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_187824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187824)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'report'
        return stypy_return_type_187824


# Assigning a type to the variable 'AdaptiveStepsize' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'AdaptiveStepsize', AdaptiveStepsize)
# Declaration of the 'RandomDisplacement' class

class RandomDisplacement(object, ):
    str_187825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', '\n    Add a random displacement of maximum size, stepsize, to the coordinates\n\n    update x inplace\n\n    Parameters\n    ----------\n    stepsize : float, optional\n        stepsize\n    random_state : None or `np.random.RandomState` instance, optional\n        The random number generator that generates the displacements\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_187826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'float')
        # Getting the type of 'None' (line 260)
        None_187827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 50), 'None')
        defaults = [float_187826, None_187827]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RandomDisplacement.__init__', ['stepsize', 'random_state'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['stepsize', 'random_state'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 261):
        
        # Assigning a Name to a Attribute (line 261):
        # Getting the type of 'stepsize' (line 261)
        stepsize_187828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'stepsize')
        # Getting the type of 'self' (line 261)
        self_187829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self')
        # Setting the type of the member 'stepsize' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_187829, 'stepsize', stepsize_187828)
        
        # Assigning a Call to a Attribute (line 262):
        
        # Assigning a Call to a Attribute (line 262):
        
        # Call to check_random_state(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'random_state' (line 262)
        random_state_187831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 47), 'random_state', False)
        # Processing the call keyword arguments (line 262)
        kwargs_187832 = {}
        # Getting the type of 'check_random_state' (line 262)
        check_random_state_187830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'check_random_state', False)
        # Calling check_random_state(args, kwargs) (line 262)
        check_random_state_call_result_187833 = invoke(stypy.reporting.localization.Localization(__file__, 262, 28), check_random_state_187830, *[random_state_187831], **kwargs_187832)
        
        # Getting the type of 'self' (line 262)
        self_187834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self')
        # Setting the type of the member 'random_state' of a type (line 262)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_187834, 'random_state', check_random_state_call_result_187833)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_localization', localization)
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_function_name', 'RandomDisplacement.__call__')
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RandomDisplacement.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RandomDisplacement.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Getting the type of 'x' (line 265)
        x_187835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'x')
        
        # Call to uniform(...): (line 265)
        # Processing the call arguments (line 265)
        
        # Getting the type of 'self' (line 265)
        self_187839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 40), 'self', False)
        # Obtaining the member 'stepsize' of a type (line 265)
        stepsize_187840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 40), self_187839, 'stepsize')
        # Applying the 'usub' unary operator (line 265)
        result___neg___187841 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 39), 'usub', stepsize_187840)
        
        # Getting the type of 'self' (line 265)
        self_187842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 55), 'self', False)
        # Obtaining the member 'stepsize' of a type (line 265)
        stepsize_187843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 55), self_187842, 'stepsize')
        
        # Call to shape(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'x' (line 265)
        x_187846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 79), 'x', False)
        # Processing the call keyword arguments (line 265)
        kwargs_187847 = {}
        # Getting the type of 'np' (line 265)
        np_187844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 70), 'np', False)
        # Obtaining the member 'shape' of a type (line 265)
        shape_187845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 70), np_187844, 'shape')
        # Calling shape(args, kwargs) (line 265)
        shape_call_result_187848 = invoke(stypy.reporting.localization.Localization(__file__, 265, 70), shape_187845, *[x_187846], **kwargs_187847)
        
        # Processing the call keyword arguments (line 265)
        kwargs_187849 = {}
        # Getting the type of 'self' (line 265)
        self_187836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'self', False)
        # Obtaining the member 'random_state' of a type (line 265)
        random_state_187837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 13), self_187836, 'random_state')
        # Obtaining the member 'uniform' of a type (line 265)
        uniform_187838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 13), random_state_187837, 'uniform')
        # Calling uniform(args, kwargs) (line 265)
        uniform_call_result_187850 = invoke(stypy.reporting.localization.Localization(__file__, 265, 13), uniform_187838, *[result___neg___187841, stepsize_187843, shape_call_result_187848], **kwargs_187849)
        
        # Applying the binary operator '+=' (line 265)
        result_iadd_187851 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 8), '+=', x_187835, uniform_call_result_187850)
        # Assigning a type to the variable 'x' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'x', result_iadd_187851)
        
        # Getting the type of 'x' (line 266)
        x_187852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'stypy_return_type', x_187852)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_187853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187853)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_187853


# Assigning a type to the variable 'RandomDisplacement' (line 247)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'RandomDisplacement', RandomDisplacement)
# Declaration of the 'MinimizerWrapper' class

class MinimizerWrapper(object, ):
    str_187854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, (-1)), 'str', '\n    wrap a minimizer function as a minimizer class\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 273)
        None_187855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 39), 'None')
        defaults = [None_187855]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MinimizerWrapper.__init__', ['minimizer', 'func'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['minimizer', 'func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 274):
        
        # Assigning a Name to a Attribute (line 274):
        # Getting the type of 'minimizer' (line 274)
        minimizer_187856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 25), 'minimizer')
        # Getting the type of 'self' (line 274)
        self_187857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self')
        # Setting the type of the member 'minimizer' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_187857, 'minimizer', minimizer_187856)
        
        # Assigning a Name to a Attribute (line 275):
        
        # Assigning a Name to a Attribute (line 275):
        # Getting the type of 'func' (line 275)
        func_187858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 20), 'func')
        # Getting the type of 'self' (line 275)
        self_187859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self')
        # Setting the type of the member 'func' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_187859, 'func', func_187858)
        
        # Assigning a Name to a Attribute (line 276):
        
        # Assigning a Name to a Attribute (line 276):
        # Getting the type of 'kwargs' (line 276)
        kwargs_187860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'kwargs')
        # Getting the type of 'self' (line 276)
        self_187861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'self')
        # Setting the type of the member 'kwargs' of a type (line 276)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), self_187861, 'kwargs', kwargs_187860)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_localization', localization)
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_function_name', 'MinimizerWrapper.__call__')
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_param_names_list', ['x0'])
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MinimizerWrapper.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MinimizerWrapper.__call__', ['x0'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x0'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 279)
        # Getting the type of 'self' (line 279)
        self_187862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'self')
        # Obtaining the member 'func' of a type (line 279)
        func_187863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 11), self_187862, 'func')
        # Getting the type of 'None' (line 279)
        None_187864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'None')
        
        (may_be_187865, more_types_in_union_187866) = may_be_none(func_187863, None_187864)

        if may_be_187865:

            if more_types_in_union_187866:
                # Runtime conditional SSA (line 279)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to minimizer(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'x0' (line 280)
            x0_187869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 34), 'x0', False)
            # Processing the call keyword arguments (line 280)
            # Getting the type of 'self' (line 280)
            self_187870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 40), 'self', False)
            # Obtaining the member 'kwargs' of a type (line 280)
            kwargs_187871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 40), self_187870, 'kwargs')
            kwargs_187872 = {'kwargs_187871': kwargs_187871}
            # Getting the type of 'self' (line 280)
            self_187867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'self', False)
            # Obtaining the member 'minimizer' of a type (line 280)
            minimizer_187868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), self_187867, 'minimizer')
            # Calling minimizer(args, kwargs) (line 280)
            minimizer_call_result_187873 = invoke(stypy.reporting.localization.Localization(__file__, 280, 19), minimizer_187868, *[x0_187869], **kwargs_187872)
            
            # Assigning a type to the variable 'stypy_return_type' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'stypy_return_type', minimizer_call_result_187873)

            if more_types_in_union_187866:
                # Runtime conditional SSA for else branch (line 279)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_187865) or more_types_in_union_187866):
            
            # Call to minimizer(...): (line 282)
            # Processing the call arguments (line 282)
            # Getting the type of 'self' (line 282)
            self_187876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 34), 'self', False)
            # Obtaining the member 'func' of a type (line 282)
            func_187877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 34), self_187876, 'func')
            # Getting the type of 'x0' (line 282)
            x0_187878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 45), 'x0', False)
            # Processing the call keyword arguments (line 282)
            # Getting the type of 'self' (line 282)
            self_187879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 51), 'self', False)
            # Obtaining the member 'kwargs' of a type (line 282)
            kwargs_187880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 51), self_187879, 'kwargs')
            kwargs_187881 = {'kwargs_187880': kwargs_187880}
            # Getting the type of 'self' (line 282)
            self_187874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'self', False)
            # Obtaining the member 'minimizer' of a type (line 282)
            minimizer_187875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 19), self_187874, 'minimizer')
            # Calling minimizer(args, kwargs) (line 282)
            minimizer_call_result_187882 = invoke(stypy.reporting.localization.Localization(__file__, 282, 19), minimizer_187875, *[func_187877, x0_187878], **kwargs_187881)
            
            # Assigning a type to the variable 'stypy_return_type' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', minimizer_call_result_187882)

            if (may_be_187865 and more_types_in_union_187866):
                # SSA join for if statement (line 279)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_187883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187883)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_187883


# Assigning a type to the variable 'MinimizerWrapper' (line 269)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 0), 'MinimizerWrapper', MinimizerWrapper)
# Declaration of the 'Metropolis' class

class Metropolis(object, ):
    str_187884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, (-1)), 'str', '\n    Metropolis acceptance criterion\n\n    Parameters\n    ----------\n    random_state : None or `np.random.RandomState` object\n        Random number generator used for acceptance test\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 294)
        None_187885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'None')
        defaults = [None_187885]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 294, 4, False)
        # Assigning a type to the variable 'self' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Metropolis.__init__', ['T', 'random_state'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['T', 'random_state'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 295):
        
        # Assigning a BinOp to a Attribute (line 295):
        float_187886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 20), 'float')
        # Getting the type of 'T' (line 295)
        T_187887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 'T')
        # Applying the binary operator 'div' (line 295)
        result_div_187888 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 20), 'div', float_187886, T_187887)
        
        # Getting the type of 'self' (line 295)
        self_187889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self')
        # Setting the type of the member 'beta' of a type (line 295)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_187889, 'beta', result_div_187888)
        
        # Assigning a Call to a Attribute (line 296):
        
        # Assigning a Call to a Attribute (line 296):
        
        # Call to check_random_state(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'random_state' (line 296)
        random_state_187891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 47), 'random_state', False)
        # Processing the call keyword arguments (line 296)
        kwargs_187892 = {}
        # Getting the type of 'check_random_state' (line 296)
        check_random_state_187890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 28), 'check_random_state', False)
        # Calling check_random_state(args, kwargs) (line 296)
        check_random_state_call_result_187893 = invoke(stypy.reporting.localization.Localization(__file__, 296, 28), check_random_state_187890, *[random_state_187891], **kwargs_187892)
        
        # Getting the type of 'self' (line 296)
        self_187894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self')
        # Setting the type of the member 'random_state' of a type (line 296)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_187894, 'random_state', check_random_state_call_result_187893)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def accept_reject(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'accept_reject'
        module_type_store = module_type_store.open_function_context('accept_reject', 298, 4, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Metropolis.accept_reject.__dict__.__setitem__('stypy_localization', localization)
        Metropolis.accept_reject.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Metropolis.accept_reject.__dict__.__setitem__('stypy_type_store', module_type_store)
        Metropolis.accept_reject.__dict__.__setitem__('stypy_function_name', 'Metropolis.accept_reject')
        Metropolis.accept_reject.__dict__.__setitem__('stypy_param_names_list', ['energy_new', 'energy_old'])
        Metropolis.accept_reject.__dict__.__setitem__('stypy_varargs_param_name', None)
        Metropolis.accept_reject.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Metropolis.accept_reject.__dict__.__setitem__('stypy_call_defaults', defaults)
        Metropolis.accept_reject.__dict__.__setitem__('stypy_call_varargs', varargs)
        Metropolis.accept_reject.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Metropolis.accept_reject.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Metropolis.accept_reject', ['energy_new', 'energy_old'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'accept_reject', localization, ['energy_new', 'energy_old'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'accept_reject(...)' code ##################

        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to exp(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Call to min(...): (line 299)
        # Processing the call arguments (line 299)
        int_187898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 23), 'int')
        
        # Getting the type of 'energy_new' (line 299)
        energy_new_187899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 'energy_new', False)
        # Getting the type of 'energy_old' (line 299)
        energy_old_187900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 41), 'energy_old', False)
        # Applying the binary operator '-' (line 299)
        result_sub_187901 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 28), '-', energy_new_187899, energy_old_187900)
        
        # Applying the 'usub' unary operator (line 299)
        result___neg___187902 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 26), 'usub', result_sub_187901)
        
        # Getting the type of 'self' (line 299)
        self_187903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 55), 'self', False)
        # Obtaining the member 'beta' of a type (line 299)
        beta_187904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 55), self_187903, 'beta')
        # Applying the binary operator '*' (line 299)
        result_mul_187905 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 26), '*', result___neg___187902, beta_187904)
        
        # Processing the call keyword arguments (line 299)
        kwargs_187906 = {}
        # Getting the type of 'min' (line 299)
        min_187897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 19), 'min', False)
        # Calling min(args, kwargs) (line 299)
        min_call_result_187907 = invoke(stypy.reporting.localization.Localization(__file__, 299, 19), min_187897, *[int_187898, result_mul_187905], **kwargs_187906)
        
        # Processing the call keyword arguments (line 299)
        kwargs_187908 = {}
        # Getting the type of 'np' (line 299)
        np_187895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'np', False)
        # Obtaining the member 'exp' of a type (line 299)
        exp_187896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), np_187895, 'exp')
        # Calling exp(args, kwargs) (line 299)
        exp_call_result_187909 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), exp_187896, *[min_call_result_187907], **kwargs_187908)
        
        # Assigning a type to the variable 'w' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'w', exp_call_result_187909)
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to rand(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_187913 = {}
        # Getting the type of 'self' (line 300)
        self_187910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'self', False)
        # Obtaining the member 'random_state' of a type (line 300)
        random_state_187911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 15), self_187910, 'random_state')
        # Obtaining the member 'rand' of a type (line 300)
        rand_187912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 15), random_state_187911, 'rand')
        # Calling rand(args, kwargs) (line 300)
        rand_call_result_187914 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), rand_187912, *[], **kwargs_187913)
        
        # Assigning a type to the variable 'rand' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'rand', rand_call_result_187914)
        
        # Getting the type of 'w' (line 301)
        w_187915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'w')
        # Getting the type of 'rand' (line 301)
        rand_187916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 20), 'rand')
        # Applying the binary operator '>=' (line 301)
        result_ge_187917 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 15), '>=', w_187915, rand_187916)
        
        # Assigning a type to the variable 'stypy_return_type' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'stypy_return_type', result_ge_187917)
        
        # ################# End of 'accept_reject(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'accept_reject' in the type store
        # Getting the type of 'stypy_return_type' (line 298)
        stypy_return_type_187918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'accept_reject'
        return stypy_return_type_187918


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Metropolis.__call__.__dict__.__setitem__('stypy_localization', localization)
        Metropolis.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Metropolis.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Metropolis.__call__.__dict__.__setitem__('stypy_function_name', 'Metropolis.__call__')
        Metropolis.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        Metropolis.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Metropolis.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Metropolis.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Metropolis.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Metropolis.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Metropolis.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Metropolis.__call__', [], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_187919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, (-1)), 'str', '\n        f_new and f_old are mandatory in kwargs\n        ')
        
        # Call to bool(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Call to accept_reject(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Obtaining the type of the subscript
        str_187923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 46), 'str', 'f_new')
        # Getting the type of 'kwargs' (line 307)
        kwargs_187924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 39), 'kwargs', False)
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___187925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 39), kwargs_187924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_187926 = invoke(stypy.reporting.localization.Localization(__file__, 307, 39), getitem___187925, str_187923)
        
        
        # Obtaining the type of the subscript
        str_187927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 27), 'str', 'f_old')
        # Getting the type of 'kwargs' (line 308)
        kwargs_187928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'kwargs', False)
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___187929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 20), kwargs_187928, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_187930 = invoke(stypy.reporting.localization.Localization(__file__, 308, 20), getitem___187929, str_187927)
        
        # Processing the call keyword arguments (line 307)
        kwargs_187931 = {}
        # Getting the type of 'self' (line 307)
        self_187921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'self', False)
        # Obtaining the member 'accept_reject' of a type (line 307)
        accept_reject_187922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 20), self_187921, 'accept_reject')
        # Calling accept_reject(args, kwargs) (line 307)
        accept_reject_call_result_187932 = invoke(stypy.reporting.localization.Localization(__file__, 307, 20), accept_reject_187922, *[subscript_call_result_187926, subscript_call_result_187930], **kwargs_187931)
        
        # Processing the call keyword arguments (line 307)
        kwargs_187933 = {}
        # Getting the type of 'bool' (line 307)
        bool_187920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'bool', False)
        # Calling bool(args, kwargs) (line 307)
        bool_call_result_187934 = invoke(stypy.reporting.localization.Localization(__file__, 307, 15), bool_187920, *[accept_reject_call_result_187932], **kwargs_187933)
        
        # Assigning a type to the variable 'stypy_return_type' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'stypy_return_type', bool_call_result_187934)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_187935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187935)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_187935


# Assigning a type to the variable 'Metropolis' (line 285)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'Metropolis', Metropolis)

@norecursion
def basinhopping(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_187936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 33), 'int')
    float_187937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 40), 'float')
    float_187938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 54), 'float')
    # Getting the type of 'None' (line 312)
    None_187939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'None')
    # Getting the type of 'None' (line 312)
    None_187940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 50), 'None')
    # Getting the type of 'None' (line 312)
    None_187941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 68), 'None')
    # Getting the type of 'None' (line 313)
    None_187942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'None')
    int_187943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 41), 'int')
    # Getting the type of 'False' (line 313)
    False_187944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 50), 'False')
    # Getting the type of 'None' (line 313)
    None_187945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 71), 'None')
    # Getting the type of 'None' (line 314)
    None_187946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 22), 'None')
    defaults = [int_187936, float_187937, float_187938, None_187939, None_187940, None_187941, None_187942, int_187943, False_187944, None_187945, None_187946]
    # Create a new context for function 'basinhopping'
    module_type_store = module_type_store.open_function_context('basinhopping', 311, 0, False)
    
    # Passed parameters checking function
    basinhopping.stypy_localization = localization
    basinhopping.stypy_type_of_self = None
    basinhopping.stypy_type_store = module_type_store
    basinhopping.stypy_function_name = 'basinhopping'
    basinhopping.stypy_param_names_list = ['func', 'x0', 'niter', 'T', 'stepsize', 'minimizer_kwargs', 'take_step', 'accept_test', 'callback', 'interval', 'disp', 'niter_success', 'seed']
    basinhopping.stypy_varargs_param_name = None
    basinhopping.stypy_kwargs_param_name = None
    basinhopping.stypy_call_defaults = defaults
    basinhopping.stypy_call_varargs = varargs
    basinhopping.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'basinhopping', ['func', 'x0', 'niter', 'T', 'stepsize', 'minimizer_kwargs', 'take_step', 'accept_test', 'callback', 'interval', 'disp', 'niter_success', 'seed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'basinhopping', localization, ['func', 'x0', 'niter', 'T', 'stepsize', 'minimizer_kwargs', 'take_step', 'accept_test', 'callback', 'interval', 'disp', 'niter_success', 'seed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'basinhopping(...)' code ##################

    str_187947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, (-1)), 'str', '\n    Find the global minimum of a function using the basin-hopping algorithm\n\n    Parameters\n    ----------\n    func : callable ``f(x, *args)``\n        Function to be optimized.  ``args`` can be passed as an optional item\n        in the dict ``minimizer_kwargs``\n    x0 : ndarray\n        Initial guess.\n    niter : integer, optional\n        The number of basin hopping iterations\n    T : float, optional\n        The "temperature" parameter for the accept or reject criterion.  Higher\n        "temperatures" mean that larger jumps in function value will be\n        accepted.  For best results ``T`` should be comparable to the\n        separation\n        (in function value) between local minima.\n    stepsize : float, optional\n        initial step size for use in the random displacement.\n    minimizer_kwargs : dict, optional\n        Extra keyword arguments to be passed to the minimizer\n        ``scipy.optimize.minimize()`` Some important options could be:\n\n            method : str\n                The minimization method (e.g. ``"L-BFGS-B"``)\n            args : tuple\n                Extra arguments passed to the objective function (``func``) and\n                its derivatives (Jacobian, Hessian).\n\n    take_step : callable ``take_step(x)``, optional\n        Replace the default step taking routine with this routine.  The default\n        step taking routine is a random displacement of the coordinates, but\n        other step taking algorithms may be better for some systems.\n        ``take_step`` can optionally have the attribute ``take_step.stepsize``.\n        If this attribute exists, then ``basinhopping`` will adjust\n        ``take_step.stepsize`` in order to try to optimize the global minimum\n        search.\n    accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional\n        Define a test which will be used to judge whether or not to accept the\n        step.  This will be used in addition to the Metropolis test based on\n        "temperature" ``T``.  The acceptable return values are True,\n        False, or ``"force accept"``. If any of the tests return False\n        then the step is rejected. If the latter, then this will override any\n        other tests in order to accept the step. This can be used, for example,\n        to forcefully escape from a local minimum that ``basinhopping`` is\n        trapped in.\n    callback : callable, ``callback(x, f, accept)``, optional\n        A callback function which will be called for all minima found.  ``x``\n        and ``f`` are the coordinates and function value of the trial minimum,\n        and ``accept`` is whether or not that minimum was accepted.  This can be\n        used, for example, to save the lowest N minima found.  Also,\n        ``callback`` can be used to specify a user defined stop criterion by\n        optionally returning True to stop the ``basinhopping`` routine.\n    interval : integer, optional\n        interval for how often to update the ``stepsize``\n    disp : bool, optional\n        Set to True to print status messages\n    niter_success : integer, optional\n        Stop the run if the global minimum candidate remains the same for this\n        number of iterations.\n    seed : int or `np.random.RandomState`, optional\n        If `seed` is not specified the `np.RandomState` singleton is used.\n        If `seed` is an int, a new `np.random.RandomState` instance is used,\n        seeded with seed.\n        If `seed` is already a `np.random.RandomState instance`, then that\n        `np.random.RandomState` instance is used.\n        Specify `seed` for repeatable minimizations. The random numbers\n        generated with this seed only affect the default Metropolis\n        `accept_test` and the default `take_step`. If you supply your own\n        `take_step` and `accept_test`, and these functions use random\n        number generation, then those functions are responsible for the state\n        of their random number generator.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a ``OptimizeResult`` object.  Important\n        attributes are: ``x`` the solution array, ``fun`` the value of the\n        function at the solution, and ``message`` which describes the cause of\n        the termination. The ``OptimzeResult`` object returned by the selected\n        minimizer at the lowest minimum is also contained within this object\n        and can be accessed through the ``lowest_optimization_result`` attribute.\n        See `OptimizeResult` for a description of other attributes.\n\n    See Also\n    --------\n    minimize :\n        The local minimization function called once for each basinhopping step.\n        ``minimizer_kwargs`` is passed to this routine.\n\n    Notes\n    -----\n    Basin-hopping is a stochastic algorithm which attempts to find the global\n    minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_\n    [4]_.  The algorithm in its current form was described by David Wales and\n    Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.\n\n    The algorithm is iterative with each cycle composed of the following\n    features\n\n    1) random perturbation of the coordinates\n\n    2) local minimization\n\n    3) accept or reject the new coordinates based on the minimized function\n       value\n\n    The acceptance test used here is the Metropolis criterion of standard Monte\n    Carlo algorithms, although there are many other possibilities [3]_.\n\n    This global minimization method has been shown to be extremely efficient\n    for a wide variety of problems in physics and chemistry.  It is\n    particularly useful when the function has many minima separated by large\n    barriers. See the Cambridge Cluster Database\n    http://www-wales.ch.cam.ac.uk/CCD.html for databases of molecular systems\n    that have been optimized primarily using basin-hopping.  This database\n    includes minimization problems exceeding 300 degrees of freedom.\n\n    See the free software program GMIN (http://www-wales.ch.cam.ac.uk/GMIN) for\n    a Fortran implementation of basin-hopping.  This implementation has many\n    different variations of the procedure described above, including more\n    advanced step taking algorithms and alternate acceptance criterion.\n\n    For stochastic global optimization there is no way to determine if the true\n    global minimum has actually been found. Instead, as a consistency check,\n    the algorithm can be run from a number of different random starting points\n    to ensure the lowest minimum found in each example has converged to the\n    global minimum.  For this reason ``basinhopping`` will by default simply\n    run for the number of iterations ``niter`` and return the lowest minimum\n    found.  It is left to the user to ensure that this is in fact the global\n    minimum.\n\n    Choosing ``stepsize``:  This is a crucial parameter in ``basinhopping`` and\n    depends on the problem being solved.  Ideally it should be comparable to\n    the typical separation between local minima of the function being\n    optimized.  ``basinhopping`` will, by default, adjust ``stepsize`` to find\n    an optimal value, but this may take many iterations.  You will get quicker\n    results if you set a sensible value for ``stepsize``.\n\n    Choosing ``T``: The parameter ``T`` is the temperature used in the\n    metropolis criterion.  Basinhopping steps are accepted with probability\n    ``1`` if ``func(xnew) < func(xold)``, or otherwise with probability::\n\n        exp( -(func(xnew) - func(xold)) / T )\n\n    So, for best results, ``T`` should to be comparable to the typical\n    difference in function values between local minima.\n\n    .. versionadded:: 0.12.0\n\n    References\n    ----------\n    .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,\n        Cambridge, UK.\n    .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and\n        the Lowest Energy Structures of Lennard-Jones Clusters Containing up to\n        110 Atoms.  Journal of Physical Chemistry A, 1997, 101, 5111.\n    .. [3] Li, Z. and Scheraga, H. A., Monte Carlo-minimization approach to the\n        multiple-minima problem in protein folding, Proc. Natl. Acad. Sci. USA,\n        1987, 84, 6611.\n    .. [4] Wales, D. J. and Scheraga, H. A., Global optimization of clusters,\n        crystals, and biomolecules, Science, 1999, 285, 1368.\n\n    Examples\n    --------\n    The following example is a one-dimensional minimization problem,  with many\n    local minima superimposed on a parabola.\n\n    >>> from scipy.optimize import basinhopping\n    >>> func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x\n    >>> x0=[1.]\n\n    Basinhopping, internally, uses a local minimization algorithm.  We will use\n    the parameter ``minimizer_kwargs`` to tell basinhopping which algorithm to\n    use and how to set up that minimizer.  This parameter will be passed to\n    ``scipy.optimize.minimize()``.\n\n    >>> minimizer_kwargs = {"method": "BFGS"}\n    >>> ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=200)\n    >>> print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))\n    global minimum: x = -0.1951, f(x0) = -1.0009\n\n    Next consider a two-dimensional minimization problem. Also, this time we\n    will use gradient information to significantly speed up the search.\n\n    >>> def func2d(x):\n    ...     f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +\n    ...                                                            0.2) * x[0]\n    ...     df = np.zeros(2)\n    ...     df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2\n    ...     df[1] = 2. * x[1] + 0.2\n    ...     return f, df\n\n    We\'ll also use a different local minimization algorithm.  Also we must tell\n    the minimizer that our function returns both energy and gradient (jacobian)\n\n    >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}\n    >>> x0 = [1.0, 1.0]\n    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=200)\n    >>> print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],\n    ...                                                           ret.x[1],\n    ...                                                           ret.fun))\n    global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109\n\n\n    Here is an example using a custom step taking routine.  Imagine you want\n    the first coordinate to take larger steps then the rest of the coordinates.\n    This can be implemented like so:\n\n    >>> class MyTakeStep(object):\n    ...    def __init__(self, stepsize=0.5):\n    ...        self.stepsize = stepsize\n    ...    def __call__(self, x):\n    ...        s = self.stepsize\n    ...        x[0] += np.random.uniform(-2.*s, 2.*s)\n    ...        x[1:] += np.random.uniform(-s, s, x[1:].shape)\n    ...        return x\n\n    Since ``MyTakeStep.stepsize`` exists basinhopping will adjust the magnitude\n    of ``stepsize`` to optimize the search.  We\'ll use the same 2-D function as\n    before\n\n    >>> mytakestep = MyTakeStep()\n    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=200, take_step=mytakestep)\n    >>> print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],\n    ...                                                           ret.x[1],\n    ...                                                           ret.fun))\n    global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109\n\n\n    Now let\'s do an example using a custom callback function which prints the\n    value of every minimum found\n\n    >>> def print_fun(x, f, accepted):\n    ...         print("at minimum %.4f accepted %d" % (f, int(accepted)))\n\n    We\'ll run it for only 10 basinhopping steps this time.\n\n    >>> np.random.seed(1)\n    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=10, callback=print_fun)\n    at minimum 0.4159 accepted 1\n    at minimum -0.9073 accepted 1\n    at minimum -0.1021 accepted 1\n    at minimum -0.1021 accepted 1\n    at minimum 0.9102 accepted 1\n    at minimum 0.9102 accepted 1\n    at minimum 2.2945 accepted 0\n    at minimum -0.1021 accepted 1\n    at minimum -1.0109 accepted 1\n    at minimum -1.0109 accepted 1\n\n\n    The minimum at -1.0109 is actually the global minimum, found already on the\n    8th iteration.\n\n    Now let\'s implement bounds on the problem using a custom ``accept_test``:\n\n    >>> class MyBounds(object):\n    ...     def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):\n    ...         self.xmax = np.array(xmax)\n    ...         self.xmin = np.array(xmin)\n    ...     def __call__(self, **kwargs):\n    ...         x = kwargs["x_new"]\n    ...         tmax = bool(np.all(x <= self.xmax))\n    ...         tmin = bool(np.all(x >= self.xmin))\n    ...         return tmax and tmin\n\n    >>> mybounds = MyBounds()\n    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=10, accept_test=mybounds)\n\n    ')
    
    # Assigning a Call to a Name (line 592):
    
    # Assigning a Call to a Name (line 592):
    
    # Call to array(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'x0' (line 592)
    x0_187950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 18), 'x0', False)
    # Processing the call keyword arguments (line 592)
    kwargs_187951 = {}
    # Getting the type of 'np' (line 592)
    np_187948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 592)
    array_187949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 9), np_187948, 'array')
    # Calling array(args, kwargs) (line 592)
    array_call_result_187952 = invoke(stypy.reporting.localization.Localization(__file__, 592, 9), array_187949, *[x0_187950], **kwargs_187951)
    
    # Assigning a type to the variable 'x0' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'x0', array_call_result_187952)
    
    # Assigning a Call to a Name (line 595):
    
    # Assigning a Call to a Name (line 595):
    
    # Call to check_random_state(...): (line 595)
    # Processing the call arguments (line 595)
    # Getting the type of 'seed' (line 595)
    seed_187954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 29), 'seed', False)
    # Processing the call keyword arguments (line 595)
    kwargs_187955 = {}
    # Getting the type of 'check_random_state' (line 595)
    check_random_state_187953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 10), 'check_random_state', False)
    # Calling check_random_state(args, kwargs) (line 595)
    check_random_state_call_result_187956 = invoke(stypy.reporting.localization.Localization(__file__, 595, 10), check_random_state_187953, *[seed_187954], **kwargs_187955)
    
    # Assigning a type to the variable 'rng' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'rng', check_random_state_call_result_187956)
    
    # Type idiom detected: calculating its left and rigth part (line 598)
    # Getting the type of 'minimizer_kwargs' (line 598)
    minimizer_kwargs_187957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 7), 'minimizer_kwargs')
    # Getting the type of 'None' (line 598)
    None_187958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 27), 'None')
    
    (may_be_187959, more_types_in_union_187960) = may_be_none(minimizer_kwargs_187957, None_187958)

    if may_be_187959:

        if more_types_in_union_187960:
            # Runtime conditional SSA (line 598)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 599):
        
        # Assigning a Call to a Name (line 599):
        
        # Call to dict(...): (line 599)
        # Processing the call keyword arguments (line 599)
        kwargs_187962 = {}
        # Getting the type of 'dict' (line 599)
        dict_187961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 27), 'dict', False)
        # Calling dict(args, kwargs) (line 599)
        dict_call_result_187963 = invoke(stypy.reporting.localization.Localization(__file__, 599, 27), dict_187961, *[], **kwargs_187962)
        
        # Assigning a type to the variable 'minimizer_kwargs' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'minimizer_kwargs', dict_call_result_187963)

        if more_types_in_union_187960:
            # SSA join for if statement (line 598)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 600):
    
    # Assigning a Call to a Name (line 600):
    
    # Call to MinimizerWrapper(...): (line 600)
    # Processing the call arguments (line 600)
    # Getting the type of 'scipy' (line 600)
    scipy_187965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 41), 'scipy', False)
    # Obtaining the member 'optimize' of a type (line 600)
    optimize_187966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 41), scipy_187965, 'optimize')
    # Obtaining the member 'minimize' of a type (line 600)
    minimize_187967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 41), optimize_187966, 'minimize')
    # Getting the type of 'func' (line 600)
    func_187968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 66), 'func', False)
    # Processing the call keyword arguments (line 600)
    # Getting the type of 'minimizer_kwargs' (line 601)
    minimizer_kwargs_187969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 43), 'minimizer_kwargs', False)
    kwargs_187970 = {'minimizer_kwargs_187969': minimizer_kwargs_187969}
    # Getting the type of 'MinimizerWrapper' (line 600)
    MinimizerWrapper_187964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 24), 'MinimizerWrapper', False)
    # Calling MinimizerWrapper(args, kwargs) (line 600)
    MinimizerWrapper_call_result_187971 = invoke(stypy.reporting.localization.Localization(__file__, 600, 24), MinimizerWrapper_187964, *[minimize_187967, func_187968], **kwargs_187970)
    
    # Assigning a type to the variable 'wrapped_minimizer' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'wrapped_minimizer', MinimizerWrapper_call_result_187971)
    
    # Type idiom detected: calculating its left and rigth part (line 604)
    # Getting the type of 'take_step' (line 604)
    take_step_187972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'take_step')
    # Getting the type of 'None' (line 604)
    None_187973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 24), 'None')
    
    (may_be_187974, more_types_in_union_187975) = may_not_be_none(take_step_187972, None_187973)

    if may_be_187974:

        if more_types_in_union_187975:
            # Runtime conditional SSA (line 604)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to isinstance(...): (line 605)
        # Processing the call arguments (line 605)
        # Getting the type of 'take_step' (line 605)
        take_step_187977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 26), 'take_step', False)
        # Getting the type of 'collections' (line 605)
        collections_187978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 37), 'collections', False)
        # Obtaining the member 'Callable' of a type (line 605)
        Callable_187979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 37), collections_187978, 'Callable')
        # Processing the call keyword arguments (line 605)
        kwargs_187980 = {}
        # Getting the type of 'isinstance' (line 605)
        isinstance_187976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 605)
        isinstance_call_result_187981 = invoke(stypy.reporting.localization.Localization(__file__, 605, 15), isinstance_187976, *[take_step_187977, Callable_187979], **kwargs_187980)
        
        # Applying the 'not' unary operator (line 605)
        result_not__187982 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 11), 'not', isinstance_call_result_187981)
        
        # Testing the type of an if condition (line 605)
        if_condition_187983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 8), result_not__187982)
        # Assigning a type to the variable 'if_condition_187983' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'if_condition_187983', if_condition_187983)
        # SSA begins for if statement (line 605)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 606)
        # Processing the call arguments (line 606)
        str_187985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 28), 'str', 'take_step must be callable')
        # Processing the call keyword arguments (line 606)
        kwargs_187986 = {}
        # Getting the type of 'TypeError' (line 606)
        TypeError_187984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 606)
        TypeError_call_result_187987 = invoke(stypy.reporting.localization.Localization(__file__, 606, 18), TypeError_187984, *[str_187985], **kwargs_187986)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 606, 12), TypeError_call_result_187987, 'raise parameter', BaseException)
        # SSA join for if statement (line 605)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 609)
        str_187988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 30), 'str', 'stepsize')
        # Getting the type of 'take_step' (line 609)
        take_step_187989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 19), 'take_step')
        
        (may_be_187990, more_types_in_union_187991) = may_provide_member(str_187988, take_step_187989)

        if may_be_187990:

            if more_types_in_union_187991:
                # Runtime conditional SSA (line 609)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'take_step' (line 609)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'take_step', remove_not_member_provider_from_union(take_step_187989, 'stepsize'))
            
            # Assigning a Call to a Name (line 610):
            
            # Assigning a Call to a Name (line 610):
            
            # Call to AdaptiveStepsize(...): (line 610)
            # Processing the call arguments (line 610)
            # Getting the type of 'take_step' (line 610)
            take_step_187993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 49), 'take_step', False)
            # Processing the call keyword arguments (line 610)
            # Getting the type of 'interval' (line 610)
            interval_187994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 69), 'interval', False)
            keyword_187995 = interval_187994
            # Getting the type of 'disp' (line 611)
            disp_187996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 57), 'disp', False)
            keyword_187997 = disp_187996
            kwargs_187998 = {'interval': keyword_187995, 'verbose': keyword_187997}
            # Getting the type of 'AdaptiveStepsize' (line 610)
            AdaptiveStepsize_187992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 32), 'AdaptiveStepsize', False)
            # Calling AdaptiveStepsize(args, kwargs) (line 610)
            AdaptiveStepsize_call_result_187999 = invoke(stypy.reporting.localization.Localization(__file__, 610, 32), AdaptiveStepsize_187992, *[take_step_187993], **kwargs_187998)
            
            # Assigning a type to the variable 'take_step_wrapped' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'take_step_wrapped', AdaptiveStepsize_call_result_187999)

            if more_types_in_union_187991:
                # Runtime conditional SSA for else branch (line 609)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_187990) or more_types_in_union_187991):
            # Assigning a type to the variable 'take_step' (line 609)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'take_step', remove_member_provider_from_union(take_step_187989, 'stepsize'))
            
            # Assigning a Name to a Name (line 613):
            
            # Assigning a Name to a Name (line 613):
            # Getting the type of 'take_step' (line 613)
            take_step_188000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 32), 'take_step')
            # Assigning a type to the variable 'take_step_wrapped' (line 613)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'take_step_wrapped', take_step_188000)

            if (may_be_187990 and more_types_in_union_187991):
                # SSA join for if statement (line 609)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_187975:
            # Runtime conditional SSA for else branch (line 604)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_187974) or more_types_in_union_187975):
        
        # Assigning a Call to a Name (line 616):
        
        # Assigning a Call to a Name (line 616):
        
        # Call to RandomDisplacement(...): (line 616)
        # Processing the call keyword arguments (line 616)
        # Getting the type of 'stepsize' (line 616)
        stepsize_188002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 47), 'stepsize', False)
        keyword_188003 = stepsize_188002
        # Getting the type of 'rng' (line 616)
        rng_188004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 70), 'rng', False)
        keyword_188005 = rng_188004
        kwargs_188006 = {'stepsize': keyword_188003, 'random_state': keyword_188005}
        # Getting the type of 'RandomDisplacement' (line 616)
        RandomDisplacement_188001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 19), 'RandomDisplacement', False)
        # Calling RandomDisplacement(args, kwargs) (line 616)
        RandomDisplacement_call_result_188007 = invoke(stypy.reporting.localization.Localization(__file__, 616, 19), RandomDisplacement_188001, *[], **kwargs_188006)
        
        # Assigning a type to the variable 'displace' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'displace', RandomDisplacement_call_result_188007)
        
        # Assigning a Call to a Name (line 617):
        
        # Assigning a Call to a Name (line 617):
        
        # Call to AdaptiveStepsize(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'displace' (line 617)
        displace_188009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 45), 'displace', False)
        # Processing the call keyword arguments (line 617)
        # Getting the type of 'interval' (line 617)
        interval_188010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 64), 'interval', False)
        keyword_188011 = interval_188010
        # Getting the type of 'disp' (line 618)
        disp_188012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 53), 'disp', False)
        keyword_188013 = disp_188012
        kwargs_188014 = {'interval': keyword_188011, 'verbose': keyword_188013}
        # Getting the type of 'AdaptiveStepsize' (line 617)
        AdaptiveStepsize_188008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 28), 'AdaptiveStepsize', False)
        # Calling AdaptiveStepsize(args, kwargs) (line 617)
        AdaptiveStepsize_call_result_188015 = invoke(stypy.reporting.localization.Localization(__file__, 617, 28), AdaptiveStepsize_188008, *[displace_188009], **kwargs_188014)
        
        # Assigning a type to the variable 'take_step_wrapped' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'take_step_wrapped', AdaptiveStepsize_call_result_188015)

        if (may_be_187974 and more_types_in_union_187975):
            # SSA join for if statement (line 604)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 621)
    # Getting the type of 'accept_test' (line 621)
    accept_test_188016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'accept_test')
    # Getting the type of 'None' (line 621)
    None_188017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 26), 'None')
    
    (may_be_188018, more_types_in_union_188019) = may_not_be_none(accept_test_188016, None_188017)

    if may_be_188018:

        if more_types_in_union_188019:
            # Runtime conditional SSA (line 621)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to isinstance(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'accept_test' (line 622)
        accept_test_188021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 26), 'accept_test', False)
        # Getting the type of 'collections' (line 622)
        collections_188022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 39), 'collections', False)
        # Obtaining the member 'Callable' of a type (line 622)
        Callable_188023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 39), collections_188022, 'Callable')
        # Processing the call keyword arguments (line 622)
        kwargs_188024 = {}
        # Getting the type of 'isinstance' (line 622)
        isinstance_188020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 622)
        isinstance_call_result_188025 = invoke(stypy.reporting.localization.Localization(__file__, 622, 15), isinstance_188020, *[accept_test_188021, Callable_188023], **kwargs_188024)
        
        # Applying the 'not' unary operator (line 622)
        result_not__188026 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 11), 'not', isinstance_call_result_188025)
        
        # Testing the type of an if condition (line 622)
        if_condition_188027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 8), result_not__188026)
        # Assigning a type to the variable 'if_condition_188027' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'if_condition_188027', if_condition_188027)
        # SSA begins for if statement (line 622)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 623)
        # Processing the call arguments (line 623)
        str_188029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 28), 'str', 'accept_test must be callable')
        # Processing the call keyword arguments (line 623)
        kwargs_188030 = {}
        # Getting the type of 'TypeError' (line 623)
        TypeError_188028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 623)
        TypeError_call_result_188031 = invoke(stypy.reporting.localization.Localization(__file__, 623, 18), TypeError_188028, *[str_188029], **kwargs_188030)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 623, 12), TypeError_call_result_188031, 'raise parameter', BaseException)
        # SSA join for if statement (line 622)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 624):
        
        # Assigning a List to a Name (line 624):
        
        # Obtaining an instance of the builtin type 'list' (line 624)
        list_188032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 624)
        # Adding element type (line 624)
        # Getting the type of 'accept_test' (line 624)
        accept_test_188033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 24), 'accept_test')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 23), list_188032, accept_test_188033)
        
        # Assigning a type to the variable 'accept_tests' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'accept_tests', list_188032)

        if more_types_in_union_188019:
            # Runtime conditional SSA for else branch (line 621)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_188018) or more_types_in_union_188019):
        
        # Assigning a List to a Name (line 626):
        
        # Assigning a List to a Name (line 626):
        
        # Obtaining an instance of the builtin type 'list' (line 626)
        list_188034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 626)
        
        # Assigning a type to the variable 'accept_tests' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'accept_tests', list_188034)

        if (may_be_188018 and more_types_in_union_188019):
            # SSA join for if statement (line 621)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 628):
    
    # Assigning a Call to a Name (line 628):
    
    # Call to Metropolis(...): (line 628)
    # Processing the call arguments (line 628)
    # Getting the type of 'T' (line 628)
    T_188036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 28), 'T', False)
    # Processing the call keyword arguments (line 628)
    # Getting the type of 'rng' (line 628)
    rng_188037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 44), 'rng', False)
    keyword_188038 = rng_188037
    kwargs_188039 = {'random_state': keyword_188038}
    # Getting the type of 'Metropolis' (line 628)
    Metropolis_188035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 17), 'Metropolis', False)
    # Calling Metropolis(args, kwargs) (line 628)
    Metropolis_call_result_188040 = invoke(stypy.reporting.localization.Localization(__file__, 628, 17), Metropolis_188035, *[T_188036], **kwargs_188039)
    
    # Assigning a type to the variable 'metropolis' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'metropolis', Metropolis_call_result_188040)
    
    # Call to append(...): (line 629)
    # Processing the call arguments (line 629)
    # Getting the type of 'metropolis' (line 629)
    metropolis_188043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 24), 'metropolis', False)
    # Processing the call keyword arguments (line 629)
    kwargs_188044 = {}
    # Getting the type of 'accept_tests' (line 629)
    accept_tests_188041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'accept_tests', False)
    # Obtaining the member 'append' of a type (line 629)
    append_188042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 4), accept_tests_188041, 'append')
    # Calling append(args, kwargs) (line 629)
    append_call_result_188045 = invoke(stypy.reporting.localization.Localization(__file__, 629, 4), append_188042, *[metropolis_188043], **kwargs_188044)
    
    
    # Type idiom detected: calculating its left and rigth part (line 631)
    # Getting the type of 'niter_success' (line 631)
    niter_success_188046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 7), 'niter_success')
    # Getting the type of 'None' (line 631)
    None_188047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 24), 'None')
    
    (may_be_188048, more_types_in_union_188049) = may_be_none(niter_success_188046, None_188047)

    if may_be_188048:

        if more_types_in_union_188049:
            # Runtime conditional SSA (line 631)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 632):
        
        # Assigning a BinOp to a Name (line 632):
        # Getting the type of 'niter' (line 632)
        niter_188050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 24), 'niter')
        int_188051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 32), 'int')
        # Applying the binary operator '+' (line 632)
        result_add_188052 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 24), '+', niter_188050, int_188051)
        
        # Assigning a type to the variable 'niter_success' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'niter_success', result_add_188052)

        if more_types_in_union_188049:
            # SSA join for if statement (line 631)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 634):
    
    # Assigning a Call to a Name (line 634):
    
    # Call to BasinHoppingRunner(...): (line 634)
    # Processing the call arguments (line 634)
    # Getting the type of 'x0' (line 634)
    x0_188054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 28), 'x0', False)
    # Getting the type of 'wrapped_minimizer' (line 634)
    wrapped_minimizer_188055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 32), 'wrapped_minimizer', False)
    # Getting the type of 'take_step_wrapped' (line 634)
    take_step_wrapped_188056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 51), 'take_step_wrapped', False)
    # Getting the type of 'accept_tests' (line 635)
    accept_tests_188057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 28), 'accept_tests', False)
    # Processing the call keyword arguments (line 634)
    # Getting the type of 'disp' (line 635)
    disp_188058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 47), 'disp', False)
    keyword_188059 = disp_188058
    kwargs_188060 = {'disp': keyword_188059}
    # Getting the type of 'BasinHoppingRunner' (line 634)
    BasinHoppingRunner_188053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 9), 'BasinHoppingRunner', False)
    # Calling BasinHoppingRunner(args, kwargs) (line 634)
    BasinHoppingRunner_call_result_188061 = invoke(stypy.reporting.localization.Localization(__file__, 634, 9), BasinHoppingRunner_188053, *[x0_188054, wrapped_minimizer_188055, take_step_wrapped_188056, accept_tests_188057], **kwargs_188060)
    
    # Assigning a type to the variable 'bh' (line 634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'bh', BasinHoppingRunner_call_result_188061)
    
    # Assigning a Tuple to a Tuple (line 638):
    
    # Assigning a Num to a Name (line 638):
    int_188062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 15), 'int')
    # Assigning a type to the variable 'tuple_assignment_187340' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'tuple_assignment_187340', int_188062)
    
    # Assigning a Num to a Name (line 638):
    int_188063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 18), 'int')
    # Assigning a type to the variable 'tuple_assignment_187341' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'tuple_assignment_187341', int_188063)
    
    # Assigning a Name to a Name (line 638):
    # Getting the type of 'tuple_assignment_187340' (line 638)
    tuple_assignment_187340_188064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'tuple_assignment_187340')
    # Assigning a type to the variable 'count' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'count', tuple_assignment_187340_188064)
    
    # Assigning a Name to a Name (line 638):
    # Getting the type of 'tuple_assignment_187341' (line 638)
    tuple_assignment_187341_188065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'tuple_assignment_187341')
    # Assigning a type to the variable 'i' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 11), 'i', tuple_assignment_187341_188065)
    
    # Assigning a List to a Name (line 639):
    
    # Assigning a List to a Name (line 639):
    
    # Obtaining an instance of the builtin type 'list' (line 639)
    list_188066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 639)
    # Adding element type (line 639)
    str_188067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 15), 'str', 'requested number of basinhopping iterations completed successfully')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 14), list_188066, str_188067)
    
    # Assigning a type to the variable 'message' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'message', list_188066)
    
    
    # Call to range(...): (line 641)
    # Processing the call arguments (line 641)
    # Getting the type of 'niter' (line 641)
    niter_188069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 19), 'niter', False)
    # Processing the call keyword arguments (line 641)
    kwargs_188070 = {}
    # Getting the type of 'range' (line 641)
    range_188068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 13), 'range', False)
    # Calling range(args, kwargs) (line 641)
    range_call_result_188071 = invoke(stypy.reporting.localization.Localization(__file__, 641, 13), range_188068, *[niter_188069], **kwargs_188070)
    
    # Testing the type of a for loop iterable (line 641)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 641, 4), range_call_result_188071)
    # Getting the type of the for loop variable (line 641)
    for_loop_var_188072 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 641, 4), range_call_result_188071)
    # Assigning a type to the variable 'i' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'i', for_loop_var_188072)
    # SSA begins for a for statement (line 641)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 642):
    
    # Assigning a Call to a Name (line 642):
    
    # Call to one_cycle(...): (line 642)
    # Processing the call keyword arguments (line 642)
    kwargs_188075 = {}
    # Getting the type of 'bh' (line 642)
    bh_188073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 25), 'bh', False)
    # Obtaining the member 'one_cycle' of a type (line 642)
    one_cycle_188074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 25), bh_188073, 'one_cycle')
    # Calling one_cycle(args, kwargs) (line 642)
    one_cycle_call_result_188076 = invoke(stypy.reporting.localization.Localization(__file__, 642, 25), one_cycle_188074, *[], **kwargs_188075)
    
    # Assigning a type to the variable 'new_global_min' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'new_global_min', one_cycle_call_result_188076)
    
    
    # Call to isinstance(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'callback' (line 644)
    callback_188078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 22), 'callback', False)
    # Getting the type of 'collections' (line 644)
    collections_188079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 32), 'collections', False)
    # Obtaining the member 'Callable' of a type (line 644)
    Callable_188080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 32), collections_188079, 'Callable')
    # Processing the call keyword arguments (line 644)
    kwargs_188081 = {}
    # Getting the type of 'isinstance' (line 644)
    isinstance_188077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 644)
    isinstance_call_result_188082 = invoke(stypy.reporting.localization.Localization(__file__, 644, 11), isinstance_188077, *[callback_188078, Callable_188080], **kwargs_188081)
    
    # Testing the type of an if condition (line 644)
    if_condition_188083 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 644, 8), isinstance_call_result_188082)
    # Assigning a type to the variable 'if_condition_188083' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'if_condition_188083', if_condition_188083)
    # SSA begins for if statement (line 644)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 646):
    
    # Assigning a Call to a Name (line 646):
    
    # Call to callback(...): (line 646)
    # Processing the call arguments (line 646)
    # Getting the type of 'bh' (line 646)
    bh_188085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 27), 'bh', False)
    # Obtaining the member 'xtrial' of a type (line 646)
    xtrial_188086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 27), bh_188085, 'xtrial')
    # Getting the type of 'bh' (line 646)
    bh_188087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 38), 'bh', False)
    # Obtaining the member 'energy_trial' of a type (line 646)
    energy_trial_188088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 38), bh_188087, 'energy_trial')
    # Getting the type of 'bh' (line 646)
    bh_188089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 55), 'bh', False)
    # Obtaining the member 'accept' of a type (line 646)
    accept_188090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 55), bh_188089, 'accept')
    # Processing the call keyword arguments (line 646)
    kwargs_188091 = {}
    # Getting the type of 'callback' (line 646)
    callback_188084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 18), 'callback', False)
    # Calling callback(args, kwargs) (line 646)
    callback_call_result_188092 = invoke(stypy.reporting.localization.Localization(__file__, 646, 18), callback_188084, *[xtrial_188086, energy_trial_188088, accept_188090], **kwargs_188091)
    
    # Assigning a type to the variable 'val' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'val', callback_call_result_188092)
    
    # Type idiom detected: calculating its left and rigth part (line 647)
    # Getting the type of 'val' (line 647)
    val_188093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'val')
    # Getting the type of 'None' (line 647)
    None_188094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 26), 'None')
    
    (may_be_188095, more_types_in_union_188096) = may_not_be_none(val_188093, None_188094)

    if may_be_188095:

        if more_types_in_union_188096:
            # Runtime conditional SSA (line 647)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'val' (line 648)
        val_188097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 19), 'val')
        # Testing the type of an if condition (line 648)
        if_condition_188098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 648, 16), val_188097)
        # Assigning a type to the variable 'if_condition_188098' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'if_condition_188098', if_condition_188098)
        # SSA begins for if statement (line 648)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 649):
        
        # Assigning a List to a Name (line 649):
        
        # Obtaining an instance of the builtin type 'list' (line 649)
        list_188099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 649)
        # Adding element type (line 649)
        str_188100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 31), 'str', 'callback function requested stop early byreturning True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 30), list_188099, str_188100)
        
        # Assigning a type to the variable 'message' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 20), 'message', list_188099)
        # SSA join for if statement (line 648)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_188096:
            # SSA join for if statement (line 647)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 644)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'count' (line 653)
    count_188101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'count')
    int_188102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 17), 'int')
    # Applying the binary operator '+=' (line 653)
    result_iadd_188103 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 8), '+=', count_188101, int_188102)
    # Assigning a type to the variable 'count' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'count', result_iadd_188103)
    
    
    # Getting the type of 'new_global_min' (line 654)
    new_global_min_188104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 11), 'new_global_min')
    # Testing the type of an if condition (line 654)
    if_condition_188105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 8), new_global_min_188104)
    # Assigning a type to the variable 'if_condition_188105' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'if_condition_188105', if_condition_188105)
    # SSA begins for if statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 655):
    
    # Assigning a Num to a Name (line 655):
    int_188106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 20), 'int')
    # Assigning a type to the variable 'count' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 12), 'count', int_188106)
    # SSA branch for the else part of an if statement (line 654)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'count' (line 656)
    count_188107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 13), 'count')
    # Getting the type of 'niter_success' (line 656)
    niter_success_188108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 21), 'niter_success')
    # Applying the binary operator '>' (line 656)
    result_gt_188109 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 13), '>', count_188107, niter_success_188108)
    
    # Testing the type of an if condition (line 656)
    if_condition_188110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 656, 13), result_gt_188109)
    # Assigning a type to the variable 'if_condition_188110' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 13), 'if_condition_188110', if_condition_188110)
    # SSA begins for if statement (line 656)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 657):
    
    # Assigning a List to a Name (line 657):
    
    # Obtaining an instance of the builtin type 'list' (line 657)
    list_188111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 657)
    # Adding element type (line 657)
    str_188112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 23), 'str', 'success condition satisfied')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 22), list_188111, str_188112)
    
    # Assigning a type to the variable 'message' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'message', list_188111)
    # SSA join for if statement (line 656)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 661):
    
    # Assigning a Attribute to a Name (line 661):
    # Getting the type of 'bh' (line 661)
    bh_188113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 10), 'bh')
    # Obtaining the member 'res' of a type (line 661)
    res_188114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 10), bh_188113, 'res')
    # Assigning a type to the variable 'res' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'res', res_188114)
    
    # Assigning a Call to a Attribute (line 662):
    
    # Assigning a Call to a Attribute (line 662):
    
    # Call to get_lowest(...): (line 662)
    # Processing the call keyword arguments (line 662)
    kwargs_188118 = {}
    # Getting the type of 'bh' (line 662)
    bh_188115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 37), 'bh', False)
    # Obtaining the member 'storage' of a type (line 662)
    storage_188116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 37), bh_188115, 'storage')
    # Obtaining the member 'get_lowest' of a type (line 662)
    get_lowest_188117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 37), storage_188116, 'get_lowest')
    # Calling get_lowest(args, kwargs) (line 662)
    get_lowest_call_result_188119 = invoke(stypy.reporting.localization.Localization(__file__, 662, 37), get_lowest_188117, *[], **kwargs_188118)
    
    # Getting the type of 'res' (line 662)
    res_188120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'res')
    # Setting the type of the member 'lowest_optimization_result' of a type (line 662)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 4), res_188120, 'lowest_optimization_result', get_lowest_call_result_188119)
    
    # Assigning a Call to a Attribute (line 663):
    
    # Assigning a Call to a Attribute (line 663):
    
    # Call to copy(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 'res' (line 663)
    res_188123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 20), 'res', False)
    # Obtaining the member 'lowest_optimization_result' of a type (line 663)
    lowest_optimization_result_188124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 20), res_188123, 'lowest_optimization_result')
    # Obtaining the member 'x' of a type (line 663)
    x_188125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 20), lowest_optimization_result_188124, 'x')
    # Processing the call keyword arguments (line 663)
    kwargs_188126 = {}
    # Getting the type of 'np' (line 663)
    np_188121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'np', False)
    # Obtaining the member 'copy' of a type (line 663)
    copy_188122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 12), np_188121, 'copy')
    # Calling copy(args, kwargs) (line 663)
    copy_call_result_188127 = invoke(stypy.reporting.localization.Localization(__file__, 663, 12), copy_188122, *[x_188125], **kwargs_188126)
    
    # Getting the type of 'res' (line 663)
    res_188128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 'res')
    # Setting the type of the member 'x' of a type (line 663)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 4), res_188128, 'x', copy_call_result_188127)
    
    # Assigning a Attribute to a Attribute (line 664):
    
    # Assigning a Attribute to a Attribute (line 664):
    # Getting the type of 'res' (line 664)
    res_188129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 14), 'res')
    # Obtaining the member 'lowest_optimization_result' of a type (line 664)
    lowest_optimization_result_188130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 14), res_188129, 'lowest_optimization_result')
    # Obtaining the member 'fun' of a type (line 664)
    fun_188131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 14), lowest_optimization_result_188130, 'fun')
    # Getting the type of 'res' (line 664)
    res_188132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'res')
    # Setting the type of the member 'fun' of a type (line 664)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 4), res_188132, 'fun', fun_188131)
    
    # Assigning a Name to a Attribute (line 665):
    
    # Assigning a Name to a Attribute (line 665):
    # Getting the type of 'message' (line 665)
    message_188133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 18), 'message')
    # Getting the type of 'res' (line 665)
    res_188134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'res')
    # Setting the type of the member 'message' of a type (line 665)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 4), res_188134, 'message', message_188133)
    
    # Assigning a BinOp to a Attribute (line 666):
    
    # Assigning a BinOp to a Attribute (line 666):
    # Getting the type of 'i' (line 666)
    i_188135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 14), 'i')
    int_188136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 18), 'int')
    # Applying the binary operator '+' (line 666)
    result_add_188137 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 14), '+', i_188135, int_188136)
    
    # Getting the type of 'res' (line 666)
    res_188138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'res')
    # Setting the type of the member 'nit' of a type (line 666)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 4), res_188138, 'nit', result_add_188137)
    # Getting the type of 'res' (line 667)
    res_188139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'stypy_return_type', res_188139)
    
    # ################# End of 'basinhopping(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'basinhopping' in the type store
    # Getting the type of 'stypy_return_type' (line 311)
    stypy_return_type_188140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_188140)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'basinhopping'
    return stypy_return_type_188140

# Assigning a type to the variable 'basinhopping' (line 311)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 0), 'basinhopping', basinhopping)

@norecursion
def _test_func2d_nograd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_test_func2d_nograd'
    module_type_store = module_type_store.open_function_context('_test_func2d_nograd', 670, 0, False)
    
    # Passed parameters checking function
    _test_func2d_nograd.stypy_localization = localization
    _test_func2d_nograd.stypy_type_of_self = None
    _test_func2d_nograd.stypy_type_store = module_type_store
    _test_func2d_nograd.stypy_function_name = '_test_func2d_nograd'
    _test_func2d_nograd.stypy_param_names_list = ['x']
    _test_func2d_nograd.stypy_varargs_param_name = None
    _test_func2d_nograd.stypy_kwargs_param_name = None
    _test_func2d_nograd.stypy_call_defaults = defaults
    _test_func2d_nograd.stypy_call_varargs = varargs
    _test_func2d_nograd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_test_func2d_nograd', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_test_func2d_nograd', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_test_func2d_nograd(...)' code ##################

    
    # Assigning a BinOp to a Name (line 671):
    
    # Assigning a BinOp to a Name (line 671):
    
    # Call to cos(...): (line 671)
    # Processing the call arguments (line 671)
    float_188142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 13), 'float')
    
    # Obtaining the type of the subscript
    int_188143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 22), 'int')
    # Getting the type of 'x' (line 671)
    x_188144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 20), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 671)
    getitem___188145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 20), x_188144, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 671)
    subscript_call_result_188146 = invoke(stypy.reporting.localization.Localization(__file__, 671, 20), getitem___188145, int_188143)
    
    # Applying the binary operator '*' (line 671)
    result_mul_188147 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 13), '*', float_188142, subscript_call_result_188146)
    
    float_188148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 27), 'float')
    # Applying the binary operator '-' (line 671)
    result_sub_188149 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 13), '-', result_mul_188147, float_188148)
    
    # Processing the call keyword arguments (line 671)
    kwargs_188150 = {}
    # Getting the type of 'cos' (line 671)
    cos_188141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 9), 'cos', False)
    # Calling cos(args, kwargs) (line 671)
    cos_call_result_188151 = invoke(stypy.reporting.localization.Localization(__file__, 671, 9), cos_188141, *[result_sub_188149], **kwargs_188150)
    
    
    # Obtaining the type of the subscript
    int_188152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 37), 'int')
    # Getting the type of 'x' (line 671)
    x_188153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 35), 'x')
    # Obtaining the member '__getitem__' of a type (line 671)
    getitem___188154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 35), x_188153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 671)
    subscript_call_result_188155 = invoke(stypy.reporting.localization.Localization(__file__, 671, 35), getitem___188154, int_188152)
    
    float_188156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 42), 'float')
    # Applying the binary operator '+' (line 671)
    result_add_188157 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 35), '+', subscript_call_result_188155, float_188156)
    
    
    # Obtaining the type of the subscript
    int_188158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 51), 'int')
    # Getting the type of 'x' (line 671)
    x_188159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 49), 'x')
    # Obtaining the member '__getitem__' of a type (line 671)
    getitem___188160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 49), x_188159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 671)
    subscript_call_result_188161 = invoke(stypy.reporting.localization.Localization(__file__, 671, 49), getitem___188160, int_188158)
    
    # Applying the binary operator '*' (line 671)
    result_mul_188162 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 34), '*', result_add_188157, subscript_call_result_188161)
    
    # Applying the binary operator '+' (line 671)
    result_add_188163 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 9), '+', cos_call_result_188151, result_mul_188162)
    
    
    # Obtaining the type of the subscript
    int_188164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 59), 'int')
    # Getting the type of 'x' (line 671)
    x_188165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 57), 'x')
    # Obtaining the member '__getitem__' of a type (line 671)
    getitem___188166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 57), x_188165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 671)
    subscript_call_result_188167 = invoke(stypy.reporting.localization.Localization(__file__, 671, 57), getitem___188166, int_188164)
    
    float_188168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 64), 'float')
    # Applying the binary operator '+' (line 671)
    result_add_188169 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 57), '+', subscript_call_result_188167, float_188168)
    
    
    # Obtaining the type of the subscript
    int_188170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 73), 'int')
    # Getting the type of 'x' (line 671)
    x_188171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 71), 'x')
    # Obtaining the member '__getitem__' of a type (line 671)
    getitem___188172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 71), x_188171, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 671)
    subscript_call_result_188173 = invoke(stypy.reporting.localization.Localization(__file__, 671, 71), getitem___188172, int_188170)
    
    # Applying the binary operator '*' (line 671)
    result_mul_188174 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 56), '*', result_add_188169, subscript_call_result_188173)
    
    # Applying the binary operator '+' (line 671)
    result_add_188175 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 54), '+', result_add_188163, result_mul_188174)
    
    float_188176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 11), 'float')
    # Applying the binary operator '+' (line 672)
    result_add_188177 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 9), '+', result_add_188175, float_188176)
    
    # Assigning a type to the variable 'f' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'f', result_add_188177)
    # Getting the type of 'f' (line 673)
    f_188178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'stypy_return_type', f_188178)
    
    # ################# End of '_test_func2d_nograd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_test_func2d_nograd' in the type store
    # Getting the type of 'stypy_return_type' (line 670)
    stypy_return_type_188179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_188179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_test_func2d_nograd'
    return stypy_return_type_188179

# Assigning a type to the variable '_test_func2d_nograd' (line 670)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 0), '_test_func2d_nograd', _test_func2d_nograd)

@norecursion
def _test_func2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_test_func2d'
    module_type_store = module_type_store.open_function_context('_test_func2d', 676, 0, False)
    
    # Passed parameters checking function
    _test_func2d.stypy_localization = localization
    _test_func2d.stypy_type_of_self = None
    _test_func2d.stypy_type_store = module_type_store
    _test_func2d.stypy_function_name = '_test_func2d'
    _test_func2d.stypy_param_names_list = ['x']
    _test_func2d.stypy_varargs_param_name = None
    _test_func2d.stypy_kwargs_param_name = None
    _test_func2d.stypy_call_defaults = defaults
    _test_func2d.stypy_call_varargs = varargs
    _test_func2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_test_func2d', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_test_func2d', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_test_func2d(...)' code ##################

    
    # Assigning a BinOp to a Name (line 677):
    
    # Assigning a BinOp to a Name (line 677):
    
    # Call to cos(...): (line 677)
    # Processing the call arguments (line 677)
    float_188181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 13), 'float')
    
    # Obtaining the type of the subscript
    int_188182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 22), 'int')
    # Getting the type of 'x' (line 677)
    x_188183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 20), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 677)
    getitem___188184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 20), x_188183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 677)
    subscript_call_result_188185 = invoke(stypy.reporting.localization.Localization(__file__, 677, 20), getitem___188184, int_188182)
    
    # Applying the binary operator '*' (line 677)
    result_mul_188186 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 13), '*', float_188181, subscript_call_result_188185)
    
    float_188187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 27), 'float')
    # Applying the binary operator '-' (line 677)
    result_sub_188188 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 13), '-', result_mul_188186, float_188187)
    
    # Processing the call keyword arguments (line 677)
    kwargs_188189 = {}
    # Getting the type of 'cos' (line 677)
    cos_188180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 9), 'cos', False)
    # Calling cos(args, kwargs) (line 677)
    cos_call_result_188190 = invoke(stypy.reporting.localization.Localization(__file__, 677, 9), cos_188180, *[result_sub_188188], **kwargs_188189)
    
    
    # Obtaining the type of the subscript
    int_188191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 37), 'int')
    # Getting the type of 'x' (line 677)
    x_188192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 35), 'x')
    # Obtaining the member '__getitem__' of a type (line 677)
    getitem___188193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 35), x_188192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 677)
    subscript_call_result_188194 = invoke(stypy.reporting.localization.Localization(__file__, 677, 35), getitem___188193, int_188191)
    
    float_188195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 42), 'float')
    # Applying the binary operator '+' (line 677)
    result_add_188196 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 35), '+', subscript_call_result_188194, float_188195)
    
    
    # Obtaining the type of the subscript
    int_188197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 51), 'int')
    # Getting the type of 'x' (line 677)
    x_188198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 49), 'x')
    # Obtaining the member '__getitem__' of a type (line 677)
    getitem___188199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 49), x_188198, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 677)
    subscript_call_result_188200 = invoke(stypy.reporting.localization.Localization(__file__, 677, 49), getitem___188199, int_188197)
    
    # Applying the binary operator '*' (line 677)
    result_mul_188201 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 34), '*', result_add_188196, subscript_call_result_188200)
    
    # Applying the binary operator '+' (line 677)
    result_add_188202 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 9), '+', cos_call_result_188190, result_mul_188201)
    
    
    # Call to cos(...): (line 677)
    # Processing the call arguments (line 677)
    float_188204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 60), 'float')
    
    # Obtaining the type of the subscript
    int_188205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 69), 'int')
    # Getting the type of 'x' (line 677)
    x_188206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 67), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 677)
    getitem___188207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 67), x_188206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 677)
    subscript_call_result_188208 = invoke(stypy.reporting.localization.Localization(__file__, 677, 67), getitem___188207, int_188205)
    
    # Applying the binary operator '*' (line 677)
    result_mul_188209 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 60), '*', float_188204, subscript_call_result_188208)
    
    float_188210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 9), 'float')
    # Applying the binary operator '-' (line 677)
    result_sub_188211 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 60), '-', result_mul_188209, float_188210)
    
    # Processing the call keyword arguments (line 677)
    kwargs_188212 = {}
    # Getting the type of 'cos' (line 677)
    cos_188203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 56), 'cos', False)
    # Calling cos(args, kwargs) (line 677)
    cos_call_result_188213 = invoke(stypy.reporting.localization.Localization(__file__, 677, 56), cos_188203, *[result_sub_188211], **kwargs_188212)
    
    # Applying the binary operator '+' (line 677)
    result_add_188214 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 54), '+', result_add_188202, cos_call_result_188213)
    
    
    # Obtaining the type of the subscript
    int_188215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 19), 'int')
    # Getting the type of 'x' (line 678)
    x_188216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 678)
    getitem___188217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 17), x_188216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 678)
    subscript_call_result_188218 = invoke(stypy.reporting.localization.Localization(__file__, 678, 17), getitem___188217, int_188215)
    
    float_188219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 24), 'float')
    # Applying the binary operator '+' (line 678)
    result_add_188220 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 17), '+', subscript_call_result_188218, float_188219)
    
    
    # Obtaining the type of the subscript
    int_188221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 33), 'int')
    # Getting the type of 'x' (line 678)
    x_188222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 31), 'x')
    # Obtaining the member '__getitem__' of a type (line 678)
    getitem___188223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 31), x_188222, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 678)
    subscript_call_result_188224 = invoke(stypy.reporting.localization.Localization(__file__, 678, 31), getitem___188223, int_188221)
    
    # Applying the binary operator '*' (line 678)
    result_mul_188225 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 16), '*', result_add_188220, subscript_call_result_188224)
    
    # Applying the binary operator '+' (line 678)
    result_add_188226 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 14), '+', result_add_188214, result_mul_188225)
    
    
    # Obtaining the type of the subscript
    int_188227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 40), 'int')
    # Getting the type of 'x' (line 678)
    x_188228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 38), 'x')
    # Obtaining the member '__getitem__' of a type (line 678)
    getitem___188229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 38), x_188228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 678)
    subscript_call_result_188230 = invoke(stypy.reporting.localization.Localization(__file__, 678, 38), getitem___188229, int_188227)
    
    
    # Obtaining the type of the subscript
    int_188231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 47), 'int')
    # Getting the type of 'x' (line 678)
    x_188232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 45), 'x')
    # Obtaining the member '__getitem__' of a type (line 678)
    getitem___188233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 45), x_188232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 678)
    subscript_call_result_188234 = invoke(stypy.reporting.localization.Localization(__file__, 678, 45), getitem___188233, int_188231)
    
    # Applying the binary operator '*' (line 678)
    result_mul_188235 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 38), '*', subscript_call_result_188230, subscript_call_result_188234)
    
    # Applying the binary operator '+' (line 678)
    result_add_188236 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 36), '+', result_add_188226, result_mul_188235)
    
    float_188237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 52), 'float')
    # Applying the binary operator '+' (line 678)
    result_add_188238 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 50), '+', result_add_188236, float_188237)
    
    # Assigning a type to the variable 'f' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'f', result_add_188238)
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Call to zeros(...): (line 679)
    # Processing the call arguments (line 679)
    int_188241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 18), 'int')
    # Processing the call keyword arguments (line 679)
    kwargs_188242 = {}
    # Getting the type of 'np' (line 679)
    np_188239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 679)
    zeros_188240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 9), np_188239, 'zeros')
    # Calling zeros(args, kwargs) (line 679)
    zeros_call_result_188243 = invoke(stypy.reporting.localization.Localization(__file__, 679, 9), zeros_188240, *[int_188241], **kwargs_188242)
    
    # Assigning a type to the variable 'df' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 4), 'df', zeros_call_result_188243)
    
    # Assigning a BinOp to a Subscript (line 680):
    
    # Assigning a BinOp to a Subscript (line 680):
    float_188244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 12), 'float')
    
    # Call to sin(...): (line 680)
    # Processing the call arguments (line 680)
    float_188246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 24), 'float')
    
    # Obtaining the type of the subscript
    int_188247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 33), 'int')
    # Getting the type of 'x' (line 680)
    x_188248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 31), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___188249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 31), x_188248, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_188250 = invoke(stypy.reporting.localization.Localization(__file__, 680, 31), getitem___188249, int_188247)
    
    # Applying the binary operator '*' (line 680)
    result_mul_188251 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 24), '*', float_188246, subscript_call_result_188250)
    
    float_188252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 38), 'float')
    # Applying the binary operator '-' (line 680)
    result_sub_188253 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 24), '-', result_mul_188251, float_188252)
    
    # Processing the call keyword arguments (line 680)
    kwargs_188254 = {}
    # Getting the type of 'sin' (line 680)
    sin_188245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 20), 'sin', False)
    # Calling sin(args, kwargs) (line 680)
    sin_call_result_188255 = invoke(stypy.reporting.localization.Localization(__file__, 680, 20), sin_188245, *[result_sub_188253], **kwargs_188254)
    
    # Applying the binary operator '*' (line 680)
    result_mul_188256 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 12), '*', float_188244, sin_call_result_188255)
    
    float_188257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 45), 'float')
    
    # Obtaining the type of the subscript
    int_188258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 52), 'int')
    # Getting the type of 'x' (line 680)
    x_188259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 50), 'x')
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___188260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 50), x_188259, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_188261 = invoke(stypy.reporting.localization.Localization(__file__, 680, 50), getitem___188260, int_188258)
    
    # Applying the binary operator '*' (line 680)
    result_mul_188262 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 45), '*', float_188257, subscript_call_result_188261)
    
    # Applying the binary operator '+' (line 680)
    result_add_188263 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 12), '+', result_mul_188256, result_mul_188262)
    
    float_188264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 57), 'float')
    # Applying the binary operator '+' (line 680)
    result_add_188265 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 55), '+', result_add_188263, float_188264)
    
    
    # Obtaining the type of the subscript
    int_188266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 65), 'int')
    # Getting the type of 'x' (line 680)
    x_188267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 63), 'x')
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___188268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 63), x_188267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_188269 = invoke(stypy.reporting.localization.Localization(__file__, 680, 63), getitem___188268, int_188266)
    
    # Applying the binary operator '+' (line 680)
    result_add_188270 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 61), '+', result_add_188265, subscript_call_result_188269)
    
    # Getting the type of 'df' (line 680)
    df_188271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'df')
    int_188272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 7), 'int')
    # Storing an element on a container (line 680)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 4), df_188271, (int_188272, result_add_188270))
    
    # Assigning a BinOp to a Subscript (line 681):
    
    # Assigning a BinOp to a Subscript (line 681):
    float_188273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 12), 'float')
    
    # Call to sin(...): (line 681)
    # Processing the call arguments (line 681)
    float_188275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 24), 'float')
    
    # Obtaining the type of the subscript
    int_188276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 33), 'int')
    # Getting the type of 'x' (line 681)
    x_188277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 31), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___188278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 31), x_188277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_188279 = invoke(stypy.reporting.localization.Localization(__file__, 681, 31), getitem___188278, int_188276)
    
    # Applying the binary operator '*' (line 681)
    result_mul_188280 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 24), '*', float_188275, subscript_call_result_188279)
    
    float_188281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 38), 'float')
    # Applying the binary operator '-' (line 681)
    result_sub_188282 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 24), '-', result_mul_188280, float_188281)
    
    # Processing the call keyword arguments (line 681)
    kwargs_188283 = {}
    # Getting the type of 'sin' (line 681)
    sin_188274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 20), 'sin', False)
    # Calling sin(args, kwargs) (line 681)
    sin_call_result_188284 = invoke(stypy.reporting.localization.Localization(__file__, 681, 20), sin_188274, *[result_sub_188282], **kwargs_188283)
    
    # Applying the binary operator '*' (line 681)
    result_mul_188285 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 12), '*', float_188273, sin_call_result_188284)
    
    float_188286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 45), 'float')
    
    # Obtaining the type of the subscript
    int_188287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 52), 'int')
    # Getting the type of 'x' (line 681)
    x_188288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 50), 'x')
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___188289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 50), x_188288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_188290 = invoke(stypy.reporting.localization.Localization(__file__, 681, 50), getitem___188289, int_188287)
    
    # Applying the binary operator '*' (line 681)
    result_mul_188291 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 45), '*', float_188286, subscript_call_result_188290)
    
    # Applying the binary operator '+' (line 681)
    result_add_188292 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 12), '+', result_mul_188285, result_mul_188291)
    
    float_188293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 57), 'float')
    # Applying the binary operator '+' (line 681)
    result_add_188294 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 55), '+', result_add_188292, float_188293)
    
    
    # Obtaining the type of the subscript
    int_188295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 65), 'int')
    # Getting the type of 'x' (line 681)
    x_188296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 63), 'x')
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___188297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 63), x_188296, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_188298 = invoke(stypy.reporting.localization.Localization(__file__, 681, 63), getitem___188297, int_188295)
    
    # Applying the binary operator '+' (line 681)
    result_add_188299 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 61), '+', result_add_188294, subscript_call_result_188298)
    
    # Getting the type of 'df' (line 681)
    df_188300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'df')
    int_188301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 7), 'int')
    # Storing an element on a container (line 681)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 4), df_188300, (int_188301, result_add_188299))
    
    # Obtaining an instance of the builtin type 'tuple' (line 682)
    tuple_188302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 682)
    # Adding element type (line 682)
    # Getting the type of 'f' (line 682)
    f_188303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 11), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 11), tuple_188302, f_188303)
    # Adding element type (line 682)
    # Getting the type of 'df' (line 682)
    df_188304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 14), 'df')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 11), tuple_188302, df_188304)
    
    # Assigning a type to the variable 'stypy_return_type' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'stypy_return_type', tuple_188302)
    
    # ################# End of '_test_func2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_test_func2d' in the type store
    # Getting the type of 'stypy_return_type' (line 676)
    stypy_return_type_188305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_188305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_test_func2d'
    return stypy_return_type_188305

# Assigning a type to the variable '_test_func2d' (line 676)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), '_test_func2d', _test_func2d)

if (__name__ == '__main__'):
    
    # Call to print(...): (line 685)
    # Processing the call arguments (line 685)
    str_188307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 10), 'str', '\n\nminimize a 2d function without gradient')
    # Processing the call keyword arguments (line 685)
    kwargs_188308 = {}
    # Getting the type of 'print' (line 685)
    print_188306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'print', False)
    # Calling print(args, kwargs) (line 685)
    print_call_result_188309 = invoke(stypy.reporting.localization.Localization(__file__, 685, 4), print_188306, *[str_188307], **kwargs_188308)
    
    
    # Assigning a Dict to a Name (line 687):
    
    # Assigning a Dict to a Name (line 687):
    
    # Obtaining an instance of the builtin type 'dict' (line 687)
    dict_188310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 687)
    # Adding element type (key, value) (line 687)
    str_188311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 14), 'str', 'method')
    str_188312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 24), 'str', 'L-BFGS-B')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 13), dict_188310, (str_188311, str_188312))
    
    # Assigning a type to the variable 'kwargs' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'kwargs', dict_188310)
    
    # Assigning a Call to a Name (line 688):
    
    # Assigning a Call to a Name (line 688):
    
    # Call to array(...): (line 688)
    # Processing the call arguments (line 688)
    
    # Obtaining an instance of the builtin type 'list' (line 688)
    list_188315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 688)
    # Adding element type (line 688)
    float_188316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 18), list_188315, float_188316)
    # Adding element type (line 688)
    float_188317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 18), list_188315, float_188317)
    
    # Processing the call keyword arguments (line 688)
    kwargs_188318 = {}
    # Getting the type of 'np' (line 688)
    np_188313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 688)
    array_188314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 9), np_188313, 'array')
    # Calling array(args, kwargs) (line 688)
    array_call_result_188319 = invoke(stypy.reporting.localization.Localization(__file__, 688, 9), array_188314, *[list_188315], **kwargs_188318)
    
    # Assigning a type to the variable 'x0' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'x0', array_call_result_188319)
    
    # Call to minimize(...): (line 689)
    # Processing the call arguments (line 689)
    # Getting the type of '_test_func2d_nograd' (line 689)
    _test_func2d_nograd_188323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 28), '_test_func2d_nograd', False)
    # Getting the type of 'x0' (line 689)
    x0_188324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 49), 'x0', False)
    # Processing the call keyword arguments (line 689)
    # Getting the type of 'kwargs' (line 689)
    kwargs_188325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 55), 'kwargs', False)
    kwargs_188326 = {'kwargs_188325': kwargs_188325}
    # Getting the type of 'scipy' (line 689)
    scipy_188320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'scipy', False)
    # Obtaining the member 'optimize' of a type (line 689)
    optimize_188321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 4), scipy_188320, 'optimize')
    # Obtaining the member 'minimize' of a type (line 689)
    minimize_188322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 4), optimize_188321, 'minimize')
    # Calling minimize(args, kwargs) (line 689)
    minimize_call_result_188327 = invoke(stypy.reporting.localization.Localization(__file__, 689, 4), minimize_188322, *[_test_func2d_nograd_188323, x0_188324], **kwargs_188326)
    
    
    # Assigning a Call to a Name (line 690):
    
    # Assigning a Call to a Name (line 690):
    
    # Call to basinhopping(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of '_test_func2d_nograd' (line 690)
    _test_func2d_nograd_188329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 23), '_test_func2d_nograd', False)
    # Getting the type of 'x0' (line 690)
    x0_188330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 44), 'x0', False)
    # Processing the call keyword arguments (line 690)
    # Getting the type of 'kwargs' (line 690)
    kwargs_188331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 65), 'kwargs', False)
    keyword_188332 = kwargs_188331
    int_188333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 29), 'int')
    keyword_188334 = int_188333
    # Getting the type of 'False' (line 691)
    False_188335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 39), 'False', False)
    keyword_188336 = False_188335
    kwargs_188337 = {'disp': keyword_188336, 'niter': keyword_188334, 'minimizer_kwargs': keyword_188332}
    # Getting the type of 'basinhopping' (line 690)
    basinhopping_188328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 10), 'basinhopping', False)
    # Calling basinhopping(args, kwargs) (line 690)
    basinhopping_call_result_188338 = invoke(stypy.reporting.localization.Localization(__file__, 690, 10), basinhopping_188328, *[_test_func2d_nograd_188329, x0_188330], **kwargs_188337)
    
    # Assigning a type to the variable 'ret' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'ret', basinhopping_call_result_188338)
    
    # Call to print(...): (line 692)
    # Processing the call arguments (line 692)
    str_188340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 10), 'str', 'minimum expected at  func([-0.195, -0.1]) = 0.0')
    # Processing the call keyword arguments (line 692)
    kwargs_188341 = {}
    # Getting the type of 'print' (line 692)
    print_188339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'print', False)
    # Calling print(args, kwargs) (line 692)
    print_call_result_188342 = invoke(stypy.reporting.localization.Localization(__file__, 692, 4), print_188339, *[str_188340], **kwargs_188341)
    
    
    # Call to print(...): (line 693)
    # Processing the call arguments (line 693)
    # Getting the type of 'ret' (line 693)
    ret_188344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 10), 'ret', False)
    # Processing the call keyword arguments (line 693)
    kwargs_188345 = {}
    # Getting the type of 'print' (line 693)
    print_188343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'print', False)
    # Calling print(args, kwargs) (line 693)
    print_call_result_188346 = invoke(stypy.reporting.localization.Localization(__file__, 693, 4), print_188343, *[ret_188344], **kwargs_188345)
    
    
    # Call to print(...): (line 695)
    # Processing the call arguments (line 695)
    str_188348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 10), 'str', '\n\ntry a harder 2d problem')
    # Processing the call keyword arguments (line 695)
    kwargs_188349 = {}
    # Getting the type of 'print' (line 695)
    print_188347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'print', False)
    # Calling print(args, kwargs) (line 695)
    print_call_result_188350 = invoke(stypy.reporting.localization.Localization(__file__, 695, 4), print_188347, *[str_188348], **kwargs_188349)
    
    
    # Assigning a Dict to a Name (line 696):
    
    # Assigning a Dict to a Name (line 696):
    
    # Obtaining an instance of the builtin type 'dict' (line 696)
    dict_188351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 696)
    # Adding element type (key, value) (line 696)
    str_188352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 14), 'str', 'method')
    str_188353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 24), 'str', 'L-BFGS-B')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 696, 13), dict_188351, (str_188352, str_188353))
    # Adding element type (key, value) (line 696)
    str_188354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 36), 'str', 'jac')
    # Getting the type of 'True' (line 696)
    True_188355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 43), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 696, 13), dict_188351, (str_188354, True_188355))
    
    # Assigning a type to the variable 'kwargs' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'kwargs', dict_188351)
    
    # Assigning a Call to a Name (line 697):
    
    # Assigning a Call to a Name (line 697):
    
    # Call to array(...): (line 697)
    # Processing the call arguments (line 697)
    
    # Obtaining an instance of the builtin type 'list' (line 697)
    list_188358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 697)
    # Adding element type (line 697)
    float_188359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 18), list_188358, float_188359)
    # Adding element type (line 697)
    float_188360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 18), list_188358, float_188360)
    
    # Processing the call keyword arguments (line 697)
    kwargs_188361 = {}
    # Getting the type of 'np' (line 697)
    np_188356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 697)
    array_188357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 9), np_188356, 'array')
    # Calling array(args, kwargs) (line 697)
    array_call_result_188362 = invoke(stypy.reporting.localization.Localization(__file__, 697, 9), array_188357, *[list_188358], **kwargs_188361)
    
    # Assigning a type to the variable 'x0' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'x0', array_call_result_188362)
    
    # Assigning a Call to a Name (line 698):
    
    # Assigning a Call to a Name (line 698):
    
    # Call to basinhopping(...): (line 698)
    # Processing the call arguments (line 698)
    # Getting the type of '_test_func2d' (line 698)
    _test_func2d_188364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 23), '_test_func2d', False)
    # Getting the type of 'x0' (line 698)
    x0_188365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 37), 'x0', False)
    # Processing the call keyword arguments (line 698)
    # Getting the type of 'kwargs' (line 698)
    kwargs_188366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 58), 'kwargs', False)
    keyword_188367 = kwargs_188366
    int_188368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 72), 'int')
    keyword_188369 = int_188368
    # Getting the type of 'False' (line 699)
    False_188370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 28), 'False', False)
    keyword_188371 = False_188370
    kwargs_188372 = {'disp': keyword_188371, 'niter': keyword_188369, 'minimizer_kwargs': keyword_188367}
    # Getting the type of 'basinhopping' (line 698)
    basinhopping_188363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 10), 'basinhopping', False)
    # Calling basinhopping(args, kwargs) (line 698)
    basinhopping_call_result_188373 = invoke(stypy.reporting.localization.Localization(__file__, 698, 10), basinhopping_188363, *[_test_func2d_188364, x0_188365], **kwargs_188372)
    
    # Assigning a type to the variable 'ret' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'ret', basinhopping_call_result_188373)
    
    # Call to print(...): (line 700)
    # Processing the call arguments (line 700)
    str_188375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 10), 'str', 'minimum expected at ~, func([-0.19415263, -0.19415263]) = 0')
    # Processing the call keyword arguments (line 700)
    kwargs_188376 = {}
    # Getting the type of 'print' (line 700)
    print_188374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'print', False)
    # Calling print(args, kwargs) (line 700)
    print_call_result_188377 = invoke(stypy.reporting.localization.Localization(__file__, 700, 4), print_188374, *[str_188375], **kwargs_188376)
    
    
    # Call to print(...): (line 701)
    # Processing the call arguments (line 701)
    # Getting the type of 'ret' (line 701)
    ret_188379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 10), 'ret', False)
    # Processing the call keyword arguments (line 701)
    kwargs_188380 = {}
    # Getting the type of 'print' (line 701)
    print_188378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'print', False)
    # Calling print(args, kwargs) (line 701)
    print_call_result_188381 = invoke(stypy.reporting.localization.Localization(__file__, 701, 4), print_188378, *[ret_188379], **kwargs_188380)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
