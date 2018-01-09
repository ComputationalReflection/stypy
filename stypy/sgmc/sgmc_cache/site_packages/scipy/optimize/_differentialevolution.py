
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: differential_evolution: The differential evolution global optimization algorithm
3: Added by Andrew Nelson 2014
4: '''
5: from __future__ import division, print_function, absolute_import
6: import numpy as np
7: from scipy.optimize import OptimizeResult, minimize
8: from scipy.optimize.optimize import _status_message
9: from scipy._lib._util import check_random_state
10: from scipy._lib.six import xrange
11: import warnings
12: 
13: 
14: __all__ = ['differential_evolution']
15: 
16: _MACHEPS = np.finfo(np.float64).eps
17: 
18: 
19: def differential_evolution(func, bounds, args=(), strategy='best1bin',
20:                            maxiter=1000, popsize=15, tol=0.01,
21:                            mutation=(0.5, 1), recombination=0.7, seed=None,
22:                            callback=None, disp=False, polish=True,
23:                            init='latinhypercube', atol=0):
24:     '''Finds the global minimum of a multivariate function.
25:     Differential Evolution is stochastic in nature (does not use gradient
26:     methods) to find the minimium, and can search large areas of candidate
27:     space, but often requires larger numbers of function evaluations than
28:     conventional gradient based techniques.
29: 
30:     The algorithm is due to Storn and Price [1]_.
31: 
32:     Parameters
33:     ----------
34:     func : callable
35:         The objective function to be minimized.  Must be in the form
36:         ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
37:         and ``args`` is a  tuple of any additional fixed parameters needed to
38:         completely specify the function.
39:     bounds : sequence
40:         Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
41:         defining the lower and upper bounds for the optimizing argument of
42:         `func`. It is required to have ``len(bounds) == len(x)``.
43:         ``len(bounds)`` is used to determine the number of parameters in ``x``.
44:     args : tuple, optional
45:         Any additional fixed parameters needed to
46:         completely specify the objective function.
47:     strategy : str, optional
48:         The differential evolution strategy to use. Should be one of:
49: 
50:             - 'best1bin'
51:             - 'best1exp'
52:             - 'rand1exp'
53:             - 'randtobest1exp'
54:             - 'best2exp'
55:             - 'rand2exp'
56:             - 'randtobest1bin'
57:             - 'best2bin'
58:             - 'rand2bin'
59:             - 'rand1bin'
60: 
61:         The default is 'best1bin'.
62:     maxiter : int, optional
63:         The maximum number of generations over which the entire population is
64:         evolved. The maximum number of function evaluations (with no polishing)
65:         is: ``(maxiter + 1) * popsize * len(x)``
66:     popsize : int, optional
67:         A multiplier for setting the total population size.  The population has
68:         ``popsize * len(x)`` individuals.
69:     tol : float, optional
70:         Relative tolerance for convergence, the solving stops when
71:         ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
72:         where and `atol` and `tol` are the absolute and relative tolerance
73:         respectively.
74:     mutation : float or tuple(float, float), optional
75:         The mutation constant. In the literature this is also known as
76:         differential weight, being denoted by F.
77:         If specified as a float it should be in the range [0, 2].
78:         If specified as a tuple ``(min, max)`` dithering is employed. Dithering
79:         randomly changes the mutation constant on a generation by generation
80:         basis. The mutation constant for that generation is taken from
81:         ``U[min, max)``. Dithering can help speed convergence significantly.
82:         Increasing the mutation constant increases the search radius, but will
83:         slow down convergence.
84:     recombination : float, optional
85:         The recombination constant, should be in the range [0, 1]. In the
86:         literature this is also known as the crossover probability, being
87:         denoted by CR. Increasing this value allows a larger number of mutants
88:         to progress into the next generation, but at the risk of population
89:         stability.
90:     seed : int or `np.random.RandomState`, optional
91:         If `seed` is not specified the `np.RandomState` singleton is used.
92:         If `seed` is an int, a new `np.random.RandomState` instance is used,
93:         seeded with seed.
94:         If `seed` is already a `np.random.RandomState instance`, then that
95:         `np.random.RandomState` instance is used.
96:         Specify `seed` for repeatable minimizations.
97:     disp : bool, optional
98:         Display status messages
99:     callback : callable, `callback(xk, convergence=val)`, optional
100:         A function to follow the progress of the minimization. ``xk`` is
101:         the current value of ``x0``. ``val`` represents the fractional
102:         value of the population convergence.  When ``val`` is greater than one
103:         the function halts. If callback returns `True`, then the minimization
104:         is halted (any polishing is still carried out).
105:     polish : bool, optional
106:         If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
107:         method is used to polish the best population member at the end, which
108:         can improve the minimization slightly.
109:     init : string, optional
110:         Specify how the population initialization is performed. Should be
111:         one of:
112: 
113:             - 'latinhypercube'
114:             - 'random'
115: 
116:         The default is 'latinhypercube'. Latin Hypercube sampling tries to
117:         maximize coverage of the available parameter space. 'random' initializes
118:         the population randomly - this has the drawback that clustering can
119:         occur, preventing the whole of parameter space being covered.
120:     atol : float, optional
121:         Absolute tolerance for convergence, the solving stops when
122:         ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
123:         where and `atol` and `tol` are the absolute and relative tolerance
124:         respectively.
125: 
126:     Returns
127:     -------
128:     res : OptimizeResult
129:         The optimization result represented as a `OptimizeResult` object.
130:         Important attributes are: ``x`` the solution array, ``success`` a
131:         Boolean flag indicating if the optimizer exited successfully and
132:         ``message`` which describes the cause of the termination. See
133:         `OptimizeResult` for a description of other attributes.  If `polish`
134:         was employed, and a lower minimum was obtained by the polishing, then
135:         OptimizeResult also contains the ``jac`` attribute.
136: 
137:     Notes
138:     -----
139:     Differential evolution is a stochastic population based method that is
140:     useful for global optimization problems. At each pass through the population
141:     the algorithm mutates each candidate solution by mixing with other candidate
142:     solutions to create a trial candidate. There are several strategies [2]_ for
143:     creating trial candidates, which suit some problems more than others. The
144:     'best1bin' strategy is a good starting point for many systems. In this
145:     strategy two members of the population are randomly chosen. Their difference
146:     is used to mutate the best member (the `best` in `best1bin`), :math:`b_0`,
147:     so far:
148: 
149:     .. math::
150: 
151:         b' = b_0 + mutation * (population[rand0] - population[rand1])
152: 
153:     A trial vector is then constructed. Starting with a randomly chosen 'i'th
154:     parameter the trial is sequentially filled (in modulo) with parameters from
155:     `b'` or the original candidate. The choice of whether to use `b'` or the
156:     original candidate is made with a binomial distribution (the 'bin' in
157:     'best1bin') - a random number in [0, 1) is generated.  If this number is
158:     less than the `recombination` constant then the parameter is loaded from
159:     `b'`, otherwise it is loaded from the original candidate.  The final
160:     parameter is always loaded from `b'`.  Once the trial candidate is built
161:     its fitness is assessed. If the trial is better than the original candidate
162:     then it takes its place. If it is also better than the best overall
163:     candidate it also replaces that.
164:     To improve your chances of finding a global minimum use higher `popsize`
165:     values, with higher `mutation` and (dithering), but lower `recombination`
166:     values. This has the effect of widening the search radius, but slowing
167:     convergence.
168: 
169:     .. versionadded:: 0.15.0
170: 
171:     Examples
172:     --------
173:     Let us consider the problem of minimizing the Rosenbrock function. This
174:     function is implemented in `rosen` in `scipy.optimize`.
175: 
176:     >>> from scipy.optimize import rosen, differential_evolution
177:     >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
178:     >>> result = differential_evolution(rosen, bounds)
179:     >>> result.x, result.fun
180:     (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)
181: 
182:     Next find the minimum of the Ackley function
183:     (http://en.wikipedia.org/wiki/Test_functions_for_optimization).
184: 
185:     >>> from scipy.optimize import differential_evolution
186:     >>> import numpy as np
187:     >>> def ackley(x):
188:     ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
189:     ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
190:     ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
191:     >>> bounds = [(-5, 5), (-5, 5)]
192:     >>> result = differential_evolution(ackley, bounds)
193:     >>> result.x, result.fun
194:     (array([ 0.,  0.]), 4.4408920985006262e-16)
195: 
196:     References
197:     ----------
198:     .. [1] Storn, R and Price, K, Differential Evolution - a Simple and
199:            Efficient Heuristic for Global Optimization over Continuous Spaces,
200:            Journal of Global Optimization, 1997, 11, 341 - 359.
201:     .. [2] http://www1.icsi.berkeley.edu/~storn/code.html
202:     .. [3] http://en.wikipedia.org/wiki/Differential_evolution
203:     '''
204: 
205:     solver = DifferentialEvolutionSolver(func, bounds, args=args,
206:                                          strategy=strategy, maxiter=maxiter,
207:                                          popsize=popsize, tol=tol,
208:                                          mutation=mutation,
209:                                          recombination=recombination,
210:                                          seed=seed, polish=polish,
211:                                          callback=callback,
212:                                          disp=disp, init=init, atol=atol)
213:     return solver.solve()
214: 
215: 
216: class DifferentialEvolutionSolver(object):
217: 
218:     '''This class implements the differential evolution solver
219: 
220:     Parameters
221:     ----------
222:     func : callable
223:         The objective function to be minimized.  Must be in the form
224:         ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
225:         and ``args`` is a  tuple of any additional fixed parameters needed to
226:         completely specify the function.
227:     bounds : sequence
228:         Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
229:         defining the lower and upper bounds for the optimizing argument of
230:         `func`. It is required to have ``len(bounds) == len(x)``.
231:         ``len(bounds)`` is used to determine the number of parameters in ``x``.
232:     args : tuple, optional
233:         Any additional fixed parameters needed to
234:         completely specify the objective function.
235:     strategy : str, optional
236:         The differential evolution strategy to use. Should be one of:
237: 
238:             - 'best1bin'
239:             - 'best1exp'
240:             - 'rand1exp'
241:             - 'randtobest1exp'
242:             - 'best2exp'
243:             - 'rand2exp'
244:             - 'randtobest1bin'
245:             - 'best2bin'
246:             - 'rand2bin'
247:             - 'rand1bin'
248: 
249:         The default is 'best1bin'
250: 
251:     maxiter : int, optional
252:         The maximum number of generations over which the entire population is
253:         evolved. The maximum number of function evaluations (with no polishing)
254:         is: ``(maxiter + 1) * popsize * len(x)``
255:     popsize : int, optional
256:         A multiplier for setting the total population size.  The population has
257:         ``popsize * len(x)`` individuals.
258:     tol : float, optional
259:         Relative tolerance for convergence, the solving stops when
260:         ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
261:         where and `atol` and `tol` are the absolute and relative tolerance
262:         respectively.
263:     mutation : float or tuple(float, float), optional
264:         The mutation constant. In the literature this is also known as
265:         differential weight, being denoted by F.
266:         If specified as a float it should be in the range [0, 2].
267:         If specified as a tuple ``(min, max)`` dithering is employed. Dithering
268:         randomly changes the mutation constant on a generation by generation
269:         basis. The mutation constant for that generation is taken from
270:         U[min, max). Dithering can help speed convergence significantly.
271:         Increasing the mutation constant increases the search radius, but will
272:         slow down convergence.
273:     recombination : float, optional
274:         The recombination constant, should be in the range [0, 1]. In the
275:         literature this is also known as the crossover probability, being
276:         denoted by CR. Increasing this value allows a larger number of mutants
277:         to progress into the next generation, but at the risk of population
278:         stability.
279:     seed : int or `np.random.RandomState`, optional
280:         If `seed` is not specified the `np.random.RandomState` singleton is
281:         used.
282:         If `seed` is an int, a new `np.random.RandomState` instance is used,
283:         seeded with `seed`.
284:         If `seed` is already a `np.random.RandomState` instance, then that
285:         `np.random.RandomState` instance is used.
286:         Specify `seed` for repeatable minimizations.
287:     disp : bool, optional
288:         Display status messages
289:     callback : callable, `callback(xk, convergence=val)`, optional
290:         A function to follow the progress of the minimization. ``xk`` is
291:         the current value of ``x0``. ``val`` represents the fractional
292:         value of the population convergence.  When ``val`` is greater than one
293:         the function halts. If callback returns `True`, then the minimization
294:         is halted (any polishing is still carried out).
295:     polish : bool, optional
296:         If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method
297:         is used to polish the best population member at the end. This requires
298:         a few more function evaluations.
299:     maxfun : int, optional
300:         Set the maximum number of function evaluations. However, it probably
301:         makes more sense to set `maxiter` instead.
302:     init : string, optional
303:         Specify which type of population initialization is performed. Should be
304:         one of:
305: 
306:             - 'latinhypercube'
307:             - 'random'
308:     atol : float, optional
309:         Absolute tolerance for convergence, the solving stops when
310:         ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
311:         where and `atol` and `tol` are the absolute and relative tolerance
312:         respectively.
313:     '''
314: 
315:     # Dispatch of mutation strategy method (binomial or exponential).
316:     _binomial = {'best1bin': '_best1',
317:                  'randtobest1bin': '_randtobest1',
318:                  'best2bin': '_best2',
319:                  'rand2bin': '_rand2',
320:                  'rand1bin': '_rand1'}
321:     _exponential = {'best1exp': '_best1',
322:                     'rand1exp': '_rand1',
323:                     'randtobest1exp': '_randtobest1',
324:                     'best2exp': '_best2',
325:                     'rand2exp': '_rand2'}
326: 
327:     def __init__(self, func, bounds, args=(),
328:                  strategy='best1bin', maxiter=1000, popsize=15,
329:                  tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
330:                  maxfun=np.inf, callback=None, disp=False, polish=True,
331:                  init='latinhypercube', atol=0):
332: 
333:         if strategy in self._binomial:
334:             self.mutation_func = getattr(self, self._binomial[strategy])
335:         elif strategy in self._exponential:
336:             self.mutation_func = getattr(self, self._exponential[strategy])
337:         else:
338:             raise ValueError("Please select a valid mutation strategy")
339:         self.strategy = strategy
340: 
341:         self.callback = callback
342:         self.polish = polish
343: 
344:         # relative and absolute tolerances for convergence
345:         self.tol, self.atol = tol, atol
346: 
347:         # Mutation constant should be in [0, 2). If specified as a sequence
348:         # then dithering is performed.
349:         self.scale = mutation
350:         if (not np.all(np.isfinite(mutation)) or
351:                 np.any(np.array(mutation) >= 2) or
352:                 np.any(np.array(mutation) < 0)):
353:             raise ValueError('The mutation constant must be a float in '
354:                              'U[0, 2), or specified as a tuple(min, max)'
355:                              ' where min < max and min, max are in U[0, 2).')
356: 
357:         self.dither = None
358:         if hasattr(mutation, '__iter__') and len(mutation) > 1:
359:             self.dither = [mutation[0], mutation[1]]
360:             self.dither.sort()
361: 
362:         self.cross_over_probability = recombination
363: 
364:         self.func = func
365:         self.args = args
366: 
367:         # convert tuple of lower and upper bounds to limits
368:         # [(low_0, high_0), ..., (low_n, high_n]
369:         #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
370:         self.limits = np.array(bounds, dtype='float').T
371:         if (np.size(self.limits, 0) != 2 or not
372:                 np.all(np.isfinite(self.limits))):
373:             raise ValueError('bounds should be a sequence containing '
374:                              'real valued (min, max) pairs for each value'
375:                              ' in x')
376: 
377:         if maxiter is None:  # the default used to be None
378:             maxiter = 1000
379:         self.maxiter = maxiter
380:         if maxfun is None:  # the default used to be None
381:             maxfun = np.inf
382:         self.maxfun = maxfun
383: 
384:         # population is scaled to between [0, 1].
385:         # We have to scale between parameter <-> population
386:         # save these arguments for _scale_parameter and
387:         # _unscale_parameter. This is an optimization
388:         self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
389:         self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])
390: 
391:         self.parameter_count = np.size(self.limits, 1)
392: 
393:         self.random_number_generator = check_random_state(seed)
394: 
395:         # default population initialization is a latin hypercube design, but
396:         # there are other population initializations possible.
397:         self.num_population_members = popsize * self.parameter_count
398: 
399:         self.population_shape = (self.num_population_members,
400:                                  self.parameter_count)
401: 
402:         self._nfev = 0
403:         if init == 'latinhypercube':
404:             self.init_population_lhs()
405:         elif init == 'random':
406:             self.init_population_random()
407:         else:
408:             raise ValueError("The population initialization method must be one"
409:                              "of 'latinhypercube' or 'random'")
410: 
411:         self.disp = disp
412: 
413:     def init_population_lhs(self):
414:         '''
415:         Initializes the population with Latin Hypercube Sampling.
416:         Latin Hypercube Sampling ensures that each parameter is uniformly
417:         sampled over its range.
418:         '''
419:         rng = self.random_number_generator
420: 
421:         # Each parameter range needs to be sampled uniformly. The scaled
422:         # parameter range ([0, 1)) needs to be split into
423:         # `self.num_population_members` segments, each of which has the following
424:         # size:
425:         segsize = 1.0 / self.num_population_members
426: 
427:         # Within each segment we sample from a uniform random distribution.
428:         # We need to do this sampling for each parameter.
429:         samples = (segsize * rng.random_sample(self.population_shape)
430: 
431:         # Offset each segment to cover the entire parameter range [0, 1)
432:                    + np.linspace(0., 1., self.num_population_members,
433:                                  endpoint=False)[:, np.newaxis])
434: 
435:         # Create an array for population of candidate solutions.
436:         self.population = np.zeros_like(samples)
437: 
438:         # Initialize population of candidate solutions by permutation of the
439:         # random samples.
440:         for j in range(self.parameter_count):
441:             order = rng.permutation(range(self.num_population_members))
442:             self.population[:, j] = samples[order, j]
443: 
444:         # reset population energies
445:         self.population_energies = (np.ones(self.num_population_members) *
446:                                     np.inf)
447: 
448:         # reset number of function evaluations counter
449:         self._nfev = 0
450: 
451:     def init_population_random(self):
452:         '''
453:         Initialises the population at random.  This type of initialization
454:         can possess clustering, Latin Hypercube sampling is generally better.
455:         '''
456:         rng = self.random_number_generator
457:         self.population = rng.random_sample(self.population_shape)
458: 
459:         # reset population energies
460:         self.population_energies = (np.ones(self.num_population_members) *
461:                                     np.inf)
462: 
463:         # reset number of function evaluations counter
464:         self._nfev = 0
465: 
466:     @property
467:     def x(self):
468:         '''
469:         The best solution from the solver
470: 
471:         Returns
472:         -------
473:         x : ndarray
474:             The best solution from the solver.
475:         '''
476:         return self._scale_parameters(self.population[0])
477: 
478:     @property
479:     def convergence(self):
480:         '''
481:         The standard deviation of the population energies divided by their
482:         mean.
483:         '''
484:         return (np.std(self.population_energies) /
485:                 np.abs(np.mean(self.population_energies) + _MACHEPS))
486: 
487:     def solve(self):
488:         '''
489:         Runs the DifferentialEvolutionSolver.
490: 
491:         Returns
492:         -------
493:         res : OptimizeResult
494:             The optimization result represented as a ``OptimizeResult`` object.
495:             Important attributes are: ``x`` the solution array, ``success`` a
496:             Boolean flag indicating if the optimizer exited successfully and
497:             ``message`` which describes the cause of the termination. See
498:             `OptimizeResult` for a description of other attributes.  If `polish`
499:             was employed, and a lower minimum was obtained by the polishing,
500:             then OptimizeResult also contains the ``jac`` attribute.
501:         '''
502:         nit, warning_flag = 0, False
503:         status_message = _status_message['success']
504: 
505:         # The population may have just been initialized (all entries are
506:         # np.inf). If it has you have to calculate the initial energies.
507:         # Although this is also done in the evolve generator it's possible
508:         # that someone can set maxiter=0, at which point we still want the
509:         # initial energies to be calculated (the following loop isn't run).
510:         if np.all(np.isinf(self.population_energies)):
511:             self._calculate_population_energies()
512: 
513:         # do the optimisation.
514:         for nit in xrange(1, self.maxiter + 1):
515:             # evolve the population by a generation
516:             try:
517:                 next(self)
518:             except StopIteration:
519:                 warning_flag = True
520:                 status_message = _status_message['maxfev']
521:                 break
522: 
523:             if self.disp:
524:                 print("differential_evolution step %d: f(x)= %g"
525:                       % (nit,
526:                          self.population_energies[0]))
527: 
528:             # should the solver terminate?
529:             convergence = self.convergence
530: 
531:             if (self.callback and
532:                     self.callback(self._scale_parameters(self.population[0]),
533:                                   convergence=self.tol / convergence) is True):
534: 
535:                 warning_flag = True
536:                 status_message = ('callback function requested stop early '
537:                                   'by returning True')
538:                 break
539: 
540:             intol = (np.std(self.population_energies) <=
541:                      self.atol +
542:                      self.tol * np.abs(np.mean(self.population_energies)))
543:             if warning_flag or intol:
544:                 break
545: 
546:         else:
547:             status_message = _status_message['maxiter']
548:             warning_flag = True
549: 
550:         DE_result = OptimizeResult(
551:             x=self.x,
552:             fun=self.population_energies[0],
553:             nfev=self._nfev,
554:             nit=nit,
555:             message=status_message,
556:             success=(warning_flag is not True))
557: 
558:         if self.polish:
559:             result = minimize(self.func,
560:                               np.copy(DE_result.x),
561:                               method='L-BFGS-B',
562:                               bounds=self.limits.T,
563:                               args=self.args)
564: 
565:             self._nfev += result.nfev
566:             DE_result.nfev = self._nfev
567: 
568:             if result.fun < DE_result.fun:
569:                 DE_result.fun = result.fun
570:                 DE_result.x = result.x
571:                 DE_result.jac = result.jac
572:                 # to keep internal state consistent
573:                 self.population_energies[0] = result.fun
574:                 self.population[0] = self._unscale_parameters(result.x)
575: 
576:         return DE_result
577: 
578:     def _calculate_population_energies(self):
579:         '''
580:         Calculate the energies of all the population members at the same time.
581:         Puts the best member in first place. Useful if the population has just
582:         been initialised.
583:         '''
584:         for index, candidate in enumerate(self.population):
585:             if self._nfev > self.maxfun:
586:                 break
587: 
588:             parameters = self._scale_parameters(candidate)
589:             self.population_energies[index] = self.func(parameters,
590:                                                         *self.args)
591:             self._nfev += 1
592: 
593:         minval = np.argmin(self.population_energies)
594: 
595:         # put the lowest energy into the best solution position.
596:         lowest_energy = self.population_energies[minval]
597:         self.population_energies[minval] = self.population_energies[0]
598:         self.population_energies[0] = lowest_energy
599: 
600:         self.population[[0, minval], :] = self.population[[minval, 0], :]
601: 
602:     def __iter__(self):
603:         return self
604: 
605:     def __next__(self):
606:         '''
607:         Evolve the population by a single generation
608: 
609:         Returns
610:         -------
611:         x : ndarray
612:             The best solution from the solver.
613:         fun : float
614:             Value of objective function obtained from the best solution.
615:         '''
616:         # the population may have just been initialized (all entries are
617:         # np.inf). If it has you have to calculate the initial energies
618:         if np.all(np.isinf(self.population_energies)):
619:             self._calculate_population_energies()
620: 
621:         if self.dither is not None:
622:             self.scale = (self.random_number_generator.rand()
623:                           * (self.dither[1] - self.dither[0]) + self.dither[0])
624: 
625:         for candidate in range(self.num_population_members):
626:             if self._nfev > self.maxfun:
627:                 raise StopIteration
628: 
629:             # create a trial solution
630:             trial = self._mutate(candidate)
631: 
632:             # ensuring that it's in the range [0, 1)
633:             self._ensure_constraint(trial)
634: 
635:             # scale from [0, 1) to the actual parameter value
636:             parameters = self._scale_parameters(trial)
637: 
638:             # determine the energy of the objective function
639:             energy = self.func(parameters, *self.args)
640:             self._nfev += 1
641: 
642:             # if the energy of the trial candidate is lower than the
643:             # original population member then replace it
644:             if energy < self.population_energies[candidate]:
645:                 self.population[candidate] = trial
646:                 self.population_energies[candidate] = energy
647: 
648:                 # if the trial candidate also has a lower energy than the
649:                 # best solution then replace that as well
650:                 if energy < self.population_energies[0]:
651:                     self.population_energies[0] = energy
652:                     self.population[0] = trial
653: 
654:         return self.x, self.population_energies[0]
655: 
656:     def next(self):
657:         '''
658:         Evolve the population by a single generation
659: 
660:         Returns
661:         -------
662:         x : ndarray
663:             The best solution from the solver.
664:         fun : float
665:             Value of objective function obtained from the best solution.
666:         '''
667:         # next() is required for compatibility with Python2.7.
668:         return self.__next__()
669: 
670:     def _scale_parameters(self, trial):
671:         '''
672:         scale from a number between 0 and 1 to parameters.
673:         '''
674:         return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
675: 
676:     def _unscale_parameters(self, parameters):
677:         '''
678:         scale from parameters to a number between 0 and 1.
679:         '''
680:         return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5
681: 
682:     def _ensure_constraint(self, trial):
683:         '''
684:         make sure the parameters lie between the limits
685:         '''
686:         for index, param in enumerate(trial):
687:             if param > 1 or param < 0:
688:                 trial[index] = self.random_number_generator.rand()
689: 
690:     def _mutate(self, candidate):
691:         '''
692:         create a trial vector based on a mutation strategy
693:         '''
694:         trial = np.copy(self.population[candidate])
695: 
696:         rng = self.random_number_generator
697: 
698:         fill_point = rng.randint(0, self.parameter_count)
699: 
700:         if (self.strategy == 'randtobest1exp' or
701:                 self.strategy == 'randtobest1bin'):
702:             bprime = self.mutation_func(candidate,
703:                                         self._select_samples(candidate, 5))
704:         else:
705:             bprime = self.mutation_func(self._select_samples(candidate, 5))
706: 
707:         if self.strategy in self._binomial:
708:             crossovers = rng.rand(self.parameter_count)
709:             crossovers = crossovers < self.cross_over_probability
710:             # the last one is always from the bprime vector for binomial
711:             # If you fill in modulo with a loop you have to set the last one to
712:             # true. If you don't use a loop then you can have any random entry
713:             # be True.
714:             crossovers[fill_point] = True
715:             trial = np.where(crossovers, bprime, trial)
716:             return trial
717: 
718:         elif self.strategy in self._exponential:
719:             i = 0
720:             while (i < self.parameter_count and
721:                    rng.rand() < self.cross_over_probability):
722: 
723:                 trial[fill_point] = bprime[fill_point]
724:                 fill_point = (fill_point + 1) % self.parameter_count
725:                 i += 1
726: 
727:             return trial
728: 
729:     def _best1(self, samples):
730:         '''
731:         best1bin, best1exp
732:         '''
733:         r0, r1 = samples[:2]
734:         return (self.population[0] + self.scale *
735:                 (self.population[r0] - self.population[r1]))
736: 
737:     def _rand1(self, samples):
738:         '''
739:         rand1bin, rand1exp
740:         '''
741:         r0, r1, r2 = samples[:3]
742:         return (self.population[r0] + self.scale *
743:                 (self.population[r1] - self.population[r2]))
744: 
745:     def _randtobest1(self, candidate, samples):
746:         '''
747:         randtobest1bin, randtobest1exp
748:         '''
749:         r0, r1 = samples[:2]
750:         bprime = np.copy(self.population[candidate])
751:         bprime += self.scale * (self.population[0] - bprime)
752:         bprime += self.scale * (self.population[r0] -
753:                                 self.population[r1])
754:         return bprime
755: 
756:     def _best2(self, samples):
757:         '''
758:         best2bin, best2exp
759:         '''
760:         r0, r1, r2, r3 = samples[:4]
761:         bprime = (self.population[0] + self.scale *
762:                   (self.population[r0] + self.population[r1] -
763:                    self.population[r2] - self.population[r3]))
764: 
765:         return bprime
766: 
767:     def _rand2(self, samples):
768:         '''
769:         rand2bin, rand2exp
770:         '''
771:         r0, r1, r2, r3, r4 = samples
772:         bprime = (self.population[r0] + self.scale *
773:                   (self.population[r1] + self.population[r2] -
774:                    self.population[r3] - self.population[r4]))
775: 
776:         return bprime
777: 
778:     def _select_samples(self, candidate, number_samples):
779:         '''
780:         obtain random integers from range(self.num_population_members),
781:         without replacement.  You can't have the original candidate either.
782:         '''
783:         idxs = list(range(self.num_population_members))
784:         idxs.remove(candidate)
785:         self.random_number_generator.shuffle(idxs)
786:         idxs = idxs[:number_samples]
787:         return idxs
788: 
789: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_188402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\ndifferential_evolution: The differential evolution global optimization algorithm\nAdded by Andrew Nelson 2014\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_188403 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_188403) is not StypyTypeError):

    if (import_188403 != 'pyd_module'):
        __import__(import_188403)
        sys_modules_188404 = sys.modules[import_188403]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_188404.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_188403)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.optimize import OptimizeResult, minimize' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_188405 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize')

if (type(import_188405) is not StypyTypeError):

    if (import_188405 != 'pyd_module'):
        __import__(import_188405)
        sys_modules_188406 = sys.modules[import_188405]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize', sys_modules_188406.module_type_store, module_type_store, ['OptimizeResult', 'minimize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_188406, sys_modules_188406.module_type_store, module_type_store)
    else:
        from scipy.optimize import OptimizeResult, minimize

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize', None, module_type_store, ['OptimizeResult', 'minimize'], [OptimizeResult, minimize])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.optimize', import_188405)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize.optimize import _status_message' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_188407 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize.optimize')

if (type(import_188407) is not StypyTypeError):

    if (import_188407 != 'pyd_module'):
        __import__(import_188407)
        sys_modules_188408 = sys.modules[import_188407]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize.optimize', sys_modules_188408.module_type_store, module_type_store, ['_status_message'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_188408, sys_modules_188408.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import _status_message

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize.optimize', None, module_type_store, ['_status_message'], [_status_message])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize.optimize', import_188407)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy._lib._util import check_random_state' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_188409 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._util')

if (type(import_188409) is not StypyTypeError):

    if (import_188409 != 'pyd_module'):
        __import__(import_188409)
        sys_modules_188410 = sys.modules[import_188409]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._util', sys_modules_188410.module_type_store, module_type_store, ['check_random_state'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_188410, sys_modules_188410.module_type_store, module_type_store)
    else:
        from scipy._lib._util import check_random_state

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._util', None, module_type_store, ['check_random_state'], [check_random_state])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._util', import_188409)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib.six import xrange' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_188411 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six')

if (type(import_188411) is not StypyTypeError):

    if (import_188411 != 'pyd_module'):
        __import__(import_188411)
        sys_modules_188412 = sys.modules[import_188411]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', sys_modules_188412.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_188412, sys_modules_188412.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', import_188411)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import warnings' statement (line 11)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'warnings', warnings, module_type_store)


# Assigning a List to a Name (line 14):

# Assigning a List to a Name (line 14):
__all__ = ['differential_evolution']
module_type_store.set_exportable_members(['differential_evolution'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_188413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_188414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'differential_evolution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_188413, str_188414)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_188413)

# Assigning a Attribute to a Name (line 16):

# Assigning a Attribute to a Name (line 16):

# Call to finfo(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'np' (line 16)
np_188417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'np', False)
# Obtaining the member 'float64' of a type (line 16)
float64_188418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), np_188417, 'float64')
# Processing the call keyword arguments (line 16)
kwargs_188419 = {}
# Getting the type of 'np' (line 16)
np_188415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'np', False)
# Obtaining the member 'finfo' of a type (line 16)
finfo_188416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), np_188415, 'finfo')
# Calling finfo(args, kwargs) (line 16)
finfo_call_result_188420 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), finfo_188416, *[float64_188418], **kwargs_188419)

# Obtaining the member 'eps' of a type (line 16)
eps_188421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), finfo_call_result_188420, 'eps')
# Assigning a type to the variable '_MACHEPS' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '_MACHEPS', eps_188421)

@norecursion
def differential_evolution(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 19)
    tuple_188422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 19)
    
    str_188423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 59), 'str', 'best1bin')
    int_188424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'int')
    int_188425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 49), 'int')
    float_188426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 57), 'float')
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_188427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    float_188428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 37), tuple_188427, float_188428)
    # Adding element type (line 21)
    int_188429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 37), tuple_188427, int_188429)
    
    float_188430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 60), 'float')
    # Getting the type of 'None' (line 21)
    None_188431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 70), 'None')
    # Getting the type of 'None' (line 22)
    None_188432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 36), 'None')
    # Getting the type of 'False' (line 22)
    False_188433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 47), 'False')
    # Getting the type of 'True' (line 22)
    True_188434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 61), 'True')
    str_188435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', 'latinhypercube')
    int_188436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 55), 'int')
    defaults = [tuple_188422, str_188423, int_188424, int_188425, float_188426, tuple_188427, float_188430, None_188431, None_188432, False_188433, True_188434, str_188435, int_188436]
    # Create a new context for function 'differential_evolution'
    module_type_store = module_type_store.open_function_context('differential_evolution', 19, 0, False)
    
    # Passed parameters checking function
    differential_evolution.stypy_localization = localization
    differential_evolution.stypy_type_of_self = None
    differential_evolution.stypy_type_store = module_type_store
    differential_evolution.stypy_function_name = 'differential_evolution'
    differential_evolution.stypy_param_names_list = ['func', 'bounds', 'args', 'strategy', 'maxiter', 'popsize', 'tol', 'mutation', 'recombination', 'seed', 'callback', 'disp', 'polish', 'init', 'atol']
    differential_evolution.stypy_varargs_param_name = None
    differential_evolution.stypy_kwargs_param_name = None
    differential_evolution.stypy_call_defaults = defaults
    differential_evolution.stypy_call_varargs = varargs
    differential_evolution.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'differential_evolution', ['func', 'bounds', 'args', 'strategy', 'maxiter', 'popsize', 'tol', 'mutation', 'recombination', 'seed', 'callback', 'disp', 'polish', 'init', 'atol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'differential_evolution', localization, ['func', 'bounds', 'args', 'strategy', 'maxiter', 'popsize', 'tol', 'mutation', 'recombination', 'seed', 'callback', 'disp', 'polish', 'init', 'atol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'differential_evolution(...)' code ##################

    str_188437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, (-1)), 'str', "Finds the global minimum of a multivariate function.\n    Differential Evolution is stochastic in nature (does not use gradient\n    methods) to find the minimium, and can search large areas of candidate\n    space, but often requires larger numbers of function evaluations than\n    conventional gradient based techniques.\n\n    The algorithm is due to Storn and Price [1]_.\n\n    Parameters\n    ----------\n    func : callable\n        The objective function to be minimized.  Must be in the form\n        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array\n        and ``args`` is a  tuple of any additional fixed parameters needed to\n        completely specify the function.\n    bounds : sequence\n        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,\n        defining the lower and upper bounds for the optimizing argument of\n        `func`. It is required to have ``len(bounds) == len(x)``.\n        ``len(bounds)`` is used to determine the number of parameters in ``x``.\n    args : tuple, optional\n        Any additional fixed parameters needed to\n        completely specify the objective function.\n    strategy : str, optional\n        The differential evolution strategy to use. Should be one of:\n\n            - 'best1bin'\n            - 'best1exp'\n            - 'rand1exp'\n            - 'randtobest1exp'\n            - 'best2exp'\n            - 'rand2exp'\n            - 'randtobest1bin'\n            - 'best2bin'\n            - 'rand2bin'\n            - 'rand1bin'\n\n        The default is 'best1bin'.\n    maxiter : int, optional\n        The maximum number of generations over which the entire population is\n        evolved. The maximum number of function evaluations (with no polishing)\n        is: ``(maxiter + 1) * popsize * len(x)``\n    popsize : int, optional\n        A multiplier for setting the total population size.  The population has\n        ``popsize * len(x)`` individuals.\n    tol : float, optional\n        Relative tolerance for convergence, the solving stops when\n        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,\n        where and `atol` and `tol` are the absolute and relative tolerance\n        respectively.\n    mutation : float or tuple(float, float), optional\n        The mutation constant. In the literature this is also known as\n        differential weight, being denoted by F.\n        If specified as a float it should be in the range [0, 2].\n        If specified as a tuple ``(min, max)`` dithering is employed. Dithering\n        randomly changes the mutation constant on a generation by generation\n        basis. The mutation constant for that generation is taken from\n        ``U[min, max)``. Dithering can help speed convergence significantly.\n        Increasing the mutation constant increases the search radius, but will\n        slow down convergence.\n    recombination : float, optional\n        The recombination constant, should be in the range [0, 1]. In the\n        literature this is also known as the crossover probability, being\n        denoted by CR. Increasing this value allows a larger number of mutants\n        to progress into the next generation, but at the risk of population\n        stability.\n    seed : int or `np.random.RandomState`, optional\n        If `seed` is not specified the `np.RandomState` singleton is used.\n        If `seed` is an int, a new `np.random.RandomState` instance is used,\n        seeded with seed.\n        If `seed` is already a `np.random.RandomState instance`, then that\n        `np.random.RandomState` instance is used.\n        Specify `seed` for repeatable minimizations.\n    disp : bool, optional\n        Display status messages\n    callback : callable, `callback(xk, convergence=val)`, optional\n        A function to follow the progress of the minimization. ``xk`` is\n        the current value of ``x0``. ``val`` represents the fractional\n        value of the population convergence.  When ``val`` is greater than one\n        the function halts. If callback returns `True`, then the minimization\n        is halted (any polishing is still carried out).\n    polish : bool, optional\n        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`\n        method is used to polish the best population member at the end, which\n        can improve the minimization slightly.\n    init : string, optional\n        Specify how the population initialization is performed. Should be\n        one of:\n\n            - 'latinhypercube'\n            - 'random'\n\n        The default is 'latinhypercube'. Latin Hypercube sampling tries to\n        maximize coverage of the available parameter space. 'random' initializes\n        the population randomly - this has the drawback that clustering can\n        occur, preventing the whole of parameter space being covered.\n    atol : float, optional\n        Absolute tolerance for convergence, the solving stops when\n        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,\n        where and `atol` and `tol` are the absolute and relative tolerance\n        respectively.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a `OptimizeResult` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.  If `polish`\n        was employed, and a lower minimum was obtained by the polishing, then\n        OptimizeResult also contains the ``jac`` attribute.\n\n    Notes\n    -----\n    Differential evolution is a stochastic population based method that is\n    useful for global optimization problems. At each pass through the population\n    the algorithm mutates each candidate solution by mixing with other candidate\n    solutions to create a trial candidate. There are several strategies [2]_ for\n    creating trial candidates, which suit some problems more than others. The\n    'best1bin' strategy is a good starting point for many systems. In this\n    strategy two members of the population are randomly chosen. Their difference\n    is used to mutate the best member (the `best` in `best1bin`), :math:`b_0`,\n    so far:\n\n    .. math::\n\n        b' = b_0 + mutation * (population[rand0] - population[rand1])\n\n    A trial vector is then constructed. Starting with a randomly chosen 'i'th\n    parameter the trial is sequentially filled (in modulo) with parameters from\n    `b'` or the original candidate. The choice of whether to use `b'` or the\n    original candidate is made with a binomial distribution (the 'bin' in\n    'best1bin') - a random number in [0, 1) is generated.  If this number is\n    less than the `recombination` constant then the parameter is loaded from\n    `b'`, otherwise it is loaded from the original candidate.  The final\n    parameter is always loaded from `b'`.  Once the trial candidate is built\n    its fitness is assessed. If the trial is better than the original candidate\n    then it takes its place. If it is also better than the best overall\n    candidate it also replaces that.\n    To improve your chances of finding a global minimum use higher `popsize`\n    values, with higher `mutation` and (dithering), but lower `recombination`\n    values. This has the effect of widening the search radius, but slowing\n    convergence.\n\n    .. versionadded:: 0.15.0\n\n    Examples\n    --------\n    Let us consider the problem of minimizing the Rosenbrock function. This\n    function is implemented in `rosen` in `scipy.optimize`.\n\n    >>> from scipy.optimize import rosen, differential_evolution\n    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]\n    >>> result = differential_evolution(rosen, bounds)\n    >>> result.x, result.fun\n    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)\n\n    Next find the minimum of the Ackley function\n    (http://en.wikipedia.org/wiki/Test_functions_for_optimization).\n\n    >>> from scipy.optimize import differential_evolution\n    >>> import numpy as np\n    >>> def ackley(x):\n    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))\n    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))\n    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e\n    >>> bounds = [(-5, 5), (-5, 5)]\n    >>> result = differential_evolution(ackley, bounds)\n    >>> result.x, result.fun\n    (array([ 0.,  0.]), 4.4408920985006262e-16)\n\n    References\n    ----------\n    .. [1] Storn, R and Price, K, Differential Evolution - a Simple and\n           Efficient Heuristic for Global Optimization over Continuous Spaces,\n           Journal of Global Optimization, 1997, 11, 341 - 359.\n    .. [2] http://www1.icsi.berkeley.edu/~storn/code.html\n    .. [3] http://en.wikipedia.org/wiki/Differential_evolution\n    ")
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to DifferentialEvolutionSolver(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'func' (line 205)
    func_188439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 41), 'func', False)
    # Getting the type of 'bounds' (line 205)
    bounds_188440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 47), 'bounds', False)
    # Processing the call keyword arguments (line 205)
    # Getting the type of 'args' (line 205)
    args_188441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 60), 'args', False)
    keyword_188442 = args_188441
    # Getting the type of 'strategy' (line 206)
    strategy_188443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 50), 'strategy', False)
    keyword_188444 = strategy_188443
    # Getting the type of 'maxiter' (line 206)
    maxiter_188445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 68), 'maxiter', False)
    keyword_188446 = maxiter_188445
    # Getting the type of 'popsize' (line 207)
    popsize_188447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 49), 'popsize', False)
    keyword_188448 = popsize_188447
    # Getting the type of 'tol' (line 207)
    tol_188449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 62), 'tol', False)
    keyword_188450 = tol_188449
    # Getting the type of 'mutation' (line 208)
    mutation_188451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 50), 'mutation', False)
    keyword_188452 = mutation_188451
    # Getting the type of 'recombination' (line 209)
    recombination_188453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 55), 'recombination', False)
    keyword_188454 = recombination_188453
    # Getting the type of 'seed' (line 210)
    seed_188455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 46), 'seed', False)
    keyword_188456 = seed_188455
    # Getting the type of 'polish' (line 210)
    polish_188457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 59), 'polish', False)
    keyword_188458 = polish_188457
    # Getting the type of 'callback' (line 211)
    callback_188459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 50), 'callback', False)
    keyword_188460 = callback_188459
    # Getting the type of 'disp' (line 212)
    disp_188461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'disp', False)
    keyword_188462 = disp_188461
    # Getting the type of 'init' (line 212)
    init_188463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 57), 'init', False)
    keyword_188464 = init_188463
    # Getting the type of 'atol' (line 212)
    atol_188465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 68), 'atol', False)
    keyword_188466 = atol_188465
    kwargs_188467 = {'disp': keyword_188462, 'atol': keyword_188466, 'recombination': keyword_188454, 'args': keyword_188442, 'popsize': keyword_188448, 'strategy': keyword_188444, 'callback': keyword_188460, 'init': keyword_188464, 'seed': keyword_188456, 'mutation': keyword_188452, 'tol': keyword_188450, 'maxiter': keyword_188446, 'polish': keyword_188458}
    # Getting the type of 'DifferentialEvolutionSolver' (line 205)
    DifferentialEvolutionSolver_188438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'DifferentialEvolutionSolver', False)
    # Calling DifferentialEvolutionSolver(args, kwargs) (line 205)
    DifferentialEvolutionSolver_call_result_188468 = invoke(stypy.reporting.localization.Localization(__file__, 205, 13), DifferentialEvolutionSolver_188438, *[func_188439, bounds_188440], **kwargs_188467)
    
    # Assigning a type to the variable 'solver' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'solver', DifferentialEvolutionSolver_call_result_188468)
    
    # Call to solve(...): (line 213)
    # Processing the call keyword arguments (line 213)
    kwargs_188471 = {}
    # Getting the type of 'solver' (line 213)
    solver_188469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'solver', False)
    # Obtaining the member 'solve' of a type (line 213)
    solve_188470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), solver_188469, 'solve')
    # Calling solve(args, kwargs) (line 213)
    solve_call_result_188472 = invoke(stypy.reporting.localization.Localization(__file__, 213, 11), solve_188470, *[], **kwargs_188471)
    
    # Assigning a type to the variable 'stypy_return_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type', solve_call_result_188472)
    
    # ################# End of 'differential_evolution(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'differential_evolution' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_188473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_188473)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'differential_evolution'
    return stypy_return_type_188473

# Assigning a type to the variable 'differential_evolution' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'differential_evolution', differential_evolution)
# Declaration of the 'DifferentialEvolutionSolver' class

class DifferentialEvolutionSolver(object, ):
    str_188474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, (-1)), 'str', "This class implements the differential evolution solver\n\n    Parameters\n    ----------\n    func : callable\n        The objective function to be minimized.  Must be in the form\n        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array\n        and ``args`` is a  tuple of any additional fixed parameters needed to\n        completely specify the function.\n    bounds : sequence\n        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,\n        defining the lower and upper bounds for the optimizing argument of\n        `func`. It is required to have ``len(bounds) == len(x)``.\n        ``len(bounds)`` is used to determine the number of parameters in ``x``.\n    args : tuple, optional\n        Any additional fixed parameters needed to\n        completely specify the objective function.\n    strategy : str, optional\n        The differential evolution strategy to use. Should be one of:\n\n            - 'best1bin'\n            - 'best1exp'\n            - 'rand1exp'\n            - 'randtobest1exp'\n            - 'best2exp'\n            - 'rand2exp'\n            - 'randtobest1bin'\n            - 'best2bin'\n            - 'rand2bin'\n            - 'rand1bin'\n\n        The default is 'best1bin'\n\n    maxiter : int, optional\n        The maximum number of generations over which the entire population is\n        evolved. The maximum number of function evaluations (with no polishing)\n        is: ``(maxiter + 1) * popsize * len(x)``\n    popsize : int, optional\n        A multiplier for setting the total population size.  The population has\n        ``popsize * len(x)`` individuals.\n    tol : float, optional\n        Relative tolerance for convergence, the solving stops when\n        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,\n        where and `atol` and `tol` are the absolute and relative tolerance\n        respectively.\n    mutation : float or tuple(float, float), optional\n        The mutation constant. In the literature this is also known as\n        differential weight, being denoted by F.\n        If specified as a float it should be in the range [0, 2].\n        If specified as a tuple ``(min, max)`` dithering is employed. Dithering\n        randomly changes the mutation constant on a generation by generation\n        basis. The mutation constant for that generation is taken from\n        U[min, max). Dithering can help speed convergence significantly.\n        Increasing the mutation constant increases the search radius, but will\n        slow down convergence.\n    recombination : float, optional\n        The recombination constant, should be in the range [0, 1]. In the\n        literature this is also known as the crossover probability, being\n        denoted by CR. Increasing this value allows a larger number of mutants\n        to progress into the next generation, but at the risk of population\n        stability.\n    seed : int or `np.random.RandomState`, optional\n        If `seed` is not specified the `np.random.RandomState` singleton is\n        used.\n        If `seed` is an int, a new `np.random.RandomState` instance is used,\n        seeded with `seed`.\n        If `seed` is already a `np.random.RandomState` instance, then that\n        `np.random.RandomState` instance is used.\n        Specify `seed` for repeatable minimizations.\n    disp : bool, optional\n        Display status messages\n    callback : callable, `callback(xk, convergence=val)`, optional\n        A function to follow the progress of the minimization. ``xk`` is\n        the current value of ``x0``. ``val`` represents the fractional\n        value of the population convergence.  When ``val`` is greater than one\n        the function halts. If callback returns `True`, then the minimization\n        is halted (any polishing is still carried out).\n    polish : bool, optional\n        If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method\n        is used to polish the best population member at the end. This requires\n        a few more function evaluations.\n    maxfun : int, optional\n        Set the maximum number of function evaluations. However, it probably\n        makes more sense to set `maxiter` instead.\n    init : string, optional\n        Specify which type of population initialization is performed. Should be\n        one of:\n\n            - 'latinhypercube'\n            - 'random'\n    atol : float, optional\n        Absolute tolerance for convergence, the solving stops when\n        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,\n        where and `atol` and `tol` are the absolute and relative tolerance\n        respectively.\n    ")
    
    # Assigning a Dict to a Name (line 316):
    
    # Assigning a Dict to a Name (line 321):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 327)
        tuple_188475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 327)
        
        str_188476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 26), 'str', 'best1bin')
        int_188477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 46), 'int')
        int_188478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 60), 'int')
        float_188479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 21), 'float')
        
        # Obtaining an instance of the builtin type 'tuple' (line 329)
        tuple_188480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 329)
        # Adding element type (line 329)
        float_188481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 37), tuple_188480, float_188481)
        # Adding element type (line 329)
        int_188482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 37), tuple_188480, int_188482)
        
        float_188483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 60), 'float')
        # Getting the type of 'None' (line 329)
        None_188484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 70), 'None')
        # Getting the type of 'np' (line 330)
        np_188485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'np')
        # Obtaining the member 'inf' of a type (line 330)
        inf_188486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 24), np_188485, 'inf')
        # Getting the type of 'None' (line 330)
        None_188487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 41), 'None')
        # Getting the type of 'False' (line 330)
        False_188488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 52), 'False')
        # Getting the type of 'True' (line 330)
        True_188489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 66), 'True')
        str_188490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 22), 'str', 'latinhypercube')
        int_188491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 45), 'int')
        defaults = [tuple_188475, str_188476, int_188477, int_188478, float_188479, tuple_188480, float_188483, None_188484, inf_188486, None_188487, False_188488, True_188489, str_188490, int_188491]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.__init__', ['func', 'bounds', 'args', 'strategy', 'maxiter', 'popsize', 'tol', 'mutation', 'recombination', 'seed', 'maxfun', 'callback', 'disp', 'polish', 'init', 'atol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['func', 'bounds', 'args', 'strategy', 'maxiter', 'popsize', 'tol', 'mutation', 'recombination', 'seed', 'maxfun', 'callback', 'disp', 'polish', 'init', 'atol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Getting the type of 'strategy' (line 333)
        strategy_188492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'strategy')
        # Getting the type of 'self' (line 333)
        self_188493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'self')
        # Obtaining the member '_binomial' of a type (line 333)
        _binomial_188494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 23), self_188493, '_binomial')
        # Applying the binary operator 'in' (line 333)
        result_contains_188495 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 11), 'in', strategy_188492, _binomial_188494)
        
        # Testing the type of an if condition (line 333)
        if_condition_188496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), result_contains_188495)
        # Assigning a type to the variable 'if_condition_188496' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_188496', if_condition_188496)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 334):
        
        # Assigning a Call to a Attribute (line 334):
        
        # Call to getattr(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'self' (line 334)
        self_188498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 41), 'self', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'strategy' (line 334)
        strategy_188499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 62), 'strategy', False)
        # Getting the type of 'self' (line 334)
        self_188500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 47), 'self', False)
        # Obtaining the member '_binomial' of a type (line 334)
        _binomial_188501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 47), self_188500, '_binomial')
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___188502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 47), _binomial_188501, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_188503 = invoke(stypy.reporting.localization.Localization(__file__, 334, 47), getitem___188502, strategy_188499)
        
        # Processing the call keyword arguments (line 334)
        kwargs_188504 = {}
        # Getting the type of 'getattr' (line 334)
        getattr_188497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 33), 'getattr', False)
        # Calling getattr(args, kwargs) (line 334)
        getattr_call_result_188505 = invoke(stypy.reporting.localization.Localization(__file__, 334, 33), getattr_188497, *[self_188498, subscript_call_result_188503], **kwargs_188504)
        
        # Getting the type of 'self' (line 334)
        self_188506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'self')
        # Setting the type of the member 'mutation_func' of a type (line 334)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), self_188506, 'mutation_func', getattr_call_result_188505)
        # SSA branch for the else part of an if statement (line 333)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'strategy' (line 335)
        strategy_188507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 13), 'strategy')
        # Getting the type of 'self' (line 335)
        self_188508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'self')
        # Obtaining the member '_exponential' of a type (line 335)
        _exponential_188509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 25), self_188508, '_exponential')
        # Applying the binary operator 'in' (line 335)
        result_contains_188510 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 13), 'in', strategy_188507, _exponential_188509)
        
        # Testing the type of an if condition (line 335)
        if_condition_188511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 13), result_contains_188510)
        # Assigning a type to the variable 'if_condition_188511' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 13), 'if_condition_188511', if_condition_188511)
        # SSA begins for if statement (line 335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 336):
        
        # Assigning a Call to a Attribute (line 336):
        
        # Call to getattr(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'self' (line 336)
        self_188513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 41), 'self', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'strategy' (line 336)
        strategy_188514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 65), 'strategy', False)
        # Getting the type of 'self' (line 336)
        self_188515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 47), 'self', False)
        # Obtaining the member '_exponential' of a type (line 336)
        _exponential_188516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 47), self_188515, '_exponential')
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___188517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 47), _exponential_188516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_188518 = invoke(stypy.reporting.localization.Localization(__file__, 336, 47), getitem___188517, strategy_188514)
        
        # Processing the call keyword arguments (line 336)
        kwargs_188519 = {}
        # Getting the type of 'getattr' (line 336)
        getattr_188512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 33), 'getattr', False)
        # Calling getattr(args, kwargs) (line 336)
        getattr_call_result_188520 = invoke(stypy.reporting.localization.Localization(__file__, 336, 33), getattr_188512, *[self_188513, subscript_call_result_188518], **kwargs_188519)
        
        # Getting the type of 'self' (line 336)
        self_188521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self')
        # Setting the type of the member 'mutation_func' of a type (line 336)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), self_188521, 'mutation_func', getattr_call_result_188520)
        # SSA branch for the else part of an if statement (line 335)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 338)
        # Processing the call arguments (line 338)
        str_188523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 29), 'str', 'Please select a valid mutation strategy')
        # Processing the call keyword arguments (line 338)
        kwargs_188524 = {}
        # Getting the type of 'ValueError' (line 338)
        ValueError_188522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 338)
        ValueError_call_result_188525 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), ValueError_188522, *[str_188523], **kwargs_188524)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 338, 12), ValueError_call_result_188525, 'raise parameter', BaseException)
        # SSA join for if statement (line 335)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 339):
        
        # Assigning a Name to a Attribute (line 339):
        # Getting the type of 'strategy' (line 339)
        strategy_188526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'strategy')
        # Getting the type of 'self' (line 339)
        self_188527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'self')
        # Setting the type of the member 'strategy' of a type (line 339)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), self_188527, 'strategy', strategy_188526)
        
        # Assigning a Name to a Attribute (line 341):
        
        # Assigning a Name to a Attribute (line 341):
        # Getting the type of 'callback' (line 341)
        callback_188528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'callback')
        # Getting the type of 'self' (line 341)
        self_188529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self')
        # Setting the type of the member 'callback' of a type (line 341)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_188529, 'callback', callback_188528)
        
        # Assigning a Name to a Attribute (line 342):
        
        # Assigning a Name to a Attribute (line 342):
        # Getting the type of 'polish' (line 342)
        polish_188530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 22), 'polish')
        # Getting the type of 'self' (line 342)
        self_188531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self')
        # Setting the type of the member 'polish' of a type (line 342)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_188531, 'polish', polish_188530)
        
        # Assigning a Tuple to a Tuple (line 345):
        
        # Assigning a Name to a Name (line 345):
        # Getting the type of 'tol' (line 345)
        tol_188532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 30), 'tol')
        # Assigning a type to the variable 'tuple_assignment_188382' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_assignment_188382', tol_188532)
        
        # Assigning a Name to a Name (line 345):
        # Getting the type of 'atol' (line 345)
        atol_188533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 35), 'atol')
        # Assigning a type to the variable 'tuple_assignment_188383' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_assignment_188383', atol_188533)
        
        # Assigning a Name to a Attribute (line 345):
        # Getting the type of 'tuple_assignment_188382' (line 345)
        tuple_assignment_188382_188534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_assignment_188382')
        # Getting the type of 'self' (line 345)
        self_188535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self')
        # Setting the type of the member 'tol' of a type (line 345)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_188535, 'tol', tuple_assignment_188382_188534)
        
        # Assigning a Name to a Attribute (line 345):
        # Getting the type of 'tuple_assignment_188383' (line 345)
        tuple_assignment_188383_188536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_assignment_188383')
        # Getting the type of 'self' (line 345)
        self_188537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 18), 'self')
        # Setting the type of the member 'atol' of a type (line 345)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 18), self_188537, 'atol', tuple_assignment_188383_188536)
        
        # Assigning a Name to a Attribute (line 349):
        
        # Assigning a Name to a Attribute (line 349):
        # Getting the type of 'mutation' (line 349)
        mutation_188538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'mutation')
        # Getting the type of 'self' (line 349)
        self_188539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'self')
        # Setting the type of the member 'scale' of a type (line 349)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), self_188539, 'scale', mutation_188538)
        
        
        # Evaluating a boolean operation
        
        
        # Call to all(...): (line 350)
        # Processing the call arguments (line 350)
        
        # Call to isfinite(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'mutation' (line 350)
        mutation_188544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 35), 'mutation', False)
        # Processing the call keyword arguments (line 350)
        kwargs_188545 = {}
        # Getting the type of 'np' (line 350)
        np_188542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 23), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 350)
        isfinite_188543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 23), np_188542, 'isfinite')
        # Calling isfinite(args, kwargs) (line 350)
        isfinite_call_result_188546 = invoke(stypy.reporting.localization.Localization(__file__, 350, 23), isfinite_188543, *[mutation_188544], **kwargs_188545)
        
        # Processing the call keyword arguments (line 350)
        kwargs_188547 = {}
        # Getting the type of 'np' (line 350)
        np_188540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 350)
        all_188541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 16), np_188540, 'all')
        # Calling all(args, kwargs) (line 350)
        all_call_result_188548 = invoke(stypy.reporting.localization.Localization(__file__, 350, 16), all_188541, *[isfinite_call_result_188546], **kwargs_188547)
        
        # Applying the 'not' unary operator (line 350)
        result_not__188549 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 12), 'not', all_call_result_188548)
        
        
        # Call to any(...): (line 351)
        # Processing the call arguments (line 351)
        
        
        # Call to array(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'mutation' (line 351)
        mutation_188554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 32), 'mutation', False)
        # Processing the call keyword arguments (line 351)
        kwargs_188555 = {}
        # Getting the type of 'np' (line 351)
        np_188552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 351)
        array_188553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 23), np_188552, 'array')
        # Calling array(args, kwargs) (line 351)
        array_call_result_188556 = invoke(stypy.reporting.localization.Localization(__file__, 351, 23), array_188553, *[mutation_188554], **kwargs_188555)
        
        int_188557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 45), 'int')
        # Applying the binary operator '>=' (line 351)
        result_ge_188558 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 23), '>=', array_call_result_188556, int_188557)
        
        # Processing the call keyword arguments (line 351)
        kwargs_188559 = {}
        # Getting the type of 'np' (line 351)
        np_188550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'np', False)
        # Obtaining the member 'any' of a type (line 351)
        any_188551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 16), np_188550, 'any')
        # Calling any(args, kwargs) (line 351)
        any_call_result_188560 = invoke(stypy.reporting.localization.Localization(__file__, 351, 16), any_188551, *[result_ge_188558], **kwargs_188559)
        
        # Applying the binary operator 'or' (line 350)
        result_or_keyword_188561 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 12), 'or', result_not__188549, any_call_result_188560)
        
        # Call to any(...): (line 352)
        # Processing the call arguments (line 352)
        
        
        # Call to array(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'mutation' (line 352)
        mutation_188566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 32), 'mutation', False)
        # Processing the call keyword arguments (line 352)
        kwargs_188567 = {}
        # Getting the type of 'np' (line 352)
        np_188564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 352)
        array_188565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 23), np_188564, 'array')
        # Calling array(args, kwargs) (line 352)
        array_call_result_188568 = invoke(stypy.reporting.localization.Localization(__file__, 352, 23), array_188565, *[mutation_188566], **kwargs_188567)
        
        int_188569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 44), 'int')
        # Applying the binary operator '<' (line 352)
        result_lt_188570 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 23), '<', array_call_result_188568, int_188569)
        
        # Processing the call keyword arguments (line 352)
        kwargs_188571 = {}
        # Getting the type of 'np' (line 352)
        np_188562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'np', False)
        # Obtaining the member 'any' of a type (line 352)
        any_188563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 16), np_188562, 'any')
        # Calling any(args, kwargs) (line 352)
        any_call_result_188572 = invoke(stypy.reporting.localization.Localization(__file__, 352, 16), any_188563, *[result_lt_188570], **kwargs_188571)
        
        # Applying the binary operator 'or' (line 350)
        result_or_keyword_188573 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 12), 'or', result_or_keyword_188561, any_call_result_188572)
        
        # Testing the type of an if condition (line 350)
        if_condition_188574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 8), result_or_keyword_188573)
        # Assigning a type to the variable 'if_condition_188574' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'if_condition_188574', if_condition_188574)
        # SSA begins for if statement (line 350)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 353)
        # Processing the call arguments (line 353)
        str_188576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 29), 'str', 'The mutation constant must be a float in U[0, 2), or specified as a tuple(min, max) where min < max and min, max are in U[0, 2).')
        # Processing the call keyword arguments (line 353)
        kwargs_188577 = {}
        # Getting the type of 'ValueError' (line 353)
        ValueError_188575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 353)
        ValueError_call_result_188578 = invoke(stypy.reporting.localization.Localization(__file__, 353, 18), ValueError_188575, *[str_188576], **kwargs_188577)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 353, 12), ValueError_call_result_188578, 'raise parameter', BaseException)
        # SSA join for if statement (line 350)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 357):
        
        # Assigning a Name to a Attribute (line 357):
        # Getting the type of 'None' (line 357)
        None_188579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'None')
        # Getting the type of 'self' (line 357)
        self_188580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'self')
        # Setting the type of the member 'dither' of a type (line 357)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), self_188580, 'dither', None_188579)
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'mutation' (line 358)
        mutation_188582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 19), 'mutation', False)
        str_188583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 29), 'str', '__iter__')
        # Processing the call keyword arguments (line 358)
        kwargs_188584 = {}
        # Getting the type of 'hasattr' (line 358)
        hasattr_188581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 358)
        hasattr_call_result_188585 = invoke(stypy.reporting.localization.Localization(__file__, 358, 11), hasattr_188581, *[mutation_188582, str_188583], **kwargs_188584)
        
        
        
        # Call to len(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'mutation' (line 358)
        mutation_188587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 49), 'mutation', False)
        # Processing the call keyword arguments (line 358)
        kwargs_188588 = {}
        # Getting the type of 'len' (line 358)
        len_188586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 45), 'len', False)
        # Calling len(args, kwargs) (line 358)
        len_call_result_188589 = invoke(stypy.reporting.localization.Localization(__file__, 358, 45), len_188586, *[mutation_188587], **kwargs_188588)
        
        int_188590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 61), 'int')
        # Applying the binary operator '>' (line 358)
        result_gt_188591 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 45), '>', len_call_result_188589, int_188590)
        
        # Applying the binary operator 'and' (line 358)
        result_and_keyword_188592 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 11), 'and', hasattr_call_result_188585, result_gt_188591)
        
        # Testing the type of an if condition (line 358)
        if_condition_188593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 358, 8), result_and_keyword_188592)
        # Assigning a type to the variable 'if_condition_188593' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'if_condition_188593', if_condition_188593)
        # SSA begins for if statement (line 358)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 359):
        
        # Assigning a List to a Attribute (line 359):
        
        # Obtaining an instance of the builtin type 'list' (line 359)
        list_188594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 359)
        # Adding element type (line 359)
        
        # Obtaining the type of the subscript
        int_188595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 36), 'int')
        # Getting the type of 'mutation' (line 359)
        mutation_188596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 27), 'mutation')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___188597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 27), mutation_188596, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_188598 = invoke(stypy.reporting.localization.Localization(__file__, 359, 27), getitem___188597, int_188595)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 26), list_188594, subscript_call_result_188598)
        # Adding element type (line 359)
        
        # Obtaining the type of the subscript
        int_188599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 49), 'int')
        # Getting the type of 'mutation' (line 359)
        mutation_188600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'mutation')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___188601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 40), mutation_188600, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_188602 = invoke(stypy.reporting.localization.Localization(__file__, 359, 40), getitem___188601, int_188599)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 26), list_188594, subscript_call_result_188602)
        
        # Getting the type of 'self' (line 359)
        self_188603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'self')
        # Setting the type of the member 'dither' of a type (line 359)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), self_188603, 'dither', list_188594)
        
        # Call to sort(...): (line 360)
        # Processing the call keyword arguments (line 360)
        kwargs_188607 = {}
        # Getting the type of 'self' (line 360)
        self_188604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self', False)
        # Obtaining the member 'dither' of a type (line 360)
        dither_188605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_188604, 'dither')
        # Obtaining the member 'sort' of a type (line 360)
        sort_188606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), dither_188605, 'sort')
        # Calling sort(args, kwargs) (line 360)
        sort_call_result_188608 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), sort_188606, *[], **kwargs_188607)
        
        # SSA join for if statement (line 358)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 362):
        
        # Assigning a Name to a Attribute (line 362):
        # Getting the type of 'recombination' (line 362)
        recombination_188609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 38), 'recombination')
        # Getting the type of 'self' (line 362)
        self_188610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'self')
        # Setting the type of the member 'cross_over_probability' of a type (line 362)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), self_188610, 'cross_over_probability', recombination_188609)
        
        # Assigning a Name to a Attribute (line 364):
        
        # Assigning a Name to a Attribute (line 364):
        # Getting the type of 'func' (line 364)
        func_188611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'func')
        # Getting the type of 'self' (line 364)
        self_188612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'self')
        # Setting the type of the member 'func' of a type (line 364)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 8), self_188612, 'func', func_188611)
        
        # Assigning a Name to a Attribute (line 365):
        
        # Assigning a Name to a Attribute (line 365):
        # Getting the type of 'args' (line 365)
        args_188613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 20), 'args')
        # Getting the type of 'self' (line 365)
        self_188614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'self')
        # Setting the type of the member 'args' of a type (line 365)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), self_188614, 'args', args_188613)
        
        # Assigning a Attribute to a Attribute (line 370):
        
        # Assigning a Attribute to a Attribute (line 370):
        
        # Call to array(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'bounds' (line 370)
        bounds_188617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 31), 'bounds', False)
        # Processing the call keyword arguments (line 370)
        str_188618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 45), 'str', 'float')
        keyword_188619 = str_188618
        kwargs_188620 = {'dtype': keyword_188619}
        # Getting the type of 'np' (line 370)
        np_188615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 370)
        array_188616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 22), np_188615, 'array')
        # Calling array(args, kwargs) (line 370)
        array_call_result_188621 = invoke(stypy.reporting.localization.Localization(__file__, 370, 22), array_188616, *[bounds_188617], **kwargs_188620)
        
        # Obtaining the member 'T' of a type (line 370)
        T_188622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 22), array_call_result_188621, 'T')
        # Getting the type of 'self' (line 370)
        self_188623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'self')
        # Setting the type of the member 'limits' of a type (line 370)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), self_188623, 'limits', T_188622)
        
        
        # Evaluating a boolean operation
        
        
        # Call to size(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'self' (line 371)
        self_188626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 20), 'self', False)
        # Obtaining the member 'limits' of a type (line 371)
        limits_188627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 20), self_188626, 'limits')
        int_188628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 33), 'int')
        # Processing the call keyword arguments (line 371)
        kwargs_188629 = {}
        # Getting the type of 'np' (line 371)
        np_188624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'np', False)
        # Obtaining the member 'size' of a type (line 371)
        size_188625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 12), np_188624, 'size')
        # Calling size(args, kwargs) (line 371)
        size_call_result_188630 = invoke(stypy.reporting.localization.Localization(__file__, 371, 12), size_188625, *[limits_188627, int_188628], **kwargs_188629)
        
        int_188631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 39), 'int')
        # Applying the binary operator '!=' (line 371)
        result_ne_188632 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 12), '!=', size_call_result_188630, int_188631)
        
        
        
        # Call to all(...): (line 372)
        # Processing the call arguments (line 372)
        
        # Call to isfinite(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'self' (line 372)
        self_188637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 35), 'self', False)
        # Obtaining the member 'limits' of a type (line 372)
        limits_188638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 35), self_188637, 'limits')
        # Processing the call keyword arguments (line 372)
        kwargs_188639 = {}
        # Getting the type of 'np' (line 372)
        np_188635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 372)
        isfinite_188636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), np_188635, 'isfinite')
        # Calling isfinite(args, kwargs) (line 372)
        isfinite_call_result_188640 = invoke(stypy.reporting.localization.Localization(__file__, 372, 23), isfinite_188636, *[limits_188638], **kwargs_188639)
        
        # Processing the call keyword arguments (line 372)
        kwargs_188641 = {}
        # Getting the type of 'np' (line 372)
        np_188633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 372)
        all_188634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), np_188633, 'all')
        # Calling all(args, kwargs) (line 372)
        all_call_result_188642 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), all_188634, *[isfinite_call_result_188640], **kwargs_188641)
        
        # Applying the 'not' unary operator (line 371)
        result_not__188643 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 44), 'not', all_call_result_188642)
        
        # Applying the binary operator 'or' (line 371)
        result_or_keyword_188644 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 12), 'or', result_ne_188632, result_not__188643)
        
        # Testing the type of an if condition (line 371)
        if_condition_188645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 8), result_or_keyword_188644)
        # Assigning a type to the variable 'if_condition_188645' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'if_condition_188645', if_condition_188645)
        # SSA begins for if statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 373)
        # Processing the call arguments (line 373)
        str_188647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 29), 'str', 'bounds should be a sequence containing real valued (min, max) pairs for each value in x')
        # Processing the call keyword arguments (line 373)
        kwargs_188648 = {}
        # Getting the type of 'ValueError' (line 373)
        ValueError_188646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 373)
        ValueError_call_result_188649 = invoke(stypy.reporting.localization.Localization(__file__, 373, 18), ValueError_188646, *[str_188647], **kwargs_188648)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 373, 12), ValueError_call_result_188649, 'raise parameter', BaseException)
        # SSA join for if statement (line 371)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 377)
        # Getting the type of 'maxiter' (line 377)
        maxiter_188650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), 'maxiter')
        # Getting the type of 'None' (line 377)
        None_188651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 22), 'None')
        
        (may_be_188652, more_types_in_union_188653) = may_be_none(maxiter_188650, None_188651)

        if may_be_188652:

            if more_types_in_union_188653:
                # Runtime conditional SSA (line 377)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 378):
            
            # Assigning a Num to a Name (line 378):
            int_188654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 22), 'int')
            # Assigning a type to the variable 'maxiter' (line 378)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'maxiter', int_188654)

            if more_types_in_union_188653:
                # SSA join for if statement (line 377)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 379):
        
        # Assigning a Name to a Attribute (line 379):
        # Getting the type of 'maxiter' (line 379)
        maxiter_188655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 23), 'maxiter')
        # Getting the type of 'self' (line 379)
        self_188656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'self')
        # Setting the type of the member 'maxiter' of a type (line 379)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), self_188656, 'maxiter', maxiter_188655)
        
        # Type idiom detected: calculating its left and rigth part (line 380)
        # Getting the type of 'maxfun' (line 380)
        maxfun_188657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 11), 'maxfun')
        # Getting the type of 'None' (line 380)
        None_188658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'None')
        
        (may_be_188659, more_types_in_union_188660) = may_be_none(maxfun_188657, None_188658)

        if may_be_188659:

            if more_types_in_union_188660:
                # Runtime conditional SSA (line 380)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 381):
            
            # Assigning a Attribute to a Name (line 381):
            # Getting the type of 'np' (line 381)
            np_188661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'np')
            # Obtaining the member 'inf' of a type (line 381)
            inf_188662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), np_188661, 'inf')
            # Assigning a type to the variable 'maxfun' (line 381)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'maxfun', inf_188662)

            if more_types_in_union_188660:
                # SSA join for if statement (line 380)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 382):
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'maxfun' (line 382)
        maxfun_188663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 22), 'maxfun')
        # Getting the type of 'self' (line 382)
        self_188664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self')
        # Setting the type of the member 'maxfun' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_188664, 'maxfun', maxfun_188663)
        
        # Assigning a BinOp to a Attribute (line 388):
        
        # Assigning a BinOp to a Attribute (line 388):
        float_188665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 28), 'float')
        
        # Obtaining the type of the subscript
        int_188666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 47), 'int')
        # Getting the type of 'self' (line 388)
        self_188667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 35), 'self')
        # Obtaining the member 'limits' of a type (line 388)
        limits_188668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 35), self_188667, 'limits')
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___188669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 35), limits_188668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_188670 = invoke(stypy.reporting.localization.Localization(__file__, 388, 35), getitem___188669, int_188666)
        
        
        # Obtaining the type of the subscript
        int_188671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 64), 'int')
        # Getting the type of 'self' (line 388)
        self_188672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 52), 'self')
        # Obtaining the member 'limits' of a type (line 388)
        limits_188673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 52), self_188672, 'limits')
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___188674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 52), limits_188673, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_188675 = invoke(stypy.reporting.localization.Localization(__file__, 388, 52), getitem___188674, int_188671)
        
        # Applying the binary operator '+' (line 388)
        result_add_188676 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 35), '+', subscript_call_result_188670, subscript_call_result_188675)
        
        # Applying the binary operator '*' (line 388)
        result_mul_188677 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 28), '*', float_188665, result_add_188676)
        
        # Getting the type of 'self' (line 388)
        self_188678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self')
        # Setting the type of the member '__scale_arg1' of a type (line 388)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_188678, '__scale_arg1', result_mul_188677)
        
        # Assigning a Call to a Attribute (line 389):
        
        # Assigning a Call to a Attribute (line 389):
        
        # Call to fabs(...): (line 389)
        # Processing the call arguments (line 389)
        
        # Obtaining the type of the subscript
        int_188681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 48), 'int')
        # Getting the type of 'self' (line 389)
        self_188682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 36), 'self', False)
        # Obtaining the member 'limits' of a type (line 389)
        limits_188683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 36), self_188682, 'limits')
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___188684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 36), limits_188683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_188685 = invoke(stypy.reporting.localization.Localization(__file__, 389, 36), getitem___188684, int_188681)
        
        
        # Obtaining the type of the subscript
        int_188686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 65), 'int')
        # Getting the type of 'self' (line 389)
        self_188687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 53), 'self', False)
        # Obtaining the member 'limits' of a type (line 389)
        limits_188688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 53), self_188687, 'limits')
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___188689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 53), limits_188688, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_188690 = invoke(stypy.reporting.localization.Localization(__file__, 389, 53), getitem___188689, int_188686)
        
        # Applying the binary operator '-' (line 389)
        result_sub_188691 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 36), '-', subscript_call_result_188685, subscript_call_result_188690)
        
        # Processing the call keyword arguments (line 389)
        kwargs_188692 = {}
        # Getting the type of 'np' (line 389)
        np_188679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 28), 'np', False)
        # Obtaining the member 'fabs' of a type (line 389)
        fabs_188680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 28), np_188679, 'fabs')
        # Calling fabs(args, kwargs) (line 389)
        fabs_call_result_188693 = invoke(stypy.reporting.localization.Localization(__file__, 389, 28), fabs_188680, *[result_sub_188691], **kwargs_188692)
        
        # Getting the type of 'self' (line 389)
        self_188694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self')
        # Setting the type of the member '__scale_arg2' of a type (line 389)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_188694, '__scale_arg2', fabs_call_result_188693)
        
        # Assigning a Call to a Attribute (line 391):
        
        # Assigning a Call to a Attribute (line 391):
        
        # Call to size(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'self' (line 391)
        self_188697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 39), 'self', False)
        # Obtaining the member 'limits' of a type (line 391)
        limits_188698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 39), self_188697, 'limits')
        int_188699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 52), 'int')
        # Processing the call keyword arguments (line 391)
        kwargs_188700 = {}
        # Getting the type of 'np' (line 391)
        np_188695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 31), 'np', False)
        # Obtaining the member 'size' of a type (line 391)
        size_188696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 31), np_188695, 'size')
        # Calling size(args, kwargs) (line 391)
        size_call_result_188701 = invoke(stypy.reporting.localization.Localization(__file__, 391, 31), size_188696, *[limits_188698, int_188699], **kwargs_188700)
        
        # Getting the type of 'self' (line 391)
        self_188702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'self')
        # Setting the type of the member 'parameter_count' of a type (line 391)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), self_188702, 'parameter_count', size_call_result_188701)
        
        # Assigning a Call to a Attribute (line 393):
        
        # Assigning a Call to a Attribute (line 393):
        
        # Call to check_random_state(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'seed' (line 393)
        seed_188704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 58), 'seed', False)
        # Processing the call keyword arguments (line 393)
        kwargs_188705 = {}
        # Getting the type of 'check_random_state' (line 393)
        check_random_state_188703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 39), 'check_random_state', False)
        # Calling check_random_state(args, kwargs) (line 393)
        check_random_state_call_result_188706 = invoke(stypy.reporting.localization.Localization(__file__, 393, 39), check_random_state_188703, *[seed_188704], **kwargs_188705)
        
        # Getting the type of 'self' (line 393)
        self_188707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'self')
        # Setting the type of the member 'random_number_generator' of a type (line 393)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), self_188707, 'random_number_generator', check_random_state_call_result_188706)
        
        # Assigning a BinOp to a Attribute (line 397):
        
        # Assigning a BinOp to a Attribute (line 397):
        # Getting the type of 'popsize' (line 397)
        popsize_188708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 38), 'popsize')
        # Getting the type of 'self' (line 397)
        self_188709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 48), 'self')
        # Obtaining the member 'parameter_count' of a type (line 397)
        parameter_count_188710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 48), self_188709, 'parameter_count')
        # Applying the binary operator '*' (line 397)
        result_mul_188711 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 38), '*', popsize_188708, parameter_count_188710)
        
        # Getting the type of 'self' (line 397)
        self_188712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self')
        # Setting the type of the member 'num_population_members' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_188712, 'num_population_members', result_mul_188711)
        
        # Assigning a Tuple to a Attribute (line 399):
        
        # Assigning a Tuple to a Attribute (line 399):
        
        # Obtaining an instance of the builtin type 'tuple' (line 399)
        tuple_188713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 399)
        # Adding element type (line 399)
        # Getting the type of 'self' (line 399)
        self_188714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 33), 'self')
        # Obtaining the member 'num_population_members' of a type (line 399)
        num_population_members_188715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 33), self_188714, 'num_population_members')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 33), tuple_188713, num_population_members_188715)
        # Adding element type (line 399)
        # Getting the type of 'self' (line 400)
        self_188716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 33), 'self')
        # Obtaining the member 'parameter_count' of a type (line 400)
        parameter_count_188717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 33), self_188716, 'parameter_count')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 33), tuple_188713, parameter_count_188717)
        
        # Getting the type of 'self' (line 399)
        self_188718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self')
        # Setting the type of the member 'population_shape' of a type (line 399)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_188718, 'population_shape', tuple_188713)
        
        # Assigning a Num to a Attribute (line 402):
        
        # Assigning a Num to a Attribute (line 402):
        int_188719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 21), 'int')
        # Getting the type of 'self' (line 402)
        self_188720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self')
        # Setting the type of the member '_nfev' of a type (line 402)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_188720, '_nfev', int_188719)
        
        
        # Getting the type of 'init' (line 403)
        init_188721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'init')
        str_188722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 19), 'str', 'latinhypercube')
        # Applying the binary operator '==' (line 403)
        result_eq_188723 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 11), '==', init_188721, str_188722)
        
        # Testing the type of an if condition (line 403)
        if_condition_188724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 8), result_eq_188723)
        # Assigning a type to the variable 'if_condition_188724' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'if_condition_188724', if_condition_188724)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to init_population_lhs(...): (line 404)
        # Processing the call keyword arguments (line 404)
        kwargs_188727 = {}
        # Getting the type of 'self' (line 404)
        self_188725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'self', False)
        # Obtaining the member 'init_population_lhs' of a type (line 404)
        init_population_lhs_188726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 12), self_188725, 'init_population_lhs')
        # Calling init_population_lhs(args, kwargs) (line 404)
        init_population_lhs_call_result_188728 = invoke(stypy.reporting.localization.Localization(__file__, 404, 12), init_population_lhs_188726, *[], **kwargs_188727)
        
        # SSA branch for the else part of an if statement (line 403)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'init' (line 405)
        init_188729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'init')
        str_188730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'str', 'random')
        # Applying the binary operator '==' (line 405)
        result_eq_188731 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 13), '==', init_188729, str_188730)
        
        # Testing the type of an if condition (line 405)
        if_condition_188732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 13), result_eq_188731)
        # Assigning a type to the variable 'if_condition_188732' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'if_condition_188732', if_condition_188732)
        # SSA begins for if statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to init_population_random(...): (line 406)
        # Processing the call keyword arguments (line 406)
        kwargs_188735 = {}
        # Getting the type of 'self' (line 406)
        self_188733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'self', False)
        # Obtaining the member 'init_population_random' of a type (line 406)
        init_population_random_188734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 12), self_188733, 'init_population_random')
        # Calling init_population_random(args, kwargs) (line 406)
        init_population_random_call_result_188736 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), init_population_random_188734, *[], **kwargs_188735)
        
        # SSA branch for the else part of an if statement (line 405)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 408)
        # Processing the call arguments (line 408)
        str_188738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 29), 'str', "The population initialization method must be oneof 'latinhypercube' or 'random'")
        # Processing the call keyword arguments (line 408)
        kwargs_188739 = {}
        # Getting the type of 'ValueError' (line 408)
        ValueError_188737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 408)
        ValueError_call_result_188740 = invoke(stypy.reporting.localization.Localization(__file__, 408, 18), ValueError_188737, *[str_188738], **kwargs_188739)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 408, 12), ValueError_call_result_188740, 'raise parameter', BaseException)
        # SSA join for if statement (line 405)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 411):
        
        # Assigning a Name to a Attribute (line 411):
        # Getting the type of 'disp' (line 411)
        disp_188741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 20), 'disp')
        # Getting the type of 'self' (line 411)
        self_188742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self')
        # Setting the type of the member 'disp' of a type (line 411)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_188742, 'disp', disp_188741)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def init_population_lhs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'init_population_lhs'
        module_type_store = module_type_store.open_function_context('init_population_lhs', 413, 4, False)
        # Assigning a type to the variable 'self' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver.init_population_lhs')
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver.init_population_lhs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.init_population_lhs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'init_population_lhs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'init_population_lhs(...)' code ##################

        str_188743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, (-1)), 'str', '\n        Initializes the population with Latin Hypercube Sampling.\n        Latin Hypercube Sampling ensures that each parameter is uniformly\n        sampled over its range.\n        ')
        
        # Assigning a Attribute to a Name (line 419):
        
        # Assigning a Attribute to a Name (line 419):
        # Getting the type of 'self' (line 419)
        self_188744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 14), 'self')
        # Obtaining the member 'random_number_generator' of a type (line 419)
        random_number_generator_188745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 14), self_188744, 'random_number_generator')
        # Assigning a type to the variable 'rng' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'rng', random_number_generator_188745)
        
        # Assigning a BinOp to a Name (line 425):
        
        # Assigning a BinOp to a Name (line 425):
        float_188746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 18), 'float')
        # Getting the type of 'self' (line 425)
        self_188747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 24), 'self')
        # Obtaining the member 'num_population_members' of a type (line 425)
        num_population_members_188748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 24), self_188747, 'num_population_members')
        # Applying the binary operator 'div' (line 425)
        result_div_188749 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 18), 'div', float_188746, num_population_members_188748)
        
        # Assigning a type to the variable 'segsize' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'segsize', result_div_188749)
        
        # Assigning a BinOp to a Name (line 429):
        
        # Assigning a BinOp to a Name (line 429):
        # Getting the type of 'segsize' (line 429)
        segsize_188750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 19), 'segsize')
        
        # Call to random_sample(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'self' (line 429)
        self_188753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 47), 'self', False)
        # Obtaining the member 'population_shape' of a type (line 429)
        population_shape_188754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 47), self_188753, 'population_shape')
        # Processing the call keyword arguments (line 429)
        kwargs_188755 = {}
        # Getting the type of 'rng' (line 429)
        rng_188751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 29), 'rng', False)
        # Obtaining the member 'random_sample' of a type (line 429)
        random_sample_188752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 29), rng_188751, 'random_sample')
        # Calling random_sample(args, kwargs) (line 429)
        random_sample_call_result_188756 = invoke(stypy.reporting.localization.Localization(__file__, 429, 29), random_sample_188752, *[population_shape_188754], **kwargs_188755)
        
        # Applying the binary operator '*' (line 429)
        result_mul_188757 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 19), '*', segsize_188750, random_sample_call_result_188756)
        
        
        # Obtaining the type of the subscript
        slice_188758 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 432, 21), None, None, None)
        # Getting the type of 'np' (line 433)
        np_188759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 52), 'np')
        # Obtaining the member 'newaxis' of a type (line 433)
        newaxis_188760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 52), np_188759, 'newaxis')
        
        # Call to linspace(...): (line 432)
        # Processing the call arguments (line 432)
        float_188763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 33), 'float')
        float_188764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 37), 'float')
        # Getting the type of 'self' (line 432)
        self_188765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 41), 'self', False)
        # Obtaining the member 'num_population_members' of a type (line 432)
        num_population_members_188766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 41), self_188765, 'num_population_members')
        # Processing the call keyword arguments (line 432)
        # Getting the type of 'False' (line 433)
        False_188767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 42), 'False', False)
        keyword_188768 = False_188767
        kwargs_188769 = {'endpoint': keyword_188768}
        # Getting the type of 'np' (line 432)
        np_188761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 21), 'np', False)
        # Obtaining the member 'linspace' of a type (line 432)
        linspace_188762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 21), np_188761, 'linspace')
        # Calling linspace(args, kwargs) (line 432)
        linspace_call_result_188770 = invoke(stypy.reporting.localization.Localization(__file__, 432, 21), linspace_188762, *[float_188763, float_188764, num_population_members_188766], **kwargs_188769)
        
        # Obtaining the member '__getitem__' of a type (line 432)
        getitem___188771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 21), linspace_call_result_188770, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 432)
        subscript_call_result_188772 = invoke(stypy.reporting.localization.Localization(__file__, 432, 21), getitem___188771, (slice_188758, newaxis_188760))
        
        # Applying the binary operator '+' (line 429)
        result_add_188773 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 19), '+', result_mul_188757, subscript_call_result_188772)
        
        # Assigning a type to the variable 'samples' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'samples', result_add_188773)
        
        # Assigning a Call to a Attribute (line 436):
        
        # Assigning a Call to a Attribute (line 436):
        
        # Call to zeros_like(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'samples' (line 436)
        samples_188776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 40), 'samples', False)
        # Processing the call keyword arguments (line 436)
        kwargs_188777 = {}
        # Getting the type of 'np' (line 436)
        np_188774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 26), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 436)
        zeros_like_188775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 26), np_188774, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 436)
        zeros_like_call_result_188778 = invoke(stypy.reporting.localization.Localization(__file__, 436, 26), zeros_like_188775, *[samples_188776], **kwargs_188777)
        
        # Getting the type of 'self' (line 436)
        self_188779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'self')
        # Setting the type of the member 'population' of a type (line 436)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), self_188779, 'population', zeros_like_call_result_188778)
        
        
        # Call to range(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'self' (line 440)
        self_188781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'self', False)
        # Obtaining the member 'parameter_count' of a type (line 440)
        parameter_count_188782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), self_188781, 'parameter_count')
        # Processing the call keyword arguments (line 440)
        kwargs_188783 = {}
        # Getting the type of 'range' (line 440)
        range_188780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 17), 'range', False)
        # Calling range(args, kwargs) (line 440)
        range_call_result_188784 = invoke(stypy.reporting.localization.Localization(__file__, 440, 17), range_188780, *[parameter_count_188782], **kwargs_188783)
        
        # Testing the type of a for loop iterable (line 440)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 8), range_call_result_188784)
        # Getting the type of the for loop variable (line 440)
        for_loop_var_188785 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 8), range_call_result_188784)
        # Assigning a type to the variable 'j' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'j', for_loop_var_188785)
        # SSA begins for a for statement (line 440)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 441):
        
        # Assigning a Call to a Name (line 441):
        
        # Call to permutation(...): (line 441)
        # Processing the call arguments (line 441)
        
        # Call to range(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'self' (line 441)
        self_188789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 42), 'self', False)
        # Obtaining the member 'num_population_members' of a type (line 441)
        num_population_members_188790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 42), self_188789, 'num_population_members')
        # Processing the call keyword arguments (line 441)
        kwargs_188791 = {}
        # Getting the type of 'range' (line 441)
        range_188788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 36), 'range', False)
        # Calling range(args, kwargs) (line 441)
        range_call_result_188792 = invoke(stypy.reporting.localization.Localization(__file__, 441, 36), range_188788, *[num_population_members_188790], **kwargs_188791)
        
        # Processing the call keyword arguments (line 441)
        kwargs_188793 = {}
        # Getting the type of 'rng' (line 441)
        rng_188786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'rng', False)
        # Obtaining the member 'permutation' of a type (line 441)
        permutation_188787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), rng_188786, 'permutation')
        # Calling permutation(args, kwargs) (line 441)
        permutation_call_result_188794 = invoke(stypy.reporting.localization.Localization(__file__, 441, 20), permutation_188787, *[range_call_result_188792], **kwargs_188793)
        
        # Assigning a type to the variable 'order' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'order', permutation_call_result_188794)
        
        # Assigning a Subscript to a Subscript (line 442):
        
        # Assigning a Subscript to a Subscript (line 442):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 442)
        tuple_188795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 442)
        # Adding element type (line 442)
        # Getting the type of 'order' (line 442)
        order_188796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 44), 'order')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 44), tuple_188795, order_188796)
        # Adding element type (line 442)
        # Getting the type of 'j' (line 442)
        j_188797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 51), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 44), tuple_188795, j_188797)
        
        # Getting the type of 'samples' (line 442)
        samples_188798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 36), 'samples')
        # Obtaining the member '__getitem__' of a type (line 442)
        getitem___188799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 36), samples_188798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 442)
        subscript_call_result_188800 = invoke(stypy.reporting.localization.Localization(__file__, 442, 36), getitem___188799, tuple_188795)
        
        # Getting the type of 'self' (line 442)
        self_188801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'self')
        # Obtaining the member 'population' of a type (line 442)
        population_188802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), self_188801, 'population')
        slice_188803 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 442, 12), None, None, None)
        # Getting the type of 'j' (line 442)
        j_188804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 31), 'j')
        # Storing an element on a container (line 442)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 12), population_188802, ((slice_188803, j_188804), subscript_call_result_188800))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Attribute (line 445):
        
        # Assigning a BinOp to a Attribute (line 445):
        
        # Call to ones(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'self' (line 445)
        self_188807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 44), 'self', False)
        # Obtaining the member 'num_population_members' of a type (line 445)
        num_population_members_188808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 44), self_188807, 'num_population_members')
        # Processing the call keyword arguments (line 445)
        kwargs_188809 = {}
        # Getting the type of 'np' (line 445)
        np_188805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 36), 'np', False)
        # Obtaining the member 'ones' of a type (line 445)
        ones_188806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 36), np_188805, 'ones')
        # Calling ones(args, kwargs) (line 445)
        ones_call_result_188810 = invoke(stypy.reporting.localization.Localization(__file__, 445, 36), ones_188806, *[num_population_members_188808], **kwargs_188809)
        
        # Getting the type of 'np' (line 446)
        np_188811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 36), 'np')
        # Obtaining the member 'inf' of a type (line 446)
        inf_188812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 36), np_188811, 'inf')
        # Applying the binary operator '*' (line 445)
        result_mul_188813 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 36), '*', ones_call_result_188810, inf_188812)
        
        # Getting the type of 'self' (line 445)
        self_188814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'self')
        # Setting the type of the member 'population_energies' of a type (line 445)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), self_188814, 'population_energies', result_mul_188813)
        
        # Assigning a Num to a Attribute (line 449):
        
        # Assigning a Num to a Attribute (line 449):
        int_188815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 21), 'int')
        # Getting the type of 'self' (line 449)
        self_188816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'self')
        # Setting the type of the member '_nfev' of a type (line 449)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), self_188816, '_nfev', int_188815)
        
        # ################# End of 'init_population_lhs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'init_population_lhs' in the type store
        # Getting the type of 'stypy_return_type' (line 413)
        stypy_return_type_188817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'init_population_lhs'
        return stypy_return_type_188817


    @norecursion
    def init_population_random(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'init_population_random'
        module_type_store = module_type_store.open_function_context('init_population_random', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver.init_population_random')
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver.init_population_random.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.init_population_random', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'init_population_random', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'init_population_random(...)' code ##################

        str_188818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, (-1)), 'str', '\n        Initialises the population at random.  This type of initialization\n        can possess clustering, Latin Hypercube sampling is generally better.\n        ')
        
        # Assigning a Attribute to a Name (line 456):
        
        # Assigning a Attribute to a Name (line 456):
        # Getting the type of 'self' (line 456)
        self_188819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'self')
        # Obtaining the member 'random_number_generator' of a type (line 456)
        random_number_generator_188820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 14), self_188819, 'random_number_generator')
        # Assigning a type to the variable 'rng' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'rng', random_number_generator_188820)
        
        # Assigning a Call to a Attribute (line 457):
        
        # Assigning a Call to a Attribute (line 457):
        
        # Call to random_sample(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'self' (line 457)
        self_188823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 44), 'self', False)
        # Obtaining the member 'population_shape' of a type (line 457)
        population_shape_188824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 44), self_188823, 'population_shape')
        # Processing the call keyword arguments (line 457)
        kwargs_188825 = {}
        # Getting the type of 'rng' (line 457)
        rng_188821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 26), 'rng', False)
        # Obtaining the member 'random_sample' of a type (line 457)
        random_sample_188822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 26), rng_188821, 'random_sample')
        # Calling random_sample(args, kwargs) (line 457)
        random_sample_call_result_188826 = invoke(stypy.reporting.localization.Localization(__file__, 457, 26), random_sample_188822, *[population_shape_188824], **kwargs_188825)
        
        # Getting the type of 'self' (line 457)
        self_188827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'self')
        # Setting the type of the member 'population' of a type (line 457)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), self_188827, 'population', random_sample_call_result_188826)
        
        # Assigning a BinOp to a Attribute (line 460):
        
        # Assigning a BinOp to a Attribute (line 460):
        
        # Call to ones(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'self' (line 460)
        self_188830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 44), 'self', False)
        # Obtaining the member 'num_population_members' of a type (line 460)
        num_population_members_188831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 44), self_188830, 'num_population_members')
        # Processing the call keyword arguments (line 460)
        kwargs_188832 = {}
        # Getting the type of 'np' (line 460)
        np_188828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 36), 'np', False)
        # Obtaining the member 'ones' of a type (line 460)
        ones_188829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 36), np_188828, 'ones')
        # Calling ones(args, kwargs) (line 460)
        ones_call_result_188833 = invoke(stypy.reporting.localization.Localization(__file__, 460, 36), ones_188829, *[num_population_members_188831], **kwargs_188832)
        
        # Getting the type of 'np' (line 461)
        np_188834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 36), 'np')
        # Obtaining the member 'inf' of a type (line 461)
        inf_188835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 36), np_188834, 'inf')
        # Applying the binary operator '*' (line 460)
        result_mul_188836 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 36), '*', ones_call_result_188833, inf_188835)
        
        # Getting the type of 'self' (line 460)
        self_188837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'self')
        # Setting the type of the member 'population_energies' of a type (line 460)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), self_188837, 'population_energies', result_mul_188836)
        
        # Assigning a Num to a Attribute (line 464):
        
        # Assigning a Num to a Attribute (line 464):
        int_188838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 21), 'int')
        # Getting the type of 'self' (line 464)
        self_188839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'self')
        # Setting the type of the member '_nfev' of a type (line 464)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 8), self_188839, '_nfev', int_188838)
        
        # ################# End of 'init_population_random(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'init_population_random' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_188840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'init_population_random'
        return stypy_return_type_188840


    @norecursion
    def x(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'x'
        module_type_store = module_type_store.open_function_context('x', 466, 4, False)
        # Assigning a type to the variable 'self' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver.x')
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver.x.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.x', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'x', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'x(...)' code ##################

        str_188841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, (-1)), 'str', '\n        The best solution from the solver\n\n        Returns\n        -------\n        x : ndarray\n            The best solution from the solver.\n        ')
        
        # Call to _scale_parameters(...): (line 476)
        # Processing the call arguments (line 476)
        
        # Obtaining the type of the subscript
        int_188844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 54), 'int')
        # Getting the type of 'self' (line 476)
        self_188845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 38), 'self', False)
        # Obtaining the member 'population' of a type (line 476)
        population_188846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 38), self_188845, 'population')
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___188847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 38), population_188846, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_188848 = invoke(stypy.reporting.localization.Localization(__file__, 476, 38), getitem___188847, int_188844)
        
        # Processing the call keyword arguments (line 476)
        kwargs_188849 = {}
        # Getting the type of 'self' (line 476)
        self_188842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'self', False)
        # Obtaining the member '_scale_parameters' of a type (line 476)
        _scale_parameters_188843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), self_188842, '_scale_parameters')
        # Calling _scale_parameters(args, kwargs) (line 476)
        _scale_parameters_call_result_188850 = invoke(stypy.reporting.localization.Localization(__file__, 476, 15), _scale_parameters_188843, *[subscript_call_result_188848], **kwargs_188849)
        
        # Assigning a type to the variable 'stypy_return_type' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'stypy_return_type', _scale_parameters_call_result_188850)
        
        # ################# End of 'x(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'x' in the type store
        # Getting the type of 'stypy_return_type' (line 466)
        stypy_return_type_188851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'x'
        return stypy_return_type_188851


    @norecursion
    def convergence(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convergence'
        module_type_store = module_type_store.open_function_context('convergence', 478, 4, False)
        # Assigning a type to the variable 'self' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver.convergence')
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver.convergence.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.convergence', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convergence', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convergence(...)' code ##################

        str_188852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, (-1)), 'str', '\n        The standard deviation of the population energies divided by their\n        mean.\n        ')
        
        # Call to std(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'self' (line 484)
        self_188855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 23), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 484)
        population_energies_188856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 23), self_188855, 'population_energies')
        # Processing the call keyword arguments (line 484)
        kwargs_188857 = {}
        # Getting the type of 'np' (line 484)
        np_188853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'np', False)
        # Obtaining the member 'std' of a type (line 484)
        std_188854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 16), np_188853, 'std')
        # Calling std(args, kwargs) (line 484)
        std_call_result_188858 = invoke(stypy.reporting.localization.Localization(__file__, 484, 16), std_188854, *[population_energies_188856], **kwargs_188857)
        
        
        # Call to abs(...): (line 485)
        # Processing the call arguments (line 485)
        
        # Call to mean(...): (line 485)
        # Processing the call arguments (line 485)
        # Getting the type of 'self' (line 485)
        self_188863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 31), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 485)
        population_energies_188864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 31), self_188863, 'population_energies')
        # Processing the call keyword arguments (line 485)
        kwargs_188865 = {}
        # Getting the type of 'np' (line 485)
        np_188861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 23), 'np', False)
        # Obtaining the member 'mean' of a type (line 485)
        mean_188862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 23), np_188861, 'mean')
        # Calling mean(args, kwargs) (line 485)
        mean_call_result_188866 = invoke(stypy.reporting.localization.Localization(__file__, 485, 23), mean_188862, *[population_energies_188864], **kwargs_188865)
        
        # Getting the type of '_MACHEPS' (line 485)
        _MACHEPS_188867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 59), '_MACHEPS', False)
        # Applying the binary operator '+' (line 485)
        result_add_188868 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 23), '+', mean_call_result_188866, _MACHEPS_188867)
        
        # Processing the call keyword arguments (line 485)
        kwargs_188869 = {}
        # Getting the type of 'np' (line 485)
        np_188859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'np', False)
        # Obtaining the member 'abs' of a type (line 485)
        abs_188860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 16), np_188859, 'abs')
        # Calling abs(args, kwargs) (line 485)
        abs_call_result_188870 = invoke(stypy.reporting.localization.Localization(__file__, 485, 16), abs_188860, *[result_add_188868], **kwargs_188869)
        
        # Applying the binary operator 'div' (line 484)
        result_div_188871 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 16), 'div', std_call_result_188858, abs_call_result_188870)
        
        # Assigning a type to the variable 'stypy_return_type' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'stypy_return_type', result_div_188871)
        
        # ################# End of 'convergence(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convergence' in the type store
        # Getting the type of 'stypy_return_type' (line 478)
        stypy_return_type_188872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188872)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convergence'
        return stypy_return_type_188872


    @norecursion
    def solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 487, 4, False)
        # Assigning a type to the variable 'self' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver.solve')
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver.solve.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.solve', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solve', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solve(...)' code ##################

        str_188873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, (-1)), 'str', '\n        Runs the DifferentialEvolutionSolver.\n\n        Returns\n        -------\n        res : OptimizeResult\n            The optimization result represented as a ``OptimizeResult`` object.\n            Important attributes are: ``x`` the solution array, ``success`` a\n            Boolean flag indicating if the optimizer exited successfully and\n            ``message`` which describes the cause of the termination. See\n            `OptimizeResult` for a description of other attributes.  If `polish`\n            was employed, and a lower minimum was obtained by the polishing,\n            then OptimizeResult also contains the ``jac`` attribute.\n        ')
        
        # Assigning a Tuple to a Tuple (line 502):
        
        # Assigning a Num to a Name (line 502):
        int_188874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 28), 'int')
        # Assigning a type to the variable 'tuple_assignment_188384' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'tuple_assignment_188384', int_188874)
        
        # Assigning a Name to a Name (line 502):
        # Getting the type of 'False' (line 502)
        False_188875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 31), 'False')
        # Assigning a type to the variable 'tuple_assignment_188385' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'tuple_assignment_188385', False_188875)
        
        # Assigning a Name to a Name (line 502):
        # Getting the type of 'tuple_assignment_188384' (line 502)
        tuple_assignment_188384_188876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'tuple_assignment_188384')
        # Assigning a type to the variable 'nit' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'nit', tuple_assignment_188384_188876)
        
        # Assigning a Name to a Name (line 502):
        # Getting the type of 'tuple_assignment_188385' (line 502)
        tuple_assignment_188385_188877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'tuple_assignment_188385')
        # Assigning a type to the variable 'warning_flag' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 13), 'warning_flag', tuple_assignment_188385_188877)
        
        # Assigning a Subscript to a Name (line 503):
        
        # Assigning a Subscript to a Name (line 503):
        
        # Obtaining the type of the subscript
        str_188878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 41), 'str', 'success')
        # Getting the type of '_status_message' (line 503)
        _status_message_188879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 25), '_status_message')
        # Obtaining the member '__getitem__' of a type (line 503)
        getitem___188880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 25), _status_message_188879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 503)
        subscript_call_result_188881 = invoke(stypy.reporting.localization.Localization(__file__, 503, 25), getitem___188880, str_188878)
        
        # Assigning a type to the variable 'status_message' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'status_message', subscript_call_result_188881)
        
        
        # Call to all(...): (line 510)
        # Processing the call arguments (line 510)
        
        # Call to isinf(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'self' (line 510)
        self_188886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 27), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 510)
        population_energies_188887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 27), self_188886, 'population_energies')
        # Processing the call keyword arguments (line 510)
        kwargs_188888 = {}
        # Getting the type of 'np' (line 510)
        np_188884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 18), 'np', False)
        # Obtaining the member 'isinf' of a type (line 510)
        isinf_188885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 18), np_188884, 'isinf')
        # Calling isinf(args, kwargs) (line 510)
        isinf_call_result_188889 = invoke(stypy.reporting.localization.Localization(__file__, 510, 18), isinf_188885, *[population_energies_188887], **kwargs_188888)
        
        # Processing the call keyword arguments (line 510)
        kwargs_188890 = {}
        # Getting the type of 'np' (line 510)
        np_188882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 11), 'np', False)
        # Obtaining the member 'all' of a type (line 510)
        all_188883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 11), np_188882, 'all')
        # Calling all(args, kwargs) (line 510)
        all_call_result_188891 = invoke(stypy.reporting.localization.Localization(__file__, 510, 11), all_188883, *[isinf_call_result_188889], **kwargs_188890)
        
        # Testing the type of an if condition (line 510)
        if_condition_188892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 8), all_call_result_188891)
        # Assigning a type to the variable 'if_condition_188892' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'if_condition_188892', if_condition_188892)
        # SSA begins for if statement (line 510)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _calculate_population_energies(...): (line 511)
        # Processing the call keyword arguments (line 511)
        kwargs_188895 = {}
        # Getting the type of 'self' (line 511)
        self_188893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'self', False)
        # Obtaining the member '_calculate_population_energies' of a type (line 511)
        _calculate_population_energies_188894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 12), self_188893, '_calculate_population_energies')
        # Calling _calculate_population_energies(args, kwargs) (line 511)
        _calculate_population_energies_call_result_188896 = invoke(stypy.reporting.localization.Localization(__file__, 511, 12), _calculate_population_energies_188894, *[], **kwargs_188895)
        
        # SSA join for if statement (line 510)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to xrange(...): (line 514)
        # Processing the call arguments (line 514)
        int_188898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 26), 'int')
        # Getting the type of 'self' (line 514)
        self_188899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 29), 'self', False)
        # Obtaining the member 'maxiter' of a type (line 514)
        maxiter_188900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 29), self_188899, 'maxiter')
        int_188901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 44), 'int')
        # Applying the binary operator '+' (line 514)
        result_add_188902 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 29), '+', maxiter_188900, int_188901)
        
        # Processing the call keyword arguments (line 514)
        kwargs_188903 = {}
        # Getting the type of 'xrange' (line 514)
        xrange_188897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 514)
        xrange_call_result_188904 = invoke(stypy.reporting.localization.Localization(__file__, 514, 19), xrange_188897, *[int_188898, result_add_188902], **kwargs_188903)
        
        # Testing the type of a for loop iterable (line 514)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 514, 8), xrange_call_result_188904)
        # Getting the type of the for loop variable (line 514)
        for_loop_var_188905 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 514, 8), xrange_call_result_188904)
        # Assigning a type to the variable 'nit' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'nit', for_loop_var_188905)
        # SSA begins for a for statement (line 514)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 516)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to next(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'self' (line 517)
        self_188907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 21), 'self', False)
        # Processing the call keyword arguments (line 517)
        kwargs_188908 = {}
        # Getting the type of 'next' (line 517)
        next_188906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 16), 'next', False)
        # Calling next(args, kwargs) (line 517)
        next_call_result_188909 = invoke(stypy.reporting.localization.Localization(__file__, 517, 16), next_188906, *[self_188907], **kwargs_188908)
        
        # SSA branch for the except part of a try statement (line 516)
        # SSA branch for the except 'StopIteration' branch of a try statement (line 516)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 519):
        
        # Assigning a Name to a Name (line 519):
        # Getting the type of 'True' (line 519)
        True_188910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 31), 'True')
        # Assigning a type to the variable 'warning_flag' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), 'warning_flag', True_188910)
        
        # Assigning a Subscript to a Name (line 520):
        
        # Assigning a Subscript to a Name (line 520):
        
        # Obtaining the type of the subscript
        str_188911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 49), 'str', 'maxfev')
        # Getting the type of '_status_message' (line 520)
        _status_message_188912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 33), '_status_message')
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___188913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 33), _status_message_188912, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 520)
        subscript_call_result_188914 = invoke(stypy.reporting.localization.Localization(__file__, 520, 33), getitem___188913, str_188911)
        
        # Assigning a type to the variable 'status_message' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'status_message', subscript_call_result_188914)
        # SSA join for try-except statement (line 516)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 523)
        self_188915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 15), 'self')
        # Obtaining the member 'disp' of a type (line 523)
        disp_188916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 15), self_188915, 'disp')
        # Testing the type of an if condition (line 523)
        if_condition_188917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 523, 12), disp_188916)
        # Assigning a type to the variable 'if_condition_188917' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'if_condition_188917', if_condition_188917)
        # SSA begins for if statement (line 523)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 524)
        # Processing the call arguments (line 524)
        str_188919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 22), 'str', 'differential_evolution step %d: f(x)= %g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 525)
        tuple_188920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 525)
        # Adding element type (line 525)
        # Getting the type of 'nit' (line 525)
        nit_188921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'nit', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 25), tuple_188920, nit_188921)
        # Adding element type (line 525)
        
        # Obtaining the type of the subscript
        int_188922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 50), 'int')
        # Getting the type of 'self' (line 526)
        self_188923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 25), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 526)
        population_energies_188924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 25), self_188923, 'population_energies')
        # Obtaining the member '__getitem__' of a type (line 526)
        getitem___188925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 25), population_energies_188924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 526)
        subscript_call_result_188926 = invoke(stypy.reporting.localization.Localization(__file__, 526, 25), getitem___188925, int_188922)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 25), tuple_188920, subscript_call_result_188926)
        
        # Applying the binary operator '%' (line 524)
        result_mod_188927 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 22), '%', str_188919, tuple_188920)
        
        # Processing the call keyword arguments (line 524)
        kwargs_188928 = {}
        # Getting the type of 'print' (line 524)
        print_188918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'print', False)
        # Calling print(args, kwargs) (line 524)
        print_call_result_188929 = invoke(stypy.reporting.localization.Localization(__file__, 524, 16), print_188918, *[result_mod_188927], **kwargs_188928)
        
        # SSA join for if statement (line 523)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 529):
        
        # Assigning a Attribute to a Name (line 529):
        # Getting the type of 'self' (line 529)
        self_188930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 26), 'self')
        # Obtaining the member 'convergence' of a type (line 529)
        convergence_188931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 26), self_188930, 'convergence')
        # Assigning a type to the variable 'convergence' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'convergence', convergence_188931)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 531)
        self_188932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'self')
        # Obtaining the member 'callback' of a type (line 531)
        callback_188933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 16), self_188932, 'callback')
        
        
        # Call to callback(...): (line 532)
        # Processing the call arguments (line 532)
        
        # Call to _scale_parameters(...): (line 532)
        # Processing the call arguments (line 532)
        
        # Obtaining the type of the subscript
        int_188938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 73), 'int')
        # Getting the type of 'self' (line 532)
        self_188939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 57), 'self', False)
        # Obtaining the member 'population' of a type (line 532)
        population_188940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 57), self_188939, 'population')
        # Obtaining the member '__getitem__' of a type (line 532)
        getitem___188941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 57), population_188940, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 532)
        subscript_call_result_188942 = invoke(stypy.reporting.localization.Localization(__file__, 532, 57), getitem___188941, int_188938)
        
        # Processing the call keyword arguments (line 532)
        kwargs_188943 = {}
        # Getting the type of 'self' (line 532)
        self_188936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 34), 'self', False)
        # Obtaining the member '_scale_parameters' of a type (line 532)
        _scale_parameters_188937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 34), self_188936, '_scale_parameters')
        # Calling _scale_parameters(args, kwargs) (line 532)
        _scale_parameters_call_result_188944 = invoke(stypy.reporting.localization.Localization(__file__, 532, 34), _scale_parameters_188937, *[subscript_call_result_188942], **kwargs_188943)
        
        # Processing the call keyword arguments (line 532)
        # Getting the type of 'self' (line 533)
        self_188945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 46), 'self', False)
        # Obtaining the member 'tol' of a type (line 533)
        tol_188946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 46), self_188945, 'tol')
        # Getting the type of 'convergence' (line 533)
        convergence_188947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 57), 'convergence', False)
        # Applying the binary operator 'div' (line 533)
        result_div_188948 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 46), 'div', tol_188946, convergence_188947)
        
        keyword_188949 = result_div_188948
        kwargs_188950 = {'convergence': keyword_188949}
        # Getting the type of 'self' (line 532)
        self_188934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 20), 'self', False)
        # Obtaining the member 'callback' of a type (line 532)
        callback_188935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 20), self_188934, 'callback')
        # Calling callback(args, kwargs) (line 532)
        callback_call_result_188951 = invoke(stypy.reporting.localization.Localization(__file__, 532, 20), callback_188935, *[_scale_parameters_call_result_188944], **kwargs_188950)
        
        # Getting the type of 'True' (line 533)
        True_188952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 73), 'True')
        # Applying the binary operator 'is' (line 532)
        result_is__188953 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 20), 'is', callback_call_result_188951, True_188952)
        
        # Applying the binary operator 'and' (line 531)
        result_and_keyword_188954 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 16), 'and', callback_188933, result_is__188953)
        
        # Testing the type of an if condition (line 531)
        if_condition_188955 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 12), result_and_keyword_188954)
        # Assigning a type to the variable 'if_condition_188955' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'if_condition_188955', if_condition_188955)
        # SSA begins for if statement (line 531)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 535):
        
        # Assigning a Name to a Name (line 535):
        # Getting the type of 'True' (line 535)
        True_188956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 31), 'True')
        # Assigning a type to the variable 'warning_flag' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 16), 'warning_flag', True_188956)
        
        # Assigning a Str to a Name (line 536):
        
        # Assigning a Str to a Name (line 536):
        str_188957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 34), 'str', 'callback function requested stop early by returning True')
        # Assigning a type to the variable 'status_message' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'status_message', str_188957)
        # SSA join for if statement (line 531)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Compare to a Name (line 540):
        
        # Assigning a Compare to a Name (line 540):
        
        
        # Call to std(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'self' (line 540)
        self_188960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 28), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 540)
        population_energies_188961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 28), self_188960, 'population_energies')
        # Processing the call keyword arguments (line 540)
        kwargs_188962 = {}
        # Getting the type of 'np' (line 540)
        np_188958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 21), 'np', False)
        # Obtaining the member 'std' of a type (line 540)
        std_188959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 21), np_188958, 'std')
        # Calling std(args, kwargs) (line 540)
        std_call_result_188963 = invoke(stypy.reporting.localization.Localization(__file__, 540, 21), std_188959, *[population_energies_188961], **kwargs_188962)
        
        # Getting the type of 'self' (line 541)
        self_188964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 21), 'self')
        # Obtaining the member 'atol' of a type (line 541)
        atol_188965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 21), self_188964, 'atol')
        # Getting the type of 'self' (line 542)
        self_188966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 21), 'self')
        # Obtaining the member 'tol' of a type (line 542)
        tol_188967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 21), self_188966, 'tol')
        
        # Call to abs(...): (line 542)
        # Processing the call arguments (line 542)
        
        # Call to mean(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'self' (line 542)
        self_188972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 47), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 542)
        population_energies_188973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 47), self_188972, 'population_energies')
        # Processing the call keyword arguments (line 542)
        kwargs_188974 = {}
        # Getting the type of 'np' (line 542)
        np_188970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 39), 'np', False)
        # Obtaining the member 'mean' of a type (line 542)
        mean_188971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 39), np_188970, 'mean')
        # Calling mean(args, kwargs) (line 542)
        mean_call_result_188975 = invoke(stypy.reporting.localization.Localization(__file__, 542, 39), mean_188971, *[population_energies_188973], **kwargs_188974)
        
        # Processing the call keyword arguments (line 542)
        kwargs_188976 = {}
        # Getting the type of 'np' (line 542)
        np_188968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'np', False)
        # Obtaining the member 'abs' of a type (line 542)
        abs_188969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 32), np_188968, 'abs')
        # Calling abs(args, kwargs) (line 542)
        abs_call_result_188977 = invoke(stypy.reporting.localization.Localization(__file__, 542, 32), abs_188969, *[mean_call_result_188975], **kwargs_188976)
        
        # Applying the binary operator '*' (line 542)
        result_mul_188978 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 21), '*', tol_188967, abs_call_result_188977)
        
        # Applying the binary operator '+' (line 541)
        result_add_188979 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 21), '+', atol_188965, result_mul_188978)
        
        # Applying the binary operator '<=' (line 540)
        result_le_188980 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 21), '<=', std_call_result_188963, result_add_188979)
        
        # Assigning a type to the variable 'intol' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'intol', result_le_188980)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'warning_flag' (line 543)
        warning_flag_188981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 15), 'warning_flag')
        # Getting the type of 'intol' (line 543)
        intol_188982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 31), 'intol')
        # Applying the binary operator 'or' (line 543)
        result_or_keyword_188983 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 15), 'or', warning_flag_188981, intol_188982)
        
        # Testing the type of an if condition (line 543)
        if_condition_188984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 543, 12), result_or_keyword_188983)
        # Assigning a type to the variable 'if_condition_188984' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'if_condition_188984', if_condition_188984)
        # SSA begins for if statement (line 543)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 543)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 514)
        module_type_store.open_ssa_branch('for loop else')
        
        # Assigning a Subscript to a Name (line 547):
        
        # Assigning a Subscript to a Name (line 547):
        
        # Obtaining the type of the subscript
        str_188985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 45), 'str', 'maxiter')
        # Getting the type of '_status_message' (line 547)
        _status_message_188986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 29), '_status_message')
        # Obtaining the member '__getitem__' of a type (line 547)
        getitem___188987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 29), _status_message_188986, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 547)
        subscript_call_result_188988 = invoke(stypy.reporting.localization.Localization(__file__, 547, 29), getitem___188987, str_188985)
        
        # Assigning a type to the variable 'status_message' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'status_message', subscript_call_result_188988)
        
        # Assigning a Name to a Name (line 548):
        
        # Assigning a Name to a Name (line 548):
        # Getting the type of 'True' (line 548)
        True_188989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 27), 'True')
        # Assigning a type to the variable 'warning_flag' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'warning_flag', True_188989)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 550):
        
        # Assigning a Call to a Name (line 550):
        
        # Call to OptimizeResult(...): (line 550)
        # Processing the call keyword arguments (line 550)
        # Getting the type of 'self' (line 551)
        self_188991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 14), 'self', False)
        # Obtaining the member 'x' of a type (line 551)
        x_188992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 14), self_188991, 'x')
        keyword_188993 = x_188992
        
        # Obtaining the type of the subscript
        int_188994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 41), 'int')
        # Getting the type of 'self' (line 552)
        self_188995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 552)
        population_energies_188996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 16), self_188995, 'population_energies')
        # Obtaining the member '__getitem__' of a type (line 552)
        getitem___188997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 16), population_energies_188996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 552)
        subscript_call_result_188998 = invoke(stypy.reporting.localization.Localization(__file__, 552, 16), getitem___188997, int_188994)
        
        keyword_188999 = subscript_call_result_188998
        # Getting the type of 'self' (line 553)
        self_189000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 17), 'self', False)
        # Obtaining the member '_nfev' of a type (line 553)
        _nfev_189001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 17), self_189000, '_nfev')
        keyword_189002 = _nfev_189001
        # Getting the type of 'nit' (line 554)
        nit_189003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'nit', False)
        keyword_189004 = nit_189003
        # Getting the type of 'status_message' (line 555)
        status_message_189005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 20), 'status_message', False)
        keyword_189006 = status_message_189005
        
        # Getting the type of 'warning_flag' (line 556)
        warning_flag_189007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 21), 'warning_flag', False)
        # Getting the type of 'True' (line 556)
        True_189008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 41), 'True', False)
        # Applying the binary operator 'isnot' (line 556)
        result_is_not_189009 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 21), 'isnot', warning_flag_189007, True_189008)
        
        keyword_189010 = result_is_not_189009
        kwargs_189011 = {'success': keyword_189010, 'nfev': keyword_189002, 'fun': keyword_188999, 'x': keyword_188993, 'message': keyword_189006, 'nit': keyword_189004}
        # Getting the type of 'OptimizeResult' (line 550)
        OptimizeResult_188990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 20), 'OptimizeResult', False)
        # Calling OptimizeResult(args, kwargs) (line 550)
        OptimizeResult_call_result_189012 = invoke(stypy.reporting.localization.Localization(__file__, 550, 20), OptimizeResult_188990, *[], **kwargs_189011)
        
        # Assigning a type to the variable 'DE_result' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'DE_result', OptimizeResult_call_result_189012)
        
        # Getting the type of 'self' (line 558)
        self_189013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 11), 'self')
        # Obtaining the member 'polish' of a type (line 558)
        polish_189014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 11), self_189013, 'polish')
        # Testing the type of an if condition (line 558)
        if_condition_189015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 558, 8), polish_189014)
        # Assigning a type to the variable 'if_condition_189015' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'if_condition_189015', if_condition_189015)
        # SSA begins for if statement (line 558)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 559):
        
        # Assigning a Call to a Name (line 559):
        
        # Call to minimize(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'self' (line 559)
        self_189017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 30), 'self', False)
        # Obtaining the member 'func' of a type (line 559)
        func_189018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 30), self_189017, 'func')
        
        # Call to copy(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'DE_result' (line 560)
        DE_result_189021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 38), 'DE_result', False)
        # Obtaining the member 'x' of a type (line 560)
        x_189022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 38), DE_result_189021, 'x')
        # Processing the call keyword arguments (line 560)
        kwargs_189023 = {}
        # Getting the type of 'np' (line 560)
        np_189019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 30), 'np', False)
        # Obtaining the member 'copy' of a type (line 560)
        copy_189020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 30), np_189019, 'copy')
        # Calling copy(args, kwargs) (line 560)
        copy_call_result_189024 = invoke(stypy.reporting.localization.Localization(__file__, 560, 30), copy_189020, *[x_189022], **kwargs_189023)
        
        # Processing the call keyword arguments (line 559)
        str_189025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 37), 'str', 'L-BFGS-B')
        keyword_189026 = str_189025
        # Getting the type of 'self' (line 562)
        self_189027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 37), 'self', False)
        # Obtaining the member 'limits' of a type (line 562)
        limits_189028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 37), self_189027, 'limits')
        # Obtaining the member 'T' of a type (line 562)
        T_189029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 37), limits_189028, 'T')
        keyword_189030 = T_189029
        # Getting the type of 'self' (line 563)
        self_189031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 35), 'self', False)
        # Obtaining the member 'args' of a type (line 563)
        args_189032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 35), self_189031, 'args')
        keyword_189033 = args_189032
        kwargs_189034 = {'args': keyword_189033, 'method': keyword_189026, 'bounds': keyword_189030}
        # Getting the type of 'minimize' (line 559)
        minimize_189016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 21), 'minimize', False)
        # Calling minimize(args, kwargs) (line 559)
        minimize_call_result_189035 = invoke(stypy.reporting.localization.Localization(__file__, 559, 21), minimize_189016, *[func_189018, copy_call_result_189024], **kwargs_189034)
        
        # Assigning a type to the variable 'result' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'result', minimize_call_result_189035)
        
        # Getting the type of 'self' (line 565)
        self_189036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'self')
        # Obtaining the member '_nfev' of a type (line 565)
        _nfev_189037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 12), self_189036, '_nfev')
        # Getting the type of 'result' (line 565)
        result_189038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 26), 'result')
        # Obtaining the member 'nfev' of a type (line 565)
        nfev_189039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 26), result_189038, 'nfev')
        # Applying the binary operator '+=' (line 565)
        result_iadd_189040 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 12), '+=', _nfev_189037, nfev_189039)
        # Getting the type of 'self' (line 565)
        self_189041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'self')
        # Setting the type of the member '_nfev' of a type (line 565)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 12), self_189041, '_nfev', result_iadd_189040)
        
        
        # Assigning a Attribute to a Attribute (line 566):
        
        # Assigning a Attribute to a Attribute (line 566):
        # Getting the type of 'self' (line 566)
        self_189042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 29), 'self')
        # Obtaining the member '_nfev' of a type (line 566)
        _nfev_189043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 29), self_189042, '_nfev')
        # Getting the type of 'DE_result' (line 566)
        DE_result_189044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'DE_result')
        # Setting the type of the member 'nfev' of a type (line 566)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 12), DE_result_189044, 'nfev', _nfev_189043)
        
        
        # Getting the type of 'result' (line 568)
        result_189045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'result')
        # Obtaining the member 'fun' of a type (line 568)
        fun_189046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 15), result_189045, 'fun')
        # Getting the type of 'DE_result' (line 568)
        DE_result_189047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 28), 'DE_result')
        # Obtaining the member 'fun' of a type (line 568)
        fun_189048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 28), DE_result_189047, 'fun')
        # Applying the binary operator '<' (line 568)
        result_lt_189049 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 15), '<', fun_189046, fun_189048)
        
        # Testing the type of an if condition (line 568)
        if_condition_189050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 12), result_lt_189049)
        # Assigning a type to the variable 'if_condition_189050' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'if_condition_189050', if_condition_189050)
        # SSA begins for if statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 569):
        
        # Assigning a Attribute to a Attribute (line 569):
        # Getting the type of 'result' (line 569)
        result_189051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 32), 'result')
        # Obtaining the member 'fun' of a type (line 569)
        fun_189052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 32), result_189051, 'fun')
        # Getting the type of 'DE_result' (line 569)
        DE_result_189053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), 'DE_result')
        # Setting the type of the member 'fun' of a type (line 569)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 16), DE_result_189053, 'fun', fun_189052)
        
        # Assigning a Attribute to a Attribute (line 570):
        
        # Assigning a Attribute to a Attribute (line 570):
        # Getting the type of 'result' (line 570)
        result_189054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 30), 'result')
        # Obtaining the member 'x' of a type (line 570)
        x_189055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 30), result_189054, 'x')
        # Getting the type of 'DE_result' (line 570)
        DE_result_189056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 16), 'DE_result')
        # Setting the type of the member 'x' of a type (line 570)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 16), DE_result_189056, 'x', x_189055)
        
        # Assigning a Attribute to a Attribute (line 571):
        
        # Assigning a Attribute to a Attribute (line 571):
        # Getting the type of 'result' (line 571)
        result_189057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 32), 'result')
        # Obtaining the member 'jac' of a type (line 571)
        jac_189058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 32), result_189057, 'jac')
        # Getting the type of 'DE_result' (line 571)
        DE_result_189059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'DE_result')
        # Setting the type of the member 'jac' of a type (line 571)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 16), DE_result_189059, 'jac', jac_189058)
        
        # Assigning a Attribute to a Subscript (line 573):
        
        # Assigning a Attribute to a Subscript (line 573):
        # Getting the type of 'result' (line 573)
        result_189060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 46), 'result')
        # Obtaining the member 'fun' of a type (line 573)
        fun_189061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 46), result_189060, 'fun')
        # Getting the type of 'self' (line 573)
        self_189062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 16), 'self')
        # Obtaining the member 'population_energies' of a type (line 573)
        population_energies_189063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 16), self_189062, 'population_energies')
        int_189064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 41), 'int')
        # Storing an element on a container (line 573)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 16), population_energies_189063, (int_189064, fun_189061))
        
        # Assigning a Call to a Subscript (line 574):
        
        # Assigning a Call to a Subscript (line 574):
        
        # Call to _unscale_parameters(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'result' (line 574)
        result_189067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 62), 'result', False)
        # Obtaining the member 'x' of a type (line 574)
        x_189068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 62), result_189067, 'x')
        # Processing the call keyword arguments (line 574)
        kwargs_189069 = {}
        # Getting the type of 'self' (line 574)
        self_189065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 37), 'self', False)
        # Obtaining the member '_unscale_parameters' of a type (line 574)
        _unscale_parameters_189066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 37), self_189065, '_unscale_parameters')
        # Calling _unscale_parameters(args, kwargs) (line 574)
        _unscale_parameters_call_result_189070 = invoke(stypy.reporting.localization.Localization(__file__, 574, 37), _unscale_parameters_189066, *[x_189068], **kwargs_189069)
        
        # Getting the type of 'self' (line 574)
        self_189071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 16), 'self')
        # Obtaining the member 'population' of a type (line 574)
        population_189072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 16), self_189071, 'population')
        int_189073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 32), 'int')
        # Storing an element on a container (line 574)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 16), population_189072, (int_189073, _unscale_parameters_call_result_189070))
        # SSA join for if statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 558)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'DE_result' (line 576)
        DE_result_189074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 15), 'DE_result')
        # Assigning a type to the variable 'stypy_return_type' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'stypy_return_type', DE_result_189074)
        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 487)
        stypy_return_type_189075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189075)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_189075


    @norecursion
    def _calculate_population_energies(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_calculate_population_energies'
        module_type_store = module_type_store.open_function_context('_calculate_population_energies', 578, 4, False)
        # Assigning a type to the variable 'self' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._calculate_population_energies')
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._calculate_population_energies.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._calculate_population_energies', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_calculate_population_energies', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_calculate_population_energies(...)' code ##################

        str_189076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, (-1)), 'str', '\n        Calculate the energies of all the population members at the same time.\n        Puts the best member in first place. Useful if the population has just\n        been initialised.\n        ')
        
        
        # Call to enumerate(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'self' (line 584)
        self_189078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 42), 'self', False)
        # Obtaining the member 'population' of a type (line 584)
        population_189079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 42), self_189078, 'population')
        # Processing the call keyword arguments (line 584)
        kwargs_189080 = {}
        # Getting the type of 'enumerate' (line 584)
        enumerate_189077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 32), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 584)
        enumerate_call_result_189081 = invoke(stypy.reporting.localization.Localization(__file__, 584, 32), enumerate_189077, *[population_189079], **kwargs_189080)
        
        # Testing the type of a for loop iterable (line 584)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 584, 8), enumerate_call_result_189081)
        # Getting the type of the for loop variable (line 584)
        for_loop_var_189082 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 584, 8), enumerate_call_result_189081)
        # Assigning a type to the variable 'index' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 8), for_loop_var_189082))
        # Assigning a type to the variable 'candidate' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'candidate', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 8), for_loop_var_189082))
        # SSA begins for a for statement (line 584)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'self' (line 585)
        self_189083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 15), 'self')
        # Obtaining the member '_nfev' of a type (line 585)
        _nfev_189084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 15), self_189083, '_nfev')
        # Getting the type of 'self' (line 585)
        self_189085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 28), 'self')
        # Obtaining the member 'maxfun' of a type (line 585)
        maxfun_189086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 28), self_189085, 'maxfun')
        # Applying the binary operator '>' (line 585)
        result_gt_189087 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 15), '>', _nfev_189084, maxfun_189086)
        
        # Testing the type of an if condition (line 585)
        if_condition_189088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 12), result_gt_189087)
        # Assigning a type to the variable 'if_condition_189088' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'if_condition_189088', if_condition_189088)
        # SSA begins for if statement (line 585)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 585)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 588):
        
        # Assigning a Call to a Name (line 588):
        
        # Call to _scale_parameters(...): (line 588)
        # Processing the call arguments (line 588)
        # Getting the type of 'candidate' (line 588)
        candidate_189091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 48), 'candidate', False)
        # Processing the call keyword arguments (line 588)
        kwargs_189092 = {}
        # Getting the type of 'self' (line 588)
        self_189089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'self', False)
        # Obtaining the member '_scale_parameters' of a type (line 588)
        _scale_parameters_189090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 25), self_189089, '_scale_parameters')
        # Calling _scale_parameters(args, kwargs) (line 588)
        _scale_parameters_call_result_189093 = invoke(stypy.reporting.localization.Localization(__file__, 588, 25), _scale_parameters_189090, *[candidate_189091], **kwargs_189092)
        
        # Assigning a type to the variable 'parameters' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'parameters', _scale_parameters_call_result_189093)
        
        # Assigning a Call to a Subscript (line 589):
        
        # Assigning a Call to a Subscript (line 589):
        
        # Call to func(...): (line 589)
        # Processing the call arguments (line 589)
        # Getting the type of 'parameters' (line 589)
        parameters_189096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 56), 'parameters', False)
        # Getting the type of 'self' (line 590)
        self_189097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 57), 'self', False)
        # Obtaining the member 'args' of a type (line 590)
        args_189098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 57), self_189097, 'args')
        # Processing the call keyword arguments (line 589)
        kwargs_189099 = {}
        # Getting the type of 'self' (line 589)
        self_189094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 46), 'self', False)
        # Obtaining the member 'func' of a type (line 589)
        func_189095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 46), self_189094, 'func')
        # Calling func(args, kwargs) (line 589)
        func_call_result_189100 = invoke(stypy.reporting.localization.Localization(__file__, 589, 46), func_189095, *[parameters_189096, args_189098], **kwargs_189099)
        
        # Getting the type of 'self' (line 589)
        self_189101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'self')
        # Obtaining the member 'population_energies' of a type (line 589)
        population_energies_189102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 12), self_189101, 'population_energies')
        # Getting the type of 'index' (line 589)
        index_189103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 37), 'index')
        # Storing an element on a container (line 589)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 12), population_energies_189102, (index_189103, func_call_result_189100))
        
        # Getting the type of 'self' (line 591)
        self_189104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'self')
        # Obtaining the member '_nfev' of a type (line 591)
        _nfev_189105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 12), self_189104, '_nfev')
        int_189106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 26), 'int')
        # Applying the binary operator '+=' (line 591)
        result_iadd_189107 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 12), '+=', _nfev_189105, int_189106)
        # Getting the type of 'self' (line 591)
        self_189108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'self')
        # Setting the type of the member '_nfev' of a type (line 591)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 12), self_189108, '_nfev', result_iadd_189107)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 593):
        
        # Assigning a Call to a Name (line 593):
        
        # Call to argmin(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'self' (line 593)
        self_189111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 27), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 593)
        population_energies_189112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 27), self_189111, 'population_energies')
        # Processing the call keyword arguments (line 593)
        kwargs_189113 = {}
        # Getting the type of 'np' (line 593)
        np_189109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 17), 'np', False)
        # Obtaining the member 'argmin' of a type (line 593)
        argmin_189110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 17), np_189109, 'argmin')
        # Calling argmin(args, kwargs) (line 593)
        argmin_call_result_189114 = invoke(stypy.reporting.localization.Localization(__file__, 593, 17), argmin_189110, *[population_energies_189112], **kwargs_189113)
        
        # Assigning a type to the variable 'minval' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'minval', argmin_call_result_189114)
        
        # Assigning a Subscript to a Name (line 596):
        
        # Assigning a Subscript to a Name (line 596):
        
        # Obtaining the type of the subscript
        # Getting the type of 'minval' (line 596)
        minval_189115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 49), 'minval')
        # Getting the type of 'self' (line 596)
        self_189116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 24), 'self')
        # Obtaining the member 'population_energies' of a type (line 596)
        population_energies_189117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 24), self_189116, 'population_energies')
        # Obtaining the member '__getitem__' of a type (line 596)
        getitem___189118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 24), population_energies_189117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 596)
        subscript_call_result_189119 = invoke(stypy.reporting.localization.Localization(__file__, 596, 24), getitem___189118, minval_189115)
        
        # Assigning a type to the variable 'lowest_energy' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'lowest_energy', subscript_call_result_189119)
        
        # Assigning a Subscript to a Subscript (line 597):
        
        # Assigning a Subscript to a Subscript (line 597):
        
        # Obtaining the type of the subscript
        int_189120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 68), 'int')
        # Getting the type of 'self' (line 597)
        self_189121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 43), 'self')
        # Obtaining the member 'population_energies' of a type (line 597)
        population_energies_189122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 43), self_189121, 'population_energies')
        # Obtaining the member '__getitem__' of a type (line 597)
        getitem___189123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 43), population_energies_189122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 597)
        subscript_call_result_189124 = invoke(stypy.reporting.localization.Localization(__file__, 597, 43), getitem___189123, int_189120)
        
        # Getting the type of 'self' (line 597)
        self_189125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'self')
        # Obtaining the member 'population_energies' of a type (line 597)
        population_energies_189126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 8), self_189125, 'population_energies')
        # Getting the type of 'minval' (line 597)
        minval_189127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 33), 'minval')
        # Storing an element on a container (line 597)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 8), population_energies_189126, (minval_189127, subscript_call_result_189124))
        
        # Assigning a Name to a Subscript (line 598):
        
        # Assigning a Name to a Subscript (line 598):
        # Getting the type of 'lowest_energy' (line 598)
        lowest_energy_189128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 38), 'lowest_energy')
        # Getting the type of 'self' (line 598)
        self_189129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'self')
        # Obtaining the member 'population_energies' of a type (line 598)
        population_energies_189130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 8), self_189129, 'population_energies')
        int_189131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 33), 'int')
        # Storing an element on a container (line 598)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 8), population_energies_189130, (int_189131, lowest_energy_189128))
        
        # Assigning a Subscript to a Subscript (line 600):
        
        # Assigning a Subscript to a Subscript (line 600):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'list' (line 600)
        list_189132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 600)
        # Adding element type (line 600)
        # Getting the type of 'minval' (line 600)
        minval_189133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 59), 'minval')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 58), list_189132, minval_189133)
        # Adding element type (line 600)
        int_189134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 58), list_189132, int_189134)
        
        slice_189135 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 600, 42), None, None, None)
        # Getting the type of 'self' (line 600)
        self_189136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 42), 'self')
        # Obtaining the member 'population' of a type (line 600)
        population_189137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 42), self_189136, 'population')
        # Obtaining the member '__getitem__' of a type (line 600)
        getitem___189138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 42), population_189137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 600)
        subscript_call_result_189139 = invoke(stypy.reporting.localization.Localization(__file__, 600, 42), getitem___189138, (list_189132, slice_189135))
        
        # Getting the type of 'self' (line 600)
        self_189140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'self')
        # Obtaining the member 'population' of a type (line 600)
        population_189141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 8), self_189140, 'population')
        
        # Obtaining an instance of the builtin type 'list' (line 600)
        list_189142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 600)
        # Adding element type (line 600)
        int_189143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 24), list_189142, int_189143)
        # Adding element type (line 600)
        # Getting the type of 'minval' (line 600)
        minval_189144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 28), 'minval')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 24), list_189142, minval_189144)
        
        slice_189145 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 600, 8), None, None, None)
        # Storing an element on a container (line 600)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 8), population_189141, ((list_189142, slice_189145), subscript_call_result_189139))
        
        # ################# End of '_calculate_population_energies(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_calculate_population_energies' in the type store
        # Getting the type of 'stypy_return_type' (line 578)
        stypy_return_type_189146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189146)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_calculate_population_energies'
        return stypy_return_type_189146


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 602, 4, False)
        # Assigning a type to the variable 'self' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver.__iter__')
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        # Getting the type of 'self' (line 603)
        self_189147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'stypy_return_type', self_189147)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 602)
        stypy_return_type_189148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189148)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_189148


    @norecursion
    def __next__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__next__'
        module_type_store = module_type_store.open_function_context('__next__', 605, 4, False)
        # Assigning a type to the variable 'self' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver.__next__')
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver.__next__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.__next__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__next__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__next__(...)' code ##################

        str_189149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, (-1)), 'str', '\n        Evolve the population by a single generation\n\n        Returns\n        -------\n        x : ndarray\n            The best solution from the solver.\n        fun : float\n            Value of objective function obtained from the best solution.\n        ')
        
        
        # Call to all(...): (line 618)
        # Processing the call arguments (line 618)
        
        # Call to isinf(...): (line 618)
        # Processing the call arguments (line 618)
        # Getting the type of 'self' (line 618)
        self_189154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 27), 'self', False)
        # Obtaining the member 'population_energies' of a type (line 618)
        population_energies_189155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 27), self_189154, 'population_energies')
        # Processing the call keyword arguments (line 618)
        kwargs_189156 = {}
        # Getting the type of 'np' (line 618)
        np_189152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 18), 'np', False)
        # Obtaining the member 'isinf' of a type (line 618)
        isinf_189153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 18), np_189152, 'isinf')
        # Calling isinf(args, kwargs) (line 618)
        isinf_call_result_189157 = invoke(stypy.reporting.localization.Localization(__file__, 618, 18), isinf_189153, *[population_energies_189155], **kwargs_189156)
        
        # Processing the call keyword arguments (line 618)
        kwargs_189158 = {}
        # Getting the type of 'np' (line 618)
        np_189150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 11), 'np', False)
        # Obtaining the member 'all' of a type (line 618)
        all_189151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 11), np_189150, 'all')
        # Calling all(args, kwargs) (line 618)
        all_call_result_189159 = invoke(stypy.reporting.localization.Localization(__file__, 618, 11), all_189151, *[isinf_call_result_189157], **kwargs_189158)
        
        # Testing the type of an if condition (line 618)
        if_condition_189160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 618, 8), all_call_result_189159)
        # Assigning a type to the variable 'if_condition_189160' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'if_condition_189160', if_condition_189160)
        # SSA begins for if statement (line 618)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _calculate_population_energies(...): (line 619)
        # Processing the call keyword arguments (line 619)
        kwargs_189163 = {}
        # Getting the type of 'self' (line 619)
        self_189161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'self', False)
        # Obtaining the member '_calculate_population_energies' of a type (line 619)
        _calculate_population_energies_189162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 12), self_189161, '_calculate_population_energies')
        # Calling _calculate_population_energies(args, kwargs) (line 619)
        _calculate_population_energies_call_result_189164 = invoke(stypy.reporting.localization.Localization(__file__, 619, 12), _calculate_population_energies_189162, *[], **kwargs_189163)
        
        # SSA join for if statement (line 618)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 621)
        self_189165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 11), 'self')
        # Obtaining the member 'dither' of a type (line 621)
        dither_189166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 11), self_189165, 'dither')
        # Getting the type of 'None' (line 621)
        None_189167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 30), 'None')
        # Applying the binary operator 'isnot' (line 621)
        result_is_not_189168 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 11), 'isnot', dither_189166, None_189167)
        
        # Testing the type of an if condition (line 621)
        if_condition_189169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 621, 8), result_is_not_189168)
        # Assigning a type to the variable 'if_condition_189169' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'if_condition_189169', if_condition_189169)
        # SSA begins for if statement (line 621)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Attribute (line 622):
        
        # Assigning a BinOp to a Attribute (line 622):
        
        # Call to rand(...): (line 622)
        # Processing the call keyword arguments (line 622)
        kwargs_189173 = {}
        # Getting the type of 'self' (line 622)
        self_189170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 26), 'self', False)
        # Obtaining the member 'random_number_generator' of a type (line 622)
        random_number_generator_189171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 26), self_189170, 'random_number_generator')
        # Obtaining the member 'rand' of a type (line 622)
        rand_189172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 26), random_number_generator_189171, 'rand')
        # Calling rand(args, kwargs) (line 622)
        rand_call_result_189174 = invoke(stypy.reporting.localization.Localization(__file__, 622, 26), rand_189172, *[], **kwargs_189173)
        
        
        # Obtaining the type of the subscript
        int_189175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 41), 'int')
        # Getting the type of 'self' (line 623)
        self_189176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 29), 'self')
        # Obtaining the member 'dither' of a type (line 623)
        dither_189177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 29), self_189176, 'dither')
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___189178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 29), dither_189177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_189179 = invoke(stypy.reporting.localization.Localization(__file__, 623, 29), getitem___189178, int_189175)
        
        
        # Obtaining the type of the subscript
        int_189180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 58), 'int')
        # Getting the type of 'self' (line 623)
        self_189181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 46), 'self')
        # Obtaining the member 'dither' of a type (line 623)
        dither_189182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 46), self_189181, 'dither')
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___189183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 46), dither_189182, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_189184 = invoke(stypy.reporting.localization.Localization(__file__, 623, 46), getitem___189183, int_189180)
        
        # Applying the binary operator '-' (line 623)
        result_sub_189185 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 29), '-', subscript_call_result_189179, subscript_call_result_189184)
        
        # Applying the binary operator '*' (line 622)
        result_mul_189186 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 26), '*', rand_call_result_189174, result_sub_189185)
        
        
        # Obtaining the type of the subscript
        int_189187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 76), 'int')
        # Getting the type of 'self' (line 623)
        self_189188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 64), 'self')
        # Obtaining the member 'dither' of a type (line 623)
        dither_189189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 64), self_189188, 'dither')
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___189190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 64), dither_189189, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_189191 = invoke(stypy.reporting.localization.Localization(__file__, 623, 64), getitem___189190, int_189187)
        
        # Applying the binary operator '+' (line 622)
        result_add_189192 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 26), '+', result_mul_189186, subscript_call_result_189191)
        
        # Getting the type of 'self' (line 622)
        self_189193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'self')
        # Setting the type of the member 'scale' of a type (line 622)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 12), self_189193, 'scale', result_add_189192)
        # SSA join for if statement (line 621)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to range(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'self' (line 625)
        self_189195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 31), 'self', False)
        # Obtaining the member 'num_population_members' of a type (line 625)
        num_population_members_189196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 31), self_189195, 'num_population_members')
        # Processing the call keyword arguments (line 625)
        kwargs_189197 = {}
        # Getting the type of 'range' (line 625)
        range_189194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 25), 'range', False)
        # Calling range(args, kwargs) (line 625)
        range_call_result_189198 = invoke(stypy.reporting.localization.Localization(__file__, 625, 25), range_189194, *[num_population_members_189196], **kwargs_189197)
        
        # Testing the type of a for loop iterable (line 625)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 625, 8), range_call_result_189198)
        # Getting the type of the for loop variable (line 625)
        for_loop_var_189199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 625, 8), range_call_result_189198)
        # Assigning a type to the variable 'candidate' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'candidate', for_loop_var_189199)
        # SSA begins for a for statement (line 625)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'self' (line 626)
        self_189200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 15), 'self')
        # Obtaining the member '_nfev' of a type (line 626)
        _nfev_189201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 15), self_189200, '_nfev')
        # Getting the type of 'self' (line 626)
        self_189202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 28), 'self')
        # Obtaining the member 'maxfun' of a type (line 626)
        maxfun_189203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 28), self_189202, 'maxfun')
        # Applying the binary operator '>' (line 626)
        result_gt_189204 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 15), '>', _nfev_189201, maxfun_189203)
        
        # Testing the type of an if condition (line 626)
        if_condition_189205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 12), result_gt_189204)
        # Assigning a type to the variable 'if_condition_189205' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'if_condition_189205', if_condition_189205)
        # SSA begins for if statement (line 626)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'StopIteration' (line 627)
        StopIteration_189206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 22), 'StopIteration')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 627, 16), StopIteration_189206, 'raise parameter', BaseException)
        # SSA join for if statement (line 626)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 630):
        
        # Assigning a Call to a Name (line 630):
        
        # Call to _mutate(...): (line 630)
        # Processing the call arguments (line 630)
        # Getting the type of 'candidate' (line 630)
        candidate_189209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 33), 'candidate', False)
        # Processing the call keyword arguments (line 630)
        kwargs_189210 = {}
        # Getting the type of 'self' (line 630)
        self_189207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 20), 'self', False)
        # Obtaining the member '_mutate' of a type (line 630)
        _mutate_189208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 20), self_189207, '_mutate')
        # Calling _mutate(args, kwargs) (line 630)
        _mutate_call_result_189211 = invoke(stypy.reporting.localization.Localization(__file__, 630, 20), _mutate_189208, *[candidate_189209], **kwargs_189210)
        
        # Assigning a type to the variable 'trial' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'trial', _mutate_call_result_189211)
        
        # Call to _ensure_constraint(...): (line 633)
        # Processing the call arguments (line 633)
        # Getting the type of 'trial' (line 633)
        trial_189214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 36), 'trial', False)
        # Processing the call keyword arguments (line 633)
        kwargs_189215 = {}
        # Getting the type of 'self' (line 633)
        self_189212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 12), 'self', False)
        # Obtaining the member '_ensure_constraint' of a type (line 633)
        _ensure_constraint_189213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 12), self_189212, '_ensure_constraint')
        # Calling _ensure_constraint(args, kwargs) (line 633)
        _ensure_constraint_call_result_189216 = invoke(stypy.reporting.localization.Localization(__file__, 633, 12), _ensure_constraint_189213, *[trial_189214], **kwargs_189215)
        
        
        # Assigning a Call to a Name (line 636):
        
        # Assigning a Call to a Name (line 636):
        
        # Call to _scale_parameters(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 'trial' (line 636)
        trial_189219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 48), 'trial', False)
        # Processing the call keyword arguments (line 636)
        kwargs_189220 = {}
        # Getting the type of 'self' (line 636)
        self_189217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 25), 'self', False)
        # Obtaining the member '_scale_parameters' of a type (line 636)
        _scale_parameters_189218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 25), self_189217, '_scale_parameters')
        # Calling _scale_parameters(args, kwargs) (line 636)
        _scale_parameters_call_result_189221 = invoke(stypy.reporting.localization.Localization(__file__, 636, 25), _scale_parameters_189218, *[trial_189219], **kwargs_189220)
        
        # Assigning a type to the variable 'parameters' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'parameters', _scale_parameters_call_result_189221)
        
        # Assigning a Call to a Name (line 639):
        
        # Assigning a Call to a Name (line 639):
        
        # Call to func(...): (line 639)
        # Processing the call arguments (line 639)
        # Getting the type of 'parameters' (line 639)
        parameters_189224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 31), 'parameters', False)
        # Getting the type of 'self' (line 639)
        self_189225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 44), 'self', False)
        # Obtaining the member 'args' of a type (line 639)
        args_189226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 44), self_189225, 'args')
        # Processing the call keyword arguments (line 639)
        kwargs_189227 = {}
        # Getting the type of 'self' (line 639)
        self_189222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 21), 'self', False)
        # Obtaining the member 'func' of a type (line 639)
        func_189223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 21), self_189222, 'func')
        # Calling func(args, kwargs) (line 639)
        func_call_result_189228 = invoke(stypy.reporting.localization.Localization(__file__, 639, 21), func_189223, *[parameters_189224, args_189226], **kwargs_189227)
        
        # Assigning a type to the variable 'energy' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'energy', func_call_result_189228)
        
        # Getting the type of 'self' (line 640)
        self_189229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'self')
        # Obtaining the member '_nfev' of a type (line 640)
        _nfev_189230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 12), self_189229, '_nfev')
        int_189231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 26), 'int')
        # Applying the binary operator '+=' (line 640)
        result_iadd_189232 = python_operator(stypy.reporting.localization.Localization(__file__, 640, 12), '+=', _nfev_189230, int_189231)
        # Getting the type of 'self' (line 640)
        self_189233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'self')
        # Setting the type of the member '_nfev' of a type (line 640)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 12), self_189233, '_nfev', result_iadd_189232)
        
        
        
        # Getting the type of 'energy' (line 644)
        energy_189234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 15), 'energy')
        
        # Obtaining the type of the subscript
        # Getting the type of 'candidate' (line 644)
        candidate_189235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 49), 'candidate')
        # Getting the type of 'self' (line 644)
        self_189236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 24), 'self')
        # Obtaining the member 'population_energies' of a type (line 644)
        population_energies_189237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 24), self_189236, 'population_energies')
        # Obtaining the member '__getitem__' of a type (line 644)
        getitem___189238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 24), population_energies_189237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 644)
        subscript_call_result_189239 = invoke(stypy.reporting.localization.Localization(__file__, 644, 24), getitem___189238, candidate_189235)
        
        # Applying the binary operator '<' (line 644)
        result_lt_189240 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 15), '<', energy_189234, subscript_call_result_189239)
        
        # Testing the type of an if condition (line 644)
        if_condition_189241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 644, 12), result_lt_189240)
        # Assigning a type to the variable 'if_condition_189241' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'if_condition_189241', if_condition_189241)
        # SSA begins for if statement (line 644)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 645):
        
        # Assigning a Name to a Subscript (line 645):
        # Getting the type of 'trial' (line 645)
        trial_189242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 45), 'trial')
        # Getting the type of 'self' (line 645)
        self_189243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 16), 'self')
        # Obtaining the member 'population' of a type (line 645)
        population_189244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 16), self_189243, 'population')
        # Getting the type of 'candidate' (line 645)
        candidate_189245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 32), 'candidate')
        # Storing an element on a container (line 645)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 16), population_189244, (candidate_189245, trial_189242))
        
        # Assigning a Name to a Subscript (line 646):
        
        # Assigning a Name to a Subscript (line 646):
        # Getting the type of 'energy' (line 646)
        energy_189246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 54), 'energy')
        # Getting the type of 'self' (line 646)
        self_189247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'self')
        # Obtaining the member 'population_energies' of a type (line 646)
        population_energies_189248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 16), self_189247, 'population_energies')
        # Getting the type of 'candidate' (line 646)
        candidate_189249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 41), 'candidate')
        # Storing an element on a container (line 646)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 646, 16), population_energies_189248, (candidate_189249, energy_189246))
        
        
        # Getting the type of 'energy' (line 650)
        energy_189250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 19), 'energy')
        
        # Obtaining the type of the subscript
        int_189251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 53), 'int')
        # Getting the type of 'self' (line 650)
        self_189252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 28), 'self')
        # Obtaining the member 'population_energies' of a type (line 650)
        population_energies_189253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 28), self_189252, 'population_energies')
        # Obtaining the member '__getitem__' of a type (line 650)
        getitem___189254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 28), population_energies_189253, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 650)
        subscript_call_result_189255 = invoke(stypy.reporting.localization.Localization(__file__, 650, 28), getitem___189254, int_189251)
        
        # Applying the binary operator '<' (line 650)
        result_lt_189256 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 19), '<', energy_189250, subscript_call_result_189255)
        
        # Testing the type of an if condition (line 650)
        if_condition_189257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 16), result_lt_189256)
        # Assigning a type to the variable 'if_condition_189257' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'if_condition_189257', if_condition_189257)
        # SSA begins for if statement (line 650)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 651):
        
        # Assigning a Name to a Subscript (line 651):
        # Getting the type of 'energy' (line 651)
        energy_189258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 50), 'energy')
        # Getting the type of 'self' (line 651)
        self_189259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 20), 'self')
        # Obtaining the member 'population_energies' of a type (line 651)
        population_energies_189260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 20), self_189259, 'population_energies')
        int_189261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 45), 'int')
        # Storing an element on a container (line 651)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 651, 20), population_energies_189260, (int_189261, energy_189258))
        
        # Assigning a Name to a Subscript (line 652):
        
        # Assigning a Name to a Subscript (line 652):
        # Getting the type of 'trial' (line 652)
        trial_189262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 41), 'trial')
        # Getting the type of 'self' (line 652)
        self_189263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 20), 'self')
        # Obtaining the member 'population' of a type (line 652)
        population_189264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 20), self_189263, 'population')
        int_189265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 36), 'int')
        # Storing an element on a container (line 652)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 20), population_189264, (int_189265, trial_189262))
        # SSA join for if statement (line 650)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 644)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 654)
        tuple_189266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 654)
        # Adding element type (line 654)
        # Getting the type of 'self' (line 654)
        self_189267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 15), 'self')
        # Obtaining the member 'x' of a type (line 654)
        x_189268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 15), self_189267, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 15), tuple_189266, x_189268)
        # Adding element type (line 654)
        
        # Obtaining the type of the subscript
        int_189269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 48), 'int')
        # Getting the type of 'self' (line 654)
        self_189270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 23), 'self')
        # Obtaining the member 'population_energies' of a type (line 654)
        population_energies_189271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 23), self_189270, 'population_energies')
        # Obtaining the member '__getitem__' of a type (line 654)
        getitem___189272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 23), population_energies_189271, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 654)
        subscript_call_result_189273 = invoke(stypy.reporting.localization.Localization(__file__, 654, 23), getitem___189272, int_189269)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 15), tuple_189266, subscript_call_result_189273)
        
        # Assigning a type to the variable 'stypy_return_type' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'stypy_return_type', tuple_189266)
        
        # ################# End of '__next__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__next__' in the type store
        # Getting the type of 'stypy_return_type' (line 605)
        stypy_return_type_189274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189274)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__next__'
        return stypy_return_type_189274


    @norecursion
    def next(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'next'
        module_type_store = module_type_store.open_function_context('next', 656, 4, False)
        # Assigning a type to the variable 'self' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver.next')
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_param_names_list', [])
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver.next.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver.next', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'next', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'next(...)' code ##################

        str_189275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, (-1)), 'str', '\n        Evolve the population by a single generation\n\n        Returns\n        -------\n        x : ndarray\n            The best solution from the solver.\n        fun : float\n            Value of objective function obtained from the best solution.\n        ')
        
        # Call to __next__(...): (line 668)
        # Processing the call keyword arguments (line 668)
        kwargs_189278 = {}
        # Getting the type of 'self' (line 668)
        self_189276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'self', False)
        # Obtaining the member '__next__' of a type (line 668)
        next___189277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 15), self_189276, '__next__')
        # Calling __next__(args, kwargs) (line 668)
        next___call_result_189279 = invoke(stypy.reporting.localization.Localization(__file__, 668, 15), next___189277, *[], **kwargs_189278)
        
        # Assigning a type to the variable 'stypy_return_type' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'stypy_return_type', next___call_result_189279)
        
        # ################# End of 'next(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'next' in the type store
        # Getting the type of 'stypy_return_type' (line 656)
        stypy_return_type_189280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'next'
        return stypy_return_type_189280


    @norecursion
    def _scale_parameters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_scale_parameters'
        module_type_store = module_type_store.open_function_context('_scale_parameters', 670, 4, False)
        # Assigning a type to the variable 'self' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._scale_parameters')
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_param_names_list', ['trial'])
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._scale_parameters.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._scale_parameters', ['trial'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_scale_parameters', localization, ['trial'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_scale_parameters(...)' code ##################

        str_189281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, (-1)), 'str', '\n        scale from a number between 0 and 1 to parameters.\n        ')
        # Getting the type of 'self' (line 674)
        self_189282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 15), 'self')
        # Obtaining the member '__scale_arg1' of a type (line 674)
        scale_arg1_189283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 15), self_189282, '__scale_arg1')
        # Getting the type of 'trial' (line 674)
        trial_189284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 36), 'trial')
        float_189285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 44), 'float')
        # Applying the binary operator '-' (line 674)
        result_sub_189286 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 36), '-', trial_189284, float_189285)
        
        # Getting the type of 'self' (line 674)
        self_189287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 51), 'self')
        # Obtaining the member '__scale_arg2' of a type (line 674)
        scale_arg2_189288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 51), self_189287, '__scale_arg2')
        # Applying the binary operator '*' (line 674)
        result_mul_189289 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 35), '*', result_sub_189286, scale_arg2_189288)
        
        # Applying the binary operator '+' (line 674)
        result_add_189290 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 15), '+', scale_arg1_189283, result_mul_189289)
        
        # Assigning a type to the variable 'stypy_return_type' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'stypy_return_type', result_add_189290)
        
        # ################# End of '_scale_parameters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_scale_parameters' in the type store
        # Getting the type of 'stypy_return_type' (line 670)
        stypy_return_type_189291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_scale_parameters'
        return stypy_return_type_189291


    @norecursion
    def _unscale_parameters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_unscale_parameters'
        module_type_store = module_type_store.open_function_context('_unscale_parameters', 676, 4, False)
        # Assigning a type to the variable 'self' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._unscale_parameters')
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_param_names_list', ['parameters'])
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._unscale_parameters.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._unscale_parameters', ['parameters'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_unscale_parameters', localization, ['parameters'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_unscale_parameters(...)' code ##################

        str_189292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, (-1)), 'str', '\n        scale from parameters to a number between 0 and 1.\n        ')
        # Getting the type of 'parameters' (line 680)
        parameters_189293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 16), 'parameters')
        # Getting the type of 'self' (line 680)
        self_189294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 29), 'self')
        # Obtaining the member '__scale_arg1' of a type (line 680)
        scale_arg1_189295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 29), self_189294, '__scale_arg1')
        # Applying the binary operator '-' (line 680)
        result_sub_189296 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 16), '-', parameters_189293, scale_arg1_189295)
        
        # Getting the type of 'self' (line 680)
        self_189297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 50), 'self')
        # Obtaining the member '__scale_arg2' of a type (line 680)
        scale_arg2_189298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 50), self_189297, '__scale_arg2')
        # Applying the binary operator 'div' (line 680)
        result_div_189299 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 15), 'div', result_sub_189296, scale_arg2_189298)
        
        float_189300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 70), 'float')
        # Applying the binary operator '+' (line 680)
        result_add_189301 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 15), '+', result_div_189299, float_189300)
        
        # Assigning a type to the variable 'stypy_return_type' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'stypy_return_type', result_add_189301)
        
        # ################# End of '_unscale_parameters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_unscale_parameters' in the type store
        # Getting the type of 'stypy_return_type' (line 676)
        stypy_return_type_189302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_unscale_parameters'
        return stypy_return_type_189302


    @norecursion
    def _ensure_constraint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ensure_constraint'
        module_type_store = module_type_store.open_function_context('_ensure_constraint', 682, 4, False)
        # Assigning a type to the variable 'self' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._ensure_constraint')
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_param_names_list', ['trial'])
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._ensure_constraint.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._ensure_constraint', ['trial'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ensure_constraint', localization, ['trial'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ensure_constraint(...)' code ##################

        str_189303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, (-1)), 'str', '\n        make sure the parameters lie between the limits\n        ')
        
        
        # Call to enumerate(...): (line 686)
        # Processing the call arguments (line 686)
        # Getting the type of 'trial' (line 686)
        trial_189305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 38), 'trial', False)
        # Processing the call keyword arguments (line 686)
        kwargs_189306 = {}
        # Getting the type of 'enumerate' (line 686)
        enumerate_189304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 28), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 686)
        enumerate_call_result_189307 = invoke(stypy.reporting.localization.Localization(__file__, 686, 28), enumerate_189304, *[trial_189305], **kwargs_189306)
        
        # Testing the type of a for loop iterable (line 686)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 686, 8), enumerate_call_result_189307)
        # Getting the type of the for loop variable (line 686)
        for_loop_var_189308 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 686, 8), enumerate_call_result_189307)
        # Assigning a type to the variable 'index' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 8), for_loop_var_189308))
        # Assigning a type to the variable 'param' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'param', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 8), for_loop_var_189308))
        # SSA begins for a for statement (line 686)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'param' (line 687)
        param_189309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 15), 'param')
        int_189310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 23), 'int')
        # Applying the binary operator '>' (line 687)
        result_gt_189311 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 15), '>', param_189309, int_189310)
        
        
        # Getting the type of 'param' (line 687)
        param_189312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 28), 'param')
        int_189313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 36), 'int')
        # Applying the binary operator '<' (line 687)
        result_lt_189314 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 28), '<', param_189312, int_189313)
        
        # Applying the binary operator 'or' (line 687)
        result_or_keyword_189315 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 15), 'or', result_gt_189311, result_lt_189314)
        
        # Testing the type of an if condition (line 687)
        if_condition_189316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 687, 12), result_or_keyword_189315)
        # Assigning a type to the variable 'if_condition_189316' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 12), 'if_condition_189316', if_condition_189316)
        # SSA begins for if statement (line 687)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 688):
        
        # Assigning a Call to a Subscript (line 688):
        
        # Call to rand(...): (line 688)
        # Processing the call keyword arguments (line 688)
        kwargs_189320 = {}
        # Getting the type of 'self' (line 688)
        self_189317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 31), 'self', False)
        # Obtaining the member 'random_number_generator' of a type (line 688)
        random_number_generator_189318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 31), self_189317, 'random_number_generator')
        # Obtaining the member 'rand' of a type (line 688)
        rand_189319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 31), random_number_generator_189318, 'rand')
        # Calling rand(args, kwargs) (line 688)
        rand_call_result_189321 = invoke(stypy.reporting.localization.Localization(__file__, 688, 31), rand_189319, *[], **kwargs_189320)
        
        # Getting the type of 'trial' (line 688)
        trial_189322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 16), 'trial')
        # Getting the type of 'index' (line 688)
        index_189323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 22), 'index')
        # Storing an element on a container (line 688)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 688, 16), trial_189322, (index_189323, rand_call_result_189321))
        # SSA join for if statement (line 687)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_ensure_constraint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ensure_constraint' in the type store
        # Getting the type of 'stypy_return_type' (line 682)
        stypy_return_type_189324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189324)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ensure_constraint'
        return stypy_return_type_189324


    @norecursion
    def _mutate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mutate'
        module_type_store = module_type_store.open_function_context('_mutate', 690, 4, False)
        # Assigning a type to the variable 'self' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._mutate')
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_param_names_list', ['candidate'])
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._mutate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._mutate', ['candidate'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mutate', localization, ['candidate'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mutate(...)' code ##################

        str_189325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, (-1)), 'str', '\n        create a trial vector based on a mutation strategy\n        ')
        
        # Assigning a Call to a Name (line 694):
        
        # Assigning a Call to a Name (line 694):
        
        # Call to copy(...): (line 694)
        # Processing the call arguments (line 694)
        
        # Obtaining the type of the subscript
        # Getting the type of 'candidate' (line 694)
        candidate_189328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 40), 'candidate', False)
        # Getting the type of 'self' (line 694)
        self_189329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 24), 'self', False)
        # Obtaining the member 'population' of a type (line 694)
        population_189330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 24), self_189329, 'population')
        # Obtaining the member '__getitem__' of a type (line 694)
        getitem___189331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 24), population_189330, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 694)
        subscript_call_result_189332 = invoke(stypy.reporting.localization.Localization(__file__, 694, 24), getitem___189331, candidate_189328)
        
        # Processing the call keyword arguments (line 694)
        kwargs_189333 = {}
        # Getting the type of 'np' (line 694)
        np_189326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'np', False)
        # Obtaining the member 'copy' of a type (line 694)
        copy_189327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), np_189326, 'copy')
        # Calling copy(args, kwargs) (line 694)
        copy_call_result_189334 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), copy_189327, *[subscript_call_result_189332], **kwargs_189333)
        
        # Assigning a type to the variable 'trial' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'trial', copy_call_result_189334)
        
        # Assigning a Attribute to a Name (line 696):
        
        # Assigning a Attribute to a Name (line 696):
        # Getting the type of 'self' (line 696)
        self_189335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 14), 'self')
        # Obtaining the member 'random_number_generator' of a type (line 696)
        random_number_generator_189336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 14), self_189335, 'random_number_generator')
        # Assigning a type to the variable 'rng' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'rng', random_number_generator_189336)
        
        # Assigning a Call to a Name (line 698):
        
        # Assigning a Call to a Name (line 698):
        
        # Call to randint(...): (line 698)
        # Processing the call arguments (line 698)
        int_189339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 33), 'int')
        # Getting the type of 'self' (line 698)
        self_189340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), 'self', False)
        # Obtaining the member 'parameter_count' of a type (line 698)
        parameter_count_189341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 36), self_189340, 'parameter_count')
        # Processing the call keyword arguments (line 698)
        kwargs_189342 = {}
        # Getting the type of 'rng' (line 698)
        rng_189337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 21), 'rng', False)
        # Obtaining the member 'randint' of a type (line 698)
        randint_189338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 21), rng_189337, 'randint')
        # Calling randint(args, kwargs) (line 698)
        randint_call_result_189343 = invoke(stypy.reporting.localization.Localization(__file__, 698, 21), randint_189338, *[int_189339, parameter_count_189341], **kwargs_189342)
        
        # Assigning a type to the variable 'fill_point' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'fill_point', randint_call_result_189343)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 700)
        self_189344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'self')
        # Obtaining the member 'strategy' of a type (line 700)
        strategy_189345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 12), self_189344, 'strategy')
        str_189346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 29), 'str', 'randtobest1exp')
        # Applying the binary operator '==' (line 700)
        result_eq_189347 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 12), '==', strategy_189345, str_189346)
        
        
        # Getting the type of 'self' (line 701)
        self_189348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'self')
        # Obtaining the member 'strategy' of a type (line 701)
        strategy_189349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 16), self_189348, 'strategy')
        str_189350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 33), 'str', 'randtobest1bin')
        # Applying the binary operator '==' (line 701)
        result_eq_189351 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 16), '==', strategy_189349, str_189350)
        
        # Applying the binary operator 'or' (line 700)
        result_or_keyword_189352 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 12), 'or', result_eq_189347, result_eq_189351)
        
        # Testing the type of an if condition (line 700)
        if_condition_189353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 700, 8), result_or_keyword_189352)
        # Assigning a type to the variable 'if_condition_189353' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'if_condition_189353', if_condition_189353)
        # SSA begins for if statement (line 700)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 702):
        
        # Assigning a Call to a Name (line 702):
        
        # Call to mutation_func(...): (line 702)
        # Processing the call arguments (line 702)
        # Getting the type of 'candidate' (line 702)
        candidate_189356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 40), 'candidate', False)
        
        # Call to _select_samples(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'candidate' (line 703)
        candidate_189359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 61), 'candidate', False)
        int_189360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 72), 'int')
        # Processing the call keyword arguments (line 703)
        kwargs_189361 = {}
        # Getting the type of 'self' (line 703)
        self_189357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 40), 'self', False)
        # Obtaining the member '_select_samples' of a type (line 703)
        _select_samples_189358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 40), self_189357, '_select_samples')
        # Calling _select_samples(args, kwargs) (line 703)
        _select_samples_call_result_189362 = invoke(stypy.reporting.localization.Localization(__file__, 703, 40), _select_samples_189358, *[candidate_189359, int_189360], **kwargs_189361)
        
        # Processing the call keyword arguments (line 702)
        kwargs_189363 = {}
        # Getting the type of 'self' (line 702)
        self_189354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 21), 'self', False)
        # Obtaining the member 'mutation_func' of a type (line 702)
        mutation_func_189355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 21), self_189354, 'mutation_func')
        # Calling mutation_func(args, kwargs) (line 702)
        mutation_func_call_result_189364 = invoke(stypy.reporting.localization.Localization(__file__, 702, 21), mutation_func_189355, *[candidate_189356, _select_samples_call_result_189362], **kwargs_189363)
        
        # Assigning a type to the variable 'bprime' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'bprime', mutation_func_call_result_189364)
        # SSA branch for the else part of an if statement (line 700)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 705):
        
        # Assigning a Call to a Name (line 705):
        
        # Call to mutation_func(...): (line 705)
        # Processing the call arguments (line 705)
        
        # Call to _select_samples(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'candidate' (line 705)
        candidate_189369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 61), 'candidate', False)
        int_189370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 72), 'int')
        # Processing the call keyword arguments (line 705)
        kwargs_189371 = {}
        # Getting the type of 'self' (line 705)
        self_189367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 40), 'self', False)
        # Obtaining the member '_select_samples' of a type (line 705)
        _select_samples_189368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 40), self_189367, '_select_samples')
        # Calling _select_samples(args, kwargs) (line 705)
        _select_samples_call_result_189372 = invoke(stypy.reporting.localization.Localization(__file__, 705, 40), _select_samples_189368, *[candidate_189369, int_189370], **kwargs_189371)
        
        # Processing the call keyword arguments (line 705)
        kwargs_189373 = {}
        # Getting the type of 'self' (line 705)
        self_189365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 21), 'self', False)
        # Obtaining the member 'mutation_func' of a type (line 705)
        mutation_func_189366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 21), self_189365, 'mutation_func')
        # Calling mutation_func(args, kwargs) (line 705)
        mutation_func_call_result_189374 = invoke(stypy.reporting.localization.Localization(__file__, 705, 21), mutation_func_189366, *[_select_samples_call_result_189372], **kwargs_189373)
        
        # Assigning a type to the variable 'bprime' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 12), 'bprime', mutation_func_call_result_189374)
        # SSA join for if statement (line 700)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 707)
        self_189375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 11), 'self')
        # Obtaining the member 'strategy' of a type (line 707)
        strategy_189376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 11), self_189375, 'strategy')
        # Getting the type of 'self' (line 707)
        self_189377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 28), 'self')
        # Obtaining the member '_binomial' of a type (line 707)
        _binomial_189378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 28), self_189377, '_binomial')
        # Applying the binary operator 'in' (line 707)
        result_contains_189379 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 11), 'in', strategy_189376, _binomial_189378)
        
        # Testing the type of an if condition (line 707)
        if_condition_189380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 707, 8), result_contains_189379)
        # Assigning a type to the variable 'if_condition_189380' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'if_condition_189380', if_condition_189380)
        # SSA begins for if statement (line 707)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 708):
        
        # Assigning a Call to a Name (line 708):
        
        # Call to rand(...): (line 708)
        # Processing the call arguments (line 708)
        # Getting the type of 'self' (line 708)
        self_189383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 34), 'self', False)
        # Obtaining the member 'parameter_count' of a type (line 708)
        parameter_count_189384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 34), self_189383, 'parameter_count')
        # Processing the call keyword arguments (line 708)
        kwargs_189385 = {}
        # Getting the type of 'rng' (line 708)
        rng_189381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 25), 'rng', False)
        # Obtaining the member 'rand' of a type (line 708)
        rand_189382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 25), rng_189381, 'rand')
        # Calling rand(args, kwargs) (line 708)
        rand_call_result_189386 = invoke(stypy.reporting.localization.Localization(__file__, 708, 25), rand_189382, *[parameter_count_189384], **kwargs_189385)
        
        # Assigning a type to the variable 'crossovers' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 12), 'crossovers', rand_call_result_189386)
        
        # Assigning a Compare to a Name (line 709):
        
        # Assigning a Compare to a Name (line 709):
        
        # Getting the type of 'crossovers' (line 709)
        crossovers_189387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 25), 'crossovers')
        # Getting the type of 'self' (line 709)
        self_189388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 38), 'self')
        # Obtaining the member 'cross_over_probability' of a type (line 709)
        cross_over_probability_189389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 38), self_189388, 'cross_over_probability')
        # Applying the binary operator '<' (line 709)
        result_lt_189390 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 25), '<', crossovers_189387, cross_over_probability_189389)
        
        # Assigning a type to the variable 'crossovers' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 12), 'crossovers', result_lt_189390)
        
        # Assigning a Name to a Subscript (line 714):
        
        # Assigning a Name to a Subscript (line 714):
        # Getting the type of 'True' (line 714)
        True_189391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 37), 'True')
        # Getting the type of 'crossovers' (line 714)
        crossovers_189392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 12), 'crossovers')
        # Getting the type of 'fill_point' (line 714)
        fill_point_189393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 23), 'fill_point')
        # Storing an element on a container (line 714)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 12), crossovers_189392, (fill_point_189393, True_189391))
        
        # Assigning a Call to a Name (line 715):
        
        # Assigning a Call to a Name (line 715):
        
        # Call to where(...): (line 715)
        # Processing the call arguments (line 715)
        # Getting the type of 'crossovers' (line 715)
        crossovers_189396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 29), 'crossovers', False)
        # Getting the type of 'bprime' (line 715)
        bprime_189397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 41), 'bprime', False)
        # Getting the type of 'trial' (line 715)
        trial_189398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 49), 'trial', False)
        # Processing the call keyword arguments (line 715)
        kwargs_189399 = {}
        # Getting the type of 'np' (line 715)
        np_189394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 20), 'np', False)
        # Obtaining the member 'where' of a type (line 715)
        where_189395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 20), np_189394, 'where')
        # Calling where(args, kwargs) (line 715)
        where_call_result_189400 = invoke(stypy.reporting.localization.Localization(__file__, 715, 20), where_189395, *[crossovers_189396, bprime_189397, trial_189398], **kwargs_189399)
        
        # Assigning a type to the variable 'trial' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 12), 'trial', where_call_result_189400)
        # Getting the type of 'trial' (line 716)
        trial_189401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 19), 'trial')
        # Assigning a type to the variable 'stypy_return_type' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'stypy_return_type', trial_189401)
        # SSA branch for the else part of an if statement (line 707)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 718)
        self_189402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 13), 'self')
        # Obtaining the member 'strategy' of a type (line 718)
        strategy_189403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 13), self_189402, 'strategy')
        # Getting the type of 'self' (line 718)
        self_189404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 30), 'self')
        # Obtaining the member '_exponential' of a type (line 718)
        _exponential_189405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 30), self_189404, '_exponential')
        # Applying the binary operator 'in' (line 718)
        result_contains_189406 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 13), 'in', strategy_189403, _exponential_189405)
        
        # Testing the type of an if condition (line 718)
        if_condition_189407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 13), result_contains_189406)
        # Assigning a type to the variable 'if_condition_189407' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 13), 'if_condition_189407', if_condition_189407)
        # SSA begins for if statement (line 718)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 719):
        
        # Assigning a Num to a Name (line 719):
        int_189408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 16), 'int')
        # Assigning a type to the variable 'i' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'i', int_189408)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 720)
        i_189409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 19), 'i')
        # Getting the type of 'self' (line 720)
        self_189410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 23), 'self')
        # Obtaining the member 'parameter_count' of a type (line 720)
        parameter_count_189411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 23), self_189410, 'parameter_count')
        # Applying the binary operator '<' (line 720)
        result_lt_189412 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 19), '<', i_189409, parameter_count_189411)
        
        
        
        # Call to rand(...): (line 721)
        # Processing the call keyword arguments (line 721)
        kwargs_189415 = {}
        # Getting the type of 'rng' (line 721)
        rng_189413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 19), 'rng', False)
        # Obtaining the member 'rand' of a type (line 721)
        rand_189414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 19), rng_189413, 'rand')
        # Calling rand(args, kwargs) (line 721)
        rand_call_result_189416 = invoke(stypy.reporting.localization.Localization(__file__, 721, 19), rand_189414, *[], **kwargs_189415)
        
        # Getting the type of 'self' (line 721)
        self_189417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 32), 'self')
        # Obtaining the member 'cross_over_probability' of a type (line 721)
        cross_over_probability_189418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 32), self_189417, 'cross_over_probability')
        # Applying the binary operator '<' (line 721)
        result_lt_189419 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 19), '<', rand_call_result_189416, cross_over_probability_189418)
        
        # Applying the binary operator 'and' (line 720)
        result_and_keyword_189420 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 19), 'and', result_lt_189412, result_lt_189419)
        
        # Testing the type of an if condition (line 720)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 720, 12), result_and_keyword_189420)
        # SSA begins for while statement (line 720)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Subscript to a Subscript (line 723):
        
        # Assigning a Subscript to a Subscript (line 723):
        
        # Obtaining the type of the subscript
        # Getting the type of 'fill_point' (line 723)
        fill_point_189421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 43), 'fill_point')
        # Getting the type of 'bprime' (line 723)
        bprime_189422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 36), 'bprime')
        # Obtaining the member '__getitem__' of a type (line 723)
        getitem___189423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 36), bprime_189422, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 723)
        subscript_call_result_189424 = invoke(stypy.reporting.localization.Localization(__file__, 723, 36), getitem___189423, fill_point_189421)
        
        # Getting the type of 'trial' (line 723)
        trial_189425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 16), 'trial')
        # Getting the type of 'fill_point' (line 723)
        fill_point_189426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 22), 'fill_point')
        # Storing an element on a container (line 723)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 723, 16), trial_189425, (fill_point_189426, subscript_call_result_189424))
        
        # Assigning a BinOp to a Name (line 724):
        
        # Assigning a BinOp to a Name (line 724):
        # Getting the type of 'fill_point' (line 724)
        fill_point_189427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 30), 'fill_point')
        int_189428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 43), 'int')
        # Applying the binary operator '+' (line 724)
        result_add_189429 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 30), '+', fill_point_189427, int_189428)
        
        # Getting the type of 'self' (line 724)
        self_189430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 48), 'self')
        # Obtaining the member 'parameter_count' of a type (line 724)
        parameter_count_189431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 48), self_189430, 'parameter_count')
        # Applying the binary operator '%' (line 724)
        result_mod_189432 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 29), '%', result_add_189429, parameter_count_189431)
        
        # Assigning a type to the variable 'fill_point' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 16), 'fill_point', result_mod_189432)
        
        # Getting the type of 'i' (line 725)
        i_189433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 16), 'i')
        int_189434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 21), 'int')
        # Applying the binary operator '+=' (line 725)
        result_iadd_189435 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 16), '+=', i_189433, int_189434)
        # Assigning a type to the variable 'i' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 16), 'i', result_iadd_189435)
        
        # SSA join for while statement (line 720)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'trial' (line 727)
        trial_189436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 19), 'trial')
        # Assigning a type to the variable 'stypy_return_type' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 12), 'stypy_return_type', trial_189436)
        # SSA join for if statement (line 718)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 707)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_mutate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mutate' in the type store
        # Getting the type of 'stypy_return_type' (line 690)
        stypy_return_type_189437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189437)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mutate'
        return stypy_return_type_189437


    @norecursion
    def _best1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_best1'
        module_type_store = module_type_store.open_function_context('_best1', 729, 4, False)
        # Assigning a type to the variable 'self' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._best1')
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_param_names_list', ['samples'])
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._best1.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._best1', ['samples'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_best1', localization, ['samples'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_best1(...)' code ##################

        str_189438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, (-1)), 'str', '\n        best1bin, best1exp\n        ')
        
        # Assigning a Subscript to a Tuple (line 733):
        
        # Assigning a Subscript to a Name (line 733):
        
        # Obtaining the type of the subscript
        int_189439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 26), 'int')
        slice_189441 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 733, 17), None, int_189440, None)
        # Getting the type of 'samples' (line 733)
        samples_189442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 17), 'samples')
        # Obtaining the member '__getitem__' of a type (line 733)
        getitem___189443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 17), samples_189442, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 733)
        subscript_call_result_189444 = invoke(stypy.reporting.localization.Localization(__file__, 733, 17), getitem___189443, slice_189441)
        
        # Obtaining the member '__getitem__' of a type (line 733)
        getitem___189445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 8), subscript_call_result_189444, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 733)
        subscript_call_result_189446 = invoke(stypy.reporting.localization.Localization(__file__, 733, 8), getitem___189445, int_189439)
        
        # Assigning a type to the variable 'tuple_var_assignment_188386' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'tuple_var_assignment_188386', subscript_call_result_189446)
        
        # Assigning a Subscript to a Name (line 733):
        
        # Obtaining the type of the subscript
        int_189447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 26), 'int')
        slice_189449 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 733, 17), None, int_189448, None)
        # Getting the type of 'samples' (line 733)
        samples_189450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 17), 'samples')
        # Obtaining the member '__getitem__' of a type (line 733)
        getitem___189451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 17), samples_189450, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 733)
        subscript_call_result_189452 = invoke(stypy.reporting.localization.Localization(__file__, 733, 17), getitem___189451, slice_189449)
        
        # Obtaining the member '__getitem__' of a type (line 733)
        getitem___189453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 8), subscript_call_result_189452, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 733)
        subscript_call_result_189454 = invoke(stypy.reporting.localization.Localization(__file__, 733, 8), getitem___189453, int_189447)
        
        # Assigning a type to the variable 'tuple_var_assignment_188387' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'tuple_var_assignment_188387', subscript_call_result_189454)
        
        # Assigning a Name to a Name (line 733):
        # Getting the type of 'tuple_var_assignment_188386' (line 733)
        tuple_var_assignment_188386_189455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'tuple_var_assignment_188386')
        # Assigning a type to the variable 'r0' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'r0', tuple_var_assignment_188386_189455)
        
        # Assigning a Name to a Name (line 733):
        # Getting the type of 'tuple_var_assignment_188387' (line 733)
        tuple_var_assignment_188387_189456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'tuple_var_assignment_188387')
        # Assigning a type to the variable 'r1' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'r1', tuple_var_assignment_188387_189456)
        
        # Obtaining the type of the subscript
        int_189457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 32), 'int')
        # Getting the type of 'self' (line 734)
        self_189458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 16), 'self')
        # Obtaining the member 'population' of a type (line 734)
        population_189459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 16), self_189458, 'population')
        # Obtaining the member '__getitem__' of a type (line 734)
        getitem___189460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 16), population_189459, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 734)
        subscript_call_result_189461 = invoke(stypy.reporting.localization.Localization(__file__, 734, 16), getitem___189460, int_189457)
        
        # Getting the type of 'self' (line 734)
        self_189462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 37), 'self')
        # Obtaining the member 'scale' of a type (line 734)
        scale_189463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 37), self_189462, 'scale')
        
        # Obtaining the type of the subscript
        # Getting the type of 'r0' (line 735)
        r0_189464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 33), 'r0')
        # Getting the type of 'self' (line 735)
        self_189465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 17), 'self')
        # Obtaining the member 'population' of a type (line 735)
        population_189466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 17), self_189465, 'population')
        # Obtaining the member '__getitem__' of a type (line 735)
        getitem___189467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 17), population_189466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 735)
        subscript_call_result_189468 = invoke(stypy.reporting.localization.Localization(__file__, 735, 17), getitem___189467, r0_189464)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r1' (line 735)
        r1_189469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 55), 'r1')
        # Getting the type of 'self' (line 735)
        self_189470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 39), 'self')
        # Obtaining the member 'population' of a type (line 735)
        population_189471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 39), self_189470, 'population')
        # Obtaining the member '__getitem__' of a type (line 735)
        getitem___189472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 39), population_189471, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 735)
        subscript_call_result_189473 = invoke(stypy.reporting.localization.Localization(__file__, 735, 39), getitem___189472, r1_189469)
        
        # Applying the binary operator '-' (line 735)
        result_sub_189474 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 17), '-', subscript_call_result_189468, subscript_call_result_189473)
        
        # Applying the binary operator '*' (line 734)
        result_mul_189475 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 37), '*', scale_189463, result_sub_189474)
        
        # Applying the binary operator '+' (line 734)
        result_add_189476 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 16), '+', subscript_call_result_189461, result_mul_189475)
        
        # Assigning a type to the variable 'stypy_return_type' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'stypy_return_type', result_add_189476)
        
        # ################# End of '_best1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_best1' in the type store
        # Getting the type of 'stypy_return_type' (line 729)
        stypy_return_type_189477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189477)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_best1'
        return stypy_return_type_189477


    @norecursion
    def _rand1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rand1'
        module_type_store = module_type_store.open_function_context('_rand1', 737, 4, False)
        # Assigning a type to the variable 'self' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._rand1')
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_param_names_list', ['samples'])
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._rand1.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._rand1', ['samples'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rand1', localization, ['samples'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rand1(...)' code ##################

        str_189478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, (-1)), 'str', '\n        rand1bin, rand1exp\n        ')
        
        # Assigning a Subscript to a Tuple (line 741):
        
        # Assigning a Subscript to a Name (line 741):
        
        # Obtaining the type of the subscript
        int_189479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 30), 'int')
        slice_189481 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 741, 21), None, int_189480, None)
        # Getting the type of 'samples' (line 741)
        samples_189482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 21), 'samples')
        # Obtaining the member '__getitem__' of a type (line 741)
        getitem___189483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 21), samples_189482, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 741)
        subscript_call_result_189484 = invoke(stypy.reporting.localization.Localization(__file__, 741, 21), getitem___189483, slice_189481)
        
        # Obtaining the member '__getitem__' of a type (line 741)
        getitem___189485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 8), subscript_call_result_189484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 741)
        subscript_call_result_189486 = invoke(stypy.reporting.localization.Localization(__file__, 741, 8), getitem___189485, int_189479)
        
        # Assigning a type to the variable 'tuple_var_assignment_188388' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'tuple_var_assignment_188388', subscript_call_result_189486)
        
        # Assigning a Subscript to a Name (line 741):
        
        # Obtaining the type of the subscript
        int_189487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 30), 'int')
        slice_189489 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 741, 21), None, int_189488, None)
        # Getting the type of 'samples' (line 741)
        samples_189490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 21), 'samples')
        # Obtaining the member '__getitem__' of a type (line 741)
        getitem___189491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 21), samples_189490, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 741)
        subscript_call_result_189492 = invoke(stypy.reporting.localization.Localization(__file__, 741, 21), getitem___189491, slice_189489)
        
        # Obtaining the member '__getitem__' of a type (line 741)
        getitem___189493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 8), subscript_call_result_189492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 741)
        subscript_call_result_189494 = invoke(stypy.reporting.localization.Localization(__file__, 741, 8), getitem___189493, int_189487)
        
        # Assigning a type to the variable 'tuple_var_assignment_188389' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'tuple_var_assignment_188389', subscript_call_result_189494)
        
        # Assigning a Subscript to a Name (line 741):
        
        # Obtaining the type of the subscript
        int_189495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 30), 'int')
        slice_189497 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 741, 21), None, int_189496, None)
        # Getting the type of 'samples' (line 741)
        samples_189498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 21), 'samples')
        # Obtaining the member '__getitem__' of a type (line 741)
        getitem___189499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 21), samples_189498, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 741)
        subscript_call_result_189500 = invoke(stypy.reporting.localization.Localization(__file__, 741, 21), getitem___189499, slice_189497)
        
        # Obtaining the member '__getitem__' of a type (line 741)
        getitem___189501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 8), subscript_call_result_189500, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 741)
        subscript_call_result_189502 = invoke(stypy.reporting.localization.Localization(__file__, 741, 8), getitem___189501, int_189495)
        
        # Assigning a type to the variable 'tuple_var_assignment_188390' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'tuple_var_assignment_188390', subscript_call_result_189502)
        
        # Assigning a Name to a Name (line 741):
        # Getting the type of 'tuple_var_assignment_188388' (line 741)
        tuple_var_assignment_188388_189503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'tuple_var_assignment_188388')
        # Assigning a type to the variable 'r0' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'r0', tuple_var_assignment_188388_189503)
        
        # Assigning a Name to a Name (line 741):
        # Getting the type of 'tuple_var_assignment_188389' (line 741)
        tuple_var_assignment_188389_189504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'tuple_var_assignment_188389')
        # Assigning a type to the variable 'r1' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'r1', tuple_var_assignment_188389_189504)
        
        # Assigning a Name to a Name (line 741):
        # Getting the type of 'tuple_var_assignment_188390' (line 741)
        tuple_var_assignment_188390_189505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'tuple_var_assignment_188390')
        # Assigning a type to the variable 'r2' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'r2', tuple_var_assignment_188390_189505)
        
        # Obtaining the type of the subscript
        # Getting the type of 'r0' (line 742)
        r0_189506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 32), 'r0')
        # Getting the type of 'self' (line 742)
        self_189507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 16), 'self')
        # Obtaining the member 'population' of a type (line 742)
        population_189508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 16), self_189507, 'population')
        # Obtaining the member '__getitem__' of a type (line 742)
        getitem___189509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 16), population_189508, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 742)
        subscript_call_result_189510 = invoke(stypy.reporting.localization.Localization(__file__, 742, 16), getitem___189509, r0_189506)
        
        # Getting the type of 'self' (line 742)
        self_189511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 38), 'self')
        # Obtaining the member 'scale' of a type (line 742)
        scale_189512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 38), self_189511, 'scale')
        
        # Obtaining the type of the subscript
        # Getting the type of 'r1' (line 743)
        r1_189513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 33), 'r1')
        # Getting the type of 'self' (line 743)
        self_189514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 17), 'self')
        # Obtaining the member 'population' of a type (line 743)
        population_189515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 17), self_189514, 'population')
        # Obtaining the member '__getitem__' of a type (line 743)
        getitem___189516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 17), population_189515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 743)
        subscript_call_result_189517 = invoke(stypy.reporting.localization.Localization(__file__, 743, 17), getitem___189516, r1_189513)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r2' (line 743)
        r2_189518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 55), 'r2')
        # Getting the type of 'self' (line 743)
        self_189519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 39), 'self')
        # Obtaining the member 'population' of a type (line 743)
        population_189520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 39), self_189519, 'population')
        # Obtaining the member '__getitem__' of a type (line 743)
        getitem___189521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 39), population_189520, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 743)
        subscript_call_result_189522 = invoke(stypy.reporting.localization.Localization(__file__, 743, 39), getitem___189521, r2_189518)
        
        # Applying the binary operator '-' (line 743)
        result_sub_189523 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 17), '-', subscript_call_result_189517, subscript_call_result_189522)
        
        # Applying the binary operator '*' (line 742)
        result_mul_189524 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 38), '*', scale_189512, result_sub_189523)
        
        # Applying the binary operator '+' (line 742)
        result_add_189525 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 16), '+', subscript_call_result_189510, result_mul_189524)
        
        # Assigning a type to the variable 'stypy_return_type' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'stypy_return_type', result_add_189525)
        
        # ################# End of '_rand1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rand1' in the type store
        # Getting the type of 'stypy_return_type' (line 737)
        stypy_return_type_189526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rand1'
        return stypy_return_type_189526


    @norecursion
    def _randtobest1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_randtobest1'
        module_type_store = module_type_store.open_function_context('_randtobest1', 745, 4, False)
        # Assigning a type to the variable 'self' (line 746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._randtobest1')
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_param_names_list', ['candidate', 'samples'])
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._randtobest1.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._randtobest1', ['candidate', 'samples'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_randtobest1', localization, ['candidate', 'samples'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_randtobest1(...)' code ##################

        str_189527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, (-1)), 'str', '\n        randtobest1bin, randtobest1exp\n        ')
        
        # Assigning a Subscript to a Tuple (line 749):
        
        # Assigning a Subscript to a Name (line 749):
        
        # Obtaining the type of the subscript
        int_189528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 26), 'int')
        slice_189530 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 749, 17), None, int_189529, None)
        # Getting the type of 'samples' (line 749)
        samples_189531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 17), 'samples')
        # Obtaining the member '__getitem__' of a type (line 749)
        getitem___189532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 17), samples_189531, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 749)
        subscript_call_result_189533 = invoke(stypy.reporting.localization.Localization(__file__, 749, 17), getitem___189532, slice_189530)
        
        # Obtaining the member '__getitem__' of a type (line 749)
        getitem___189534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 8), subscript_call_result_189533, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 749)
        subscript_call_result_189535 = invoke(stypy.reporting.localization.Localization(__file__, 749, 8), getitem___189534, int_189528)
        
        # Assigning a type to the variable 'tuple_var_assignment_188391' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'tuple_var_assignment_188391', subscript_call_result_189535)
        
        # Assigning a Subscript to a Name (line 749):
        
        # Obtaining the type of the subscript
        int_189536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 26), 'int')
        slice_189538 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 749, 17), None, int_189537, None)
        # Getting the type of 'samples' (line 749)
        samples_189539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 17), 'samples')
        # Obtaining the member '__getitem__' of a type (line 749)
        getitem___189540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 17), samples_189539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 749)
        subscript_call_result_189541 = invoke(stypy.reporting.localization.Localization(__file__, 749, 17), getitem___189540, slice_189538)
        
        # Obtaining the member '__getitem__' of a type (line 749)
        getitem___189542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 8), subscript_call_result_189541, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 749)
        subscript_call_result_189543 = invoke(stypy.reporting.localization.Localization(__file__, 749, 8), getitem___189542, int_189536)
        
        # Assigning a type to the variable 'tuple_var_assignment_188392' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'tuple_var_assignment_188392', subscript_call_result_189543)
        
        # Assigning a Name to a Name (line 749):
        # Getting the type of 'tuple_var_assignment_188391' (line 749)
        tuple_var_assignment_188391_189544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'tuple_var_assignment_188391')
        # Assigning a type to the variable 'r0' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'r0', tuple_var_assignment_188391_189544)
        
        # Assigning a Name to a Name (line 749):
        # Getting the type of 'tuple_var_assignment_188392' (line 749)
        tuple_var_assignment_188392_189545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'tuple_var_assignment_188392')
        # Assigning a type to the variable 'r1' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 12), 'r1', tuple_var_assignment_188392_189545)
        
        # Assigning a Call to a Name (line 750):
        
        # Assigning a Call to a Name (line 750):
        
        # Call to copy(...): (line 750)
        # Processing the call arguments (line 750)
        
        # Obtaining the type of the subscript
        # Getting the type of 'candidate' (line 750)
        candidate_189548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 41), 'candidate', False)
        # Getting the type of 'self' (line 750)
        self_189549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 25), 'self', False)
        # Obtaining the member 'population' of a type (line 750)
        population_189550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 25), self_189549, 'population')
        # Obtaining the member '__getitem__' of a type (line 750)
        getitem___189551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 25), population_189550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 750)
        subscript_call_result_189552 = invoke(stypy.reporting.localization.Localization(__file__, 750, 25), getitem___189551, candidate_189548)
        
        # Processing the call keyword arguments (line 750)
        kwargs_189553 = {}
        # Getting the type of 'np' (line 750)
        np_189546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 17), 'np', False)
        # Obtaining the member 'copy' of a type (line 750)
        copy_189547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 17), np_189546, 'copy')
        # Calling copy(args, kwargs) (line 750)
        copy_call_result_189554 = invoke(stypy.reporting.localization.Localization(__file__, 750, 17), copy_189547, *[subscript_call_result_189552], **kwargs_189553)
        
        # Assigning a type to the variable 'bprime' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'bprime', copy_call_result_189554)
        
        # Getting the type of 'bprime' (line 751)
        bprime_189555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'bprime')
        # Getting the type of 'self' (line 751)
        self_189556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 18), 'self')
        # Obtaining the member 'scale' of a type (line 751)
        scale_189557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 18), self_189556, 'scale')
        
        # Obtaining the type of the subscript
        int_189558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 48), 'int')
        # Getting the type of 'self' (line 751)
        self_189559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 32), 'self')
        # Obtaining the member 'population' of a type (line 751)
        population_189560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 32), self_189559, 'population')
        # Obtaining the member '__getitem__' of a type (line 751)
        getitem___189561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 32), population_189560, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 751)
        subscript_call_result_189562 = invoke(stypy.reporting.localization.Localization(__file__, 751, 32), getitem___189561, int_189558)
        
        # Getting the type of 'bprime' (line 751)
        bprime_189563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 53), 'bprime')
        # Applying the binary operator '-' (line 751)
        result_sub_189564 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 32), '-', subscript_call_result_189562, bprime_189563)
        
        # Applying the binary operator '*' (line 751)
        result_mul_189565 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 18), '*', scale_189557, result_sub_189564)
        
        # Applying the binary operator '+=' (line 751)
        result_iadd_189566 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 8), '+=', bprime_189555, result_mul_189565)
        # Assigning a type to the variable 'bprime' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'bprime', result_iadd_189566)
        
        
        # Getting the type of 'bprime' (line 752)
        bprime_189567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'bprime')
        # Getting the type of 'self' (line 752)
        self_189568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 18), 'self')
        # Obtaining the member 'scale' of a type (line 752)
        scale_189569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 18), self_189568, 'scale')
        
        # Obtaining the type of the subscript
        # Getting the type of 'r0' (line 752)
        r0_189570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 48), 'r0')
        # Getting the type of 'self' (line 752)
        self_189571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 32), 'self')
        # Obtaining the member 'population' of a type (line 752)
        population_189572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 32), self_189571, 'population')
        # Obtaining the member '__getitem__' of a type (line 752)
        getitem___189573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 32), population_189572, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 752)
        subscript_call_result_189574 = invoke(stypy.reporting.localization.Localization(__file__, 752, 32), getitem___189573, r0_189570)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r1' (line 753)
        r1_189575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 48), 'r1')
        # Getting the type of 'self' (line 753)
        self_189576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 32), 'self')
        # Obtaining the member 'population' of a type (line 753)
        population_189577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 32), self_189576, 'population')
        # Obtaining the member '__getitem__' of a type (line 753)
        getitem___189578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 32), population_189577, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 753)
        subscript_call_result_189579 = invoke(stypy.reporting.localization.Localization(__file__, 753, 32), getitem___189578, r1_189575)
        
        # Applying the binary operator '-' (line 752)
        result_sub_189580 = python_operator(stypy.reporting.localization.Localization(__file__, 752, 32), '-', subscript_call_result_189574, subscript_call_result_189579)
        
        # Applying the binary operator '*' (line 752)
        result_mul_189581 = python_operator(stypy.reporting.localization.Localization(__file__, 752, 18), '*', scale_189569, result_sub_189580)
        
        # Applying the binary operator '+=' (line 752)
        result_iadd_189582 = python_operator(stypy.reporting.localization.Localization(__file__, 752, 8), '+=', bprime_189567, result_mul_189581)
        # Assigning a type to the variable 'bprime' (line 752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'bprime', result_iadd_189582)
        
        # Getting the type of 'bprime' (line 754)
        bprime_189583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 15), 'bprime')
        # Assigning a type to the variable 'stypy_return_type' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'stypy_return_type', bprime_189583)
        
        # ################# End of '_randtobest1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_randtobest1' in the type store
        # Getting the type of 'stypy_return_type' (line 745)
        stypy_return_type_189584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_randtobest1'
        return stypy_return_type_189584


    @norecursion
    def _best2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_best2'
        module_type_store = module_type_store.open_function_context('_best2', 756, 4, False)
        # Assigning a type to the variable 'self' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._best2')
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_param_names_list', ['samples'])
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._best2.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._best2', ['samples'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_best2', localization, ['samples'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_best2(...)' code ##################

        str_189585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, (-1)), 'str', '\n        best2bin, best2exp\n        ')
        
        # Assigning a Subscript to a Tuple (line 760):
        
        # Assigning a Subscript to a Name (line 760):
        
        # Obtaining the type of the subscript
        int_189586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 34), 'int')
        slice_189588 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 760, 25), None, int_189587, None)
        # Getting the type of 'samples' (line 760)
        samples_189589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 25), 'samples')
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___189590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 25), samples_189589, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_189591 = invoke(stypy.reporting.localization.Localization(__file__, 760, 25), getitem___189590, slice_189588)
        
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___189592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), subscript_call_result_189591, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_189593 = invoke(stypy.reporting.localization.Localization(__file__, 760, 8), getitem___189592, int_189586)
        
        # Assigning a type to the variable 'tuple_var_assignment_188393' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'tuple_var_assignment_188393', subscript_call_result_189593)
        
        # Assigning a Subscript to a Name (line 760):
        
        # Obtaining the type of the subscript
        int_189594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 34), 'int')
        slice_189596 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 760, 25), None, int_189595, None)
        # Getting the type of 'samples' (line 760)
        samples_189597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 25), 'samples')
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___189598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 25), samples_189597, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_189599 = invoke(stypy.reporting.localization.Localization(__file__, 760, 25), getitem___189598, slice_189596)
        
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___189600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), subscript_call_result_189599, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_189601 = invoke(stypy.reporting.localization.Localization(__file__, 760, 8), getitem___189600, int_189594)
        
        # Assigning a type to the variable 'tuple_var_assignment_188394' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'tuple_var_assignment_188394', subscript_call_result_189601)
        
        # Assigning a Subscript to a Name (line 760):
        
        # Obtaining the type of the subscript
        int_189602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 34), 'int')
        slice_189604 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 760, 25), None, int_189603, None)
        # Getting the type of 'samples' (line 760)
        samples_189605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 25), 'samples')
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___189606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 25), samples_189605, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_189607 = invoke(stypy.reporting.localization.Localization(__file__, 760, 25), getitem___189606, slice_189604)
        
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___189608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), subscript_call_result_189607, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_189609 = invoke(stypy.reporting.localization.Localization(__file__, 760, 8), getitem___189608, int_189602)
        
        # Assigning a type to the variable 'tuple_var_assignment_188395' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'tuple_var_assignment_188395', subscript_call_result_189609)
        
        # Assigning a Subscript to a Name (line 760):
        
        # Obtaining the type of the subscript
        int_189610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 8), 'int')
        
        # Obtaining the type of the subscript
        int_189611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 34), 'int')
        slice_189612 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 760, 25), None, int_189611, None)
        # Getting the type of 'samples' (line 760)
        samples_189613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 25), 'samples')
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___189614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 25), samples_189613, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_189615 = invoke(stypy.reporting.localization.Localization(__file__, 760, 25), getitem___189614, slice_189612)
        
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___189616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), subscript_call_result_189615, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_189617 = invoke(stypy.reporting.localization.Localization(__file__, 760, 8), getitem___189616, int_189610)
        
        # Assigning a type to the variable 'tuple_var_assignment_188396' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'tuple_var_assignment_188396', subscript_call_result_189617)
        
        # Assigning a Name to a Name (line 760):
        # Getting the type of 'tuple_var_assignment_188393' (line 760)
        tuple_var_assignment_188393_189618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'tuple_var_assignment_188393')
        # Assigning a type to the variable 'r0' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'r0', tuple_var_assignment_188393_189618)
        
        # Assigning a Name to a Name (line 760):
        # Getting the type of 'tuple_var_assignment_188394' (line 760)
        tuple_var_assignment_188394_189619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'tuple_var_assignment_188394')
        # Assigning a type to the variable 'r1' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'r1', tuple_var_assignment_188394_189619)
        
        # Assigning a Name to a Name (line 760):
        # Getting the type of 'tuple_var_assignment_188395' (line 760)
        tuple_var_assignment_188395_189620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'tuple_var_assignment_188395')
        # Assigning a type to the variable 'r2' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 16), 'r2', tuple_var_assignment_188395_189620)
        
        # Assigning a Name to a Name (line 760):
        # Getting the type of 'tuple_var_assignment_188396' (line 760)
        tuple_var_assignment_188396_189621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'tuple_var_assignment_188396')
        # Assigning a type to the variable 'r3' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 20), 'r3', tuple_var_assignment_188396_189621)
        
        # Assigning a BinOp to a Name (line 761):
        
        # Assigning a BinOp to a Name (line 761):
        
        # Obtaining the type of the subscript
        int_189622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 34), 'int')
        # Getting the type of 'self' (line 761)
        self_189623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 18), 'self')
        # Obtaining the member 'population' of a type (line 761)
        population_189624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 18), self_189623, 'population')
        # Obtaining the member '__getitem__' of a type (line 761)
        getitem___189625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 18), population_189624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 761)
        subscript_call_result_189626 = invoke(stypy.reporting.localization.Localization(__file__, 761, 18), getitem___189625, int_189622)
        
        # Getting the type of 'self' (line 761)
        self_189627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 39), 'self')
        # Obtaining the member 'scale' of a type (line 761)
        scale_189628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 39), self_189627, 'scale')
        
        # Obtaining the type of the subscript
        # Getting the type of 'r0' (line 762)
        r0_189629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 35), 'r0')
        # Getting the type of 'self' (line 762)
        self_189630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 19), 'self')
        # Obtaining the member 'population' of a type (line 762)
        population_189631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 19), self_189630, 'population')
        # Obtaining the member '__getitem__' of a type (line 762)
        getitem___189632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 19), population_189631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 762)
        subscript_call_result_189633 = invoke(stypy.reporting.localization.Localization(__file__, 762, 19), getitem___189632, r0_189629)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r1' (line 762)
        r1_189634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 57), 'r1')
        # Getting the type of 'self' (line 762)
        self_189635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 41), 'self')
        # Obtaining the member 'population' of a type (line 762)
        population_189636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 41), self_189635, 'population')
        # Obtaining the member '__getitem__' of a type (line 762)
        getitem___189637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 41), population_189636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 762)
        subscript_call_result_189638 = invoke(stypy.reporting.localization.Localization(__file__, 762, 41), getitem___189637, r1_189634)
        
        # Applying the binary operator '+' (line 762)
        result_add_189639 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 19), '+', subscript_call_result_189633, subscript_call_result_189638)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r2' (line 763)
        r2_189640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 35), 'r2')
        # Getting the type of 'self' (line 763)
        self_189641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 19), 'self')
        # Obtaining the member 'population' of a type (line 763)
        population_189642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 19), self_189641, 'population')
        # Obtaining the member '__getitem__' of a type (line 763)
        getitem___189643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 19), population_189642, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 763)
        subscript_call_result_189644 = invoke(stypy.reporting.localization.Localization(__file__, 763, 19), getitem___189643, r2_189640)
        
        # Applying the binary operator '-' (line 762)
        result_sub_189645 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 61), '-', result_add_189639, subscript_call_result_189644)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r3' (line 763)
        r3_189646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 57), 'r3')
        # Getting the type of 'self' (line 763)
        self_189647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 41), 'self')
        # Obtaining the member 'population' of a type (line 763)
        population_189648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 41), self_189647, 'population')
        # Obtaining the member '__getitem__' of a type (line 763)
        getitem___189649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 41), population_189648, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 763)
        subscript_call_result_189650 = invoke(stypy.reporting.localization.Localization(__file__, 763, 41), getitem___189649, r3_189646)
        
        # Applying the binary operator '-' (line 763)
        result_sub_189651 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 39), '-', result_sub_189645, subscript_call_result_189650)
        
        # Applying the binary operator '*' (line 761)
        result_mul_189652 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 39), '*', scale_189628, result_sub_189651)
        
        # Applying the binary operator '+' (line 761)
        result_add_189653 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 18), '+', subscript_call_result_189626, result_mul_189652)
        
        # Assigning a type to the variable 'bprime' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'bprime', result_add_189653)
        # Getting the type of 'bprime' (line 765)
        bprime_189654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 15), 'bprime')
        # Assigning a type to the variable 'stypy_return_type' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 8), 'stypy_return_type', bprime_189654)
        
        # ################# End of '_best2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_best2' in the type store
        # Getting the type of 'stypy_return_type' (line 756)
        stypy_return_type_189655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189655)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_best2'
        return stypy_return_type_189655


    @norecursion
    def _rand2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rand2'
        module_type_store = module_type_store.open_function_context('_rand2', 767, 4, False)
        # Assigning a type to the variable 'self' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._rand2')
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_param_names_list', ['samples'])
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._rand2.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._rand2', ['samples'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rand2', localization, ['samples'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rand2(...)' code ##################

        str_189656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, (-1)), 'str', '\n        rand2bin, rand2exp\n        ')
        
        # Assigning a Name to a Tuple (line 771):
        
        # Assigning a Subscript to a Name (line 771):
        
        # Obtaining the type of the subscript
        int_189657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 8), 'int')
        # Getting the type of 'samples' (line 771)
        samples_189658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 29), 'samples')
        # Obtaining the member '__getitem__' of a type (line 771)
        getitem___189659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 8), samples_189658, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 771)
        subscript_call_result_189660 = invoke(stypy.reporting.localization.Localization(__file__, 771, 8), getitem___189659, int_189657)
        
        # Assigning a type to the variable 'tuple_var_assignment_188397' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188397', subscript_call_result_189660)
        
        # Assigning a Subscript to a Name (line 771):
        
        # Obtaining the type of the subscript
        int_189661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 8), 'int')
        # Getting the type of 'samples' (line 771)
        samples_189662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 29), 'samples')
        # Obtaining the member '__getitem__' of a type (line 771)
        getitem___189663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 8), samples_189662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 771)
        subscript_call_result_189664 = invoke(stypy.reporting.localization.Localization(__file__, 771, 8), getitem___189663, int_189661)
        
        # Assigning a type to the variable 'tuple_var_assignment_188398' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188398', subscript_call_result_189664)
        
        # Assigning a Subscript to a Name (line 771):
        
        # Obtaining the type of the subscript
        int_189665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 8), 'int')
        # Getting the type of 'samples' (line 771)
        samples_189666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 29), 'samples')
        # Obtaining the member '__getitem__' of a type (line 771)
        getitem___189667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 8), samples_189666, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 771)
        subscript_call_result_189668 = invoke(stypy.reporting.localization.Localization(__file__, 771, 8), getitem___189667, int_189665)
        
        # Assigning a type to the variable 'tuple_var_assignment_188399' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188399', subscript_call_result_189668)
        
        # Assigning a Subscript to a Name (line 771):
        
        # Obtaining the type of the subscript
        int_189669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 8), 'int')
        # Getting the type of 'samples' (line 771)
        samples_189670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 29), 'samples')
        # Obtaining the member '__getitem__' of a type (line 771)
        getitem___189671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 8), samples_189670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 771)
        subscript_call_result_189672 = invoke(stypy.reporting.localization.Localization(__file__, 771, 8), getitem___189671, int_189669)
        
        # Assigning a type to the variable 'tuple_var_assignment_188400' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188400', subscript_call_result_189672)
        
        # Assigning a Subscript to a Name (line 771):
        
        # Obtaining the type of the subscript
        int_189673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 8), 'int')
        # Getting the type of 'samples' (line 771)
        samples_189674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 29), 'samples')
        # Obtaining the member '__getitem__' of a type (line 771)
        getitem___189675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 8), samples_189674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 771)
        subscript_call_result_189676 = invoke(stypy.reporting.localization.Localization(__file__, 771, 8), getitem___189675, int_189673)
        
        # Assigning a type to the variable 'tuple_var_assignment_188401' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188401', subscript_call_result_189676)
        
        # Assigning a Name to a Name (line 771):
        # Getting the type of 'tuple_var_assignment_188397' (line 771)
        tuple_var_assignment_188397_189677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188397')
        # Assigning a type to the variable 'r0' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'r0', tuple_var_assignment_188397_189677)
        
        # Assigning a Name to a Name (line 771):
        # Getting the type of 'tuple_var_assignment_188398' (line 771)
        tuple_var_assignment_188398_189678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188398')
        # Assigning a type to the variable 'r1' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 'r1', tuple_var_assignment_188398_189678)
        
        # Assigning a Name to a Name (line 771):
        # Getting the type of 'tuple_var_assignment_188399' (line 771)
        tuple_var_assignment_188399_189679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188399')
        # Assigning a type to the variable 'r2' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 16), 'r2', tuple_var_assignment_188399_189679)
        
        # Assigning a Name to a Name (line 771):
        # Getting the type of 'tuple_var_assignment_188400' (line 771)
        tuple_var_assignment_188400_189680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188400')
        # Assigning a type to the variable 'r3' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 20), 'r3', tuple_var_assignment_188400_189680)
        
        # Assigning a Name to a Name (line 771):
        # Getting the type of 'tuple_var_assignment_188401' (line 771)
        tuple_var_assignment_188401_189681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'tuple_var_assignment_188401')
        # Assigning a type to the variable 'r4' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 24), 'r4', tuple_var_assignment_188401_189681)
        
        # Assigning a BinOp to a Name (line 772):
        
        # Assigning a BinOp to a Name (line 772):
        
        # Obtaining the type of the subscript
        # Getting the type of 'r0' (line 772)
        r0_189682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 34), 'r0')
        # Getting the type of 'self' (line 772)
        self_189683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 18), 'self')
        # Obtaining the member 'population' of a type (line 772)
        population_189684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 18), self_189683, 'population')
        # Obtaining the member '__getitem__' of a type (line 772)
        getitem___189685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 18), population_189684, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 772)
        subscript_call_result_189686 = invoke(stypy.reporting.localization.Localization(__file__, 772, 18), getitem___189685, r0_189682)
        
        # Getting the type of 'self' (line 772)
        self_189687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 40), 'self')
        # Obtaining the member 'scale' of a type (line 772)
        scale_189688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 40), self_189687, 'scale')
        
        # Obtaining the type of the subscript
        # Getting the type of 'r1' (line 773)
        r1_189689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 35), 'r1')
        # Getting the type of 'self' (line 773)
        self_189690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 19), 'self')
        # Obtaining the member 'population' of a type (line 773)
        population_189691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 19), self_189690, 'population')
        # Obtaining the member '__getitem__' of a type (line 773)
        getitem___189692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 19), population_189691, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 773)
        subscript_call_result_189693 = invoke(stypy.reporting.localization.Localization(__file__, 773, 19), getitem___189692, r1_189689)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r2' (line 773)
        r2_189694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 57), 'r2')
        # Getting the type of 'self' (line 773)
        self_189695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 41), 'self')
        # Obtaining the member 'population' of a type (line 773)
        population_189696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 41), self_189695, 'population')
        # Obtaining the member '__getitem__' of a type (line 773)
        getitem___189697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 41), population_189696, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 773)
        subscript_call_result_189698 = invoke(stypy.reporting.localization.Localization(__file__, 773, 41), getitem___189697, r2_189694)
        
        # Applying the binary operator '+' (line 773)
        result_add_189699 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 19), '+', subscript_call_result_189693, subscript_call_result_189698)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r3' (line 774)
        r3_189700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 35), 'r3')
        # Getting the type of 'self' (line 774)
        self_189701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 19), 'self')
        # Obtaining the member 'population' of a type (line 774)
        population_189702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 19), self_189701, 'population')
        # Obtaining the member '__getitem__' of a type (line 774)
        getitem___189703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 19), population_189702, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 774)
        subscript_call_result_189704 = invoke(stypy.reporting.localization.Localization(__file__, 774, 19), getitem___189703, r3_189700)
        
        # Applying the binary operator '-' (line 773)
        result_sub_189705 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 61), '-', result_add_189699, subscript_call_result_189704)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'r4' (line 774)
        r4_189706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 57), 'r4')
        # Getting the type of 'self' (line 774)
        self_189707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 41), 'self')
        # Obtaining the member 'population' of a type (line 774)
        population_189708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 41), self_189707, 'population')
        # Obtaining the member '__getitem__' of a type (line 774)
        getitem___189709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 41), population_189708, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 774)
        subscript_call_result_189710 = invoke(stypy.reporting.localization.Localization(__file__, 774, 41), getitem___189709, r4_189706)
        
        # Applying the binary operator '-' (line 774)
        result_sub_189711 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 39), '-', result_sub_189705, subscript_call_result_189710)
        
        # Applying the binary operator '*' (line 772)
        result_mul_189712 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 40), '*', scale_189688, result_sub_189711)
        
        # Applying the binary operator '+' (line 772)
        result_add_189713 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 18), '+', subscript_call_result_189686, result_mul_189712)
        
        # Assigning a type to the variable 'bprime' (line 772)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 8), 'bprime', result_add_189713)
        # Getting the type of 'bprime' (line 776)
        bprime_189714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 15), 'bprime')
        # Assigning a type to the variable 'stypy_return_type' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'stypy_return_type', bprime_189714)
        
        # ################# End of '_rand2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rand2' in the type store
        # Getting the type of 'stypy_return_type' (line 767)
        stypy_return_type_189715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rand2'
        return stypy_return_type_189715


    @norecursion
    def _select_samples(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_select_samples'
        module_type_store = module_type_store.open_function_context('_select_samples', 778, 4, False)
        # Assigning a type to the variable 'self' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_localization', localization)
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_type_store', module_type_store)
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_function_name', 'DifferentialEvolutionSolver._select_samples')
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_param_names_list', ['candidate', 'number_samples'])
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_varargs_param_name', None)
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_call_defaults', defaults)
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_call_varargs', varargs)
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DifferentialEvolutionSolver._select_samples.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DifferentialEvolutionSolver._select_samples', ['candidate', 'number_samples'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_select_samples', localization, ['candidate', 'number_samples'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_select_samples(...)' code ##################

        str_189716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, (-1)), 'str', "\n        obtain random integers from range(self.num_population_members),\n        without replacement.  You can't have the original candidate either.\n        ")
        
        # Assigning a Call to a Name (line 783):
        
        # Assigning a Call to a Name (line 783):
        
        # Call to list(...): (line 783)
        # Processing the call arguments (line 783)
        
        # Call to range(...): (line 783)
        # Processing the call arguments (line 783)
        # Getting the type of 'self' (line 783)
        self_189719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 26), 'self', False)
        # Obtaining the member 'num_population_members' of a type (line 783)
        num_population_members_189720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 26), self_189719, 'num_population_members')
        # Processing the call keyword arguments (line 783)
        kwargs_189721 = {}
        # Getting the type of 'range' (line 783)
        range_189718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 20), 'range', False)
        # Calling range(args, kwargs) (line 783)
        range_call_result_189722 = invoke(stypy.reporting.localization.Localization(__file__, 783, 20), range_189718, *[num_population_members_189720], **kwargs_189721)
        
        # Processing the call keyword arguments (line 783)
        kwargs_189723 = {}
        # Getting the type of 'list' (line 783)
        list_189717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 15), 'list', False)
        # Calling list(args, kwargs) (line 783)
        list_call_result_189724 = invoke(stypy.reporting.localization.Localization(__file__, 783, 15), list_189717, *[range_call_result_189722], **kwargs_189723)
        
        # Assigning a type to the variable 'idxs' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'idxs', list_call_result_189724)
        
        # Call to remove(...): (line 784)
        # Processing the call arguments (line 784)
        # Getting the type of 'candidate' (line 784)
        candidate_189727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 20), 'candidate', False)
        # Processing the call keyword arguments (line 784)
        kwargs_189728 = {}
        # Getting the type of 'idxs' (line 784)
        idxs_189725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 8), 'idxs', False)
        # Obtaining the member 'remove' of a type (line 784)
        remove_189726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 8), idxs_189725, 'remove')
        # Calling remove(args, kwargs) (line 784)
        remove_call_result_189729 = invoke(stypy.reporting.localization.Localization(__file__, 784, 8), remove_189726, *[candidate_189727], **kwargs_189728)
        
        
        # Call to shuffle(...): (line 785)
        # Processing the call arguments (line 785)
        # Getting the type of 'idxs' (line 785)
        idxs_189733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 45), 'idxs', False)
        # Processing the call keyword arguments (line 785)
        kwargs_189734 = {}
        # Getting the type of 'self' (line 785)
        self_189730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'self', False)
        # Obtaining the member 'random_number_generator' of a type (line 785)
        random_number_generator_189731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 8), self_189730, 'random_number_generator')
        # Obtaining the member 'shuffle' of a type (line 785)
        shuffle_189732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 8), random_number_generator_189731, 'shuffle')
        # Calling shuffle(args, kwargs) (line 785)
        shuffle_call_result_189735 = invoke(stypy.reporting.localization.Localization(__file__, 785, 8), shuffle_189732, *[idxs_189733], **kwargs_189734)
        
        
        # Assigning a Subscript to a Name (line 786):
        
        # Assigning a Subscript to a Name (line 786):
        
        # Obtaining the type of the subscript
        # Getting the type of 'number_samples' (line 786)
        number_samples_189736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 21), 'number_samples')
        slice_189737 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 786, 15), None, number_samples_189736, None)
        # Getting the type of 'idxs' (line 786)
        idxs_189738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 15), 'idxs')
        # Obtaining the member '__getitem__' of a type (line 786)
        getitem___189739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 15), idxs_189738, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 786)
        subscript_call_result_189740 = invoke(stypy.reporting.localization.Localization(__file__, 786, 15), getitem___189739, slice_189737)
        
        # Assigning a type to the variable 'idxs' (line 786)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'idxs', subscript_call_result_189740)
        # Getting the type of 'idxs' (line 787)
        idxs_189741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 15), 'idxs')
        # Assigning a type to the variable 'stypy_return_type' (line 787)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'stypy_return_type', idxs_189741)
        
        # ################# End of '_select_samples(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_select_samples' in the type store
        # Getting the type of 'stypy_return_type' (line 778)
        stypy_return_type_189742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189742)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_select_samples'
        return stypy_return_type_189742


# Assigning a type to the variable 'DifferentialEvolutionSolver' (line 216)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), 'DifferentialEvolutionSolver', DifferentialEvolutionSolver)

# Assigning a Dict to a Name (line 316):

# Obtaining an instance of the builtin type 'dict' (line 316)
dict_189743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 316)
# Adding element type (key, value) (line 316)
str_189744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 17), 'str', 'best1bin')
str_189745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 29), 'str', '_best1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 16), dict_189743, (str_189744, str_189745))
# Adding element type (key, value) (line 316)
str_189746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 17), 'str', 'randtobest1bin')
str_189747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 35), 'str', '_randtobest1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 16), dict_189743, (str_189746, str_189747))
# Adding element type (key, value) (line 316)
str_189748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 17), 'str', 'best2bin')
str_189749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 29), 'str', '_best2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 16), dict_189743, (str_189748, str_189749))
# Adding element type (key, value) (line 316)
str_189750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 17), 'str', 'rand2bin')
str_189751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 29), 'str', '_rand2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 16), dict_189743, (str_189750, str_189751))
# Adding element type (key, value) (line 316)
str_189752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 17), 'str', 'rand1bin')
str_189753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 29), 'str', '_rand1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 16), dict_189743, (str_189752, str_189753))

# Getting the type of 'DifferentialEvolutionSolver'
DifferentialEvolutionSolver_189754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DifferentialEvolutionSolver')
# Setting the type of the member '_binomial' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DifferentialEvolutionSolver_189754, '_binomial', dict_189743)

# Assigning a Dict to a Name (line 321):

# Obtaining an instance of the builtin type 'dict' (line 321)
dict_189755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 321)
# Adding element type (key, value) (line 321)
str_189756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 20), 'str', 'best1exp')
str_189757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 32), 'str', '_best1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 19), dict_189755, (str_189756, str_189757))
# Adding element type (key, value) (line 321)
str_189758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 20), 'str', 'rand1exp')
str_189759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 32), 'str', '_rand1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 19), dict_189755, (str_189758, str_189759))
# Adding element type (key, value) (line 321)
str_189760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'str', 'randtobest1exp')
str_189761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 38), 'str', '_randtobest1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 19), dict_189755, (str_189760, str_189761))
# Adding element type (key, value) (line 321)
str_189762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 20), 'str', 'best2exp')
str_189763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 32), 'str', '_best2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 19), dict_189755, (str_189762, str_189763))
# Adding element type (key, value) (line 321)
str_189764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 20), 'str', 'rand2exp')
str_189765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 32), 'str', '_rand2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 19), dict_189755, (str_189764, str_189765))

# Getting the type of 'DifferentialEvolutionSolver'
DifferentialEvolutionSolver_189766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DifferentialEvolutionSolver')
# Setting the type of the member '_exponential' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DifferentialEvolutionSolver_189766, '_exponential', dict_189755)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
