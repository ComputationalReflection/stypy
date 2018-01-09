
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit tests for the differential global minimization algorithm.
3: '''
4: from scipy.optimize import _differentialevolution
5: from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
6: from scipy.optimize import differential_evolution
7: import numpy as np
8: from scipy.optimize import rosen
9: from numpy.testing import (assert_equal, assert_allclose,
10:                            assert_almost_equal,
11:                            assert_string_equal, assert_)
12: from pytest import raises as assert_raises
13: 
14: class TestDifferentialEvolutionSolver(object):
15: 
16:     def setup_method(self):
17:         self.old_seterr = np.seterr(invalid='raise')
18:         self.limits = np.array([[0., 0.],
19:                                 [2., 2.]])
20:         self.bounds = [(0., 2.), (0., 2.)]
21: 
22:         self.dummy_solver = DifferentialEvolutionSolver(self.quadratic,
23:                                                         [(0, 100)])
24: 
25:         # dummy_solver2 will be used to test mutation strategies
26:         self.dummy_solver2 = DifferentialEvolutionSolver(self.quadratic,
27:                                                          [(0, 1)],
28:                                                          popsize=7,
29:                                                          mutation=0.5)
30:         # create a population that's only 7 members long
31:         # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
32:         population = np.atleast_2d(np.arange(0.1, 0.8, 0.1)).T
33:         self.dummy_solver2.population = population
34: 
35:     def teardown_method(self):
36:         np.seterr(**self.old_seterr)
37: 
38:     def quadratic(self, x):
39:         return x[0]**2
40: 
41:     def test__strategy_resolves(self):
42:         # test that the correct mutation function is resolved by
43:         # different requested strategy arguments
44:         solver = DifferentialEvolutionSolver(rosen,
45:                                              self.bounds,
46:                                              strategy='best1exp')
47:         assert_equal(solver.strategy, 'best1exp')
48:         assert_equal(solver.mutation_func.__name__, '_best1')
49: 
50:         solver = DifferentialEvolutionSolver(rosen,
51:                                              self.bounds,
52:                                              strategy='best1bin')
53:         assert_equal(solver.strategy, 'best1bin')
54:         assert_equal(solver.mutation_func.__name__, '_best1')
55: 
56:         solver = DifferentialEvolutionSolver(rosen,
57:                                              self.bounds,
58:                                              strategy='rand1bin')
59:         assert_equal(solver.strategy, 'rand1bin')
60:         assert_equal(solver.mutation_func.__name__, '_rand1')
61: 
62:         solver = DifferentialEvolutionSolver(rosen,
63:                                              self.bounds,
64:                                              strategy='rand1exp')
65:         assert_equal(solver.strategy, 'rand1exp')
66:         assert_equal(solver.mutation_func.__name__, '_rand1')
67: 
68:         solver = DifferentialEvolutionSolver(rosen,
69:                                              self.bounds,
70:                                              strategy='rand2exp')
71:         assert_equal(solver.strategy, 'rand2exp')
72:         assert_equal(solver.mutation_func.__name__, '_rand2')
73: 
74:         solver = DifferentialEvolutionSolver(rosen,
75:                                              self.bounds,
76:                                              strategy='best2bin')
77:         assert_equal(solver.strategy, 'best2bin')
78:         assert_equal(solver.mutation_func.__name__, '_best2')
79: 
80:         solver = DifferentialEvolutionSolver(rosen,
81:                                              self.bounds,
82:                                              strategy='rand2bin')
83:         assert_equal(solver.strategy, 'rand2bin')
84:         assert_equal(solver.mutation_func.__name__, '_rand2')
85: 
86:         solver = DifferentialEvolutionSolver(rosen,
87:                                              self.bounds,
88:                                              strategy='rand2exp')
89:         assert_equal(solver.strategy, 'rand2exp')
90:         assert_equal(solver.mutation_func.__name__, '_rand2')
91: 
92:         solver = DifferentialEvolutionSolver(rosen,
93:                                              self.bounds,
94:                                              strategy='randtobest1bin')
95:         assert_equal(solver.strategy, 'randtobest1bin')
96:         assert_equal(solver.mutation_func.__name__, '_randtobest1')
97: 
98:         solver = DifferentialEvolutionSolver(rosen,
99:                                              self.bounds,
100:                                              strategy='randtobest1exp')
101:         assert_equal(solver.strategy, 'randtobest1exp')
102:         assert_equal(solver.mutation_func.__name__, '_randtobest1')
103: 
104:     def test__mutate1(self):
105:         # strategies */1/*, i.e. rand/1/bin, best/1/exp, etc.
106:         result = np.array([0.05])
107:         trial = self.dummy_solver2._best1((2, 3, 4, 5, 6))
108:         assert_allclose(trial, result)
109: 
110:         result = np.array([0.25])
111:         trial = self.dummy_solver2._rand1((2, 3, 4, 5, 6))
112:         assert_allclose(trial, result)
113: 
114:     def test__mutate2(self):
115:         # strategies */2/*, i.e. rand/2/bin, best/2/exp, etc.
116:         # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
117: 
118:         result = np.array([-0.1])
119:         trial = self.dummy_solver2._best2((2, 3, 4, 5, 6))
120:         assert_allclose(trial, result)
121: 
122:         result = np.array([0.1])
123:         trial = self.dummy_solver2._rand2((2, 3, 4, 5, 6))
124:         assert_allclose(trial, result)
125: 
126:     def test__randtobest1(self):
127:         # strategies randtobest/1/*
128:         result = np.array([0.1])
129:         trial = self.dummy_solver2._randtobest1(1, (2, 3, 4, 5, 6))
130:         assert_allclose(trial, result)
131: 
132:     def test_can_init_with_dithering(self):
133:         mutation = (0.5, 1)
134:         solver = DifferentialEvolutionSolver(self.quadratic,
135:                                              self.bounds,
136:                                              mutation=mutation)
137: 
138:         assert_equal(solver.dither, list(mutation))
139: 
140:     def test_invalid_mutation_values_arent_accepted(self):
141:         func = rosen
142:         mutation = (0.5, 3)
143:         assert_raises(ValueError,
144:                           DifferentialEvolutionSolver,
145:                           func,
146:                           self.bounds,
147:                           mutation=mutation)
148: 
149:         mutation = (-1, 1)
150:         assert_raises(ValueError,
151:                           DifferentialEvolutionSolver,
152:                           func,
153:                           self.bounds,
154:                           mutation=mutation)
155: 
156:         mutation = (0.1, np.nan)
157:         assert_raises(ValueError,
158:                           DifferentialEvolutionSolver,
159:                           func,
160:                           self.bounds,
161:                           mutation=mutation)
162: 
163:         mutation = 0.5
164:         solver = DifferentialEvolutionSolver(func,
165:                                              self.bounds,
166:                                              mutation=mutation)
167:         assert_equal(0.5, solver.scale)
168:         assert_equal(None, solver.dither)
169: 
170:     def test__scale_parameters(self):
171:         trial = np.array([0.3])
172:         assert_equal(30, self.dummy_solver._scale_parameters(trial))
173: 
174:         # it should also work with the limits reversed
175:         self.dummy_solver.limits = np.array([[100], [0.]])
176:         assert_equal(30, self.dummy_solver._scale_parameters(trial))
177: 
178:     def test__unscale_parameters(self):
179:         trial = np.array([30])
180:         assert_equal(0.3, self.dummy_solver._unscale_parameters(trial))
181: 
182:         # it should also work with the limits reversed
183:         self.dummy_solver.limits = np.array([[100], [0.]])
184:         assert_equal(0.3, self.dummy_solver._unscale_parameters(trial))
185: 
186:     def test__ensure_constraint(self):
187:         trial = np.array([1.1, -100, 2., 300., -0.00001])
188:         self.dummy_solver._ensure_constraint(trial)
189:         assert_equal(np.all(trial <= 1), True)
190: 
191:     def test_differential_evolution(self):
192:         # test that the Jmin of DifferentialEvolutionSolver
193:         # is the same as the function evaluation
194:         solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)])
195:         result = solver.solve()
196:         assert_almost_equal(result.fun, self.quadratic(result.x))
197: 
198:     def test_best_solution_retrieval(self):
199:         # test that the getter property method for the best solution works.
200:         solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)])
201:         result = solver.solve()
202:         assert_almost_equal(result.x, solver.x)
203: 
204:     def test_callback_terminates(self):
205:         # test that if the callback returns true, then the minimization halts
206:         bounds = [(0, 2), (0, 2)]
207: 
208:         def callback(param, convergence=0.):
209:             return True
210: 
211:         result = differential_evolution(rosen, bounds, callback=callback)
212: 
213:         assert_string_equal(result.message,
214:                                 'callback function requested stop early '
215:                                 'by returning True')
216: 
217:     def test_args_tuple_is_passed(self):
218:         # test that the args tuple is passed to the cost function properly.
219:         bounds = [(-10, 10)]
220:         args = (1., 2., 3.)
221: 
222:         def quadratic(x, *args):
223:             if type(args) != tuple:
224:                 raise ValueError('args should be a tuple')
225:             return args[0] + args[1] * x + args[2] * x**2.
226: 
227:         result = differential_evolution(quadratic,
228:                                         bounds,
229:                                         args=args,
230:                                         polish=True)
231:         assert_almost_equal(result.fun, 2 / 3.)
232: 
233:     def test_init_with_invalid_strategy(self):
234:         # test that passing an invalid strategy raises ValueError
235:         func = rosen
236:         bounds = [(-3, 3)]
237:         assert_raises(ValueError,
238:                           differential_evolution,
239:                           func,
240:                           bounds,
241:                           strategy='abc')
242: 
243:     def test_bounds_checking(self):
244:         # test that the bounds checking works
245:         func = rosen
246:         bounds = [(-3, None)]
247:         assert_raises(ValueError,
248:                           differential_evolution,
249:                           func,
250:                           bounds)
251:         bounds = [(-3)]
252:         assert_raises(ValueError,
253:                           differential_evolution,
254:                           func,
255:                           bounds)
256:         bounds = [(-3, 3), (3, 4, 5)]
257:         assert_raises(ValueError,
258:                           differential_evolution,
259:                           func,
260:                           bounds)
261: 
262:     def test_select_samples(self):
263:         # select_samples should return 5 separate random numbers.
264:         limits = np.arange(12., dtype='float64').reshape(2, 6)
265:         bounds = list(zip(limits[0, :], limits[1, :]))
266:         solver = DifferentialEvolutionSolver(None, bounds, popsize=1)
267:         candidate = 0
268:         r1, r2, r3, r4, r5 = solver._select_samples(candidate, 5)
269:         assert_equal(
270:             len(np.unique(np.array([candidate, r1, r2, r3, r4, r5]))), 6)
271: 
272:     def test_maxiter_stops_solve(self):
273:         # test that if the maximum number of iterations is exceeded
274:         # the solver stops.
275:         solver = DifferentialEvolutionSolver(rosen, self.bounds, maxiter=1)
276:         result = solver.solve()
277:         assert_equal(result.success, False)
278:         assert_equal(result.message,
279:                         'Maximum number of iterations has been exceeded.')
280: 
281:     def test_maxfun_stops_solve(self):
282:         # test that if the maximum number of function evaluations is exceeded
283:         # during initialisation the solver stops
284:         solver = DifferentialEvolutionSolver(rosen, self.bounds, maxfun=1,
285:                                              polish=False)
286:         result = solver.solve()
287: 
288:         assert_equal(result.nfev, 2)
289:         assert_equal(result.success, False)
290:         assert_equal(result.message,
291:                      'Maximum number of function evaluations has '
292:                      'been exceeded.')
293: 
294:         # test that if the maximum number of function evaluations is exceeded
295:         # during the actual minimisation, then the solver stops.
296:         # Have to turn polishing off, as this will still occur even if maxfun
297:         # is reached. For popsize=5 and len(bounds)=2, then there are only 10
298:         # function evaluations during initialisation.
299:         solver = DifferentialEvolutionSolver(rosen,
300:                                              self.bounds,
301:                                              popsize=5,
302:                                              polish=False,
303:                                              maxfun=40)
304:         result = solver.solve()
305: 
306:         assert_equal(result.nfev, 41)
307:         assert_equal(result.success, False)
308:         assert_equal(result.message,
309:                          'Maximum number of function evaluations has '
310:                               'been exceeded.')
311: 
312:     def test_quadratic(self):
313:         # test the quadratic function from object
314:         solver = DifferentialEvolutionSolver(self.quadratic,
315:                                              [(-100, 100)],
316:                                              tol=0.02)
317:         solver.solve()
318:         assert_equal(np.argmin(solver.population_energies), 0)
319: 
320:     def test_quadratic_from_diff_ev(self):
321:         # test the quadratic function from differential_evolution function
322:         differential_evolution(self.quadratic,
323:                                [(-100, 100)],
324:                                tol=0.02)
325: 
326:     def test_seed_gives_repeatability(self):
327:         result = differential_evolution(self.quadratic,
328:                                         [(-100, 100)],
329:                                         polish=False,
330:                                         seed=1,
331:                                         tol=0.5)
332:         result2 = differential_evolution(self.quadratic,
333:                                         [(-100, 100)],
334:                                         polish=False,
335:                                         seed=1,
336:                                         tol=0.5)
337:         assert_equal(result.x, result2.x)
338: 
339:     def test_exp_runs(self):
340:         # test whether exponential mutation loop runs
341:         solver = DifferentialEvolutionSolver(rosen,
342:                                              self.bounds,
343:                                              strategy='best1exp',
344:                                              maxiter=1)
345: 
346:         solver.solve()
347: 
348:     def test_gh_4511_regression(self):
349:         # This modification of the differential evolution docstring example
350:         # uses a custom popsize that had triggered an off-by-one error.
351:         # Because we do not care about solving the optimization problem in
352:         # this test, we use maxiter=1 to reduce the testing time.
353:         bounds = [(-5, 5), (-5, 5)]
354:         result = differential_evolution(rosen, bounds, popsize=1815, maxiter=1)
355: 
356:     def test_calculate_population_energies(self):
357:         # if popsize is 2 then the overall generation has size (4,)
358:         solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=2)
359:         solver._calculate_population_energies()
360: 
361:         assert_equal(np.argmin(solver.population_energies), 0)
362: 
363:         # initial calculation of the energies should require 4 nfev.
364:         assert_equal(solver._nfev, 4)
365: 
366:     def test_iteration(self):
367:         # test that DifferentialEvolutionSolver is iterable
368:         # if popsize is 2 then the overall generation has size (4,)
369:         solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=2,
370:                                              maxfun=8)
371:         x, fun = next(solver)
372:         assert_equal(np.size(x, 0), 2)
373: 
374:         # 4 nfev are required for initial calculation of energies, 4 nfev are
375:         # required for the evolution of the 4 population members.
376:         assert_equal(solver._nfev, 8)
377: 
378:         # the next generation should halt because it exceeds maxfun
379:         assert_raises(StopIteration, next, solver)
380: 
381:         # check a proper minimisation can be done by an iterable solver
382:         solver = DifferentialEvolutionSolver(rosen, self.bounds)
383:         for i, soln in enumerate(solver):
384:             x_current, fun_current = soln
385:             # need to have this otherwise the solver would never stop.
386:             if i == 1000:
387:                 break
388: 
389:         assert_almost_equal(fun_current, 0)
390: 
391:     def test_convergence(self):
392:         solver = DifferentialEvolutionSolver(rosen, self.bounds, tol=0.2,
393:                                              polish=False)
394:         solver.solve()
395:         assert_(solver.convergence < 0.2)
396: 
397:     def test_maxiter_none_GH5731(self):
398:         # Pre 0.17 the previous default for maxiter and maxfun was None.
399:         # the numerical defaults are now 1000 and np.inf. However, some scripts
400:         # will still supply None for both of those, this will raise a TypeError
401:         # in the solve method.
402:         solver = DifferentialEvolutionSolver(rosen, self.bounds, maxiter=None,
403:                                              maxfun=None)
404:         solver.solve()
405: 
406:     def test_population_initiation(self):
407:         # test the different modes of population initiation
408: 
409:         # init must be either 'latinhypercube' or 'random'
410:         # raising ValueError is something else is passed in
411:         assert_raises(ValueError,
412:                       DifferentialEvolutionSolver,
413:                       *(rosen, self.bounds),
414:                       **{'init': 'rubbish'})
415: 
416:         solver = DifferentialEvolutionSolver(rosen, self.bounds)
417: 
418:         # check that population initiation:
419:         # 1) resets _nfev to 0
420:         # 2) all population energies are np.inf
421:         solver.init_population_random()
422:         assert_equal(solver._nfev, 0)
423:         assert_(np.all(np.isinf(solver.population_energies)))
424: 
425:         solver.init_population_lhs()
426:         assert_equal(solver._nfev, 0)
427:         assert_(np.all(np.isinf(solver.population_energies)))
428: 
429: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_238301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nUnit tests for the differential global minimization algorithm.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.optimize import _differentialevolution' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_238302 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize')

if (type(import_238302) is not StypyTypeError):

    if (import_238302 != 'pyd_module'):
        __import__(import_238302)
        sys_modules_238303 = sys.modules[import_238302]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', sys_modules_238303.module_type_store, module_type_store, ['_differentialevolution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_238303, sys_modules_238303.module_type_store, module_type_store)
    else:
        from scipy.optimize import _differentialevolution

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', None, module_type_store, ['_differentialevolution'], [_differentialevolution])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', import_238302)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.optimize._differentialevolution import DifferentialEvolutionSolver' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_238304 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize._differentialevolution')

if (type(import_238304) is not StypyTypeError):

    if (import_238304 != 'pyd_module'):
        __import__(import_238304)
        sys_modules_238305 = sys.modules[import_238304]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize._differentialevolution', sys_modules_238305.module_type_store, module_type_store, ['DifferentialEvolutionSolver'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_238305, sys_modules_238305.module_type_store, module_type_store)
    else:
        from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize._differentialevolution', None, module_type_store, ['DifferentialEvolutionSolver'], [DifferentialEvolutionSolver])

else:
    # Assigning a type to the variable 'scipy.optimize._differentialevolution' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize._differentialevolution', import_238304)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.optimize import differential_evolution' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_238306 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize')

if (type(import_238306) is not StypyTypeError):

    if (import_238306 != 'pyd_module'):
        __import__(import_238306)
        sys_modules_238307 = sys.modules[import_238306]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', sys_modules_238307.module_type_store, module_type_store, ['differential_evolution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_238307, sys_modules_238307.module_type_store, module_type_store)
    else:
        from scipy.optimize import differential_evolution

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', None, module_type_store, ['differential_evolution'], [differential_evolution])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', import_238306)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_238308 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_238308) is not StypyTypeError):

    if (import_238308 != 'pyd_module'):
        __import__(import_238308)
        sys_modules_238309 = sys.modules[import_238308]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_238309.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_238308)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize import rosen' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_238310 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize')

if (type(import_238310) is not StypyTypeError):

    if (import_238310 != 'pyd_module'):
        __import__(import_238310)
        sys_modules_238311 = sys.modules[import_238310]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', sys_modules_238311.module_type_store, module_type_store, ['rosen'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_238311, sys_modules_238311.module_type_store, module_type_store)
    else:
        from scipy.optimize import rosen

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', None, module_type_store, ['rosen'], [rosen])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize', import_238310)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_equal, assert_allclose, assert_almost_equal, assert_string_equal, assert_' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_238312 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_238312) is not StypyTypeError):

    if (import_238312 != 'pyd_module'):
        __import__(import_238312)
        sys_modules_238313 = sys.modules[import_238312]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_238313.module_type_store, module_type_store, ['assert_equal', 'assert_allclose', 'assert_almost_equal', 'assert_string_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_238313, sys_modules_238313.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose, assert_almost_equal, assert_string_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose', 'assert_almost_equal', 'assert_string_equal', 'assert_'], [assert_equal, assert_allclose, assert_almost_equal, assert_string_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_238312)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from pytest import assert_raises' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_238314 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest')

if (type(import_238314) is not StypyTypeError):

    if (import_238314 != 'pyd_module'):
        __import__(import_238314)
        sys_modules_238315 = sys.modules[import_238314]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', sys_modules_238315.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_238315, sys_modules_238315.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', import_238314)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'TestDifferentialEvolutionSolver' class

class TestDifferentialEvolutionSolver(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.setup_method')
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Call to a Attribute (line 17):
        
        # Assigning a Call to a Attribute (line 17):
        
        # Call to seterr(...): (line 17)
        # Processing the call keyword arguments (line 17)
        str_238318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 44), 'str', 'raise')
        keyword_238319 = str_238318
        kwargs_238320 = {'invalid': keyword_238319}
        # Getting the type of 'np' (line 17)
        np_238316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 26), 'np', False)
        # Obtaining the member 'seterr' of a type (line 17)
        seterr_238317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 26), np_238316, 'seterr')
        # Calling seterr(args, kwargs) (line 17)
        seterr_call_result_238321 = invoke(stypy.reporting.localization.Localization(__file__, 17, 26), seterr_238317, *[], **kwargs_238320)
        
        # Getting the type of 'self' (line 17)
        self_238322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'old_seterr' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_238322, 'old_seterr', seterr_call_result_238321)
        
        # Assigning a Call to a Attribute (line 18):
        
        # Assigning a Call to a Attribute (line 18):
        
        # Call to array(...): (line 18)
        # Processing the call arguments (line 18)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_238325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_238326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        # Adding element type (line 18)
        float_238327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 32), list_238326, float_238327)
        # Adding element type (line 18)
        float_238328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 32), list_238326, float_238328)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 31), list_238325, list_238326)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_238329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        float_238330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 32), list_238329, float_238330)
        # Adding element type (line 19)
        float_238331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 32), list_238329, float_238331)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 31), list_238325, list_238329)
        
        # Processing the call keyword arguments (line 18)
        kwargs_238332 = {}
        # Getting the type of 'np' (line 18)
        np_238323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 18)
        array_238324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 22), np_238323, 'array')
        # Calling array(args, kwargs) (line 18)
        array_call_result_238333 = invoke(stypy.reporting.localization.Localization(__file__, 18, 22), array_238324, *[list_238325], **kwargs_238332)
        
        # Getting the type of 'self' (line 18)
        self_238334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'limits' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_238334, 'limits', array_call_result_238333)
        
        # Assigning a List to a Attribute (line 20):
        
        # Assigning a List to a Attribute (line 20):
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_238335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_238336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        float_238337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 24), tuple_238336, float_238337)
        # Adding element type (line 20)
        float_238338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 24), tuple_238336, float_238338)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 22), list_238335, tuple_238336)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_238339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        float_238340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 34), tuple_238339, float_238340)
        # Adding element type (line 20)
        float_238341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 34), tuple_238339, float_238341)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 22), list_238335, tuple_238339)
        
        # Getting the type of 'self' (line 20)
        self_238342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'bounds' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_238342, 'bounds', list_238335)
        
        # Assigning a Call to a Attribute (line 22):
        
        # Assigning a Call to a Attribute (line 22):
        
        # Call to DifferentialEvolutionSolver(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'self' (line 22)
        self_238344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 56), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 22)
        quadratic_238345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 56), self_238344, 'quadratic')
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_238346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_238347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        int_238348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 58), tuple_238347, int_238348)
        # Adding element type (line 23)
        int_238349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 58), tuple_238347, int_238349)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 56), list_238346, tuple_238347)
        
        # Processing the call keyword arguments (line 22)
        kwargs_238350 = {}
        # Getting the type of 'DifferentialEvolutionSolver' (line 22)
        DifferentialEvolutionSolver_238343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 22)
        DifferentialEvolutionSolver_call_result_238351 = invoke(stypy.reporting.localization.Localization(__file__, 22, 28), DifferentialEvolutionSolver_238343, *[quadratic_238345, list_238346], **kwargs_238350)
        
        # Getting the type of 'self' (line 22)
        self_238352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'dummy_solver' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_238352, 'dummy_solver', DifferentialEvolutionSolver_call_result_238351)
        
        # Assigning a Call to a Attribute (line 26):
        
        # Assigning a Call to a Attribute (line 26):
        
        # Call to DifferentialEvolutionSolver(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'self' (line 26)
        self_238354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 57), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 26)
        quadratic_238355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 57), self_238354, 'quadratic')
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_238356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_238357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        int_238358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 59), tuple_238357, int_238358)
        # Adding element type (line 27)
        int_238359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 59), tuple_238357, int_238359)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 57), list_238356, tuple_238357)
        
        # Processing the call keyword arguments (line 26)
        int_238360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 65), 'int')
        keyword_238361 = int_238360
        float_238362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 66), 'float')
        keyword_238363 = float_238362
        kwargs_238364 = {'popsize': keyword_238361, 'mutation': keyword_238363}
        # Getting the type of 'DifferentialEvolutionSolver' (line 26)
        DifferentialEvolutionSolver_238353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 26)
        DifferentialEvolutionSolver_call_result_238365 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), DifferentialEvolutionSolver_238353, *[quadratic_238355, list_238356], **kwargs_238364)
        
        # Getting the type of 'self' (line 26)
        self_238366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'dummy_solver2' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_238366, 'dummy_solver2', DifferentialEvolutionSolver_call_result_238365)
        
        # Assigning a Attribute to a Name (line 32):
        
        # Assigning a Attribute to a Name (line 32):
        
        # Call to atleast_2d(...): (line 32)
        # Processing the call arguments (line 32)
        
        # Call to arange(...): (line 32)
        # Processing the call arguments (line 32)
        float_238371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 45), 'float')
        float_238372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 50), 'float')
        float_238373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 55), 'float')
        # Processing the call keyword arguments (line 32)
        kwargs_238374 = {}
        # Getting the type of 'np' (line 32)
        np_238369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'np', False)
        # Obtaining the member 'arange' of a type (line 32)
        arange_238370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 35), np_238369, 'arange')
        # Calling arange(args, kwargs) (line 32)
        arange_call_result_238375 = invoke(stypy.reporting.localization.Localization(__file__, 32, 35), arange_238370, *[float_238371, float_238372, float_238373], **kwargs_238374)
        
        # Processing the call keyword arguments (line 32)
        kwargs_238376 = {}
        # Getting the type of 'np' (line 32)
        np_238367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 32)
        atleast_2d_238368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 21), np_238367, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 32)
        atleast_2d_call_result_238377 = invoke(stypy.reporting.localization.Localization(__file__, 32, 21), atleast_2d_238368, *[arange_call_result_238375], **kwargs_238376)
        
        # Obtaining the member 'T' of a type (line 32)
        T_238378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 21), atleast_2d_call_result_238377, 'T')
        # Assigning a type to the variable 'population' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'population', T_238378)
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'population' (line 33)
        population_238379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'population')
        # Getting the type of 'self' (line 33)
        self_238380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Obtaining the member 'dummy_solver2' of a type (line 33)
        dummy_solver2_238381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_238380, 'dummy_solver2')
        # Setting the type of the member 'population' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), dummy_solver2_238381, 'population', population_238379)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_238382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238382)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_238382


    @norecursion
    def teardown_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'teardown_method'
        module_type_store = module_type_store.open_function_context('teardown_method', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.teardown_method')
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.teardown_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.teardown_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'teardown_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'teardown_method(...)' code ##################

        
        # Call to seterr(...): (line 36)
        # Processing the call keyword arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_238385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'self', False)
        # Obtaining the member 'old_seterr' of a type (line 36)
        old_seterr_238386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 20), self_238385, 'old_seterr')
        kwargs_238387 = {'old_seterr_238386': old_seterr_238386}
        # Getting the type of 'np' (line 36)
        np_238383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'np', False)
        # Obtaining the member 'seterr' of a type (line 36)
        seterr_238384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), np_238383, 'seterr')
        # Calling seterr(args, kwargs) (line 36)
        seterr_call_result_238388 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), seterr_238384, *[], **kwargs_238387)
        
        
        # ################# End of 'teardown_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'teardown_method' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_238389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'teardown_method'
        return stypy_return_type_238389


    @norecursion
    def quadratic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'quadratic'
        module_type_store = module_type_store.open_function_context('quadratic', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.quadratic')
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.quadratic.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.quadratic', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'quadratic', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'quadratic(...)' code ##################

        
        # Obtaining the type of the subscript
        int_238390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'int')
        # Getting the type of 'x' (line 39)
        x_238391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'x')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___238392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), x_238391, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_238393 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), getitem___238392, int_238390)
        
        int_238394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'int')
        # Applying the binary operator '**' (line 39)
        result_pow_238395 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '**', subscript_call_result_238393, int_238394)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', result_pow_238395)
        
        # ################# End of 'quadratic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'quadratic' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_238396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238396)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'quadratic'
        return stypy_return_type_238396


    @norecursion
    def test__strategy_resolves(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__strategy_resolves'
        module_type_store = module_type_store.open_function_context('test__strategy_resolves', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test__strategy_resolves')
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test__strategy_resolves.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test__strategy_resolves', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__strategy_resolves', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__strategy_resolves(...)' code ##################

        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to DifferentialEvolutionSolver(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'rosen' (line 44)
        rosen_238398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 45), 'rosen', False)
        # Getting the type of 'self' (line 45)
        self_238399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 45)
        bounds_238400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 45), self_238399, 'bounds')
        # Processing the call keyword arguments (line 44)
        str_238401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 54), 'str', 'best1exp')
        keyword_238402 = str_238401
        kwargs_238403 = {'strategy': keyword_238402}
        # Getting the type of 'DifferentialEvolutionSolver' (line 44)
        DifferentialEvolutionSolver_238397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 44)
        DifferentialEvolutionSolver_call_result_238404 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), DifferentialEvolutionSolver_238397, *[rosen_238398, bounds_238400], **kwargs_238403)
        
        # Assigning a type to the variable 'solver' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'solver', DifferentialEvolutionSolver_call_result_238404)
        
        # Call to assert_equal(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'solver' (line 47)
        solver_238406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 47)
        strategy_238407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 21), solver_238406, 'strategy')
        str_238408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'str', 'best1exp')
        # Processing the call keyword arguments (line 47)
        kwargs_238409 = {}
        # Getting the type of 'assert_equal' (line 47)
        assert_equal_238405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 47)
        assert_equal_call_result_238410 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_equal_238405, *[strategy_238407, str_238408], **kwargs_238409)
        
        
        # Call to assert_equal(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'solver' (line 48)
        solver_238412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 48)
        mutation_func_238413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 21), solver_238412, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 48)
        name___238414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 21), mutation_func_238413, '__name__')
        str_238415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 52), 'str', '_best1')
        # Processing the call keyword arguments (line 48)
        kwargs_238416 = {}
        # Getting the type of 'assert_equal' (line 48)
        assert_equal_238411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 48)
        assert_equal_call_result_238417 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_equal_238411, *[name___238414, str_238415], **kwargs_238416)
        
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to DifferentialEvolutionSolver(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'rosen' (line 50)
        rosen_238419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'rosen', False)
        # Getting the type of 'self' (line 51)
        self_238420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 51)
        bounds_238421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 45), self_238420, 'bounds')
        # Processing the call keyword arguments (line 50)
        str_238422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 54), 'str', 'best1bin')
        keyword_238423 = str_238422
        kwargs_238424 = {'strategy': keyword_238423}
        # Getting the type of 'DifferentialEvolutionSolver' (line 50)
        DifferentialEvolutionSolver_238418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 50)
        DifferentialEvolutionSolver_call_result_238425 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), DifferentialEvolutionSolver_238418, *[rosen_238419, bounds_238421], **kwargs_238424)
        
        # Assigning a type to the variable 'solver' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'solver', DifferentialEvolutionSolver_call_result_238425)
        
        # Call to assert_equal(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'solver' (line 53)
        solver_238427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 53)
        strategy_238428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 21), solver_238427, 'strategy')
        str_238429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'str', 'best1bin')
        # Processing the call keyword arguments (line 53)
        kwargs_238430 = {}
        # Getting the type of 'assert_equal' (line 53)
        assert_equal_238426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 53)
        assert_equal_call_result_238431 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert_equal_238426, *[strategy_238428, str_238429], **kwargs_238430)
        
        
        # Call to assert_equal(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'solver' (line 54)
        solver_238433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 54)
        mutation_func_238434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), solver_238433, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 54)
        name___238435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), mutation_func_238434, '__name__')
        str_238436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 52), 'str', '_best1')
        # Processing the call keyword arguments (line 54)
        kwargs_238437 = {}
        # Getting the type of 'assert_equal' (line 54)
        assert_equal_238432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 54)
        assert_equal_call_result_238438 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert_equal_238432, *[name___238435, str_238436], **kwargs_238437)
        
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to DifferentialEvolutionSolver(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'rosen' (line 56)
        rosen_238440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 45), 'rosen', False)
        # Getting the type of 'self' (line 57)
        self_238441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 57)
        bounds_238442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 45), self_238441, 'bounds')
        # Processing the call keyword arguments (line 56)
        str_238443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 54), 'str', 'rand1bin')
        keyword_238444 = str_238443
        kwargs_238445 = {'strategy': keyword_238444}
        # Getting the type of 'DifferentialEvolutionSolver' (line 56)
        DifferentialEvolutionSolver_238439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 56)
        DifferentialEvolutionSolver_call_result_238446 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), DifferentialEvolutionSolver_238439, *[rosen_238440, bounds_238442], **kwargs_238445)
        
        # Assigning a type to the variable 'solver' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'solver', DifferentialEvolutionSolver_call_result_238446)
        
        # Call to assert_equal(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'solver' (line 59)
        solver_238448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 59)
        strategy_238449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 21), solver_238448, 'strategy')
        str_238450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 38), 'str', 'rand1bin')
        # Processing the call keyword arguments (line 59)
        kwargs_238451 = {}
        # Getting the type of 'assert_equal' (line 59)
        assert_equal_238447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 59)
        assert_equal_call_result_238452 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert_equal_238447, *[strategy_238449, str_238450], **kwargs_238451)
        
        
        # Call to assert_equal(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'solver' (line 60)
        solver_238454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 60)
        mutation_func_238455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 21), solver_238454, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 60)
        name___238456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 21), mutation_func_238455, '__name__')
        str_238457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 52), 'str', '_rand1')
        # Processing the call keyword arguments (line 60)
        kwargs_238458 = {}
        # Getting the type of 'assert_equal' (line 60)
        assert_equal_238453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 60)
        assert_equal_call_result_238459 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_equal_238453, *[name___238456, str_238457], **kwargs_238458)
        
        
        # Assigning a Call to a Name (line 62):
        
        # Assigning a Call to a Name (line 62):
        
        # Call to DifferentialEvolutionSolver(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'rosen' (line 62)
        rosen_238461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 45), 'rosen', False)
        # Getting the type of 'self' (line 63)
        self_238462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 63)
        bounds_238463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 45), self_238462, 'bounds')
        # Processing the call keyword arguments (line 62)
        str_238464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 54), 'str', 'rand1exp')
        keyword_238465 = str_238464
        kwargs_238466 = {'strategy': keyword_238465}
        # Getting the type of 'DifferentialEvolutionSolver' (line 62)
        DifferentialEvolutionSolver_238460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 62)
        DifferentialEvolutionSolver_call_result_238467 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), DifferentialEvolutionSolver_238460, *[rosen_238461, bounds_238463], **kwargs_238466)
        
        # Assigning a type to the variable 'solver' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'solver', DifferentialEvolutionSolver_call_result_238467)
        
        # Call to assert_equal(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'solver' (line 65)
        solver_238469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 65)
        strategy_238470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), solver_238469, 'strategy')
        str_238471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'str', 'rand1exp')
        # Processing the call keyword arguments (line 65)
        kwargs_238472 = {}
        # Getting the type of 'assert_equal' (line 65)
        assert_equal_238468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 65)
        assert_equal_call_result_238473 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_equal_238468, *[strategy_238470, str_238471], **kwargs_238472)
        
        
        # Call to assert_equal(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'solver' (line 66)
        solver_238475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 66)
        mutation_func_238476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), solver_238475, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 66)
        name___238477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), mutation_func_238476, '__name__')
        str_238478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 52), 'str', '_rand1')
        # Processing the call keyword arguments (line 66)
        kwargs_238479 = {}
        # Getting the type of 'assert_equal' (line 66)
        assert_equal_238474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 66)
        assert_equal_call_result_238480 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_equal_238474, *[name___238477, str_238478], **kwargs_238479)
        
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to DifferentialEvolutionSolver(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'rosen' (line 68)
        rosen_238482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 45), 'rosen', False)
        # Getting the type of 'self' (line 69)
        self_238483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 69)
        bounds_238484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 45), self_238483, 'bounds')
        # Processing the call keyword arguments (line 68)
        str_238485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 54), 'str', 'rand2exp')
        keyword_238486 = str_238485
        kwargs_238487 = {'strategy': keyword_238486}
        # Getting the type of 'DifferentialEvolutionSolver' (line 68)
        DifferentialEvolutionSolver_238481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 68)
        DifferentialEvolutionSolver_call_result_238488 = invoke(stypy.reporting.localization.Localization(__file__, 68, 17), DifferentialEvolutionSolver_238481, *[rosen_238482, bounds_238484], **kwargs_238487)
        
        # Assigning a type to the variable 'solver' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'solver', DifferentialEvolutionSolver_call_result_238488)
        
        # Call to assert_equal(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'solver' (line 71)
        solver_238490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 71)
        strategy_238491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 21), solver_238490, 'strategy')
        str_238492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'str', 'rand2exp')
        # Processing the call keyword arguments (line 71)
        kwargs_238493 = {}
        # Getting the type of 'assert_equal' (line 71)
        assert_equal_238489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 71)
        assert_equal_call_result_238494 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert_equal_238489, *[strategy_238491, str_238492], **kwargs_238493)
        
        
        # Call to assert_equal(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'solver' (line 72)
        solver_238496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 72)
        mutation_func_238497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 21), solver_238496, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 72)
        name___238498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 21), mutation_func_238497, '__name__')
        str_238499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 52), 'str', '_rand2')
        # Processing the call keyword arguments (line 72)
        kwargs_238500 = {}
        # Getting the type of 'assert_equal' (line 72)
        assert_equal_238495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 72)
        assert_equal_call_result_238501 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert_equal_238495, *[name___238498, str_238499], **kwargs_238500)
        
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to DifferentialEvolutionSolver(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'rosen' (line 74)
        rosen_238503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 45), 'rosen', False)
        # Getting the type of 'self' (line 75)
        self_238504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 75)
        bounds_238505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 45), self_238504, 'bounds')
        # Processing the call keyword arguments (line 74)
        str_238506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 54), 'str', 'best2bin')
        keyword_238507 = str_238506
        kwargs_238508 = {'strategy': keyword_238507}
        # Getting the type of 'DifferentialEvolutionSolver' (line 74)
        DifferentialEvolutionSolver_238502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 74)
        DifferentialEvolutionSolver_call_result_238509 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), DifferentialEvolutionSolver_238502, *[rosen_238503, bounds_238505], **kwargs_238508)
        
        # Assigning a type to the variable 'solver' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'solver', DifferentialEvolutionSolver_call_result_238509)
        
        # Call to assert_equal(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'solver' (line 77)
        solver_238511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 77)
        strategy_238512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 21), solver_238511, 'strategy')
        str_238513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 38), 'str', 'best2bin')
        # Processing the call keyword arguments (line 77)
        kwargs_238514 = {}
        # Getting the type of 'assert_equal' (line 77)
        assert_equal_238510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 77)
        assert_equal_call_result_238515 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), assert_equal_238510, *[strategy_238512, str_238513], **kwargs_238514)
        
        
        # Call to assert_equal(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'solver' (line 78)
        solver_238517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 78)
        mutation_func_238518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), solver_238517, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 78)
        name___238519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), mutation_func_238518, '__name__')
        str_238520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 52), 'str', '_best2')
        # Processing the call keyword arguments (line 78)
        kwargs_238521 = {}
        # Getting the type of 'assert_equal' (line 78)
        assert_equal_238516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 78)
        assert_equal_call_result_238522 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assert_equal_238516, *[name___238519, str_238520], **kwargs_238521)
        
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to DifferentialEvolutionSolver(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'rosen' (line 80)
        rosen_238524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'rosen', False)
        # Getting the type of 'self' (line 81)
        self_238525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 81)
        bounds_238526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 45), self_238525, 'bounds')
        # Processing the call keyword arguments (line 80)
        str_238527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 54), 'str', 'rand2bin')
        keyword_238528 = str_238527
        kwargs_238529 = {'strategy': keyword_238528}
        # Getting the type of 'DifferentialEvolutionSolver' (line 80)
        DifferentialEvolutionSolver_238523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 80)
        DifferentialEvolutionSolver_call_result_238530 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), DifferentialEvolutionSolver_238523, *[rosen_238524, bounds_238526], **kwargs_238529)
        
        # Assigning a type to the variable 'solver' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'solver', DifferentialEvolutionSolver_call_result_238530)
        
        # Call to assert_equal(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'solver' (line 83)
        solver_238532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 83)
        strategy_238533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 21), solver_238532, 'strategy')
        str_238534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 38), 'str', 'rand2bin')
        # Processing the call keyword arguments (line 83)
        kwargs_238535 = {}
        # Getting the type of 'assert_equal' (line 83)
        assert_equal_238531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 83)
        assert_equal_call_result_238536 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assert_equal_238531, *[strategy_238533, str_238534], **kwargs_238535)
        
        
        # Call to assert_equal(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'solver' (line 84)
        solver_238538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 84)
        mutation_func_238539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), solver_238538, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 84)
        name___238540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), mutation_func_238539, '__name__')
        str_238541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 52), 'str', '_rand2')
        # Processing the call keyword arguments (line 84)
        kwargs_238542 = {}
        # Getting the type of 'assert_equal' (line 84)
        assert_equal_238537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 84)
        assert_equal_call_result_238543 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_equal_238537, *[name___238540, str_238541], **kwargs_238542)
        
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to DifferentialEvolutionSolver(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'rosen' (line 86)
        rosen_238545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 45), 'rosen', False)
        # Getting the type of 'self' (line 87)
        self_238546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 87)
        bounds_238547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 45), self_238546, 'bounds')
        # Processing the call keyword arguments (line 86)
        str_238548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 54), 'str', 'rand2exp')
        keyword_238549 = str_238548
        kwargs_238550 = {'strategy': keyword_238549}
        # Getting the type of 'DifferentialEvolutionSolver' (line 86)
        DifferentialEvolutionSolver_238544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 86)
        DifferentialEvolutionSolver_call_result_238551 = invoke(stypy.reporting.localization.Localization(__file__, 86, 17), DifferentialEvolutionSolver_238544, *[rosen_238545, bounds_238547], **kwargs_238550)
        
        # Assigning a type to the variable 'solver' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'solver', DifferentialEvolutionSolver_call_result_238551)
        
        # Call to assert_equal(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'solver' (line 89)
        solver_238553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 89)
        strategy_238554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 21), solver_238553, 'strategy')
        str_238555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 38), 'str', 'rand2exp')
        # Processing the call keyword arguments (line 89)
        kwargs_238556 = {}
        # Getting the type of 'assert_equal' (line 89)
        assert_equal_238552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 89)
        assert_equal_call_result_238557 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert_equal_238552, *[strategy_238554, str_238555], **kwargs_238556)
        
        
        # Call to assert_equal(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'solver' (line 90)
        solver_238559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 90)
        mutation_func_238560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 21), solver_238559, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 90)
        name___238561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 21), mutation_func_238560, '__name__')
        str_238562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 52), 'str', '_rand2')
        # Processing the call keyword arguments (line 90)
        kwargs_238563 = {}
        # Getting the type of 'assert_equal' (line 90)
        assert_equal_238558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 90)
        assert_equal_call_result_238564 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_equal_238558, *[name___238561, str_238562], **kwargs_238563)
        
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to DifferentialEvolutionSolver(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'rosen' (line 92)
        rosen_238566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 45), 'rosen', False)
        # Getting the type of 'self' (line 93)
        self_238567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 93)
        bounds_238568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 45), self_238567, 'bounds')
        # Processing the call keyword arguments (line 92)
        str_238569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 54), 'str', 'randtobest1bin')
        keyword_238570 = str_238569
        kwargs_238571 = {'strategy': keyword_238570}
        # Getting the type of 'DifferentialEvolutionSolver' (line 92)
        DifferentialEvolutionSolver_238565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 92)
        DifferentialEvolutionSolver_call_result_238572 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), DifferentialEvolutionSolver_238565, *[rosen_238566, bounds_238568], **kwargs_238571)
        
        # Assigning a type to the variable 'solver' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'solver', DifferentialEvolutionSolver_call_result_238572)
        
        # Call to assert_equal(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'solver' (line 95)
        solver_238574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 95)
        strategy_238575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 21), solver_238574, 'strategy')
        str_238576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 38), 'str', 'randtobest1bin')
        # Processing the call keyword arguments (line 95)
        kwargs_238577 = {}
        # Getting the type of 'assert_equal' (line 95)
        assert_equal_238573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 95)
        assert_equal_call_result_238578 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assert_equal_238573, *[strategy_238575, str_238576], **kwargs_238577)
        
        
        # Call to assert_equal(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'solver' (line 96)
        solver_238580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 96)
        mutation_func_238581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 21), solver_238580, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 96)
        name___238582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 21), mutation_func_238581, '__name__')
        str_238583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 52), 'str', '_randtobest1')
        # Processing the call keyword arguments (line 96)
        kwargs_238584 = {}
        # Getting the type of 'assert_equal' (line 96)
        assert_equal_238579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 96)
        assert_equal_call_result_238585 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assert_equal_238579, *[name___238582, str_238583], **kwargs_238584)
        
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to DifferentialEvolutionSolver(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'rosen' (line 98)
        rosen_238587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 45), 'rosen', False)
        # Getting the type of 'self' (line 99)
        self_238588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 99)
        bounds_238589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 45), self_238588, 'bounds')
        # Processing the call keyword arguments (line 98)
        str_238590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 54), 'str', 'randtobest1exp')
        keyword_238591 = str_238590
        kwargs_238592 = {'strategy': keyword_238591}
        # Getting the type of 'DifferentialEvolutionSolver' (line 98)
        DifferentialEvolutionSolver_238586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 98)
        DifferentialEvolutionSolver_call_result_238593 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), DifferentialEvolutionSolver_238586, *[rosen_238587, bounds_238589], **kwargs_238592)
        
        # Assigning a type to the variable 'solver' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'solver', DifferentialEvolutionSolver_call_result_238593)
        
        # Call to assert_equal(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'solver' (line 101)
        solver_238595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'solver', False)
        # Obtaining the member 'strategy' of a type (line 101)
        strategy_238596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 21), solver_238595, 'strategy')
        str_238597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 38), 'str', 'randtobest1exp')
        # Processing the call keyword arguments (line 101)
        kwargs_238598 = {}
        # Getting the type of 'assert_equal' (line 101)
        assert_equal_238594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 101)
        assert_equal_call_result_238599 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assert_equal_238594, *[strategy_238596, str_238597], **kwargs_238598)
        
        
        # Call to assert_equal(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'solver' (line 102)
        solver_238601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 21), 'solver', False)
        # Obtaining the member 'mutation_func' of a type (line 102)
        mutation_func_238602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 21), solver_238601, 'mutation_func')
        # Obtaining the member '__name__' of a type (line 102)
        name___238603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 21), mutation_func_238602, '__name__')
        str_238604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 52), 'str', '_randtobest1')
        # Processing the call keyword arguments (line 102)
        kwargs_238605 = {}
        # Getting the type of 'assert_equal' (line 102)
        assert_equal_238600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 102)
        assert_equal_call_result_238606 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_equal_238600, *[name___238603, str_238604], **kwargs_238605)
        
        
        # ################# End of 'test__strategy_resolves(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__strategy_resolves' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_238607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__strategy_resolves'
        return stypy_return_type_238607


    @norecursion
    def test__mutate1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__mutate1'
        module_type_store = module_type_store.open_function_context('test__mutate1', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test__mutate1')
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test__mutate1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test__mutate1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__mutate1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__mutate1(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to array(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_238610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        float_238611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 26), list_238610, float_238611)
        
        # Processing the call keyword arguments (line 106)
        kwargs_238612 = {}
        # Getting the type of 'np' (line 106)
        np_238608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 106)
        array_238609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 17), np_238608, 'array')
        # Calling array(args, kwargs) (line 106)
        array_call_result_238613 = invoke(stypy.reporting.localization.Localization(__file__, 106, 17), array_238609, *[list_238610], **kwargs_238612)
        
        # Assigning a type to the variable 'result' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'result', array_call_result_238613)
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to _best1(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Obtaining an instance of the builtin type 'tuple' (line 107)
        tuple_238617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 107)
        # Adding element type (line 107)
        int_238618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 43), tuple_238617, int_238618)
        # Adding element type (line 107)
        int_238619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 43), tuple_238617, int_238619)
        # Adding element type (line 107)
        int_238620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 43), tuple_238617, int_238620)
        # Adding element type (line 107)
        int_238621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 43), tuple_238617, int_238621)
        # Adding element type (line 107)
        int_238622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 43), tuple_238617, int_238622)
        
        # Processing the call keyword arguments (line 107)
        kwargs_238623 = {}
        # Getting the type of 'self' (line 107)
        self_238614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'self', False)
        # Obtaining the member 'dummy_solver2' of a type (line 107)
        dummy_solver2_238615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), self_238614, 'dummy_solver2')
        # Obtaining the member '_best1' of a type (line 107)
        _best1_238616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), dummy_solver2_238615, '_best1')
        # Calling _best1(args, kwargs) (line 107)
        _best1_call_result_238624 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), _best1_238616, *[tuple_238617], **kwargs_238623)
        
        # Assigning a type to the variable 'trial' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'trial', _best1_call_result_238624)
        
        # Call to assert_allclose(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'trial' (line 108)
        trial_238626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'trial', False)
        # Getting the type of 'result' (line 108)
        result_238627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 31), 'result', False)
        # Processing the call keyword arguments (line 108)
        kwargs_238628 = {}
        # Getting the type of 'assert_allclose' (line 108)
        assert_allclose_238625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 108)
        assert_allclose_call_result_238629 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assert_allclose_238625, *[trial_238626, result_238627], **kwargs_238628)
        
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to array(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_238632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        float_238633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 26), list_238632, float_238633)
        
        # Processing the call keyword arguments (line 110)
        kwargs_238634 = {}
        # Getting the type of 'np' (line 110)
        np_238630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 110)
        array_238631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), np_238630, 'array')
        # Calling array(args, kwargs) (line 110)
        array_call_result_238635 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), array_238631, *[list_238632], **kwargs_238634)
        
        # Assigning a type to the variable 'result' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'result', array_call_result_238635)
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to _rand1(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'tuple' (line 111)
        tuple_238639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 111)
        # Adding element type (line 111)
        int_238640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 43), tuple_238639, int_238640)
        # Adding element type (line 111)
        int_238641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 43), tuple_238639, int_238641)
        # Adding element type (line 111)
        int_238642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 43), tuple_238639, int_238642)
        # Adding element type (line 111)
        int_238643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 43), tuple_238639, int_238643)
        # Adding element type (line 111)
        int_238644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 43), tuple_238639, int_238644)
        
        # Processing the call keyword arguments (line 111)
        kwargs_238645 = {}
        # Getting the type of 'self' (line 111)
        self_238636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'self', False)
        # Obtaining the member 'dummy_solver2' of a type (line 111)
        dummy_solver2_238637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), self_238636, 'dummy_solver2')
        # Obtaining the member '_rand1' of a type (line 111)
        _rand1_238638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), dummy_solver2_238637, '_rand1')
        # Calling _rand1(args, kwargs) (line 111)
        _rand1_call_result_238646 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), _rand1_238638, *[tuple_238639], **kwargs_238645)
        
        # Assigning a type to the variable 'trial' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'trial', _rand1_call_result_238646)
        
        # Call to assert_allclose(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'trial' (line 112)
        trial_238648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'trial', False)
        # Getting the type of 'result' (line 112)
        result_238649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'result', False)
        # Processing the call keyword arguments (line 112)
        kwargs_238650 = {}
        # Getting the type of 'assert_allclose' (line 112)
        assert_allclose_238647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 112)
        assert_allclose_call_result_238651 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert_allclose_238647, *[trial_238648, result_238649], **kwargs_238650)
        
        
        # ################# End of 'test__mutate1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__mutate1' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_238652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__mutate1'
        return stypy_return_type_238652


    @norecursion
    def test__mutate2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__mutate2'
        module_type_store = module_type_store.open_function_context('test__mutate2', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test__mutate2')
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test__mutate2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test__mutate2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__mutate2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__mutate2(...)' code ##################

        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to array(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_238655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        float_238656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 26), list_238655, float_238656)
        
        # Processing the call keyword arguments (line 118)
        kwargs_238657 = {}
        # Getting the type of 'np' (line 118)
        np_238653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 118)
        array_238654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 17), np_238653, 'array')
        # Calling array(args, kwargs) (line 118)
        array_call_result_238658 = invoke(stypy.reporting.localization.Localization(__file__, 118, 17), array_238654, *[list_238655], **kwargs_238657)
        
        # Assigning a type to the variable 'result' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'result', array_call_result_238658)
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to _best2(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_238662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        int_238663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 43), tuple_238662, int_238663)
        # Adding element type (line 119)
        int_238664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 43), tuple_238662, int_238664)
        # Adding element type (line 119)
        int_238665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 43), tuple_238662, int_238665)
        # Adding element type (line 119)
        int_238666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 43), tuple_238662, int_238666)
        # Adding element type (line 119)
        int_238667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 43), tuple_238662, int_238667)
        
        # Processing the call keyword arguments (line 119)
        kwargs_238668 = {}
        # Getting the type of 'self' (line 119)
        self_238659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'self', False)
        # Obtaining the member 'dummy_solver2' of a type (line 119)
        dummy_solver2_238660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), self_238659, 'dummy_solver2')
        # Obtaining the member '_best2' of a type (line 119)
        _best2_238661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), dummy_solver2_238660, '_best2')
        # Calling _best2(args, kwargs) (line 119)
        _best2_call_result_238669 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), _best2_238661, *[tuple_238662], **kwargs_238668)
        
        # Assigning a type to the variable 'trial' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'trial', _best2_call_result_238669)
        
        # Call to assert_allclose(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'trial' (line 120)
        trial_238671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'trial', False)
        # Getting the type of 'result' (line 120)
        result_238672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'result', False)
        # Processing the call keyword arguments (line 120)
        kwargs_238673 = {}
        # Getting the type of 'assert_allclose' (line 120)
        assert_allclose_238670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 120)
        assert_allclose_call_result_238674 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), assert_allclose_238670, *[trial_238671, result_238672], **kwargs_238673)
        
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to array(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_238677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        float_238678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 26), list_238677, float_238678)
        
        # Processing the call keyword arguments (line 122)
        kwargs_238679 = {}
        # Getting the type of 'np' (line 122)
        np_238675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 122)
        array_238676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), np_238675, 'array')
        # Calling array(args, kwargs) (line 122)
        array_call_result_238680 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), array_238676, *[list_238677], **kwargs_238679)
        
        # Assigning a type to the variable 'result' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'result', array_call_result_238680)
        
        # Assigning a Call to a Name (line 123):
        
        # Assigning a Call to a Name (line 123):
        
        # Call to _rand2(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_238684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        int_238685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 43), tuple_238684, int_238685)
        # Adding element type (line 123)
        int_238686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 43), tuple_238684, int_238686)
        # Adding element type (line 123)
        int_238687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 43), tuple_238684, int_238687)
        # Adding element type (line 123)
        int_238688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 43), tuple_238684, int_238688)
        # Adding element type (line 123)
        int_238689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 43), tuple_238684, int_238689)
        
        # Processing the call keyword arguments (line 123)
        kwargs_238690 = {}
        # Getting the type of 'self' (line 123)
        self_238681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'self', False)
        # Obtaining the member 'dummy_solver2' of a type (line 123)
        dummy_solver2_238682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), self_238681, 'dummy_solver2')
        # Obtaining the member '_rand2' of a type (line 123)
        _rand2_238683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), dummy_solver2_238682, '_rand2')
        # Calling _rand2(args, kwargs) (line 123)
        _rand2_call_result_238691 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), _rand2_238683, *[tuple_238684], **kwargs_238690)
        
        # Assigning a type to the variable 'trial' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'trial', _rand2_call_result_238691)
        
        # Call to assert_allclose(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'trial' (line 124)
        trial_238693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'trial', False)
        # Getting the type of 'result' (line 124)
        result_238694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'result', False)
        # Processing the call keyword arguments (line 124)
        kwargs_238695 = {}
        # Getting the type of 'assert_allclose' (line 124)
        assert_allclose_238692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 124)
        assert_allclose_call_result_238696 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assert_allclose_238692, *[trial_238693, result_238694], **kwargs_238695)
        
        
        # ################# End of 'test__mutate2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__mutate2' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_238697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238697)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__mutate2'
        return stypy_return_type_238697


    @norecursion
    def test__randtobest1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__randtobest1'
        module_type_store = module_type_store.open_function_context('test__randtobest1', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test__randtobest1')
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test__randtobest1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test__randtobest1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__randtobest1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__randtobest1(...)' code ##################

        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to array(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_238700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        float_238701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 26), list_238700, float_238701)
        
        # Processing the call keyword arguments (line 128)
        kwargs_238702 = {}
        # Getting the type of 'np' (line 128)
        np_238698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 128)
        array_238699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 17), np_238698, 'array')
        # Calling array(args, kwargs) (line 128)
        array_call_result_238703 = invoke(stypy.reporting.localization.Localization(__file__, 128, 17), array_238699, *[list_238700], **kwargs_238702)
        
        # Assigning a type to the variable 'result' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'result', array_call_result_238703)
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to _randtobest1(...): (line 129)
        # Processing the call arguments (line 129)
        int_238707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 48), 'int')
        
        # Obtaining an instance of the builtin type 'tuple' (line 129)
        tuple_238708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 129)
        # Adding element type (line 129)
        int_238709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 52), tuple_238708, int_238709)
        # Adding element type (line 129)
        int_238710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 52), tuple_238708, int_238710)
        # Adding element type (line 129)
        int_238711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 52), tuple_238708, int_238711)
        # Adding element type (line 129)
        int_238712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 52), tuple_238708, int_238712)
        # Adding element type (line 129)
        int_238713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 52), tuple_238708, int_238713)
        
        # Processing the call keyword arguments (line 129)
        kwargs_238714 = {}
        # Getting the type of 'self' (line 129)
        self_238704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'self', False)
        # Obtaining the member 'dummy_solver2' of a type (line 129)
        dummy_solver2_238705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), self_238704, 'dummy_solver2')
        # Obtaining the member '_randtobest1' of a type (line 129)
        _randtobest1_238706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), dummy_solver2_238705, '_randtobest1')
        # Calling _randtobest1(args, kwargs) (line 129)
        _randtobest1_call_result_238715 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), _randtobest1_238706, *[int_238707, tuple_238708], **kwargs_238714)
        
        # Assigning a type to the variable 'trial' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'trial', _randtobest1_call_result_238715)
        
        # Call to assert_allclose(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'trial' (line 130)
        trial_238717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'trial', False)
        # Getting the type of 'result' (line 130)
        result_238718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), 'result', False)
        # Processing the call keyword arguments (line 130)
        kwargs_238719 = {}
        # Getting the type of 'assert_allclose' (line 130)
        assert_allclose_238716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 130)
        assert_allclose_call_result_238720 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assert_allclose_238716, *[trial_238717, result_238718], **kwargs_238719)
        
        
        # ################# End of 'test__randtobest1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__randtobest1' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_238721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238721)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__randtobest1'
        return stypy_return_type_238721


    @norecursion
    def test_can_init_with_dithering(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_can_init_with_dithering'
        module_type_store = module_type_store.open_function_context('test_can_init_with_dithering', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_can_init_with_dithering')
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_can_init_with_dithering.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_can_init_with_dithering', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_can_init_with_dithering', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_can_init_with_dithering(...)' code ##################

        
        # Assigning a Tuple to a Name (line 133):
        
        # Assigning a Tuple to a Name (line 133):
        
        # Obtaining an instance of the builtin type 'tuple' (line 133)
        tuple_238722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 133)
        # Adding element type (line 133)
        float_238723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 20), tuple_238722, float_238723)
        # Adding element type (line 133)
        int_238724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 20), tuple_238722, int_238724)
        
        # Assigning a type to the variable 'mutation' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'mutation', tuple_238722)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to DifferentialEvolutionSolver(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'self' (line 134)
        self_238726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 134)
        quadratic_238727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 45), self_238726, 'quadratic')
        # Getting the type of 'self' (line 135)
        self_238728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 135)
        bounds_238729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 45), self_238728, 'bounds')
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'mutation' (line 136)
        mutation_238730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 54), 'mutation', False)
        keyword_238731 = mutation_238730
        kwargs_238732 = {'mutation': keyword_238731}
        # Getting the type of 'DifferentialEvolutionSolver' (line 134)
        DifferentialEvolutionSolver_238725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 134)
        DifferentialEvolutionSolver_call_result_238733 = invoke(stypy.reporting.localization.Localization(__file__, 134, 17), DifferentialEvolutionSolver_238725, *[quadratic_238727, bounds_238729], **kwargs_238732)
        
        # Assigning a type to the variable 'solver' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'solver', DifferentialEvolutionSolver_call_result_238733)
        
        # Call to assert_equal(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'solver' (line 138)
        solver_238735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'solver', False)
        # Obtaining the member 'dither' of a type (line 138)
        dither_238736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 21), solver_238735, 'dither')
        
        # Call to list(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'mutation' (line 138)
        mutation_238738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 41), 'mutation', False)
        # Processing the call keyword arguments (line 138)
        kwargs_238739 = {}
        # Getting the type of 'list' (line 138)
        list_238737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 36), 'list', False)
        # Calling list(args, kwargs) (line 138)
        list_call_result_238740 = invoke(stypy.reporting.localization.Localization(__file__, 138, 36), list_238737, *[mutation_238738], **kwargs_238739)
        
        # Processing the call keyword arguments (line 138)
        kwargs_238741 = {}
        # Getting the type of 'assert_equal' (line 138)
        assert_equal_238734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 138)
        assert_equal_call_result_238742 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), assert_equal_238734, *[dither_238736, list_call_result_238740], **kwargs_238741)
        
        
        # ################# End of 'test_can_init_with_dithering(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_can_init_with_dithering' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_238743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238743)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_can_init_with_dithering'
        return stypy_return_type_238743


    @norecursion
    def test_invalid_mutation_values_arent_accepted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_invalid_mutation_values_arent_accepted'
        module_type_store = module_type_store.open_function_context('test_invalid_mutation_values_arent_accepted', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted')
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_invalid_mutation_values_arent_accepted', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_invalid_mutation_values_arent_accepted', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_invalid_mutation_values_arent_accepted(...)' code ##################

        
        # Assigning a Name to a Name (line 141):
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'rosen' (line 141)
        rosen_238744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'rosen')
        # Assigning a type to the variable 'func' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'func', rosen_238744)
        
        # Assigning a Tuple to a Name (line 142):
        
        # Assigning a Tuple to a Name (line 142):
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_238745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        float_238746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 20), tuple_238745, float_238746)
        # Adding element type (line 142)
        int_238747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 20), tuple_238745, int_238747)
        
        # Assigning a type to the variable 'mutation' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'mutation', tuple_238745)
        
        # Call to assert_raises(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'ValueError' (line 143)
        ValueError_238749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'ValueError', False)
        # Getting the type of 'DifferentialEvolutionSolver' (line 144)
        DifferentialEvolutionSolver_238750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'DifferentialEvolutionSolver', False)
        # Getting the type of 'func' (line 145)
        func_238751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 26), 'func', False)
        # Getting the type of 'self' (line 146)
        self_238752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'self', False)
        # Obtaining the member 'bounds' of a type (line 146)
        bounds_238753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 26), self_238752, 'bounds')
        # Processing the call keyword arguments (line 143)
        # Getting the type of 'mutation' (line 147)
        mutation_238754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 35), 'mutation', False)
        keyword_238755 = mutation_238754
        kwargs_238756 = {'mutation': keyword_238755}
        # Getting the type of 'assert_raises' (line 143)
        assert_raises_238748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 143)
        assert_raises_call_result_238757 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assert_raises_238748, *[ValueError_238749, DifferentialEvolutionSolver_238750, func_238751, bounds_238753], **kwargs_238756)
        
        
        # Assigning a Tuple to a Name (line 149):
        
        # Assigning a Tuple to a Name (line 149):
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_238758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        int_238759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 20), tuple_238758, int_238759)
        # Adding element type (line 149)
        int_238760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 20), tuple_238758, int_238760)
        
        # Assigning a type to the variable 'mutation' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'mutation', tuple_238758)
        
        # Call to assert_raises(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'ValueError' (line 150)
        ValueError_238762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'ValueError', False)
        # Getting the type of 'DifferentialEvolutionSolver' (line 151)
        DifferentialEvolutionSolver_238763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'DifferentialEvolutionSolver', False)
        # Getting the type of 'func' (line 152)
        func_238764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 26), 'func', False)
        # Getting the type of 'self' (line 153)
        self_238765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'self', False)
        # Obtaining the member 'bounds' of a type (line 153)
        bounds_238766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 26), self_238765, 'bounds')
        # Processing the call keyword arguments (line 150)
        # Getting the type of 'mutation' (line 154)
        mutation_238767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 35), 'mutation', False)
        keyword_238768 = mutation_238767
        kwargs_238769 = {'mutation': keyword_238768}
        # Getting the type of 'assert_raises' (line 150)
        assert_raises_238761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 150)
        assert_raises_call_result_238770 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), assert_raises_238761, *[ValueError_238762, DifferentialEvolutionSolver_238763, func_238764, bounds_238766], **kwargs_238769)
        
        
        # Assigning a Tuple to a Name (line 156):
        
        # Assigning a Tuple to a Name (line 156):
        
        # Obtaining an instance of the builtin type 'tuple' (line 156)
        tuple_238771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 156)
        # Adding element type (line 156)
        float_238772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), tuple_238771, float_238772)
        # Adding element type (line 156)
        # Getting the type of 'np' (line 156)
        np_238773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'np')
        # Obtaining the member 'nan' of a type (line 156)
        nan_238774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 25), np_238773, 'nan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 20), tuple_238771, nan_238774)
        
        # Assigning a type to the variable 'mutation' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'mutation', tuple_238771)
        
        # Call to assert_raises(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'ValueError' (line 157)
        ValueError_238776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'ValueError', False)
        # Getting the type of 'DifferentialEvolutionSolver' (line 158)
        DifferentialEvolutionSolver_238777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'DifferentialEvolutionSolver', False)
        # Getting the type of 'func' (line 159)
        func_238778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 26), 'func', False)
        # Getting the type of 'self' (line 160)
        self_238779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'self', False)
        # Obtaining the member 'bounds' of a type (line 160)
        bounds_238780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 26), self_238779, 'bounds')
        # Processing the call keyword arguments (line 157)
        # Getting the type of 'mutation' (line 161)
        mutation_238781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'mutation', False)
        keyword_238782 = mutation_238781
        kwargs_238783 = {'mutation': keyword_238782}
        # Getting the type of 'assert_raises' (line 157)
        assert_raises_238775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 157)
        assert_raises_call_result_238784 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), assert_raises_238775, *[ValueError_238776, DifferentialEvolutionSolver_238777, func_238778, bounds_238780], **kwargs_238783)
        
        
        # Assigning a Num to a Name (line 163):
        
        # Assigning a Num to a Name (line 163):
        float_238785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 19), 'float')
        # Assigning a type to the variable 'mutation' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'mutation', float_238785)
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to DifferentialEvolutionSolver(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'func' (line 164)
        func_238787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 45), 'func', False)
        # Getting the type of 'self' (line 165)
        self_238788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 165)
        bounds_238789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 45), self_238788, 'bounds')
        # Processing the call keyword arguments (line 164)
        # Getting the type of 'mutation' (line 166)
        mutation_238790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 54), 'mutation', False)
        keyword_238791 = mutation_238790
        kwargs_238792 = {'mutation': keyword_238791}
        # Getting the type of 'DifferentialEvolutionSolver' (line 164)
        DifferentialEvolutionSolver_238786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 164)
        DifferentialEvolutionSolver_call_result_238793 = invoke(stypy.reporting.localization.Localization(__file__, 164, 17), DifferentialEvolutionSolver_238786, *[func_238787, bounds_238789], **kwargs_238792)
        
        # Assigning a type to the variable 'solver' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'solver', DifferentialEvolutionSolver_call_result_238793)
        
        # Call to assert_equal(...): (line 167)
        # Processing the call arguments (line 167)
        float_238795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 21), 'float')
        # Getting the type of 'solver' (line 167)
        solver_238796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 26), 'solver', False)
        # Obtaining the member 'scale' of a type (line 167)
        scale_238797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 26), solver_238796, 'scale')
        # Processing the call keyword arguments (line 167)
        kwargs_238798 = {}
        # Getting the type of 'assert_equal' (line 167)
        assert_equal_238794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 167)
        assert_equal_call_result_238799 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), assert_equal_238794, *[float_238795, scale_238797], **kwargs_238798)
        
        
        # Call to assert_equal(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'None' (line 168)
        None_238801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'None', False)
        # Getting the type of 'solver' (line 168)
        solver_238802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'solver', False)
        # Obtaining the member 'dither' of a type (line 168)
        dither_238803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 27), solver_238802, 'dither')
        # Processing the call keyword arguments (line 168)
        kwargs_238804 = {}
        # Getting the type of 'assert_equal' (line 168)
        assert_equal_238800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 168)
        assert_equal_call_result_238805 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), assert_equal_238800, *[None_238801, dither_238803], **kwargs_238804)
        
        
        # ################# End of 'test_invalid_mutation_values_arent_accepted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_invalid_mutation_values_arent_accepted' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_238806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238806)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_invalid_mutation_values_arent_accepted'
        return stypy_return_type_238806


    @norecursion
    def test__scale_parameters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__scale_parameters'
        module_type_store = module_type_store.open_function_context('test__scale_parameters', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test__scale_parameters')
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test__scale_parameters.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test__scale_parameters', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__scale_parameters', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__scale_parameters(...)' code ##################

        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to array(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_238809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        float_238810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), list_238809, float_238810)
        
        # Processing the call keyword arguments (line 171)
        kwargs_238811 = {}
        # Getting the type of 'np' (line 171)
        np_238807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 171)
        array_238808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), np_238807, 'array')
        # Calling array(args, kwargs) (line 171)
        array_call_result_238812 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), array_238808, *[list_238809], **kwargs_238811)
        
        # Assigning a type to the variable 'trial' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'trial', array_call_result_238812)
        
        # Call to assert_equal(...): (line 172)
        # Processing the call arguments (line 172)
        int_238814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 21), 'int')
        
        # Call to _scale_parameters(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'trial' (line 172)
        trial_238818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 61), 'trial', False)
        # Processing the call keyword arguments (line 172)
        kwargs_238819 = {}
        # Getting the type of 'self' (line 172)
        self_238815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'self', False)
        # Obtaining the member 'dummy_solver' of a type (line 172)
        dummy_solver_238816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), self_238815, 'dummy_solver')
        # Obtaining the member '_scale_parameters' of a type (line 172)
        _scale_parameters_238817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), dummy_solver_238816, '_scale_parameters')
        # Calling _scale_parameters(args, kwargs) (line 172)
        _scale_parameters_call_result_238820 = invoke(stypy.reporting.localization.Localization(__file__, 172, 25), _scale_parameters_238817, *[trial_238818], **kwargs_238819)
        
        # Processing the call keyword arguments (line 172)
        kwargs_238821 = {}
        # Getting the type of 'assert_equal' (line 172)
        assert_equal_238813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 172)
        assert_equal_call_result_238822 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), assert_equal_238813, *[int_238814, _scale_parameters_call_result_238820], **kwargs_238821)
        
        
        # Assigning a Call to a Attribute (line 175):
        
        # Assigning a Call to a Attribute (line 175):
        
        # Call to array(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_238825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_238826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        int_238827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 45), list_238826, int_238827)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 44), list_238825, list_238826)
        # Adding element type (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_238828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        float_238829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 52), list_238828, float_238829)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 44), list_238825, list_238828)
        
        # Processing the call keyword arguments (line 175)
        kwargs_238830 = {}
        # Getting the type of 'np' (line 175)
        np_238823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 175)
        array_238824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 35), np_238823, 'array')
        # Calling array(args, kwargs) (line 175)
        array_call_result_238831 = invoke(stypy.reporting.localization.Localization(__file__, 175, 35), array_238824, *[list_238825], **kwargs_238830)
        
        # Getting the type of 'self' (line 175)
        self_238832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Obtaining the member 'dummy_solver' of a type (line 175)
        dummy_solver_238833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_238832, 'dummy_solver')
        # Setting the type of the member 'limits' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), dummy_solver_238833, 'limits', array_call_result_238831)
        
        # Call to assert_equal(...): (line 176)
        # Processing the call arguments (line 176)
        int_238835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 21), 'int')
        
        # Call to _scale_parameters(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'trial' (line 176)
        trial_238839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 61), 'trial', False)
        # Processing the call keyword arguments (line 176)
        kwargs_238840 = {}
        # Getting the type of 'self' (line 176)
        self_238836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'self', False)
        # Obtaining the member 'dummy_solver' of a type (line 176)
        dummy_solver_238837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), self_238836, 'dummy_solver')
        # Obtaining the member '_scale_parameters' of a type (line 176)
        _scale_parameters_238838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), dummy_solver_238837, '_scale_parameters')
        # Calling _scale_parameters(args, kwargs) (line 176)
        _scale_parameters_call_result_238841 = invoke(stypy.reporting.localization.Localization(__file__, 176, 25), _scale_parameters_238838, *[trial_238839], **kwargs_238840)
        
        # Processing the call keyword arguments (line 176)
        kwargs_238842 = {}
        # Getting the type of 'assert_equal' (line 176)
        assert_equal_238834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 176)
        assert_equal_call_result_238843 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assert_equal_238834, *[int_238835, _scale_parameters_call_result_238841], **kwargs_238842)
        
        
        # ################# End of 'test__scale_parameters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__scale_parameters' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_238844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__scale_parameters'
        return stypy_return_type_238844


    @norecursion
    def test__unscale_parameters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__unscale_parameters'
        module_type_store = module_type_store.open_function_context('test__unscale_parameters', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test__unscale_parameters')
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test__unscale_parameters.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test__unscale_parameters', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__unscale_parameters', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__unscale_parameters(...)' code ##################

        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to array(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_238847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        int_238848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 25), list_238847, int_238848)
        
        # Processing the call keyword arguments (line 179)
        kwargs_238849 = {}
        # Getting the type of 'np' (line 179)
        np_238845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 179)
        array_238846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), np_238845, 'array')
        # Calling array(args, kwargs) (line 179)
        array_call_result_238850 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), array_238846, *[list_238847], **kwargs_238849)
        
        # Assigning a type to the variable 'trial' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'trial', array_call_result_238850)
        
        # Call to assert_equal(...): (line 180)
        # Processing the call arguments (line 180)
        float_238852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 21), 'float')
        
        # Call to _unscale_parameters(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'trial' (line 180)
        trial_238856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 64), 'trial', False)
        # Processing the call keyword arguments (line 180)
        kwargs_238857 = {}
        # Getting the type of 'self' (line 180)
        self_238853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 26), 'self', False)
        # Obtaining the member 'dummy_solver' of a type (line 180)
        dummy_solver_238854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 26), self_238853, 'dummy_solver')
        # Obtaining the member '_unscale_parameters' of a type (line 180)
        _unscale_parameters_238855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 26), dummy_solver_238854, '_unscale_parameters')
        # Calling _unscale_parameters(args, kwargs) (line 180)
        _unscale_parameters_call_result_238858 = invoke(stypy.reporting.localization.Localization(__file__, 180, 26), _unscale_parameters_238855, *[trial_238856], **kwargs_238857)
        
        # Processing the call keyword arguments (line 180)
        kwargs_238859 = {}
        # Getting the type of 'assert_equal' (line 180)
        assert_equal_238851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 180)
        assert_equal_call_result_238860 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), assert_equal_238851, *[float_238852, _unscale_parameters_call_result_238858], **kwargs_238859)
        
        
        # Assigning a Call to a Attribute (line 183):
        
        # Assigning a Call to a Attribute (line 183):
        
        # Call to array(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_238863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_238864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        int_238865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 45), list_238864, int_238865)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 44), list_238863, list_238864)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_238866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        float_238867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 52), list_238866, float_238867)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 44), list_238863, list_238866)
        
        # Processing the call keyword arguments (line 183)
        kwargs_238868 = {}
        # Getting the type of 'np' (line 183)
        np_238861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 183)
        array_238862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 35), np_238861, 'array')
        # Calling array(args, kwargs) (line 183)
        array_call_result_238869 = invoke(stypy.reporting.localization.Localization(__file__, 183, 35), array_238862, *[list_238863], **kwargs_238868)
        
        # Getting the type of 'self' (line 183)
        self_238870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self')
        # Obtaining the member 'dummy_solver' of a type (line 183)
        dummy_solver_238871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_238870, 'dummy_solver')
        # Setting the type of the member 'limits' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), dummy_solver_238871, 'limits', array_call_result_238869)
        
        # Call to assert_equal(...): (line 184)
        # Processing the call arguments (line 184)
        float_238873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 21), 'float')
        
        # Call to _unscale_parameters(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'trial' (line 184)
        trial_238877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 64), 'trial', False)
        # Processing the call keyword arguments (line 184)
        kwargs_238878 = {}
        # Getting the type of 'self' (line 184)
        self_238874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'self', False)
        # Obtaining the member 'dummy_solver' of a type (line 184)
        dummy_solver_238875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 26), self_238874, 'dummy_solver')
        # Obtaining the member '_unscale_parameters' of a type (line 184)
        _unscale_parameters_238876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 26), dummy_solver_238875, '_unscale_parameters')
        # Calling _unscale_parameters(args, kwargs) (line 184)
        _unscale_parameters_call_result_238879 = invoke(stypy.reporting.localization.Localization(__file__, 184, 26), _unscale_parameters_238876, *[trial_238877], **kwargs_238878)
        
        # Processing the call keyword arguments (line 184)
        kwargs_238880 = {}
        # Getting the type of 'assert_equal' (line 184)
        assert_equal_238872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 184)
        assert_equal_call_result_238881 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), assert_equal_238872, *[float_238873, _unscale_parameters_call_result_238879], **kwargs_238880)
        
        
        # ################# End of 'test__unscale_parameters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__unscale_parameters' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_238882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238882)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__unscale_parameters'
        return stypy_return_type_238882


    @norecursion
    def test__ensure_constraint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__ensure_constraint'
        module_type_store = module_type_store.open_function_context('test__ensure_constraint', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test__ensure_constraint')
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test__ensure_constraint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test__ensure_constraint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__ensure_constraint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__ensure_constraint(...)' code ##################

        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to array(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Obtaining an instance of the builtin type 'list' (line 187)
        list_238885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 187)
        # Adding element type (line 187)
        float_238886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 25), list_238885, float_238886)
        # Adding element type (line 187)
        int_238887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 25), list_238885, int_238887)
        # Adding element type (line 187)
        float_238888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 25), list_238885, float_238888)
        # Adding element type (line 187)
        float_238889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 25), list_238885, float_238889)
        # Adding element type (line 187)
        float_238890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 25), list_238885, float_238890)
        
        # Processing the call keyword arguments (line 187)
        kwargs_238891 = {}
        # Getting the type of 'np' (line 187)
        np_238883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 187)
        array_238884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), np_238883, 'array')
        # Calling array(args, kwargs) (line 187)
        array_call_result_238892 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), array_238884, *[list_238885], **kwargs_238891)
        
        # Assigning a type to the variable 'trial' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'trial', array_call_result_238892)
        
        # Call to _ensure_constraint(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'trial' (line 188)
        trial_238896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 45), 'trial', False)
        # Processing the call keyword arguments (line 188)
        kwargs_238897 = {}
        # Getting the type of 'self' (line 188)
        self_238893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self', False)
        # Obtaining the member 'dummy_solver' of a type (line 188)
        dummy_solver_238894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_238893, 'dummy_solver')
        # Obtaining the member '_ensure_constraint' of a type (line 188)
        _ensure_constraint_238895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), dummy_solver_238894, '_ensure_constraint')
        # Calling _ensure_constraint(args, kwargs) (line 188)
        _ensure_constraint_call_result_238898 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), _ensure_constraint_238895, *[trial_238896], **kwargs_238897)
        
        
        # Call to assert_equal(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Call to all(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Getting the type of 'trial' (line 189)
        trial_238902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 28), 'trial', False)
        int_238903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 37), 'int')
        # Applying the binary operator '<=' (line 189)
        result_le_238904 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 28), '<=', trial_238902, int_238903)
        
        # Processing the call keyword arguments (line 189)
        kwargs_238905 = {}
        # Getting the type of 'np' (line 189)
        np_238900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'np', False)
        # Obtaining the member 'all' of a type (line 189)
        all_238901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 21), np_238900, 'all')
        # Calling all(args, kwargs) (line 189)
        all_call_result_238906 = invoke(stypy.reporting.localization.Localization(__file__, 189, 21), all_238901, *[result_le_238904], **kwargs_238905)
        
        # Getting the type of 'True' (line 189)
        True_238907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 41), 'True', False)
        # Processing the call keyword arguments (line 189)
        kwargs_238908 = {}
        # Getting the type of 'assert_equal' (line 189)
        assert_equal_238899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 189)
        assert_equal_call_result_238909 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assert_equal_238899, *[all_call_result_238906, True_238907], **kwargs_238908)
        
        
        # ################# End of 'test__ensure_constraint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__ensure_constraint' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_238910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__ensure_constraint'
        return stypy_return_type_238910


    @norecursion
    def test_differential_evolution(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_differential_evolution'
        module_type_store = module_type_store.open_function_context('test_differential_evolution', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_differential_evolution')
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_differential_evolution.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_differential_evolution', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_differential_evolution', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_differential_evolution(...)' code ##################

        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to DifferentialEvolutionSolver(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'self' (line 194)
        self_238912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 45), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 194)
        quadratic_238913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 45), self_238912, 'quadratic')
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_238914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        
        # Obtaining an instance of the builtin type 'tuple' (line 194)
        tuple_238915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 194)
        # Adding element type (line 194)
        int_238916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 63), tuple_238915, int_238916)
        # Adding element type (line 194)
        int_238917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 63), tuple_238915, int_238917)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 61), list_238914, tuple_238915)
        
        # Processing the call keyword arguments (line 194)
        kwargs_238918 = {}
        # Getting the type of 'DifferentialEvolutionSolver' (line 194)
        DifferentialEvolutionSolver_238911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 194)
        DifferentialEvolutionSolver_call_result_238919 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), DifferentialEvolutionSolver_238911, *[quadratic_238913, list_238914], **kwargs_238918)
        
        # Assigning a type to the variable 'solver' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'solver', DifferentialEvolutionSolver_call_result_238919)
        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to solve(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_238922 = {}
        # Getting the type of 'solver' (line 195)
        solver_238920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), 'solver', False)
        # Obtaining the member 'solve' of a type (line 195)
        solve_238921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 17), solver_238920, 'solve')
        # Calling solve(args, kwargs) (line 195)
        solve_call_result_238923 = invoke(stypy.reporting.localization.Localization(__file__, 195, 17), solve_238921, *[], **kwargs_238922)
        
        # Assigning a type to the variable 'result' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'result', solve_call_result_238923)
        
        # Call to assert_almost_equal(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'result' (line 196)
        result_238925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'result', False)
        # Obtaining the member 'fun' of a type (line 196)
        fun_238926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 28), result_238925, 'fun')
        
        # Call to quadratic(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'result' (line 196)
        result_238929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 55), 'result', False)
        # Obtaining the member 'x' of a type (line 196)
        x_238930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 55), result_238929, 'x')
        # Processing the call keyword arguments (line 196)
        kwargs_238931 = {}
        # Getting the type of 'self' (line 196)
        self_238927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 40), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 196)
        quadratic_238928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 40), self_238927, 'quadratic')
        # Calling quadratic(args, kwargs) (line 196)
        quadratic_call_result_238932 = invoke(stypy.reporting.localization.Localization(__file__, 196, 40), quadratic_238928, *[x_238930], **kwargs_238931)
        
        # Processing the call keyword arguments (line 196)
        kwargs_238933 = {}
        # Getting the type of 'assert_almost_equal' (line 196)
        assert_almost_equal_238924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 196)
        assert_almost_equal_call_result_238934 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), assert_almost_equal_238924, *[fun_238926, quadratic_call_result_238932], **kwargs_238933)
        
        
        # ################# End of 'test_differential_evolution(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_differential_evolution' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_238935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238935)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_differential_evolution'
        return stypy_return_type_238935


    @norecursion
    def test_best_solution_retrieval(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_best_solution_retrieval'
        module_type_store = module_type_store.open_function_context('test_best_solution_retrieval', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_best_solution_retrieval')
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_best_solution_retrieval.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_best_solution_retrieval', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_best_solution_retrieval', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_best_solution_retrieval(...)' code ##################

        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to DifferentialEvolutionSolver(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'self' (line 200)
        self_238937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 45), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 200)
        quadratic_238938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 45), self_238937, 'quadratic')
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_238939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_238940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        int_238941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 63), tuple_238940, int_238941)
        # Adding element type (line 200)
        int_238942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 63), tuple_238940, int_238942)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 61), list_238939, tuple_238940)
        
        # Processing the call keyword arguments (line 200)
        kwargs_238943 = {}
        # Getting the type of 'DifferentialEvolutionSolver' (line 200)
        DifferentialEvolutionSolver_238936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 200)
        DifferentialEvolutionSolver_call_result_238944 = invoke(stypy.reporting.localization.Localization(__file__, 200, 17), DifferentialEvolutionSolver_238936, *[quadratic_238938, list_238939], **kwargs_238943)
        
        # Assigning a type to the variable 'solver' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'solver', DifferentialEvolutionSolver_call_result_238944)
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to solve(...): (line 201)
        # Processing the call keyword arguments (line 201)
        kwargs_238947 = {}
        # Getting the type of 'solver' (line 201)
        solver_238945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 17), 'solver', False)
        # Obtaining the member 'solve' of a type (line 201)
        solve_238946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 17), solver_238945, 'solve')
        # Calling solve(args, kwargs) (line 201)
        solve_call_result_238948 = invoke(stypy.reporting.localization.Localization(__file__, 201, 17), solve_238946, *[], **kwargs_238947)
        
        # Assigning a type to the variable 'result' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'result', solve_call_result_238948)
        
        # Call to assert_almost_equal(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'result' (line 202)
        result_238950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'result', False)
        # Obtaining the member 'x' of a type (line 202)
        x_238951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 28), result_238950, 'x')
        # Getting the type of 'solver' (line 202)
        solver_238952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'solver', False)
        # Obtaining the member 'x' of a type (line 202)
        x_238953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 38), solver_238952, 'x')
        # Processing the call keyword arguments (line 202)
        kwargs_238954 = {}
        # Getting the type of 'assert_almost_equal' (line 202)
        assert_almost_equal_238949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 202)
        assert_almost_equal_call_result_238955 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assert_almost_equal_238949, *[x_238951, x_238953], **kwargs_238954)
        
        
        # ################# End of 'test_best_solution_retrieval(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_best_solution_retrieval' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_238956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238956)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_best_solution_retrieval'
        return stypy_return_type_238956


    @norecursion
    def test_callback_terminates(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_callback_terminates'
        module_type_store = module_type_store.open_function_context('test_callback_terminates', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_callback_terminates')
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_callback_terminates.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_callback_terminates', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_callback_terminates', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_callback_terminates(...)' code ##################

        
        # Assigning a List to a Name (line 206):
        
        # Assigning a List to a Name (line 206):
        
        # Obtaining an instance of the builtin type 'list' (line 206)
        list_238957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 206)
        # Adding element type (line 206)
        
        # Obtaining an instance of the builtin type 'tuple' (line 206)
        tuple_238958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 206)
        # Adding element type (line 206)
        int_238959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_238958, int_238959)
        # Adding element type (line 206)
        int_238960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_238958, int_238960)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 17), list_238957, tuple_238958)
        # Adding element type (line 206)
        
        # Obtaining an instance of the builtin type 'tuple' (line 206)
        tuple_238961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 206)
        # Adding element type (line 206)
        int_238962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 27), tuple_238961, int_238962)
        # Adding element type (line 206)
        int_238963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 27), tuple_238961, int_238963)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 17), list_238957, tuple_238961)
        
        # Assigning a type to the variable 'bounds' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'bounds', list_238957)

        @norecursion
        def callback(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            float_238964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 40), 'float')
            defaults = [float_238964]
            # Create a new context for function 'callback'
            module_type_store = module_type_store.open_function_context('callback', 208, 8, False)
            
            # Passed parameters checking function
            callback.stypy_localization = localization
            callback.stypy_type_of_self = None
            callback.stypy_type_store = module_type_store
            callback.stypy_function_name = 'callback'
            callback.stypy_param_names_list = ['param', 'convergence']
            callback.stypy_varargs_param_name = None
            callback.stypy_kwargs_param_name = None
            callback.stypy_call_defaults = defaults
            callback.stypy_call_varargs = varargs
            callback.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'callback', ['param', 'convergence'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'callback', localization, ['param', 'convergence'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'callback(...)' code ##################

            # Getting the type of 'True' (line 209)
            True_238965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'stypy_return_type', True_238965)
            
            # ################# End of 'callback(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'callback' in the type store
            # Getting the type of 'stypy_return_type' (line 208)
            stypy_return_type_238966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_238966)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'callback'
            return stypy_return_type_238966

        # Assigning a type to the variable 'callback' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'callback', callback)
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to differential_evolution(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'rosen' (line 211)
        rosen_238968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 40), 'rosen', False)
        # Getting the type of 'bounds' (line 211)
        bounds_238969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 47), 'bounds', False)
        # Processing the call keyword arguments (line 211)
        # Getting the type of 'callback' (line 211)
        callback_238970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 64), 'callback', False)
        keyword_238971 = callback_238970
        kwargs_238972 = {'callback': keyword_238971}
        # Getting the type of 'differential_evolution' (line 211)
        differential_evolution_238967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 17), 'differential_evolution', False)
        # Calling differential_evolution(args, kwargs) (line 211)
        differential_evolution_call_result_238973 = invoke(stypy.reporting.localization.Localization(__file__, 211, 17), differential_evolution_238967, *[rosen_238968, bounds_238969], **kwargs_238972)
        
        # Assigning a type to the variable 'result' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'result', differential_evolution_call_result_238973)
        
        # Call to assert_string_equal(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'result' (line 213)
        result_238975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'result', False)
        # Obtaining the member 'message' of a type (line 213)
        message_238976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 28), result_238975, 'message')
        str_238977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 32), 'str', 'callback function requested stop early by returning True')
        # Processing the call keyword arguments (line 213)
        kwargs_238978 = {}
        # Getting the type of 'assert_string_equal' (line 213)
        assert_string_equal_238974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'assert_string_equal', False)
        # Calling assert_string_equal(args, kwargs) (line 213)
        assert_string_equal_call_result_238979 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assert_string_equal_238974, *[message_238976, str_238977], **kwargs_238978)
        
        
        # ################# End of 'test_callback_terminates(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_callback_terminates' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_238980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_callback_terminates'
        return stypy_return_type_238980


    @norecursion
    def test_args_tuple_is_passed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_args_tuple_is_passed'
        module_type_store = module_type_store.open_function_context('test_args_tuple_is_passed', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_args_tuple_is_passed')
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_args_tuple_is_passed.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_args_tuple_is_passed', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_args_tuple_is_passed', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_args_tuple_is_passed(...)' code ##################

        
        # Assigning a List to a Name (line 219):
        
        # Assigning a List to a Name (line 219):
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_238981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        
        # Obtaining an instance of the builtin type 'tuple' (line 219)
        tuple_238982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 219)
        # Adding element type (line 219)
        int_238983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 19), tuple_238982, int_238983)
        # Adding element type (line 219)
        int_238984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 19), tuple_238982, int_238984)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 17), list_238981, tuple_238982)
        
        # Assigning a type to the variable 'bounds' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'bounds', list_238981)
        
        # Assigning a Tuple to a Name (line 220):
        
        # Assigning a Tuple to a Name (line 220):
        
        # Obtaining an instance of the builtin type 'tuple' (line 220)
        tuple_238985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 220)
        # Adding element type (line 220)
        float_238986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 16), tuple_238985, float_238986)
        # Adding element type (line 220)
        float_238987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 16), tuple_238985, float_238987)
        # Adding element type (line 220)
        float_238988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 16), tuple_238985, float_238988)
        
        # Assigning a type to the variable 'args' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'args', tuple_238985)

        @norecursion
        def quadratic(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'quadratic'
            module_type_store = module_type_store.open_function_context('quadratic', 222, 8, False)
            
            # Passed parameters checking function
            quadratic.stypy_localization = localization
            quadratic.stypy_type_of_self = None
            quadratic.stypy_type_store = module_type_store
            quadratic.stypy_function_name = 'quadratic'
            quadratic.stypy_param_names_list = ['x']
            quadratic.stypy_varargs_param_name = 'args'
            quadratic.stypy_kwargs_param_name = None
            quadratic.stypy_call_defaults = defaults
            quadratic.stypy_call_varargs = varargs
            quadratic.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'quadratic', ['x'], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'quadratic', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'quadratic(...)' code ##################

            
            # Type idiom detected: calculating its left and rigth part (line 223)
            # Getting the type of 'args' (line 223)
            args_238989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'args')
            # Getting the type of 'tuple' (line 223)
            tuple_238990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'tuple')
            
            (may_be_238991, more_types_in_union_238992) = may_be_type(args_238989, tuple_238990)

            if may_be_238991:

                if more_types_in_union_238992:
                    # Runtime conditional SSA (line 223)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'args' (line 223)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'args', tuple_238990())
                
                # Call to ValueError(...): (line 224)
                # Processing the call arguments (line 224)
                str_238994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 33), 'str', 'args should be a tuple')
                # Processing the call keyword arguments (line 224)
                kwargs_238995 = {}
                # Getting the type of 'ValueError' (line 224)
                ValueError_238993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 22), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 224)
                ValueError_call_result_238996 = invoke(stypy.reporting.localization.Localization(__file__, 224, 22), ValueError_238993, *[str_238994], **kwargs_238995)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 224, 16), ValueError_call_result_238996, 'raise parameter', BaseException)

                if more_types_in_union_238992:
                    # SSA join for if statement (line 223)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Obtaining the type of the subscript
            int_238997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 24), 'int')
            # Getting the type of 'args' (line 225)
            args_238998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'args')
            # Obtaining the member '__getitem__' of a type (line 225)
            getitem___238999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 19), args_238998, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 225)
            subscript_call_result_239000 = invoke(stypy.reporting.localization.Localization(__file__, 225, 19), getitem___238999, int_238997)
            
            
            # Obtaining the type of the subscript
            int_239001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 34), 'int')
            # Getting the type of 'args' (line 225)
            args_239002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 29), 'args')
            # Obtaining the member '__getitem__' of a type (line 225)
            getitem___239003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 29), args_239002, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 225)
            subscript_call_result_239004 = invoke(stypy.reporting.localization.Localization(__file__, 225, 29), getitem___239003, int_239001)
            
            # Getting the type of 'x' (line 225)
            x_239005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 39), 'x')
            # Applying the binary operator '*' (line 225)
            result_mul_239006 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 29), '*', subscript_call_result_239004, x_239005)
            
            # Applying the binary operator '+' (line 225)
            result_add_239007 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 19), '+', subscript_call_result_239000, result_mul_239006)
            
            
            # Obtaining the type of the subscript
            int_239008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 48), 'int')
            # Getting the type of 'args' (line 225)
            args_239009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 43), 'args')
            # Obtaining the member '__getitem__' of a type (line 225)
            getitem___239010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 43), args_239009, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 225)
            subscript_call_result_239011 = invoke(stypy.reporting.localization.Localization(__file__, 225, 43), getitem___239010, int_239008)
            
            # Getting the type of 'x' (line 225)
            x_239012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 53), 'x')
            float_239013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 56), 'float')
            # Applying the binary operator '**' (line 225)
            result_pow_239014 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 53), '**', x_239012, float_239013)
            
            # Applying the binary operator '*' (line 225)
            result_mul_239015 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 43), '*', subscript_call_result_239011, result_pow_239014)
            
            # Applying the binary operator '+' (line 225)
            result_add_239016 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 41), '+', result_add_239007, result_mul_239015)
            
            # Assigning a type to the variable 'stypy_return_type' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'stypy_return_type', result_add_239016)
            
            # ################# End of 'quadratic(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'quadratic' in the type store
            # Getting the type of 'stypy_return_type' (line 222)
            stypy_return_type_239017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_239017)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'quadratic'
            return stypy_return_type_239017

        # Assigning a type to the variable 'quadratic' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'quadratic', quadratic)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to differential_evolution(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'quadratic' (line 227)
        quadratic_239019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 40), 'quadratic', False)
        # Getting the type of 'bounds' (line 228)
        bounds_239020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 40), 'bounds', False)
        # Processing the call keyword arguments (line 227)
        # Getting the type of 'args' (line 229)
        args_239021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 45), 'args', False)
        keyword_239022 = args_239021
        # Getting the type of 'True' (line 230)
        True_239023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 47), 'True', False)
        keyword_239024 = True_239023
        kwargs_239025 = {'polish': keyword_239024, 'args': keyword_239022}
        # Getting the type of 'differential_evolution' (line 227)
        differential_evolution_239018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 17), 'differential_evolution', False)
        # Calling differential_evolution(args, kwargs) (line 227)
        differential_evolution_call_result_239026 = invoke(stypy.reporting.localization.Localization(__file__, 227, 17), differential_evolution_239018, *[quadratic_239019, bounds_239020], **kwargs_239025)
        
        # Assigning a type to the variable 'result' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'result', differential_evolution_call_result_239026)
        
        # Call to assert_almost_equal(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'result' (line 231)
        result_239028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 28), 'result', False)
        # Obtaining the member 'fun' of a type (line 231)
        fun_239029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 28), result_239028, 'fun')
        int_239030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 40), 'int')
        float_239031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'float')
        # Applying the binary operator 'div' (line 231)
        result_div_239032 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 40), 'div', int_239030, float_239031)
        
        # Processing the call keyword arguments (line 231)
        kwargs_239033 = {}
        # Getting the type of 'assert_almost_equal' (line 231)
        assert_almost_equal_239027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 231)
        assert_almost_equal_call_result_239034 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), assert_almost_equal_239027, *[fun_239029, result_div_239032], **kwargs_239033)
        
        
        # ################# End of 'test_args_tuple_is_passed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_args_tuple_is_passed' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_239035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239035)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_args_tuple_is_passed'
        return stypy_return_type_239035


    @norecursion
    def test_init_with_invalid_strategy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_init_with_invalid_strategy'
        module_type_store = module_type_store.open_function_context('test_init_with_invalid_strategy', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_init_with_invalid_strategy')
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_init_with_invalid_strategy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_init_with_invalid_strategy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_init_with_invalid_strategy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_init_with_invalid_strategy(...)' code ##################

        
        # Assigning a Name to a Name (line 235):
        
        # Assigning a Name to a Name (line 235):
        # Getting the type of 'rosen' (line 235)
        rosen_239036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'rosen')
        # Assigning a type to the variable 'func' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'func', rosen_239036)
        
        # Assigning a List to a Name (line 236):
        
        # Assigning a List to a Name (line 236):
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_239037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'tuple' (line 236)
        tuple_239038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 236)
        # Adding element type (line 236)
        int_239039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 19), tuple_239038, int_239039)
        # Adding element type (line 236)
        int_239040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 19), tuple_239038, int_239040)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 17), list_239037, tuple_239038)
        
        # Assigning a type to the variable 'bounds' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'bounds', list_239037)
        
        # Call to assert_raises(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'ValueError' (line 237)
        ValueError_239042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'ValueError', False)
        # Getting the type of 'differential_evolution' (line 238)
        differential_evolution_239043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 26), 'differential_evolution', False)
        # Getting the type of 'func' (line 239)
        func_239044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 26), 'func', False)
        # Getting the type of 'bounds' (line 240)
        bounds_239045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 26), 'bounds', False)
        # Processing the call keyword arguments (line 237)
        str_239046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 35), 'str', 'abc')
        keyword_239047 = str_239046
        kwargs_239048 = {'strategy': keyword_239047}
        # Getting the type of 'assert_raises' (line 237)
        assert_raises_239041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 237)
        assert_raises_call_result_239049 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), assert_raises_239041, *[ValueError_239042, differential_evolution_239043, func_239044, bounds_239045], **kwargs_239048)
        
        
        # ################# End of 'test_init_with_invalid_strategy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_init_with_invalid_strategy' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_239050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_init_with_invalid_strategy'
        return stypy_return_type_239050


    @norecursion
    def test_bounds_checking(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bounds_checking'
        module_type_store = module_type_store.open_function_context('test_bounds_checking', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_bounds_checking')
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_bounds_checking.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_bounds_checking', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bounds_checking', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bounds_checking(...)' code ##################

        
        # Assigning a Name to a Name (line 245):
        
        # Assigning a Name to a Name (line 245):
        # Getting the type of 'rosen' (line 245)
        rosen_239051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'rosen')
        # Assigning a type to the variable 'func' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'func', rosen_239051)
        
        # Assigning a List to a Name (line 246):
        
        # Assigning a List to a Name (line 246):
        
        # Obtaining an instance of the builtin type 'list' (line 246)
        list_239052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 246)
        # Adding element type (line 246)
        
        # Obtaining an instance of the builtin type 'tuple' (line 246)
        tuple_239053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 246)
        # Adding element type (line 246)
        int_239054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), tuple_239053, int_239054)
        # Adding element type (line 246)
        # Getting the type of 'None' (line 246)
        None_239055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), tuple_239053, None_239055)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 17), list_239052, tuple_239053)
        
        # Assigning a type to the variable 'bounds' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'bounds', list_239052)
        
        # Call to assert_raises(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'ValueError' (line 247)
        ValueError_239057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'ValueError', False)
        # Getting the type of 'differential_evolution' (line 248)
        differential_evolution_239058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 26), 'differential_evolution', False)
        # Getting the type of 'func' (line 249)
        func_239059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'func', False)
        # Getting the type of 'bounds' (line 250)
        bounds_239060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 26), 'bounds', False)
        # Processing the call keyword arguments (line 247)
        kwargs_239061 = {}
        # Getting the type of 'assert_raises' (line 247)
        assert_raises_239056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 247)
        assert_raises_call_result_239062 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), assert_raises_239056, *[ValueError_239057, differential_evolution_239058, func_239059, bounds_239060], **kwargs_239061)
        
        
        # Assigning a List to a Name (line 251):
        
        # Assigning a List to a Name (line 251):
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_239063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        int_239064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 17), list_239063, int_239064)
        
        # Assigning a type to the variable 'bounds' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'bounds', list_239063)
        
        # Call to assert_raises(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'ValueError' (line 252)
        ValueError_239066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 22), 'ValueError', False)
        # Getting the type of 'differential_evolution' (line 253)
        differential_evolution_239067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'differential_evolution', False)
        # Getting the type of 'func' (line 254)
        func_239068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'func', False)
        # Getting the type of 'bounds' (line 255)
        bounds_239069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 26), 'bounds', False)
        # Processing the call keyword arguments (line 252)
        kwargs_239070 = {}
        # Getting the type of 'assert_raises' (line 252)
        assert_raises_239065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 252)
        assert_raises_call_result_239071 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), assert_raises_239065, *[ValueError_239066, differential_evolution_239067, func_239068, bounds_239069], **kwargs_239070)
        
        
        # Assigning a List to a Name (line 256):
        
        # Assigning a List to a Name (line 256):
        
        # Obtaining an instance of the builtin type 'list' (line 256)
        list_239072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 256)
        # Adding element type (line 256)
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_239073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        int_239074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), tuple_239073, int_239074)
        # Adding element type (line 256)
        int_239075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), tuple_239073, int_239075)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 17), list_239072, tuple_239073)
        # Adding element type (line 256)
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_239076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        int_239077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 28), tuple_239076, int_239077)
        # Adding element type (line 256)
        int_239078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 28), tuple_239076, int_239078)
        # Adding element type (line 256)
        int_239079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 28), tuple_239076, int_239079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 17), list_239072, tuple_239076)
        
        # Assigning a type to the variable 'bounds' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'bounds', list_239072)
        
        # Call to assert_raises(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'ValueError' (line 257)
        ValueError_239081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'ValueError', False)
        # Getting the type of 'differential_evolution' (line 258)
        differential_evolution_239082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 26), 'differential_evolution', False)
        # Getting the type of 'func' (line 259)
        func_239083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'func', False)
        # Getting the type of 'bounds' (line 260)
        bounds_239084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 26), 'bounds', False)
        # Processing the call keyword arguments (line 257)
        kwargs_239085 = {}
        # Getting the type of 'assert_raises' (line 257)
        assert_raises_239080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 257)
        assert_raises_call_result_239086 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), assert_raises_239080, *[ValueError_239081, differential_evolution_239082, func_239083, bounds_239084], **kwargs_239085)
        
        
        # ################# End of 'test_bounds_checking(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bounds_checking' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_239087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239087)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bounds_checking'
        return stypy_return_type_239087


    @norecursion
    def test_select_samples(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_select_samples'
        module_type_store = module_type_store.open_function_context('test_select_samples', 262, 4, False)
        # Assigning a type to the variable 'self' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_select_samples')
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_select_samples.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_select_samples', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_select_samples', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_select_samples(...)' code ##################

        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to reshape(...): (line 264)
        # Processing the call arguments (line 264)
        int_239096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 57), 'int')
        int_239097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 60), 'int')
        # Processing the call keyword arguments (line 264)
        kwargs_239098 = {}
        
        # Call to arange(...): (line 264)
        # Processing the call arguments (line 264)
        float_239090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 27), 'float')
        # Processing the call keyword arguments (line 264)
        str_239091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 38), 'str', 'float64')
        keyword_239092 = str_239091
        kwargs_239093 = {'dtype': keyword_239092}
        # Getting the type of 'np' (line 264)
        np_239088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 264)
        arange_239089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 17), np_239088, 'arange')
        # Calling arange(args, kwargs) (line 264)
        arange_call_result_239094 = invoke(stypy.reporting.localization.Localization(__file__, 264, 17), arange_239089, *[float_239090], **kwargs_239093)
        
        # Obtaining the member 'reshape' of a type (line 264)
        reshape_239095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 17), arange_call_result_239094, 'reshape')
        # Calling reshape(args, kwargs) (line 264)
        reshape_call_result_239099 = invoke(stypy.reporting.localization.Localization(__file__, 264, 17), reshape_239095, *[int_239096, int_239097], **kwargs_239098)
        
        # Assigning a type to the variable 'limits' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'limits', reshape_call_result_239099)
        
        # Assigning a Call to a Name (line 265):
        
        # Assigning a Call to a Name (line 265):
        
        # Call to list(...): (line 265)
        # Processing the call arguments (line 265)
        
        # Call to zip(...): (line 265)
        # Processing the call arguments (line 265)
        
        # Obtaining the type of the subscript
        int_239102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 33), 'int')
        slice_239103 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 26), None, None, None)
        # Getting the type of 'limits' (line 265)
        limits_239104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 26), 'limits', False)
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___239105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 26), limits_239104, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_239106 = invoke(stypy.reporting.localization.Localization(__file__, 265, 26), getitem___239105, (int_239102, slice_239103))
        
        
        # Obtaining the type of the subscript
        int_239107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 47), 'int')
        slice_239108 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 40), None, None, None)
        # Getting the type of 'limits' (line 265)
        limits_239109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 40), 'limits', False)
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___239110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 40), limits_239109, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_239111 = invoke(stypy.reporting.localization.Localization(__file__, 265, 40), getitem___239110, (int_239107, slice_239108))
        
        # Processing the call keyword arguments (line 265)
        kwargs_239112 = {}
        # Getting the type of 'zip' (line 265)
        zip_239101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'zip', False)
        # Calling zip(args, kwargs) (line 265)
        zip_call_result_239113 = invoke(stypy.reporting.localization.Localization(__file__, 265, 22), zip_239101, *[subscript_call_result_239106, subscript_call_result_239111], **kwargs_239112)
        
        # Processing the call keyword arguments (line 265)
        kwargs_239114 = {}
        # Getting the type of 'list' (line 265)
        list_239100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'list', False)
        # Calling list(args, kwargs) (line 265)
        list_call_result_239115 = invoke(stypy.reporting.localization.Localization(__file__, 265, 17), list_239100, *[zip_call_result_239113], **kwargs_239114)
        
        # Assigning a type to the variable 'bounds' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'bounds', list_call_result_239115)
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to DifferentialEvolutionSolver(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'None' (line 266)
        None_239117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 45), 'None', False)
        # Getting the type of 'bounds' (line 266)
        bounds_239118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 51), 'bounds', False)
        # Processing the call keyword arguments (line 266)
        int_239119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 67), 'int')
        keyword_239120 = int_239119
        kwargs_239121 = {'popsize': keyword_239120}
        # Getting the type of 'DifferentialEvolutionSolver' (line 266)
        DifferentialEvolutionSolver_239116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 266)
        DifferentialEvolutionSolver_call_result_239122 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), DifferentialEvolutionSolver_239116, *[None_239117, bounds_239118], **kwargs_239121)
        
        # Assigning a type to the variable 'solver' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'solver', DifferentialEvolutionSolver_call_result_239122)
        
        # Assigning a Num to a Name (line 267):
        
        # Assigning a Num to a Name (line 267):
        int_239123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 20), 'int')
        # Assigning a type to the variable 'candidate' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'candidate', int_239123)
        
        # Assigning a Call to a Tuple (line 268):
        
        # Assigning a Subscript to a Name (line 268):
        
        # Obtaining the type of the subscript
        int_239124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        
        # Call to _select_samples(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'candidate' (line 268)
        candidate_239127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 52), 'candidate', False)
        int_239128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 63), 'int')
        # Processing the call keyword arguments (line 268)
        kwargs_239129 = {}
        # Getting the type of 'solver' (line 268)
        solver_239125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'solver', False)
        # Obtaining the member '_select_samples' of a type (line 268)
        _select_samples_239126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 29), solver_239125, '_select_samples')
        # Calling _select_samples(args, kwargs) (line 268)
        _select_samples_call_result_239130 = invoke(stypy.reporting.localization.Localization(__file__, 268, 29), _select_samples_239126, *[candidate_239127, int_239128], **kwargs_239129)
        
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___239131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), _select_samples_call_result_239130, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_239132 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), getitem___239131, int_239124)
        
        # Assigning a type to the variable 'tuple_var_assignment_238292' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238292', subscript_call_result_239132)
        
        # Assigning a Subscript to a Name (line 268):
        
        # Obtaining the type of the subscript
        int_239133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        
        # Call to _select_samples(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'candidate' (line 268)
        candidate_239136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 52), 'candidate', False)
        int_239137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 63), 'int')
        # Processing the call keyword arguments (line 268)
        kwargs_239138 = {}
        # Getting the type of 'solver' (line 268)
        solver_239134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'solver', False)
        # Obtaining the member '_select_samples' of a type (line 268)
        _select_samples_239135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 29), solver_239134, '_select_samples')
        # Calling _select_samples(args, kwargs) (line 268)
        _select_samples_call_result_239139 = invoke(stypy.reporting.localization.Localization(__file__, 268, 29), _select_samples_239135, *[candidate_239136, int_239137], **kwargs_239138)
        
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___239140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), _select_samples_call_result_239139, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_239141 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), getitem___239140, int_239133)
        
        # Assigning a type to the variable 'tuple_var_assignment_238293' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238293', subscript_call_result_239141)
        
        # Assigning a Subscript to a Name (line 268):
        
        # Obtaining the type of the subscript
        int_239142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        
        # Call to _select_samples(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'candidate' (line 268)
        candidate_239145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 52), 'candidate', False)
        int_239146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 63), 'int')
        # Processing the call keyword arguments (line 268)
        kwargs_239147 = {}
        # Getting the type of 'solver' (line 268)
        solver_239143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'solver', False)
        # Obtaining the member '_select_samples' of a type (line 268)
        _select_samples_239144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 29), solver_239143, '_select_samples')
        # Calling _select_samples(args, kwargs) (line 268)
        _select_samples_call_result_239148 = invoke(stypy.reporting.localization.Localization(__file__, 268, 29), _select_samples_239144, *[candidate_239145, int_239146], **kwargs_239147)
        
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___239149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), _select_samples_call_result_239148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_239150 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), getitem___239149, int_239142)
        
        # Assigning a type to the variable 'tuple_var_assignment_238294' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238294', subscript_call_result_239150)
        
        # Assigning a Subscript to a Name (line 268):
        
        # Obtaining the type of the subscript
        int_239151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        
        # Call to _select_samples(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'candidate' (line 268)
        candidate_239154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 52), 'candidate', False)
        int_239155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 63), 'int')
        # Processing the call keyword arguments (line 268)
        kwargs_239156 = {}
        # Getting the type of 'solver' (line 268)
        solver_239152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'solver', False)
        # Obtaining the member '_select_samples' of a type (line 268)
        _select_samples_239153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 29), solver_239152, '_select_samples')
        # Calling _select_samples(args, kwargs) (line 268)
        _select_samples_call_result_239157 = invoke(stypy.reporting.localization.Localization(__file__, 268, 29), _select_samples_239153, *[candidate_239154, int_239155], **kwargs_239156)
        
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___239158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), _select_samples_call_result_239157, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_239159 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), getitem___239158, int_239151)
        
        # Assigning a type to the variable 'tuple_var_assignment_238295' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238295', subscript_call_result_239159)
        
        # Assigning a Subscript to a Name (line 268):
        
        # Obtaining the type of the subscript
        int_239160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        
        # Call to _select_samples(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'candidate' (line 268)
        candidate_239163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 52), 'candidate', False)
        int_239164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 63), 'int')
        # Processing the call keyword arguments (line 268)
        kwargs_239165 = {}
        # Getting the type of 'solver' (line 268)
        solver_239161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'solver', False)
        # Obtaining the member '_select_samples' of a type (line 268)
        _select_samples_239162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 29), solver_239161, '_select_samples')
        # Calling _select_samples(args, kwargs) (line 268)
        _select_samples_call_result_239166 = invoke(stypy.reporting.localization.Localization(__file__, 268, 29), _select_samples_239162, *[candidate_239163, int_239164], **kwargs_239165)
        
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___239167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), _select_samples_call_result_239166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_239168 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), getitem___239167, int_239160)
        
        # Assigning a type to the variable 'tuple_var_assignment_238296' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238296', subscript_call_result_239168)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'tuple_var_assignment_238292' (line 268)
        tuple_var_assignment_238292_239169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238292')
        # Assigning a type to the variable 'r1' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'r1', tuple_var_assignment_238292_239169)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'tuple_var_assignment_238293' (line 268)
        tuple_var_assignment_238293_239170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238293')
        # Assigning a type to the variable 'r2' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'r2', tuple_var_assignment_238293_239170)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'tuple_var_assignment_238294' (line 268)
        tuple_var_assignment_238294_239171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238294')
        # Assigning a type to the variable 'r3' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'r3', tuple_var_assignment_238294_239171)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'tuple_var_assignment_238295' (line 268)
        tuple_var_assignment_238295_239172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238295')
        # Assigning a type to the variable 'r4' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'r4', tuple_var_assignment_238295_239172)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'tuple_var_assignment_238296' (line 268)
        tuple_var_assignment_238296_239173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_238296')
        # Assigning a type to the variable 'r5' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 24), 'r5', tuple_var_assignment_238296_239173)
        
        # Call to assert_equal(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Call to len(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Call to unique(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Call to array(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Obtaining an instance of the builtin type 'list' (line 270)
        list_239180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 270)
        # Adding element type (line 270)
        # Getting the type of 'candidate' (line 270)
        candidate_239181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 36), 'candidate', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 35), list_239180, candidate_239181)
        # Adding element type (line 270)
        # Getting the type of 'r1' (line 270)
        r1_239182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 47), 'r1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 35), list_239180, r1_239182)
        # Adding element type (line 270)
        # Getting the type of 'r2' (line 270)
        r2_239183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 51), 'r2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 35), list_239180, r2_239183)
        # Adding element type (line 270)
        # Getting the type of 'r3' (line 270)
        r3_239184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 55), 'r3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 35), list_239180, r3_239184)
        # Adding element type (line 270)
        # Getting the type of 'r4' (line 270)
        r4_239185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 59), 'r4', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 35), list_239180, r4_239185)
        # Adding element type (line 270)
        # Getting the type of 'r5' (line 270)
        r5_239186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 63), 'r5', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 35), list_239180, r5_239186)
        
        # Processing the call keyword arguments (line 270)
        kwargs_239187 = {}
        # Getting the type of 'np' (line 270)
        np_239178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 26), 'np', False)
        # Obtaining the member 'array' of a type (line 270)
        array_239179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 26), np_239178, 'array')
        # Calling array(args, kwargs) (line 270)
        array_call_result_239188 = invoke(stypy.reporting.localization.Localization(__file__, 270, 26), array_239179, *[list_239180], **kwargs_239187)
        
        # Processing the call keyword arguments (line 270)
        kwargs_239189 = {}
        # Getting the type of 'np' (line 270)
        np_239176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'np', False)
        # Obtaining the member 'unique' of a type (line 270)
        unique_239177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), np_239176, 'unique')
        # Calling unique(args, kwargs) (line 270)
        unique_call_result_239190 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), unique_239177, *[array_call_result_239188], **kwargs_239189)
        
        # Processing the call keyword arguments (line 270)
        kwargs_239191 = {}
        # Getting the type of 'len' (line 270)
        len_239175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'len', False)
        # Calling len(args, kwargs) (line 270)
        len_call_result_239192 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), len_239175, *[unique_call_result_239190], **kwargs_239191)
        
        int_239193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 71), 'int')
        # Processing the call keyword arguments (line 269)
        kwargs_239194 = {}
        # Getting the type of 'assert_equal' (line 269)
        assert_equal_239174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 269)
        assert_equal_call_result_239195 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), assert_equal_239174, *[len_call_result_239192, int_239193], **kwargs_239194)
        
        
        # ################# End of 'test_select_samples(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_select_samples' in the type store
        # Getting the type of 'stypy_return_type' (line 262)
        stypy_return_type_239196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_select_samples'
        return stypy_return_type_239196


    @norecursion
    def test_maxiter_stops_solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_maxiter_stops_solve'
        module_type_store = module_type_store.open_function_context('test_maxiter_stops_solve', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_maxiter_stops_solve')
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_maxiter_stops_solve.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_maxiter_stops_solve', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_maxiter_stops_solve', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_maxiter_stops_solve(...)' code ##################

        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to DifferentialEvolutionSolver(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'rosen' (line 275)
        rosen_239198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 45), 'rosen', False)
        # Getting the type of 'self' (line 275)
        self_239199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 52), 'self', False)
        # Obtaining the member 'bounds' of a type (line 275)
        bounds_239200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 52), self_239199, 'bounds')
        # Processing the call keyword arguments (line 275)
        int_239201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 73), 'int')
        keyword_239202 = int_239201
        kwargs_239203 = {'maxiter': keyword_239202}
        # Getting the type of 'DifferentialEvolutionSolver' (line 275)
        DifferentialEvolutionSolver_239197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 275)
        DifferentialEvolutionSolver_call_result_239204 = invoke(stypy.reporting.localization.Localization(__file__, 275, 17), DifferentialEvolutionSolver_239197, *[rosen_239198, bounds_239200], **kwargs_239203)
        
        # Assigning a type to the variable 'solver' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'solver', DifferentialEvolutionSolver_call_result_239204)
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to solve(...): (line 276)
        # Processing the call keyword arguments (line 276)
        kwargs_239207 = {}
        # Getting the type of 'solver' (line 276)
        solver_239205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 17), 'solver', False)
        # Obtaining the member 'solve' of a type (line 276)
        solve_239206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 17), solver_239205, 'solve')
        # Calling solve(args, kwargs) (line 276)
        solve_call_result_239208 = invoke(stypy.reporting.localization.Localization(__file__, 276, 17), solve_239206, *[], **kwargs_239207)
        
        # Assigning a type to the variable 'result' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'result', solve_call_result_239208)
        
        # Call to assert_equal(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'result' (line 277)
        result_239210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'result', False)
        # Obtaining the member 'success' of a type (line 277)
        success_239211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 21), result_239210, 'success')
        # Getting the type of 'False' (line 277)
        False_239212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 37), 'False', False)
        # Processing the call keyword arguments (line 277)
        kwargs_239213 = {}
        # Getting the type of 'assert_equal' (line 277)
        assert_equal_239209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 277)
        assert_equal_call_result_239214 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), assert_equal_239209, *[success_239211, False_239212], **kwargs_239213)
        
        
        # Call to assert_equal(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'result' (line 278)
        result_239216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'result', False)
        # Obtaining the member 'message' of a type (line 278)
        message_239217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 21), result_239216, 'message')
        str_239218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 24), 'str', 'Maximum number of iterations has been exceeded.')
        # Processing the call keyword arguments (line 278)
        kwargs_239219 = {}
        # Getting the type of 'assert_equal' (line 278)
        assert_equal_239215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 278)
        assert_equal_call_result_239220 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), assert_equal_239215, *[message_239217, str_239218], **kwargs_239219)
        
        
        # ################# End of 'test_maxiter_stops_solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_maxiter_stops_solve' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_239221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239221)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_maxiter_stops_solve'
        return stypy_return_type_239221


    @norecursion
    def test_maxfun_stops_solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_maxfun_stops_solve'
        module_type_store = module_type_store.open_function_context('test_maxfun_stops_solve', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_maxfun_stops_solve')
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_maxfun_stops_solve.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_maxfun_stops_solve', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_maxfun_stops_solve', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_maxfun_stops_solve(...)' code ##################

        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to DifferentialEvolutionSolver(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'rosen' (line 284)
        rosen_239223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 45), 'rosen', False)
        # Getting the type of 'self' (line 284)
        self_239224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 52), 'self', False)
        # Obtaining the member 'bounds' of a type (line 284)
        bounds_239225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 52), self_239224, 'bounds')
        # Processing the call keyword arguments (line 284)
        int_239226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 72), 'int')
        keyword_239227 = int_239226
        # Getting the type of 'False' (line 285)
        False_239228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 52), 'False', False)
        keyword_239229 = False_239228
        kwargs_239230 = {'polish': keyword_239229, 'maxfun': keyword_239227}
        # Getting the type of 'DifferentialEvolutionSolver' (line 284)
        DifferentialEvolutionSolver_239222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 284)
        DifferentialEvolutionSolver_call_result_239231 = invoke(stypy.reporting.localization.Localization(__file__, 284, 17), DifferentialEvolutionSolver_239222, *[rosen_239223, bounds_239225], **kwargs_239230)
        
        # Assigning a type to the variable 'solver' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'solver', DifferentialEvolutionSolver_call_result_239231)
        
        # Assigning a Call to a Name (line 286):
        
        # Assigning a Call to a Name (line 286):
        
        # Call to solve(...): (line 286)
        # Processing the call keyword arguments (line 286)
        kwargs_239234 = {}
        # Getting the type of 'solver' (line 286)
        solver_239232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 17), 'solver', False)
        # Obtaining the member 'solve' of a type (line 286)
        solve_239233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 17), solver_239232, 'solve')
        # Calling solve(args, kwargs) (line 286)
        solve_call_result_239235 = invoke(stypy.reporting.localization.Localization(__file__, 286, 17), solve_239233, *[], **kwargs_239234)
        
        # Assigning a type to the variable 'result' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'result', solve_call_result_239235)
        
        # Call to assert_equal(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'result' (line 288)
        result_239237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'result', False)
        # Obtaining the member 'nfev' of a type (line 288)
        nfev_239238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 21), result_239237, 'nfev')
        int_239239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 34), 'int')
        # Processing the call keyword arguments (line 288)
        kwargs_239240 = {}
        # Getting the type of 'assert_equal' (line 288)
        assert_equal_239236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 288)
        assert_equal_call_result_239241 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), assert_equal_239236, *[nfev_239238, int_239239], **kwargs_239240)
        
        
        # Call to assert_equal(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'result' (line 289)
        result_239243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 21), 'result', False)
        # Obtaining the member 'success' of a type (line 289)
        success_239244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 21), result_239243, 'success')
        # Getting the type of 'False' (line 289)
        False_239245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 37), 'False', False)
        # Processing the call keyword arguments (line 289)
        kwargs_239246 = {}
        # Getting the type of 'assert_equal' (line 289)
        assert_equal_239242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 289)
        assert_equal_call_result_239247 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), assert_equal_239242, *[success_239244, False_239245], **kwargs_239246)
        
        
        # Call to assert_equal(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'result' (line 290)
        result_239249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 21), 'result', False)
        # Obtaining the member 'message' of a type (line 290)
        message_239250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 21), result_239249, 'message')
        str_239251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 21), 'str', 'Maximum number of function evaluations has been exceeded.')
        # Processing the call keyword arguments (line 290)
        kwargs_239252 = {}
        # Getting the type of 'assert_equal' (line 290)
        assert_equal_239248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 290)
        assert_equal_call_result_239253 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), assert_equal_239248, *[message_239250, str_239251], **kwargs_239252)
        
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to DifferentialEvolutionSolver(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'rosen' (line 299)
        rosen_239255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 45), 'rosen', False)
        # Getting the type of 'self' (line 300)
        self_239256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 300)
        bounds_239257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 45), self_239256, 'bounds')
        # Processing the call keyword arguments (line 299)
        int_239258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 53), 'int')
        keyword_239259 = int_239258
        # Getting the type of 'False' (line 302)
        False_239260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 52), 'False', False)
        keyword_239261 = False_239260
        int_239262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 52), 'int')
        keyword_239263 = int_239262
        kwargs_239264 = {'polish': keyword_239261, 'popsize': keyword_239259, 'maxfun': keyword_239263}
        # Getting the type of 'DifferentialEvolutionSolver' (line 299)
        DifferentialEvolutionSolver_239254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 299)
        DifferentialEvolutionSolver_call_result_239265 = invoke(stypy.reporting.localization.Localization(__file__, 299, 17), DifferentialEvolutionSolver_239254, *[rosen_239255, bounds_239257], **kwargs_239264)
        
        # Assigning a type to the variable 'solver' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'solver', DifferentialEvolutionSolver_call_result_239265)
        
        # Assigning a Call to a Name (line 304):
        
        # Assigning a Call to a Name (line 304):
        
        # Call to solve(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_239268 = {}
        # Getting the type of 'solver' (line 304)
        solver_239266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 17), 'solver', False)
        # Obtaining the member 'solve' of a type (line 304)
        solve_239267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 17), solver_239266, 'solve')
        # Calling solve(args, kwargs) (line 304)
        solve_call_result_239269 = invoke(stypy.reporting.localization.Localization(__file__, 304, 17), solve_239267, *[], **kwargs_239268)
        
        # Assigning a type to the variable 'result' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'result', solve_call_result_239269)
        
        # Call to assert_equal(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'result' (line 306)
        result_239271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'result', False)
        # Obtaining the member 'nfev' of a type (line 306)
        nfev_239272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 21), result_239271, 'nfev')
        int_239273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 34), 'int')
        # Processing the call keyword arguments (line 306)
        kwargs_239274 = {}
        # Getting the type of 'assert_equal' (line 306)
        assert_equal_239270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 306)
        assert_equal_call_result_239275 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), assert_equal_239270, *[nfev_239272, int_239273], **kwargs_239274)
        
        
        # Call to assert_equal(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'result' (line 307)
        result_239277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 21), 'result', False)
        # Obtaining the member 'success' of a type (line 307)
        success_239278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 21), result_239277, 'success')
        # Getting the type of 'False' (line 307)
        False_239279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 37), 'False', False)
        # Processing the call keyword arguments (line 307)
        kwargs_239280 = {}
        # Getting the type of 'assert_equal' (line 307)
        assert_equal_239276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 307)
        assert_equal_call_result_239281 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), assert_equal_239276, *[success_239278, False_239279], **kwargs_239280)
        
        
        # Call to assert_equal(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'result' (line 308)
        result_239283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 21), 'result', False)
        # Obtaining the member 'message' of a type (line 308)
        message_239284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 21), result_239283, 'message')
        str_239285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 25), 'str', 'Maximum number of function evaluations has been exceeded.')
        # Processing the call keyword arguments (line 308)
        kwargs_239286 = {}
        # Getting the type of 'assert_equal' (line 308)
        assert_equal_239282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 308)
        assert_equal_call_result_239287 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), assert_equal_239282, *[message_239284, str_239285], **kwargs_239286)
        
        
        # ################# End of 'test_maxfun_stops_solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_maxfun_stops_solve' in the type store
        # Getting the type of 'stypy_return_type' (line 281)
        stypy_return_type_239288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239288)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_maxfun_stops_solve'
        return stypy_return_type_239288


    @norecursion
    def test_quadratic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadratic'
        module_type_store = module_type_store.open_function_context('test_quadratic', 312, 4, False)
        # Assigning a type to the variable 'self' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_quadratic')
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_quadratic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_quadratic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadratic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadratic(...)' code ##################

        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to DifferentialEvolutionSolver(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'self' (line 314)
        self_239290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 45), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 314)
        quadratic_239291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 45), self_239290, 'quadratic')
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_239292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        
        # Obtaining an instance of the builtin type 'tuple' (line 315)
        tuple_239293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 315)
        # Adding element type (line 315)
        int_239294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 47), tuple_239293, int_239294)
        # Adding element type (line 315)
        int_239295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 47), tuple_239293, int_239295)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 45), list_239292, tuple_239293)
        
        # Processing the call keyword arguments (line 314)
        float_239296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 49), 'float')
        keyword_239297 = float_239296
        kwargs_239298 = {'tol': keyword_239297}
        # Getting the type of 'DifferentialEvolutionSolver' (line 314)
        DifferentialEvolutionSolver_239289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 314)
        DifferentialEvolutionSolver_call_result_239299 = invoke(stypy.reporting.localization.Localization(__file__, 314, 17), DifferentialEvolutionSolver_239289, *[quadratic_239291, list_239292], **kwargs_239298)
        
        # Assigning a type to the variable 'solver' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'solver', DifferentialEvolutionSolver_call_result_239299)
        
        # Call to solve(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_239302 = {}
        # Getting the type of 'solver' (line 317)
        solver_239300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'solver', False)
        # Obtaining the member 'solve' of a type (line 317)
        solve_239301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), solver_239300, 'solve')
        # Calling solve(args, kwargs) (line 317)
        solve_call_result_239303 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), solve_239301, *[], **kwargs_239302)
        
        
        # Call to assert_equal(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Call to argmin(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'solver' (line 318)
        solver_239307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 31), 'solver', False)
        # Obtaining the member 'population_energies' of a type (line 318)
        population_energies_239308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 31), solver_239307, 'population_energies')
        # Processing the call keyword arguments (line 318)
        kwargs_239309 = {}
        # Getting the type of 'np' (line 318)
        np_239305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'np', False)
        # Obtaining the member 'argmin' of a type (line 318)
        argmin_239306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 21), np_239305, 'argmin')
        # Calling argmin(args, kwargs) (line 318)
        argmin_call_result_239310 = invoke(stypy.reporting.localization.Localization(__file__, 318, 21), argmin_239306, *[population_energies_239308], **kwargs_239309)
        
        int_239311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 60), 'int')
        # Processing the call keyword arguments (line 318)
        kwargs_239312 = {}
        # Getting the type of 'assert_equal' (line 318)
        assert_equal_239304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 318)
        assert_equal_call_result_239313 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), assert_equal_239304, *[argmin_call_result_239310, int_239311], **kwargs_239312)
        
        
        # ################# End of 'test_quadratic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadratic' in the type store
        # Getting the type of 'stypy_return_type' (line 312)
        stypy_return_type_239314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239314)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadratic'
        return stypy_return_type_239314


    @norecursion
    def test_quadratic_from_diff_ev(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadratic_from_diff_ev'
        module_type_store = module_type_store.open_function_context('test_quadratic_from_diff_ev', 320, 4, False)
        # Assigning a type to the variable 'self' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev')
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_quadratic_from_diff_ev', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadratic_from_diff_ev', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadratic_from_diff_ev(...)' code ##################

        
        # Call to differential_evolution(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'self' (line 322)
        self_239316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 31), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 322)
        quadratic_239317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 31), self_239316, 'quadratic')
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_239318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'tuple' (line 323)
        tuple_239319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 323)
        # Adding element type (line 323)
        int_239320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 33), tuple_239319, int_239320)
        # Adding element type (line 323)
        int_239321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 33), tuple_239319, int_239321)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 31), list_239318, tuple_239319)
        
        # Processing the call keyword arguments (line 322)
        float_239322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 35), 'float')
        keyword_239323 = float_239322
        kwargs_239324 = {'tol': keyword_239323}
        # Getting the type of 'differential_evolution' (line 322)
        differential_evolution_239315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'differential_evolution', False)
        # Calling differential_evolution(args, kwargs) (line 322)
        differential_evolution_call_result_239325 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), differential_evolution_239315, *[quadratic_239317, list_239318], **kwargs_239324)
        
        
        # ################# End of 'test_quadratic_from_diff_ev(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadratic_from_diff_ev' in the type store
        # Getting the type of 'stypy_return_type' (line 320)
        stypy_return_type_239326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadratic_from_diff_ev'
        return stypy_return_type_239326


    @norecursion
    def test_seed_gives_repeatability(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_seed_gives_repeatability'
        module_type_store = module_type_store.open_function_context('test_seed_gives_repeatability', 326, 4, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_seed_gives_repeatability')
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_seed_gives_repeatability.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_seed_gives_repeatability', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_seed_gives_repeatability', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_seed_gives_repeatability(...)' code ##################

        
        # Assigning a Call to a Name (line 327):
        
        # Assigning a Call to a Name (line 327):
        
        # Call to differential_evolution(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'self' (line 327)
        self_239328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 40), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 327)
        quadratic_239329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 40), self_239328, 'quadratic')
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_239330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        # Adding element type (line 328)
        
        # Obtaining an instance of the builtin type 'tuple' (line 328)
        tuple_239331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 328)
        # Adding element type (line 328)
        int_239332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 42), tuple_239331, int_239332)
        # Adding element type (line 328)
        int_239333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 42), tuple_239331, int_239333)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 40), list_239330, tuple_239331)
        
        # Processing the call keyword arguments (line 327)
        # Getting the type of 'False' (line 329)
        False_239334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 47), 'False', False)
        keyword_239335 = False_239334
        int_239336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 45), 'int')
        keyword_239337 = int_239336
        float_239338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 44), 'float')
        keyword_239339 = float_239338
        kwargs_239340 = {'polish': keyword_239335, 'seed': keyword_239337, 'tol': keyword_239339}
        # Getting the type of 'differential_evolution' (line 327)
        differential_evolution_239327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 17), 'differential_evolution', False)
        # Calling differential_evolution(args, kwargs) (line 327)
        differential_evolution_call_result_239341 = invoke(stypy.reporting.localization.Localization(__file__, 327, 17), differential_evolution_239327, *[quadratic_239329, list_239330], **kwargs_239340)
        
        # Assigning a type to the variable 'result' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'result', differential_evolution_call_result_239341)
        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to differential_evolution(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'self' (line 332)
        self_239343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 41), 'self', False)
        # Obtaining the member 'quadratic' of a type (line 332)
        quadratic_239344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 41), self_239343, 'quadratic')
        
        # Obtaining an instance of the builtin type 'list' (line 333)
        list_239345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 333)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_239346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        int_239347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 42), tuple_239346, int_239347)
        # Adding element type (line 333)
        int_239348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 42), tuple_239346, int_239348)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 40), list_239345, tuple_239346)
        
        # Processing the call keyword arguments (line 332)
        # Getting the type of 'False' (line 334)
        False_239349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 47), 'False', False)
        keyword_239350 = False_239349
        int_239351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 45), 'int')
        keyword_239352 = int_239351
        float_239353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 44), 'float')
        keyword_239354 = float_239353
        kwargs_239355 = {'polish': keyword_239350, 'seed': keyword_239352, 'tol': keyword_239354}
        # Getting the type of 'differential_evolution' (line 332)
        differential_evolution_239342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'differential_evolution', False)
        # Calling differential_evolution(args, kwargs) (line 332)
        differential_evolution_call_result_239356 = invoke(stypy.reporting.localization.Localization(__file__, 332, 18), differential_evolution_239342, *[quadratic_239344, list_239345], **kwargs_239355)
        
        # Assigning a type to the variable 'result2' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'result2', differential_evolution_call_result_239356)
        
        # Call to assert_equal(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'result' (line 337)
        result_239358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'result', False)
        # Obtaining the member 'x' of a type (line 337)
        x_239359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 21), result_239358, 'x')
        # Getting the type of 'result2' (line 337)
        result2_239360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'result2', False)
        # Obtaining the member 'x' of a type (line 337)
        x_239361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 31), result2_239360, 'x')
        # Processing the call keyword arguments (line 337)
        kwargs_239362 = {}
        # Getting the type of 'assert_equal' (line 337)
        assert_equal_239357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 337)
        assert_equal_call_result_239363 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), assert_equal_239357, *[x_239359, x_239361], **kwargs_239362)
        
        
        # ################# End of 'test_seed_gives_repeatability(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_seed_gives_repeatability' in the type store
        # Getting the type of 'stypy_return_type' (line 326)
        stypy_return_type_239364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_seed_gives_repeatability'
        return stypy_return_type_239364


    @norecursion
    def test_exp_runs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_exp_runs'
        module_type_store = module_type_store.open_function_context('test_exp_runs', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_exp_runs')
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_exp_runs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_exp_runs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_exp_runs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_exp_runs(...)' code ##################

        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Call to DifferentialEvolutionSolver(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'rosen' (line 341)
        rosen_239366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 45), 'rosen', False)
        # Getting the type of 'self' (line 342)
        self_239367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 45), 'self', False)
        # Obtaining the member 'bounds' of a type (line 342)
        bounds_239368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 45), self_239367, 'bounds')
        # Processing the call keyword arguments (line 341)
        str_239369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 54), 'str', 'best1exp')
        keyword_239370 = str_239369
        int_239371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 53), 'int')
        keyword_239372 = int_239371
        kwargs_239373 = {'maxiter': keyword_239372, 'strategy': keyword_239370}
        # Getting the type of 'DifferentialEvolutionSolver' (line 341)
        DifferentialEvolutionSolver_239365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 341)
        DifferentialEvolutionSolver_call_result_239374 = invoke(stypy.reporting.localization.Localization(__file__, 341, 17), DifferentialEvolutionSolver_239365, *[rosen_239366, bounds_239368], **kwargs_239373)
        
        # Assigning a type to the variable 'solver' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'solver', DifferentialEvolutionSolver_call_result_239374)
        
        # Call to solve(...): (line 346)
        # Processing the call keyword arguments (line 346)
        kwargs_239377 = {}
        # Getting the type of 'solver' (line 346)
        solver_239375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'solver', False)
        # Obtaining the member 'solve' of a type (line 346)
        solve_239376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), solver_239375, 'solve')
        # Calling solve(args, kwargs) (line 346)
        solve_call_result_239378 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), solve_239376, *[], **kwargs_239377)
        
        
        # ################# End of 'test_exp_runs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_exp_runs' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_239379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239379)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_exp_runs'
        return stypy_return_type_239379


    @norecursion
    def test_gh_4511_regression(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gh_4511_regression'
        module_type_store = module_type_store.open_function_context('test_gh_4511_regression', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_gh_4511_regression')
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_gh_4511_regression.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_gh_4511_regression', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gh_4511_regression', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gh_4511_regression(...)' code ##################

        
        # Assigning a List to a Name (line 353):
        
        # Assigning a List to a Name (line 353):
        
        # Obtaining an instance of the builtin type 'list' (line 353)
        list_239380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 353)
        # Adding element type (line 353)
        
        # Obtaining an instance of the builtin type 'tuple' (line 353)
        tuple_239381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 353)
        # Adding element type (line 353)
        int_239382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 19), tuple_239381, int_239382)
        # Adding element type (line 353)
        int_239383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 19), tuple_239381, int_239383)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 17), list_239380, tuple_239381)
        # Adding element type (line 353)
        
        # Obtaining an instance of the builtin type 'tuple' (line 353)
        tuple_239384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 353)
        # Adding element type (line 353)
        int_239385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 28), tuple_239384, int_239385)
        # Adding element type (line 353)
        int_239386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 28), tuple_239384, int_239386)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 17), list_239380, tuple_239384)
        
        # Assigning a type to the variable 'bounds' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'bounds', list_239380)
        
        # Assigning a Call to a Name (line 354):
        
        # Assigning a Call to a Name (line 354):
        
        # Call to differential_evolution(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'rosen' (line 354)
        rosen_239388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 40), 'rosen', False)
        # Getting the type of 'bounds' (line 354)
        bounds_239389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 47), 'bounds', False)
        # Processing the call keyword arguments (line 354)
        int_239390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 63), 'int')
        keyword_239391 = int_239390
        int_239392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 77), 'int')
        keyword_239393 = int_239392
        kwargs_239394 = {'popsize': keyword_239391, 'maxiter': keyword_239393}
        # Getting the type of 'differential_evolution' (line 354)
        differential_evolution_239387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 17), 'differential_evolution', False)
        # Calling differential_evolution(args, kwargs) (line 354)
        differential_evolution_call_result_239395 = invoke(stypy.reporting.localization.Localization(__file__, 354, 17), differential_evolution_239387, *[rosen_239388, bounds_239389], **kwargs_239394)
        
        # Assigning a type to the variable 'result' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'result', differential_evolution_call_result_239395)
        
        # ################# End of 'test_gh_4511_regression(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gh_4511_regression' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_239396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239396)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gh_4511_regression'
        return stypy_return_type_239396


    @norecursion
    def test_calculate_population_energies(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_calculate_population_energies'
        module_type_store = module_type_store.open_function_context('test_calculate_population_energies', 356, 4, False)
        # Assigning a type to the variable 'self' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_calculate_population_energies')
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_calculate_population_energies.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_calculate_population_energies', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_calculate_population_energies', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_calculate_population_energies(...)' code ##################

        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to DifferentialEvolutionSolver(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'rosen' (line 358)
        rosen_239398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 45), 'rosen', False)
        # Getting the type of 'self' (line 358)
        self_239399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 52), 'self', False)
        # Obtaining the member 'bounds' of a type (line 358)
        bounds_239400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 52), self_239399, 'bounds')
        # Processing the call keyword arguments (line 358)
        int_239401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 73), 'int')
        keyword_239402 = int_239401
        kwargs_239403 = {'popsize': keyword_239402}
        # Getting the type of 'DifferentialEvolutionSolver' (line 358)
        DifferentialEvolutionSolver_239397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 358)
        DifferentialEvolutionSolver_call_result_239404 = invoke(stypy.reporting.localization.Localization(__file__, 358, 17), DifferentialEvolutionSolver_239397, *[rosen_239398, bounds_239400], **kwargs_239403)
        
        # Assigning a type to the variable 'solver' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'solver', DifferentialEvolutionSolver_call_result_239404)
        
        # Call to _calculate_population_energies(...): (line 359)
        # Processing the call keyword arguments (line 359)
        kwargs_239407 = {}
        # Getting the type of 'solver' (line 359)
        solver_239405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'solver', False)
        # Obtaining the member '_calculate_population_energies' of a type (line 359)
        _calculate_population_energies_239406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), solver_239405, '_calculate_population_energies')
        # Calling _calculate_population_energies(args, kwargs) (line 359)
        _calculate_population_energies_call_result_239408 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), _calculate_population_energies_239406, *[], **kwargs_239407)
        
        
        # Call to assert_equal(...): (line 361)
        # Processing the call arguments (line 361)
        
        # Call to argmin(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'solver' (line 361)
        solver_239412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 31), 'solver', False)
        # Obtaining the member 'population_energies' of a type (line 361)
        population_energies_239413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 31), solver_239412, 'population_energies')
        # Processing the call keyword arguments (line 361)
        kwargs_239414 = {}
        # Getting the type of 'np' (line 361)
        np_239410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'np', False)
        # Obtaining the member 'argmin' of a type (line 361)
        argmin_239411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 21), np_239410, 'argmin')
        # Calling argmin(args, kwargs) (line 361)
        argmin_call_result_239415 = invoke(stypy.reporting.localization.Localization(__file__, 361, 21), argmin_239411, *[population_energies_239413], **kwargs_239414)
        
        int_239416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 60), 'int')
        # Processing the call keyword arguments (line 361)
        kwargs_239417 = {}
        # Getting the type of 'assert_equal' (line 361)
        assert_equal_239409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 361)
        assert_equal_call_result_239418 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), assert_equal_239409, *[argmin_call_result_239415, int_239416], **kwargs_239417)
        
        
        # Call to assert_equal(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'solver' (line 364)
        solver_239420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 21), 'solver', False)
        # Obtaining the member '_nfev' of a type (line 364)
        _nfev_239421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 21), solver_239420, '_nfev')
        int_239422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 35), 'int')
        # Processing the call keyword arguments (line 364)
        kwargs_239423 = {}
        # Getting the type of 'assert_equal' (line 364)
        assert_equal_239419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 364)
        assert_equal_call_result_239424 = invoke(stypy.reporting.localization.Localization(__file__, 364, 8), assert_equal_239419, *[_nfev_239421, int_239422], **kwargs_239423)
        
        
        # ################# End of 'test_calculate_population_energies(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_calculate_population_energies' in the type store
        # Getting the type of 'stypy_return_type' (line 356)
        stypy_return_type_239425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239425)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_calculate_population_energies'
        return stypy_return_type_239425


    @norecursion
    def test_iteration(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_iteration'
        module_type_store = module_type_store.open_function_context('test_iteration', 366, 4, False)
        # Assigning a type to the variable 'self' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_iteration')
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_iteration.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_iteration', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_iteration', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_iteration(...)' code ##################

        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to DifferentialEvolutionSolver(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'rosen' (line 369)
        rosen_239427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 45), 'rosen', False)
        # Getting the type of 'self' (line 369)
        self_239428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 52), 'self', False)
        # Obtaining the member 'bounds' of a type (line 369)
        bounds_239429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 52), self_239428, 'bounds')
        # Processing the call keyword arguments (line 369)
        int_239430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 73), 'int')
        keyword_239431 = int_239430
        int_239432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 52), 'int')
        keyword_239433 = int_239432
        kwargs_239434 = {'popsize': keyword_239431, 'maxfun': keyword_239433}
        # Getting the type of 'DifferentialEvolutionSolver' (line 369)
        DifferentialEvolutionSolver_239426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 369)
        DifferentialEvolutionSolver_call_result_239435 = invoke(stypy.reporting.localization.Localization(__file__, 369, 17), DifferentialEvolutionSolver_239426, *[rosen_239427, bounds_239429], **kwargs_239434)
        
        # Assigning a type to the variable 'solver' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'solver', DifferentialEvolutionSolver_call_result_239435)
        
        # Assigning a Call to a Tuple (line 371):
        
        # Assigning a Subscript to a Name (line 371):
        
        # Obtaining the type of the subscript
        int_239436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 8), 'int')
        
        # Call to next(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'solver' (line 371)
        solver_239438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 22), 'solver', False)
        # Processing the call keyword arguments (line 371)
        kwargs_239439 = {}
        # Getting the type of 'next' (line 371)
        next_239437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'next', False)
        # Calling next(args, kwargs) (line 371)
        next_call_result_239440 = invoke(stypy.reporting.localization.Localization(__file__, 371, 17), next_239437, *[solver_239438], **kwargs_239439)
        
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___239441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), next_call_result_239440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_239442 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), getitem___239441, int_239436)
        
        # Assigning a type to the variable 'tuple_var_assignment_238297' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_238297', subscript_call_result_239442)
        
        # Assigning a Subscript to a Name (line 371):
        
        # Obtaining the type of the subscript
        int_239443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 8), 'int')
        
        # Call to next(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'solver' (line 371)
        solver_239445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 22), 'solver', False)
        # Processing the call keyword arguments (line 371)
        kwargs_239446 = {}
        # Getting the type of 'next' (line 371)
        next_239444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'next', False)
        # Calling next(args, kwargs) (line 371)
        next_call_result_239447 = invoke(stypy.reporting.localization.Localization(__file__, 371, 17), next_239444, *[solver_239445], **kwargs_239446)
        
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___239448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), next_call_result_239447, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_239449 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), getitem___239448, int_239443)
        
        # Assigning a type to the variable 'tuple_var_assignment_238298' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_238298', subscript_call_result_239449)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'tuple_var_assignment_238297' (line 371)
        tuple_var_assignment_238297_239450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_238297')
        # Assigning a type to the variable 'x' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'x', tuple_var_assignment_238297_239450)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'tuple_var_assignment_238298' (line 371)
        tuple_var_assignment_238298_239451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_238298')
        # Assigning a type to the variable 'fun' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'fun', tuple_var_assignment_238298_239451)
        
        # Call to assert_equal(...): (line 372)
        # Processing the call arguments (line 372)
        
        # Call to size(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'x' (line 372)
        x_239455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 29), 'x', False)
        int_239456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 32), 'int')
        # Processing the call keyword arguments (line 372)
        kwargs_239457 = {}
        # Getting the type of 'np' (line 372)
        np_239453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 21), 'np', False)
        # Obtaining the member 'size' of a type (line 372)
        size_239454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 21), np_239453, 'size')
        # Calling size(args, kwargs) (line 372)
        size_call_result_239458 = invoke(stypy.reporting.localization.Localization(__file__, 372, 21), size_239454, *[x_239455, int_239456], **kwargs_239457)
        
        int_239459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 36), 'int')
        # Processing the call keyword arguments (line 372)
        kwargs_239460 = {}
        # Getting the type of 'assert_equal' (line 372)
        assert_equal_239452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 372)
        assert_equal_call_result_239461 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), assert_equal_239452, *[size_call_result_239458, int_239459], **kwargs_239460)
        
        
        # Call to assert_equal(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'solver' (line 376)
        solver_239463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 21), 'solver', False)
        # Obtaining the member '_nfev' of a type (line 376)
        _nfev_239464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 21), solver_239463, '_nfev')
        int_239465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 35), 'int')
        # Processing the call keyword arguments (line 376)
        kwargs_239466 = {}
        # Getting the type of 'assert_equal' (line 376)
        assert_equal_239462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 376)
        assert_equal_call_result_239467 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), assert_equal_239462, *[_nfev_239464, int_239465], **kwargs_239466)
        
        
        # Call to assert_raises(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'StopIteration' (line 379)
        StopIteration_239469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 22), 'StopIteration', False)
        # Getting the type of 'next' (line 379)
        next_239470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 37), 'next', False)
        # Getting the type of 'solver' (line 379)
        solver_239471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 43), 'solver', False)
        # Processing the call keyword arguments (line 379)
        kwargs_239472 = {}
        # Getting the type of 'assert_raises' (line 379)
        assert_raises_239468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 379)
        assert_raises_call_result_239473 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), assert_raises_239468, *[StopIteration_239469, next_239470, solver_239471], **kwargs_239472)
        
        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Call to DifferentialEvolutionSolver(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'rosen' (line 382)
        rosen_239475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 45), 'rosen', False)
        # Getting the type of 'self' (line 382)
        self_239476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 52), 'self', False)
        # Obtaining the member 'bounds' of a type (line 382)
        bounds_239477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 52), self_239476, 'bounds')
        # Processing the call keyword arguments (line 382)
        kwargs_239478 = {}
        # Getting the type of 'DifferentialEvolutionSolver' (line 382)
        DifferentialEvolutionSolver_239474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 382)
        DifferentialEvolutionSolver_call_result_239479 = invoke(stypy.reporting.localization.Localization(__file__, 382, 17), DifferentialEvolutionSolver_239474, *[rosen_239475, bounds_239477], **kwargs_239478)
        
        # Assigning a type to the variable 'solver' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'solver', DifferentialEvolutionSolver_call_result_239479)
        
        
        # Call to enumerate(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'solver' (line 383)
        solver_239481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 33), 'solver', False)
        # Processing the call keyword arguments (line 383)
        kwargs_239482 = {}
        # Getting the type of 'enumerate' (line 383)
        enumerate_239480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 23), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 383)
        enumerate_call_result_239483 = invoke(stypy.reporting.localization.Localization(__file__, 383, 23), enumerate_239480, *[solver_239481], **kwargs_239482)
        
        # Testing the type of a for loop iterable (line 383)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 383, 8), enumerate_call_result_239483)
        # Getting the type of the for loop variable (line 383)
        for_loop_var_239484 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 383, 8), enumerate_call_result_239483)
        # Assigning a type to the variable 'i' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), for_loop_var_239484))
        # Assigning a type to the variable 'soln' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'soln', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), for_loop_var_239484))
        # SSA begins for a for statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Tuple (line 384):
        
        # Assigning a Subscript to a Name (line 384):
        
        # Obtaining the type of the subscript
        int_239485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 12), 'int')
        # Getting the type of 'soln' (line 384)
        soln_239486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 37), 'soln')
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___239487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 12), soln_239486, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_239488 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), getitem___239487, int_239485)
        
        # Assigning a type to the variable 'tuple_var_assignment_238299' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'tuple_var_assignment_238299', subscript_call_result_239488)
        
        # Assigning a Subscript to a Name (line 384):
        
        # Obtaining the type of the subscript
        int_239489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 12), 'int')
        # Getting the type of 'soln' (line 384)
        soln_239490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 37), 'soln')
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___239491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 12), soln_239490, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_239492 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), getitem___239491, int_239489)
        
        # Assigning a type to the variable 'tuple_var_assignment_238300' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'tuple_var_assignment_238300', subscript_call_result_239492)
        
        # Assigning a Name to a Name (line 384):
        # Getting the type of 'tuple_var_assignment_238299' (line 384)
        tuple_var_assignment_238299_239493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'tuple_var_assignment_238299')
        # Assigning a type to the variable 'x_current' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'x_current', tuple_var_assignment_238299_239493)
        
        # Assigning a Name to a Name (line 384):
        # Getting the type of 'tuple_var_assignment_238300' (line 384)
        tuple_var_assignment_238300_239494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'tuple_var_assignment_238300')
        # Assigning a type to the variable 'fun_current' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 23), 'fun_current', tuple_var_assignment_238300_239494)
        
        
        # Getting the type of 'i' (line 386)
        i_239495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'i')
        int_239496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 20), 'int')
        # Applying the binary operator '==' (line 386)
        result_eq_239497 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 15), '==', i_239495, int_239496)
        
        # Testing the type of an if condition (line 386)
        if_condition_239498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 12), result_eq_239497)
        # Assigning a type to the variable 'if_condition_239498' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'if_condition_239498', if_condition_239498)
        # SSA begins for if statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 386)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_almost_equal(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'fun_current' (line 389)
        fun_current_239500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 28), 'fun_current', False)
        int_239501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 41), 'int')
        # Processing the call keyword arguments (line 389)
        kwargs_239502 = {}
        # Getting the type of 'assert_almost_equal' (line 389)
        assert_almost_equal_239499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 389)
        assert_almost_equal_call_result_239503 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), assert_almost_equal_239499, *[fun_current_239500, int_239501], **kwargs_239502)
        
        
        # ################# End of 'test_iteration(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_iteration' in the type store
        # Getting the type of 'stypy_return_type' (line 366)
        stypy_return_type_239504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239504)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_iteration'
        return stypy_return_type_239504


    @norecursion
    def test_convergence(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_convergence'
        module_type_store = module_type_store.open_function_context('test_convergence', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_convergence')
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_convergence.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_convergence', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_convergence', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_convergence(...)' code ##################

        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to DifferentialEvolutionSolver(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'rosen' (line 392)
        rosen_239506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 45), 'rosen', False)
        # Getting the type of 'self' (line 392)
        self_239507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 52), 'self', False)
        # Obtaining the member 'bounds' of a type (line 392)
        bounds_239508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 52), self_239507, 'bounds')
        # Processing the call keyword arguments (line 392)
        float_239509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 69), 'float')
        keyword_239510 = float_239509
        # Getting the type of 'False' (line 393)
        False_239511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 52), 'False', False)
        keyword_239512 = False_239511
        kwargs_239513 = {'polish': keyword_239512, 'tol': keyword_239510}
        # Getting the type of 'DifferentialEvolutionSolver' (line 392)
        DifferentialEvolutionSolver_239505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 392)
        DifferentialEvolutionSolver_call_result_239514 = invoke(stypy.reporting.localization.Localization(__file__, 392, 17), DifferentialEvolutionSolver_239505, *[rosen_239506, bounds_239508], **kwargs_239513)
        
        # Assigning a type to the variable 'solver' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'solver', DifferentialEvolutionSolver_call_result_239514)
        
        # Call to solve(...): (line 394)
        # Processing the call keyword arguments (line 394)
        kwargs_239517 = {}
        # Getting the type of 'solver' (line 394)
        solver_239515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'solver', False)
        # Obtaining the member 'solve' of a type (line 394)
        solve_239516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), solver_239515, 'solve')
        # Calling solve(args, kwargs) (line 394)
        solve_call_result_239518 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), solve_239516, *[], **kwargs_239517)
        
        
        # Call to assert_(...): (line 395)
        # Processing the call arguments (line 395)
        
        # Getting the type of 'solver' (line 395)
        solver_239520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'solver', False)
        # Obtaining the member 'convergence' of a type (line 395)
        convergence_239521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 16), solver_239520, 'convergence')
        float_239522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 37), 'float')
        # Applying the binary operator '<' (line 395)
        result_lt_239523 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 16), '<', convergence_239521, float_239522)
        
        # Processing the call keyword arguments (line 395)
        kwargs_239524 = {}
        # Getting the type of 'assert_' (line 395)
        assert__239519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 395)
        assert__call_result_239525 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), assert__239519, *[result_lt_239523], **kwargs_239524)
        
        
        # ################# End of 'test_convergence(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_convergence' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_239526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_convergence'
        return stypy_return_type_239526


    @norecursion
    def test_maxiter_none_GH5731(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_maxiter_none_GH5731'
        module_type_store = module_type_store.open_function_context('test_maxiter_none_GH5731', 397, 4, False)
        # Assigning a type to the variable 'self' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_maxiter_none_GH5731')
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_maxiter_none_GH5731.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_maxiter_none_GH5731', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_maxiter_none_GH5731', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_maxiter_none_GH5731(...)' code ##################

        
        # Assigning a Call to a Name (line 402):
        
        # Assigning a Call to a Name (line 402):
        
        # Call to DifferentialEvolutionSolver(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'rosen' (line 402)
        rosen_239528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 45), 'rosen', False)
        # Getting the type of 'self' (line 402)
        self_239529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 52), 'self', False)
        # Obtaining the member 'bounds' of a type (line 402)
        bounds_239530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 52), self_239529, 'bounds')
        # Processing the call keyword arguments (line 402)
        # Getting the type of 'None' (line 402)
        None_239531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 73), 'None', False)
        keyword_239532 = None_239531
        # Getting the type of 'None' (line 403)
        None_239533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 52), 'None', False)
        keyword_239534 = None_239533
        kwargs_239535 = {'maxfun': keyword_239534, 'maxiter': keyword_239532}
        # Getting the type of 'DifferentialEvolutionSolver' (line 402)
        DifferentialEvolutionSolver_239527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 402)
        DifferentialEvolutionSolver_call_result_239536 = invoke(stypy.reporting.localization.Localization(__file__, 402, 17), DifferentialEvolutionSolver_239527, *[rosen_239528, bounds_239530], **kwargs_239535)
        
        # Assigning a type to the variable 'solver' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'solver', DifferentialEvolutionSolver_call_result_239536)
        
        # Call to solve(...): (line 404)
        # Processing the call keyword arguments (line 404)
        kwargs_239539 = {}
        # Getting the type of 'solver' (line 404)
        solver_239537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'solver', False)
        # Obtaining the member 'solve' of a type (line 404)
        solve_239538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), solver_239537, 'solve')
        # Calling solve(args, kwargs) (line 404)
        solve_call_result_239540 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), solve_239538, *[], **kwargs_239539)
        
        
        # ################# End of 'test_maxiter_none_GH5731(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_maxiter_none_GH5731' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_239541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239541)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_maxiter_none_GH5731'
        return stypy_return_type_239541


    @norecursion
    def test_population_initiation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_population_initiation'
        module_type_store = module_type_store.open_function_context('test_population_initiation', 406, 4, False)
        # Assigning a type to the variable 'self' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_localization', localization)
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_function_name', 'TestDifferentialEvolutionSolver.test_population_initiation')
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_param_names_list', [])
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDifferentialEvolutionSolver.test_population_initiation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.test_population_initiation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_population_initiation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_population_initiation(...)' code ##################

        
        # Call to assert_raises(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'ValueError' (line 411)
        ValueError_239543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 22), 'ValueError', False)
        # Getting the type of 'DifferentialEvolutionSolver' (line 412)
        DifferentialEvolutionSolver_239544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 22), 'DifferentialEvolutionSolver', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 413)
        tuple_239545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 413)
        # Adding element type (line 413)
        # Getting the type of 'rosen' (line 413)
        rosen_239546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 24), 'rosen', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 24), tuple_239545, rosen_239546)
        # Adding element type (line 413)
        # Getting the type of 'self' (line 413)
        self_239547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 31), 'self', False)
        # Obtaining the member 'bounds' of a type (line 413)
        bounds_239548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 31), self_239547, 'bounds')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 24), tuple_239545, bounds_239548)
        
        # Processing the call keyword arguments (line 411)
        
        # Obtaining an instance of the builtin type 'dict' (line 414)
        dict_239549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 414)
        # Adding element type (key, value) (line 414)
        str_239550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 25), 'str', 'init')
        str_239551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 33), 'str', 'rubbish')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 24), dict_239549, (str_239550, str_239551))
        
        kwargs_239552 = {'dict_239549': dict_239549}
        # Getting the type of 'assert_raises' (line 411)
        assert_raises_239542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 411)
        assert_raises_call_result_239553 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), assert_raises_239542, *[ValueError_239543, DifferentialEvolutionSolver_239544, tuple_239545], **kwargs_239552)
        
        
        # Assigning a Call to a Name (line 416):
        
        # Assigning a Call to a Name (line 416):
        
        # Call to DifferentialEvolutionSolver(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'rosen' (line 416)
        rosen_239555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 45), 'rosen', False)
        # Getting the type of 'self' (line 416)
        self_239556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 52), 'self', False)
        # Obtaining the member 'bounds' of a type (line 416)
        bounds_239557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 52), self_239556, 'bounds')
        # Processing the call keyword arguments (line 416)
        kwargs_239558 = {}
        # Getting the type of 'DifferentialEvolutionSolver' (line 416)
        DifferentialEvolutionSolver_239554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 17), 'DifferentialEvolutionSolver', False)
        # Calling DifferentialEvolutionSolver(args, kwargs) (line 416)
        DifferentialEvolutionSolver_call_result_239559 = invoke(stypy.reporting.localization.Localization(__file__, 416, 17), DifferentialEvolutionSolver_239554, *[rosen_239555, bounds_239557], **kwargs_239558)
        
        # Assigning a type to the variable 'solver' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'solver', DifferentialEvolutionSolver_call_result_239559)
        
        # Call to init_population_random(...): (line 421)
        # Processing the call keyword arguments (line 421)
        kwargs_239562 = {}
        # Getting the type of 'solver' (line 421)
        solver_239560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'solver', False)
        # Obtaining the member 'init_population_random' of a type (line 421)
        init_population_random_239561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), solver_239560, 'init_population_random')
        # Calling init_population_random(args, kwargs) (line 421)
        init_population_random_call_result_239563 = invoke(stypy.reporting.localization.Localization(__file__, 421, 8), init_population_random_239561, *[], **kwargs_239562)
        
        
        # Call to assert_equal(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'solver' (line 422)
        solver_239565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 21), 'solver', False)
        # Obtaining the member '_nfev' of a type (line 422)
        _nfev_239566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 21), solver_239565, '_nfev')
        int_239567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 35), 'int')
        # Processing the call keyword arguments (line 422)
        kwargs_239568 = {}
        # Getting the type of 'assert_equal' (line 422)
        assert_equal_239564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 422)
        assert_equal_call_result_239569 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), assert_equal_239564, *[_nfev_239566, int_239567], **kwargs_239568)
        
        
        # Call to assert_(...): (line 423)
        # Processing the call arguments (line 423)
        
        # Call to all(...): (line 423)
        # Processing the call arguments (line 423)
        
        # Call to isinf(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'solver' (line 423)
        solver_239575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 32), 'solver', False)
        # Obtaining the member 'population_energies' of a type (line 423)
        population_energies_239576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 32), solver_239575, 'population_energies')
        # Processing the call keyword arguments (line 423)
        kwargs_239577 = {}
        # Getting the type of 'np' (line 423)
        np_239573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 23), 'np', False)
        # Obtaining the member 'isinf' of a type (line 423)
        isinf_239574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 23), np_239573, 'isinf')
        # Calling isinf(args, kwargs) (line 423)
        isinf_call_result_239578 = invoke(stypy.reporting.localization.Localization(__file__, 423, 23), isinf_239574, *[population_energies_239576], **kwargs_239577)
        
        # Processing the call keyword arguments (line 423)
        kwargs_239579 = {}
        # Getting the type of 'np' (line 423)
        np_239571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 423)
        all_239572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), np_239571, 'all')
        # Calling all(args, kwargs) (line 423)
        all_call_result_239580 = invoke(stypy.reporting.localization.Localization(__file__, 423, 16), all_239572, *[isinf_call_result_239578], **kwargs_239579)
        
        # Processing the call keyword arguments (line 423)
        kwargs_239581 = {}
        # Getting the type of 'assert_' (line 423)
        assert__239570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 423)
        assert__call_result_239582 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), assert__239570, *[all_call_result_239580], **kwargs_239581)
        
        
        # Call to init_population_lhs(...): (line 425)
        # Processing the call keyword arguments (line 425)
        kwargs_239585 = {}
        # Getting the type of 'solver' (line 425)
        solver_239583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'solver', False)
        # Obtaining the member 'init_population_lhs' of a type (line 425)
        init_population_lhs_239584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 8), solver_239583, 'init_population_lhs')
        # Calling init_population_lhs(args, kwargs) (line 425)
        init_population_lhs_call_result_239586 = invoke(stypy.reporting.localization.Localization(__file__, 425, 8), init_population_lhs_239584, *[], **kwargs_239585)
        
        
        # Call to assert_equal(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'solver' (line 426)
        solver_239588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 21), 'solver', False)
        # Obtaining the member '_nfev' of a type (line 426)
        _nfev_239589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 21), solver_239588, '_nfev')
        int_239590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 35), 'int')
        # Processing the call keyword arguments (line 426)
        kwargs_239591 = {}
        # Getting the type of 'assert_equal' (line 426)
        assert_equal_239587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 426)
        assert_equal_call_result_239592 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), assert_equal_239587, *[_nfev_239589, int_239590], **kwargs_239591)
        
        
        # Call to assert_(...): (line 427)
        # Processing the call arguments (line 427)
        
        # Call to all(...): (line 427)
        # Processing the call arguments (line 427)
        
        # Call to isinf(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'solver' (line 427)
        solver_239598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 32), 'solver', False)
        # Obtaining the member 'population_energies' of a type (line 427)
        population_energies_239599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 32), solver_239598, 'population_energies')
        # Processing the call keyword arguments (line 427)
        kwargs_239600 = {}
        # Getting the type of 'np' (line 427)
        np_239596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 23), 'np', False)
        # Obtaining the member 'isinf' of a type (line 427)
        isinf_239597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 23), np_239596, 'isinf')
        # Calling isinf(args, kwargs) (line 427)
        isinf_call_result_239601 = invoke(stypy.reporting.localization.Localization(__file__, 427, 23), isinf_239597, *[population_energies_239599], **kwargs_239600)
        
        # Processing the call keyword arguments (line 427)
        kwargs_239602 = {}
        # Getting the type of 'np' (line 427)
        np_239594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 427)
        all_239595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 16), np_239594, 'all')
        # Calling all(args, kwargs) (line 427)
        all_call_result_239603 = invoke(stypy.reporting.localization.Localization(__file__, 427, 16), all_239595, *[isinf_call_result_239601], **kwargs_239602)
        
        # Processing the call keyword arguments (line 427)
        kwargs_239604 = {}
        # Getting the type of 'assert_' (line 427)
        assert__239593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 427)
        assert__call_result_239605 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), assert__239593, *[all_call_result_239603], **kwargs_239604)
        
        
        # ################# End of 'test_population_initiation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_population_initiation' in the type store
        # Getting the type of 'stypy_return_type' (line 406)
        stypy_return_type_239606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_239606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_population_initiation'
        return stypy_return_type_239606


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDifferentialEvolutionSolver.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDifferentialEvolutionSolver' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TestDifferentialEvolutionSolver', TestDifferentialEvolutionSolver)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
