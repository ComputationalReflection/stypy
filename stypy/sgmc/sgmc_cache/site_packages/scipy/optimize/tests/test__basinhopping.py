
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit tests for the basin hopping global minimization algorithm.
3: '''
4: from __future__ import division, print_function, absolute_import
5: import copy
6: 
7: from numpy.testing import assert_almost_equal, assert_equal, assert_
8: from pytest import raises as assert_raises
9: import numpy as np
10: from numpy import cos, sin
11: 
12: from scipy.optimize import basinhopping, OptimizeResult
13: from scipy.optimize._basinhopping import (
14:     Storage, RandomDisplacement, Metropolis, AdaptiveStepsize)
15: 
16: 
17: def func1d(x):
18:     f = cos(14.5 * x - 0.3) + (x + 0.2) * x
19:     df = np.array(-14.5 * sin(14.5 * x - 0.3) + 2. * x + 0.2)
20:     return f, df
21: 
22: 
23: def func1d_nograd(x):
24:     f = cos(14.5 * x - 0.3) + (x + 0.2) * x
25:     df = np.array(-14.5 * sin(14.5 * x - 0.3) + 2. * x + 0.2)
26:     return f, df
27: 
28: 
29: def func2d_nograd(x):
30:     f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
31:     return f
32: 
33: 
34: def func2d(x):
35:     f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
36:     df = np.zeros(2)
37:     df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
38:     df[1] = 2. * x[1] + 0.2
39:     return f, df
40: 
41: def func2d_easyderiv(x):
42:     f = 2.0*x[0]**2 + 2.0*x[0]*x[1] + 2.0*x[1]**2 - 6.0*x[0]
43:     df = np.zeros(2)
44:     df[0] = 4.0*x[0] + 2.0*x[1] - 6.0
45:     df[1] = 2.0*x[0] + 4.0*x[1]
46: 
47:     return f, df
48: 
49: class MyTakeStep1(RandomDisplacement):
50:     '''use a copy of displace, but have it set a special parameter to
51:     make sure it's actually being used.'''
52:     def __init__(self):
53:         self.been_called = False
54:         super(MyTakeStep1, self).__init__()
55: 
56:     def __call__(self, x):
57:         self.been_called = True
58:         return super(MyTakeStep1, self).__call__(x)
59: 
60: 
61: def myTakeStep2(x):
62:     '''redo RandomDisplacement in function form without the attribute stepsize
63:     to make sure still everything works ok
64:     '''
65:     s = 0.5
66:     x += np.random.uniform(-s, s, np.shape(x))
67:     return x
68: 
69: 
70: class MyAcceptTest(object):
71:     '''pass a custom accept test
72: 
73:     This does nothing but make sure it's being used and ensure all the
74:     possible return values are accepted
75:     '''
76:     def __init__(self):
77:         self.been_called = False
78:         self.ncalls = 0
79:         self.testres = [False, 'force accept', True, np.bool_(True),
80:                         np.bool_(False), [], {}, 0, 1]
81: 
82:     def __call__(self, **kwargs):
83:         self.been_called = True
84:         self.ncalls += 1
85:         if self.ncalls - 1 < len(self.testres):
86:             return self.testres[self.ncalls - 1]
87:         else:
88:             return True
89: 
90: 
91: class MyCallBack(object):
92:     '''pass a custom callback function
93: 
94:     This makes sure it's being used.  It also returns True after 10
95:     steps to ensure that it's stopping early.
96: 
97:     '''
98:     def __init__(self):
99:         self.been_called = False
100:         self.ncalls = 0
101: 
102:     def __call__(self, x, f, accepted):
103:         self.been_called = True
104:         self.ncalls += 1
105:         if self.ncalls == 10:
106:             return True
107: 
108: 
109: class TestBasinHopping(object):
110: 
111:     def setup_method(self):
112:         ''' Tests setup.
113: 
114:         Run tests based on the 1-D and 2-D functions described above.
115:         '''
116:         self.x0 = (1.0, [1.0, 1.0])
117:         self.sol = (-0.195, np.array([-0.195, -0.1]))
118:         
119:         self.tol = 3  # number of decimal places
120: 
121:         self.niter = 100
122:         self.disp = False
123: 
124:         # fix random seed
125:         np.random.seed(1234)
126: 
127:         self.kwargs = {"method": "L-BFGS-B", "jac": True}
128:         self.kwargs_nograd = {"method": "L-BFGS-B"}
129: 
130:     def test_TypeError(self):
131:         # test the TypeErrors are raised on bad input
132:         i = 1
133:         # if take_step is passed, it must be callable
134:         assert_raises(TypeError, basinhopping, func2d, self.x0[i],
135:                           take_step=1)
136:         # if accept_test is passed, it must be callable
137:         assert_raises(TypeError, basinhopping, func2d, self.x0[i],
138:                           accept_test=1)
139: 
140:     def test_1d_grad(self):
141:         # test 1d minimizations with gradient
142:         i = 0
143:         res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs,
144:                            niter=self.niter, disp=self.disp)
145:         assert_almost_equal(res.x, self.sol[i], self.tol)
146: 
147:     def test_2d(self):
148:         # test 2d minimizations with gradient
149:         i = 1
150:         res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
151:                            niter=self.niter, disp=self.disp)
152:         assert_almost_equal(res.x, self.sol[i], self.tol)
153:         assert_(res.nfev > 0)
154: 
155:     def test_njev(self):
156:         # test njev is returned correctly
157:         i = 1
158:         minimizer_kwargs = self.kwargs.copy()
159:         # L-BFGS-B doesn't use njev, but BFGS does
160:         minimizer_kwargs["method"] = "BFGS"
161:         res = basinhopping(func2d, self.x0[i],
162:                            minimizer_kwargs=minimizer_kwargs, niter=self.niter,
163:                            disp=self.disp)
164:         assert_(res.nfev > 0)
165:         assert_equal(res.nfev, res.njev)
166: 
167:     def test_jac(self):
168:         # test jacobian returned
169:         minimizer_kwargs = self.kwargs.copy()
170:         # BFGS returns a Jacobian
171:         minimizer_kwargs["method"] = "BFGS"
172: 
173:         res = basinhopping(func2d_easyderiv, [0.0, 0.0],
174:                            minimizer_kwargs=minimizer_kwargs, niter=self.niter,
175:                            disp=self.disp)
176: 
177:         assert_(hasattr(res.lowest_optimization_result, "jac"))
178: 
179:         #in this case, the jacobian is just [df/dx, df/dy]
180:         _, jacobian = func2d_easyderiv(res.x)
181:         assert_almost_equal(res.lowest_optimization_result.jac, jacobian, self.tol)
182: 
183:     def test_2d_nograd(self):
184:         # test 2d minimizations without gradient
185:         i = 1
186:         res = basinhopping(func2d_nograd, self.x0[i],
187:                            minimizer_kwargs=self.kwargs_nograd,
188:                            niter=self.niter, disp=self.disp)
189:         assert_almost_equal(res.x, self.sol[i], self.tol)
190: 
191:     def test_all_minimizers(self):
192:         # test 2d minimizations with gradient.  Nelder-Mead, Powell and COBYLA
193:         # don't accept jac=True, so aren't included here.
194:         i = 1
195:         methods = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP']
196:         minimizer_kwargs = copy.copy(self.kwargs)
197:         for method in methods:
198:             minimizer_kwargs["method"] = method
199:             res = basinhopping(func2d, self.x0[i],
200:                                minimizer_kwargs=minimizer_kwargs,
201:                                niter=self.niter, disp=self.disp)
202:             assert_almost_equal(res.x, self.sol[i], self.tol)
203: 
204:     def test_all_nograd_minimizers(self):
205:         # test 2d minimizations without gradient.  Newton-CG requires jac=True,
206:         # so not included here.
207:         i = 1
208:         methods = ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP',
209:                    'Nelder-Mead', 'Powell', 'COBYLA']
210:         minimizer_kwargs = copy.copy(self.kwargs_nograd)
211:         for method in methods:
212:             minimizer_kwargs["method"] = method
213:             res = basinhopping(func2d_nograd, self.x0[i],
214:                                minimizer_kwargs=minimizer_kwargs,
215:                                niter=self.niter, disp=self.disp)
216:             tol = self.tol
217:             if method == 'COBYLA':
218:                 tol = 2
219:             assert_almost_equal(res.x, self.sol[i], decimal=tol)
220: 
221:     def test_pass_takestep(self):
222:         # test that passing a custom takestep works
223:         # also test that the stepsize is being adjusted
224:         takestep = MyTakeStep1()
225:         initial_step_size = takestep.stepsize
226:         i = 1
227:         res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
228:                            niter=self.niter, disp=self.disp,
229:                            take_step=takestep)
230:         assert_almost_equal(res.x, self.sol[i], self.tol)
231:         assert_(takestep.been_called)
232:         # make sure that the built in adaptive step size has been used
233:         assert_(initial_step_size != takestep.stepsize)
234: 
235:     def test_pass_simple_takestep(self):
236:         # test that passing a custom takestep without attribute stepsize
237:         takestep = myTakeStep2
238:         i = 1
239:         res = basinhopping(func2d_nograd, self.x0[i],
240:                            minimizer_kwargs=self.kwargs_nograd,
241:                            niter=self.niter, disp=self.disp,
242:                            take_step=takestep)
243:         assert_almost_equal(res.x, self.sol[i], self.tol)
244: 
245:     def test_pass_accept_test(self):
246:         # test passing a custom accept test
247:         # makes sure it's being used and ensures all the possible return values
248:         # are accepted.
249:         accept_test = MyAcceptTest()
250:         i = 1
251:         # there's no point in running it more than a few steps.
252:         basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
253:                      niter=10, disp=self.disp, accept_test=accept_test)
254:         assert_(accept_test.been_called)
255: 
256:     def test_pass_callback(self):
257:         # test passing a custom callback function
258:         # This makes sure it's being used.  It also returns True after 10 steps
259:         # to ensure that it's stopping early.
260:         callback = MyCallBack()
261:         i = 1
262:         # there's no point in running it more than a few steps.
263:         res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
264:                            niter=30, disp=self.disp, callback=callback)
265:         assert_(callback.been_called)
266:         assert_("callback" in res.message[0])
267:         assert_equal(res.nit, 10)
268: 
269:     def test_minimizer_fail(self):
270:         # test if a minimizer fails
271:         i = 1
272:         self.kwargs["options"] = dict(maxiter=0)
273:         self.niter = 10
274:         res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
275:                            niter=self.niter, disp=self.disp)
276:         # the number of failed minimizations should be the number of
277:         # iterations + 1
278:         assert_equal(res.nit + 1, res.minimization_failures)
279: 
280:     def test_niter_zero(self):
281:         # gh5915, what happens if you call basinhopping with niter=0
282:         i = 0
283:         res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs,
284:                            niter=0, disp=self.disp)
285: 
286:     def test_seed_reproducibility(self):
287:         # seed should ensure reproducibility between runs
288:         minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}
289: 
290:         f_1 = []
291: 
292:         def callback(x, f, accepted):
293:             f_1.append(f)
294: 
295:         basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs,
296:                      niter=10, callback=callback, seed=10)
297: 
298:         f_2 = []
299: 
300:         def callback2(x, f, accepted):
301:             f_2.append(f)
302: 
303:         basinhopping(func2d, [1.0, 1.0], minimizer_kwargs=minimizer_kwargs,
304:                      niter=10, callback=callback2, seed=10)
305:         assert_equal(np.array(f_1), np.array(f_2))
306: 
307: 
308: class Test_Storage(object):
309:     def setup_method(self):
310:         self.x0 = np.array(1)
311:         self.f0 = 0
312: 
313:         minres = OptimizeResult()
314:         minres.x = self.x0
315:         minres.fun = self.f0
316: 
317:         self.storage = Storage(minres)
318: 
319:     def test_higher_f_rejected(self):
320:         new_minres = OptimizeResult()
321:         new_minres.x = self.x0 + 1
322:         new_minres.fun = self.f0 + 1
323: 
324:         ret = self.storage.update(new_minres)
325:         minres = self.storage.get_lowest()
326:         assert_equal(self.x0, minres.x)
327:         assert_equal(self.f0, minres.fun)
328:         assert_(not ret)
329: 
330:     def test_lower_f_accepted(self):
331:         new_minres = OptimizeResult()
332:         new_minres.x = self.x0 + 1
333:         new_minres.fun = self.f0 - 1
334: 
335:         ret = self.storage.update(new_minres)
336:         minres = self.storage.get_lowest()
337:         assert_(self.x0 != minres.x)
338:         assert_(self.f0 != minres.fun)
339:         assert_(ret)
340: 
341: 
342: class Test_RandomDisplacement(object):
343:     def setup_method(self):
344:         self.stepsize = 1.0
345:         self.displace = RandomDisplacement(stepsize=self.stepsize)
346:         self.N = 300000
347:         self.x0 = np.zeros([self.N])
348: 
349:     def test_random(self):
350:         # the mean should be 0
351:         # the variance should be (2*stepsize)**2 / 12
352:         # note these tests are random, they will fail from time to time
353:         x = self.displace(self.x0)
354:         v = (2. * self.stepsize) ** 2 / 12
355:         assert_almost_equal(np.mean(x), 0., 1)
356:         assert_almost_equal(np.var(x), v, 1)
357: 
358: 
359: class Test_Metropolis(object):
360:     def setup_method(self):
361:         self.T = 2.
362:         self.met = Metropolis(self.T)
363: 
364:     def test_boolean_return(self):
365:         # the return must be a bool.  else an error will be raised in
366:         # basinhopping
367:         ret = self.met(f_new=0., f_old=1.)
368:         assert isinstance(ret, bool)
369: 
370:     def test_lower_f_accepted(self):
371:         assert_(self.met(f_new=0., f_old=1.))
372: 
373:     def test_KeyError(self):
374:         # should raise KeyError if kwargs f_old or f_new is not passed
375:         assert_raises(KeyError, self.met, f_old=1.)
376:         assert_raises(KeyError, self.met, f_new=1.)
377: 
378:     def test_accept(self):
379:         # test that steps are randomly accepted for f_new > f_old
380:         one_accept = False
381:         one_reject = False
382:         for i in range(1000):
383:             if one_accept and one_reject:
384:                 break
385:             ret = self.met(f_new=1., f_old=0.5)
386:             if ret:
387:                 one_accept = True
388:             else:
389:                 one_reject = True
390:         assert_(one_accept)
391:         assert_(one_reject)
392: 
393:     def test_GH7495(self):
394:         # an overflow in exp was producing a RuntimeWarning
395:         # create own object here in case someone changes self.T
396:         met = Metropolis(2)
397:         with np.errstate(over='raise'):
398:             met.accept_reject(0, 2000)
399: 
400: 
401: class Test_AdaptiveStepsize(object):
402:     def setup_method(self):
403:         self.stepsize = 1.
404:         self.ts = RandomDisplacement(stepsize=self.stepsize)
405:         self.target_accept_rate = 0.5
406:         self.takestep = AdaptiveStepsize(takestep=self.ts, verbose=False,
407:                                           accept_rate=self.target_accept_rate)
408: 
409:     def test_adaptive_increase(self):
410:         # if few steps are rejected, the stepsize should increase
411:         x = 0.
412:         self.takestep(x)
413:         self.takestep.report(False)
414:         for i in range(self.takestep.interval):
415:             self.takestep(x)
416:             self.takestep.report(True)
417:         assert_(self.ts.stepsize > self.stepsize)
418: 
419:     def test_adaptive_decrease(self):
420:         # if few steps are rejected, the stepsize should increase
421:         x = 0.
422:         self.takestep(x)
423:         self.takestep.report(True)
424:         for i in range(self.takestep.interval):
425:             self.takestep(x)
426:             self.takestep.report(False)
427:         assert_(self.ts.stepsize < self.stepsize)
428: 
429:     def test_all_accepted(self):
430:         # test that everything works OK if all steps were accepted
431:         x = 0.
432:         for i in range(self.takestep.interval + 1):
433:             self.takestep(x)
434:             self.takestep.report(True)
435:         assert_(self.ts.stepsize > self.stepsize)
436: 
437:     def test_all_rejected(self):
438:         # test that everything works OK if all steps were rejected
439:         x = 0.
440:         for i in range(self.takestep.interval + 1):
441:             self.takestep(x)
442:             self.takestep.report(False)
443:         assert_(self.ts.stepsize < self.stepsize)
444: 
445: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_236798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nUnit tests for the basin hopping global minimization algorithm.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import copy' statement (line 5)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_almost_equal, assert_equal, assert_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236799 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_236799) is not StypyTypeError):

    if (import_236799 != 'pyd_module'):
        __import__(import_236799)
        sys_modules_236800 = sys.modules[import_236799]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_236800.module_type_store, module_type_store, ['assert_almost_equal', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_236800, sys_modules_236800.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_equal', 'assert_'], [assert_almost_equal, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_236799)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236801 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_236801) is not StypyTypeError):

    if (import_236801 != 'pyd_module'):
        __import__(import_236801)
        sys_modules_236802 = sys.modules[import_236801]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_236802.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_236802, sys_modules_236802.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_236801)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236803 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_236803) is not StypyTypeError):

    if (import_236803 != 'pyd_module'):
        __import__(import_236803)
        sys_modules_236804 = sys.modules[import_236803]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_236804.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_236803)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy import cos, sin' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236805 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_236805) is not StypyTypeError):

    if (import_236805 != 'pyd_module'):
        __import__(import_236805)
        sys_modules_236806 = sys.modules[import_236805]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', sys_modules_236806.module_type_store, module_type_store, ['cos', 'sin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_236806, sys_modules_236806.module_type_store, module_type_store)
    else:
        from numpy import cos, sin

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', None, module_type_store, ['cos', 'sin'], [cos, sin])

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_236805)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.optimize import basinhopping, OptimizeResult' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236807 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize')

if (type(import_236807) is not StypyTypeError):

    if (import_236807 != 'pyd_module'):
        __import__(import_236807)
        sys_modules_236808 = sys.modules[import_236807]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize', sys_modules_236808.module_type_store, module_type_store, ['basinhopping', 'OptimizeResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_236808, sys_modules_236808.module_type_store, module_type_store)
    else:
        from scipy.optimize import basinhopping, OptimizeResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize', None, module_type_store, ['basinhopping', 'OptimizeResult'], [basinhopping, OptimizeResult])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize', import_236807)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.optimize._basinhopping import Storage, RandomDisplacement, Metropolis, AdaptiveStepsize' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_236809 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.optimize._basinhopping')

if (type(import_236809) is not StypyTypeError):

    if (import_236809 != 'pyd_module'):
        __import__(import_236809)
        sys_modules_236810 = sys.modules[import_236809]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.optimize._basinhopping', sys_modules_236810.module_type_store, module_type_store, ['Storage', 'RandomDisplacement', 'Metropolis', 'AdaptiveStepsize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_236810, sys_modules_236810.module_type_store, module_type_store)
    else:
        from scipy.optimize._basinhopping import Storage, RandomDisplacement, Metropolis, AdaptiveStepsize

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.optimize._basinhopping', None, module_type_store, ['Storage', 'RandomDisplacement', 'Metropolis', 'AdaptiveStepsize'], [Storage, RandomDisplacement, Metropolis, AdaptiveStepsize])

else:
    # Assigning a type to the variable 'scipy.optimize._basinhopping' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.optimize._basinhopping', import_236809)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def func1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func1d'
    module_type_store = module_type_store.open_function_context('func1d', 17, 0, False)
    
    # Passed parameters checking function
    func1d.stypy_localization = localization
    func1d.stypy_type_of_self = None
    func1d.stypy_type_store = module_type_store
    func1d.stypy_function_name = 'func1d'
    func1d.stypy_param_names_list = ['x']
    func1d.stypy_varargs_param_name = None
    func1d.stypy_kwargs_param_name = None
    func1d.stypy_call_defaults = defaults
    func1d.stypy_call_varargs = varargs
    func1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func1d', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func1d', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func1d(...)' code ##################

    
    # Assigning a BinOp to a Name (line 18):
    
    # Assigning a BinOp to a Name (line 18):
    
    # Call to cos(...): (line 18)
    # Processing the call arguments (line 18)
    float_236812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'float')
    # Getting the type of 'x' (line 18)
    x_236813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'x', False)
    # Applying the binary operator '*' (line 18)
    result_mul_236814 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 12), '*', float_236812, x_236813)
    
    float_236815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'float')
    # Applying the binary operator '-' (line 18)
    result_sub_236816 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 12), '-', result_mul_236814, float_236815)
    
    # Processing the call keyword arguments (line 18)
    kwargs_236817 = {}
    # Getting the type of 'cos' (line 18)
    cos_236811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'cos', False)
    # Calling cos(args, kwargs) (line 18)
    cos_call_result_236818 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), cos_236811, *[result_sub_236816], **kwargs_236817)
    
    # Getting the type of 'x' (line 18)
    x_236819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'x')
    float_236820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'float')
    # Applying the binary operator '+' (line 18)
    result_add_236821 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 31), '+', x_236819, float_236820)
    
    # Getting the type of 'x' (line 18)
    x_236822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 42), 'x')
    # Applying the binary operator '*' (line 18)
    result_mul_236823 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 30), '*', result_add_236821, x_236822)
    
    # Applying the binary operator '+' (line 18)
    result_add_236824 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 8), '+', cos_call_result_236818, result_mul_236823)
    
    # Assigning a type to the variable 'f' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'f', result_add_236824)
    
    # Assigning a Call to a Name (line 19):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to array(...): (line 19)
    # Processing the call arguments (line 19)
    float_236827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'float')
    
    # Call to sin(...): (line 19)
    # Processing the call arguments (line 19)
    float_236829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 30), 'float')
    # Getting the type of 'x' (line 19)
    x_236830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 37), 'x', False)
    # Applying the binary operator '*' (line 19)
    result_mul_236831 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 30), '*', float_236829, x_236830)
    
    float_236832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'float')
    # Applying the binary operator '-' (line 19)
    result_sub_236833 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 30), '-', result_mul_236831, float_236832)
    
    # Processing the call keyword arguments (line 19)
    kwargs_236834 = {}
    # Getting the type of 'sin' (line 19)
    sin_236828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 26), 'sin', False)
    # Calling sin(args, kwargs) (line 19)
    sin_call_result_236835 = invoke(stypy.reporting.localization.Localization(__file__, 19, 26), sin_236828, *[result_sub_236833], **kwargs_236834)
    
    # Applying the binary operator '*' (line 19)
    result_mul_236836 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 18), '*', float_236827, sin_call_result_236835)
    
    float_236837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 48), 'float')
    # Getting the type of 'x' (line 19)
    x_236838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 53), 'x', False)
    # Applying the binary operator '*' (line 19)
    result_mul_236839 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 48), '*', float_236837, x_236838)
    
    # Applying the binary operator '+' (line 19)
    result_add_236840 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 18), '+', result_mul_236836, result_mul_236839)
    
    float_236841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 57), 'float')
    # Applying the binary operator '+' (line 19)
    result_add_236842 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 55), '+', result_add_236840, float_236841)
    
    # Processing the call keyword arguments (line 19)
    kwargs_236843 = {}
    # Getting the type of 'np' (line 19)
    np_236825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 19)
    array_236826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 9), np_236825, 'array')
    # Calling array(args, kwargs) (line 19)
    array_call_result_236844 = invoke(stypy.reporting.localization.Localization(__file__, 19, 9), array_236826, *[result_add_236842], **kwargs_236843)
    
    # Assigning a type to the variable 'df' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'df', array_call_result_236844)
    
    # Obtaining an instance of the builtin type 'tuple' (line 20)
    tuple_236845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 20)
    # Adding element type (line 20)
    # Getting the type of 'f' (line 20)
    f_236846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 11), tuple_236845, f_236846)
    # Adding element type (line 20)
    # Getting the type of 'df' (line 20)
    df_236847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'df')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 11), tuple_236845, df_236847)
    
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type', tuple_236845)
    
    # ################# End of 'func1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func1d' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_236848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_236848)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func1d'
    return stypy_return_type_236848

# Assigning a type to the variable 'func1d' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'func1d', func1d)

@norecursion
def func1d_nograd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func1d_nograd'
    module_type_store = module_type_store.open_function_context('func1d_nograd', 23, 0, False)
    
    # Passed parameters checking function
    func1d_nograd.stypy_localization = localization
    func1d_nograd.stypy_type_of_self = None
    func1d_nograd.stypy_type_store = module_type_store
    func1d_nograd.stypy_function_name = 'func1d_nograd'
    func1d_nograd.stypy_param_names_list = ['x']
    func1d_nograd.stypy_varargs_param_name = None
    func1d_nograd.stypy_kwargs_param_name = None
    func1d_nograd.stypy_call_defaults = defaults
    func1d_nograd.stypy_call_varargs = varargs
    func1d_nograd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func1d_nograd', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func1d_nograd', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func1d_nograd(...)' code ##################

    
    # Assigning a BinOp to a Name (line 24):
    
    # Assigning a BinOp to a Name (line 24):
    
    # Call to cos(...): (line 24)
    # Processing the call arguments (line 24)
    float_236850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'float')
    # Getting the type of 'x' (line 24)
    x_236851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'x', False)
    # Applying the binary operator '*' (line 24)
    result_mul_236852 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 12), '*', float_236850, x_236851)
    
    float_236853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'float')
    # Applying the binary operator '-' (line 24)
    result_sub_236854 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 12), '-', result_mul_236852, float_236853)
    
    # Processing the call keyword arguments (line 24)
    kwargs_236855 = {}
    # Getting the type of 'cos' (line 24)
    cos_236849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'cos', False)
    # Calling cos(args, kwargs) (line 24)
    cos_call_result_236856 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), cos_236849, *[result_sub_236854], **kwargs_236855)
    
    # Getting the type of 'x' (line 24)
    x_236857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'x')
    float_236858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 35), 'float')
    # Applying the binary operator '+' (line 24)
    result_add_236859 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 31), '+', x_236857, float_236858)
    
    # Getting the type of 'x' (line 24)
    x_236860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'x')
    # Applying the binary operator '*' (line 24)
    result_mul_236861 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 30), '*', result_add_236859, x_236860)
    
    # Applying the binary operator '+' (line 24)
    result_add_236862 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 8), '+', cos_call_result_236856, result_mul_236861)
    
    # Assigning a type to the variable 'f' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'f', result_add_236862)
    
    # Assigning a Call to a Name (line 25):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to array(...): (line 25)
    # Processing the call arguments (line 25)
    float_236865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'float')
    
    # Call to sin(...): (line 25)
    # Processing the call arguments (line 25)
    float_236867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'float')
    # Getting the type of 'x' (line 25)
    x_236868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), 'x', False)
    # Applying the binary operator '*' (line 25)
    result_mul_236869 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 30), '*', float_236867, x_236868)
    
    float_236870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'float')
    # Applying the binary operator '-' (line 25)
    result_sub_236871 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 30), '-', result_mul_236869, float_236870)
    
    # Processing the call keyword arguments (line 25)
    kwargs_236872 = {}
    # Getting the type of 'sin' (line 25)
    sin_236866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'sin', False)
    # Calling sin(args, kwargs) (line 25)
    sin_call_result_236873 = invoke(stypy.reporting.localization.Localization(__file__, 25, 26), sin_236866, *[result_sub_236871], **kwargs_236872)
    
    # Applying the binary operator '*' (line 25)
    result_mul_236874 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 18), '*', float_236865, sin_call_result_236873)
    
    float_236875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 48), 'float')
    # Getting the type of 'x' (line 25)
    x_236876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 53), 'x', False)
    # Applying the binary operator '*' (line 25)
    result_mul_236877 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 48), '*', float_236875, x_236876)
    
    # Applying the binary operator '+' (line 25)
    result_add_236878 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 18), '+', result_mul_236874, result_mul_236877)
    
    float_236879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 57), 'float')
    # Applying the binary operator '+' (line 25)
    result_add_236880 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 55), '+', result_add_236878, float_236879)
    
    # Processing the call keyword arguments (line 25)
    kwargs_236881 = {}
    # Getting the type of 'np' (line 25)
    np_236863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 25)
    array_236864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 9), np_236863, 'array')
    # Calling array(args, kwargs) (line 25)
    array_call_result_236882 = invoke(stypy.reporting.localization.Localization(__file__, 25, 9), array_236864, *[result_add_236880], **kwargs_236881)
    
    # Assigning a type to the variable 'df' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'df', array_call_result_236882)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_236883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'f' (line 26)
    f_236884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 11), tuple_236883, f_236884)
    # Adding element type (line 26)
    # Getting the type of 'df' (line 26)
    df_236885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'df')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 11), tuple_236883, df_236885)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', tuple_236883)
    
    # ################# End of 'func1d_nograd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func1d_nograd' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_236886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_236886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func1d_nograd'
    return stypy_return_type_236886

# Assigning a type to the variable 'func1d_nograd' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'func1d_nograd', func1d_nograd)

@norecursion
def func2d_nograd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func2d_nograd'
    module_type_store = module_type_store.open_function_context('func2d_nograd', 29, 0, False)
    
    # Passed parameters checking function
    func2d_nograd.stypy_localization = localization
    func2d_nograd.stypy_type_of_self = None
    func2d_nograd.stypy_type_store = module_type_store
    func2d_nograd.stypy_function_name = 'func2d_nograd'
    func2d_nograd.stypy_param_names_list = ['x']
    func2d_nograd.stypy_varargs_param_name = None
    func2d_nograd.stypy_kwargs_param_name = None
    func2d_nograd.stypy_call_defaults = defaults
    func2d_nograd.stypy_call_varargs = varargs
    func2d_nograd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func2d_nograd', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func2d_nograd', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func2d_nograd(...)' code ##################

    
    # Assigning a BinOp to a Name (line 30):
    
    # Assigning a BinOp to a Name (line 30):
    
    # Call to cos(...): (line 30)
    # Processing the call arguments (line 30)
    float_236888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'float')
    
    # Obtaining the type of the subscript
    int_236889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'int')
    # Getting the type of 'x' (line 30)
    x_236890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___236891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 19), x_236890, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_236892 = invoke(stypy.reporting.localization.Localization(__file__, 30, 19), getitem___236891, int_236889)
    
    # Applying the binary operator '*' (line 30)
    result_mul_236893 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), '*', float_236888, subscript_call_result_236892)
    
    float_236894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 26), 'float')
    # Applying the binary operator '-' (line 30)
    result_sub_236895 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), '-', result_mul_236893, float_236894)
    
    # Processing the call keyword arguments (line 30)
    kwargs_236896 = {}
    # Getting the type of 'cos' (line 30)
    cos_236887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'cos', False)
    # Calling cos(args, kwargs) (line 30)
    cos_call_result_236897 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), cos_236887, *[result_sub_236895], **kwargs_236896)
    
    
    # Obtaining the type of the subscript
    int_236898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'int')
    # Getting the type of 'x' (line 30)
    x_236899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'x')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___236900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 34), x_236899, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_236901 = invoke(stypy.reporting.localization.Localization(__file__, 30, 34), getitem___236900, int_236898)
    
    float_236902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 41), 'float')
    # Applying the binary operator '+' (line 30)
    result_add_236903 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 34), '+', subscript_call_result_236901, float_236902)
    
    
    # Obtaining the type of the subscript
    int_236904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 50), 'int')
    # Getting the type of 'x' (line 30)
    x_236905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 48), 'x')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___236906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 48), x_236905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_236907 = invoke(stypy.reporting.localization.Localization(__file__, 30, 48), getitem___236906, int_236904)
    
    # Applying the binary operator '*' (line 30)
    result_mul_236908 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 33), '*', result_add_236903, subscript_call_result_236907)
    
    # Applying the binary operator '+' (line 30)
    result_add_236909 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 8), '+', cos_call_result_236897, result_mul_236908)
    
    
    # Obtaining the type of the subscript
    int_236910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 58), 'int')
    # Getting the type of 'x' (line 30)
    x_236911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 56), 'x')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___236912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 56), x_236911, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_236913 = invoke(stypy.reporting.localization.Localization(__file__, 30, 56), getitem___236912, int_236910)
    
    float_236914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 63), 'float')
    # Applying the binary operator '+' (line 30)
    result_add_236915 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 56), '+', subscript_call_result_236913, float_236914)
    
    
    # Obtaining the type of the subscript
    int_236916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 72), 'int')
    # Getting the type of 'x' (line 30)
    x_236917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 70), 'x')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___236918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 70), x_236917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_236919 = invoke(stypy.reporting.localization.Localization(__file__, 30, 70), getitem___236918, int_236916)
    
    # Applying the binary operator '*' (line 30)
    result_mul_236920 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 55), '*', result_add_236915, subscript_call_result_236919)
    
    # Applying the binary operator '+' (line 30)
    result_add_236921 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 53), '+', result_add_236909, result_mul_236920)
    
    # Assigning a type to the variable 'f' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'f', result_add_236921)
    # Getting the type of 'f' (line 31)
    f_236922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', f_236922)
    
    # ################# End of 'func2d_nograd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func2d_nograd' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_236923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_236923)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func2d_nograd'
    return stypy_return_type_236923

# Assigning a type to the variable 'func2d_nograd' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'func2d_nograd', func2d_nograd)

@norecursion
def func2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func2d'
    module_type_store = module_type_store.open_function_context('func2d', 34, 0, False)
    
    # Passed parameters checking function
    func2d.stypy_localization = localization
    func2d.stypy_type_of_self = None
    func2d.stypy_type_store = module_type_store
    func2d.stypy_function_name = 'func2d'
    func2d.stypy_param_names_list = ['x']
    func2d.stypy_varargs_param_name = None
    func2d.stypy_kwargs_param_name = None
    func2d.stypy_call_defaults = defaults
    func2d.stypy_call_varargs = varargs
    func2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func2d', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func2d', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func2d(...)' code ##################

    
    # Assigning a BinOp to a Name (line 35):
    
    # Assigning a BinOp to a Name (line 35):
    
    # Call to cos(...): (line 35)
    # Processing the call arguments (line 35)
    float_236925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 12), 'float')
    
    # Obtaining the type of the subscript
    int_236926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'int')
    # Getting the type of 'x' (line 35)
    x_236927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___236928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 19), x_236927, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_236929 = invoke(stypy.reporting.localization.Localization(__file__, 35, 19), getitem___236928, int_236926)
    
    # Applying the binary operator '*' (line 35)
    result_mul_236930 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 12), '*', float_236925, subscript_call_result_236929)
    
    float_236931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'float')
    # Applying the binary operator '-' (line 35)
    result_sub_236932 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 12), '-', result_mul_236930, float_236931)
    
    # Processing the call keyword arguments (line 35)
    kwargs_236933 = {}
    # Getting the type of 'cos' (line 35)
    cos_236924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'cos', False)
    # Calling cos(args, kwargs) (line 35)
    cos_call_result_236934 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), cos_236924, *[result_sub_236932], **kwargs_236933)
    
    
    # Obtaining the type of the subscript
    int_236935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'int')
    # Getting the type of 'x' (line 35)
    x_236936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'x')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___236937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), x_236936, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_236938 = invoke(stypy.reporting.localization.Localization(__file__, 35, 34), getitem___236937, int_236935)
    
    float_236939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 41), 'float')
    # Applying the binary operator '+' (line 35)
    result_add_236940 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 34), '+', subscript_call_result_236938, float_236939)
    
    
    # Obtaining the type of the subscript
    int_236941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 50), 'int')
    # Getting the type of 'x' (line 35)
    x_236942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 48), 'x')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___236943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 48), x_236942, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_236944 = invoke(stypy.reporting.localization.Localization(__file__, 35, 48), getitem___236943, int_236941)
    
    # Applying the binary operator '*' (line 35)
    result_mul_236945 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 33), '*', result_add_236940, subscript_call_result_236944)
    
    # Applying the binary operator '+' (line 35)
    result_add_236946 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 8), '+', cos_call_result_236934, result_mul_236945)
    
    
    # Obtaining the type of the subscript
    int_236947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 58), 'int')
    # Getting the type of 'x' (line 35)
    x_236948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 56), 'x')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___236949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 56), x_236948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_236950 = invoke(stypy.reporting.localization.Localization(__file__, 35, 56), getitem___236949, int_236947)
    
    float_236951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 63), 'float')
    # Applying the binary operator '+' (line 35)
    result_add_236952 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 56), '+', subscript_call_result_236950, float_236951)
    
    
    # Obtaining the type of the subscript
    int_236953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 72), 'int')
    # Getting the type of 'x' (line 35)
    x_236954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 70), 'x')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___236955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 70), x_236954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_236956 = invoke(stypy.reporting.localization.Localization(__file__, 35, 70), getitem___236955, int_236953)
    
    # Applying the binary operator '*' (line 35)
    result_mul_236957 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 55), '*', result_add_236952, subscript_call_result_236956)
    
    # Applying the binary operator '+' (line 35)
    result_add_236958 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 53), '+', result_add_236946, result_mul_236957)
    
    # Assigning a type to the variable 'f' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'f', result_add_236958)
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to zeros(...): (line 36)
    # Processing the call arguments (line 36)
    int_236961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'int')
    # Processing the call keyword arguments (line 36)
    kwargs_236962 = {}
    # Getting the type of 'np' (line 36)
    np_236959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 36)
    zeros_236960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 9), np_236959, 'zeros')
    # Calling zeros(args, kwargs) (line 36)
    zeros_call_result_236963 = invoke(stypy.reporting.localization.Localization(__file__, 36, 9), zeros_236960, *[int_236961], **kwargs_236962)
    
    # Assigning a type to the variable 'df' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'df', zeros_call_result_236963)
    
    # Assigning a BinOp to a Subscript (line 37):
    
    # Assigning a BinOp to a Subscript (line 37):
    float_236964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 12), 'float')
    
    # Call to sin(...): (line 37)
    # Processing the call arguments (line 37)
    float_236966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'float')
    
    # Obtaining the type of the subscript
    int_236967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'int')
    # Getting the type of 'x' (line 37)
    x_236968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___236969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 31), x_236968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_236970 = invoke(stypy.reporting.localization.Localization(__file__, 37, 31), getitem___236969, int_236967)
    
    # Applying the binary operator '*' (line 37)
    result_mul_236971 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 24), '*', float_236966, subscript_call_result_236970)
    
    float_236972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 38), 'float')
    # Applying the binary operator '-' (line 37)
    result_sub_236973 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 24), '-', result_mul_236971, float_236972)
    
    # Processing the call keyword arguments (line 37)
    kwargs_236974 = {}
    # Getting the type of 'sin' (line 37)
    sin_236965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'sin', False)
    # Calling sin(args, kwargs) (line 37)
    sin_call_result_236975 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), sin_236965, *[result_sub_236973], **kwargs_236974)
    
    # Applying the binary operator '*' (line 37)
    result_mul_236976 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), '*', float_236964, sin_call_result_236975)
    
    float_236977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 45), 'float')
    
    # Obtaining the type of the subscript
    int_236978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 52), 'int')
    # Getting the type of 'x' (line 37)
    x_236979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 50), 'x')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___236980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 50), x_236979, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_236981 = invoke(stypy.reporting.localization.Localization(__file__, 37, 50), getitem___236980, int_236978)
    
    # Applying the binary operator '*' (line 37)
    result_mul_236982 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 45), '*', float_236977, subscript_call_result_236981)
    
    # Applying the binary operator '+' (line 37)
    result_add_236983 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), '+', result_mul_236976, result_mul_236982)
    
    float_236984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 57), 'float')
    # Applying the binary operator '+' (line 37)
    result_add_236985 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 55), '+', result_add_236983, float_236984)
    
    # Getting the type of 'df' (line 37)
    df_236986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'df')
    int_236987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 7), 'int')
    # Storing an element on a container (line 37)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), df_236986, (int_236987, result_add_236985))
    
    # Assigning a BinOp to a Subscript (line 38):
    
    # Assigning a BinOp to a Subscript (line 38):
    float_236988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 12), 'float')
    
    # Obtaining the type of the subscript
    int_236989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 19), 'int')
    # Getting the type of 'x' (line 38)
    x_236990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___236991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), x_236990, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_236992 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), getitem___236991, int_236989)
    
    # Applying the binary operator '*' (line 38)
    result_mul_236993 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 12), '*', float_236988, subscript_call_result_236992)
    
    float_236994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'float')
    # Applying the binary operator '+' (line 38)
    result_add_236995 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 12), '+', result_mul_236993, float_236994)
    
    # Getting the type of 'df' (line 38)
    df_236996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'df')
    int_236997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 7), 'int')
    # Storing an element on a container (line 38)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 4), df_236996, (int_236997, result_add_236995))
    
    # Obtaining an instance of the builtin type 'tuple' (line 39)
    tuple_236998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'f' (line 39)
    f_236999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 11), tuple_236998, f_236999)
    # Adding element type (line 39)
    # Getting the type of 'df' (line 39)
    df_237000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'df')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 11), tuple_236998, df_237000)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type', tuple_236998)
    
    # ################# End of 'func2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func2d' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_237001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_237001)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func2d'
    return stypy_return_type_237001

# Assigning a type to the variable 'func2d' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'func2d', func2d)

@norecursion
def func2d_easyderiv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func2d_easyderiv'
    module_type_store = module_type_store.open_function_context('func2d_easyderiv', 41, 0, False)
    
    # Passed parameters checking function
    func2d_easyderiv.stypy_localization = localization
    func2d_easyderiv.stypy_type_of_self = None
    func2d_easyderiv.stypy_type_store = module_type_store
    func2d_easyderiv.stypy_function_name = 'func2d_easyderiv'
    func2d_easyderiv.stypy_param_names_list = ['x']
    func2d_easyderiv.stypy_varargs_param_name = None
    func2d_easyderiv.stypy_kwargs_param_name = None
    func2d_easyderiv.stypy_call_defaults = defaults
    func2d_easyderiv.stypy_call_varargs = varargs
    func2d_easyderiv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func2d_easyderiv', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func2d_easyderiv', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func2d_easyderiv(...)' code ##################

    
    # Assigning a BinOp to a Name (line 42):
    
    # Assigning a BinOp to a Name (line 42):
    float_237002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'float')
    
    # Obtaining the type of the subscript
    int_237003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'int')
    # Getting the type of 'x' (line 42)
    x_237004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___237005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), x_237004, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_237006 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), getitem___237005, int_237003)
    
    int_237007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'int')
    # Applying the binary operator '**' (line 42)
    result_pow_237008 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 12), '**', subscript_call_result_237006, int_237007)
    
    # Applying the binary operator '*' (line 42)
    result_mul_237009 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 8), '*', float_237002, result_pow_237008)
    
    float_237010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 22), 'float')
    
    # Obtaining the type of the subscript
    int_237011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 28), 'int')
    # Getting the type of 'x' (line 42)
    x_237012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'x')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___237013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 26), x_237012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_237014 = invoke(stypy.reporting.localization.Localization(__file__, 42, 26), getitem___237013, int_237011)
    
    # Applying the binary operator '*' (line 42)
    result_mul_237015 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 22), '*', float_237010, subscript_call_result_237014)
    
    
    # Obtaining the type of the subscript
    int_237016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'int')
    # Getting the type of 'x' (line 42)
    x_237017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'x')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___237018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 31), x_237017, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_237019 = invoke(stypy.reporting.localization.Localization(__file__, 42, 31), getitem___237018, int_237016)
    
    # Applying the binary operator '*' (line 42)
    result_mul_237020 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 30), '*', result_mul_237015, subscript_call_result_237019)
    
    # Applying the binary operator '+' (line 42)
    result_add_237021 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 8), '+', result_mul_237009, result_mul_237020)
    
    float_237022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 38), 'float')
    
    # Obtaining the type of the subscript
    int_237023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 44), 'int')
    # Getting the type of 'x' (line 42)
    x_237024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 42), 'x')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___237025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 42), x_237024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_237026 = invoke(stypy.reporting.localization.Localization(__file__, 42, 42), getitem___237025, int_237023)
    
    int_237027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 48), 'int')
    # Applying the binary operator '**' (line 42)
    result_pow_237028 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 42), '**', subscript_call_result_237026, int_237027)
    
    # Applying the binary operator '*' (line 42)
    result_mul_237029 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 38), '*', float_237022, result_pow_237028)
    
    # Applying the binary operator '+' (line 42)
    result_add_237030 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 36), '+', result_add_237021, result_mul_237029)
    
    float_237031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 52), 'float')
    
    # Obtaining the type of the subscript
    int_237032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 58), 'int')
    # Getting the type of 'x' (line 42)
    x_237033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 56), 'x')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___237034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 56), x_237033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_237035 = invoke(stypy.reporting.localization.Localization(__file__, 42, 56), getitem___237034, int_237032)
    
    # Applying the binary operator '*' (line 42)
    result_mul_237036 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 52), '*', float_237031, subscript_call_result_237035)
    
    # Applying the binary operator '-' (line 42)
    result_sub_237037 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 50), '-', result_add_237030, result_mul_237036)
    
    # Assigning a type to the variable 'f' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'f', result_sub_237037)
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to zeros(...): (line 43)
    # Processing the call arguments (line 43)
    int_237040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_237041 = {}
    # Getting the type of 'np' (line 43)
    np_237038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 43)
    zeros_237039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 9), np_237038, 'zeros')
    # Calling zeros(args, kwargs) (line 43)
    zeros_call_result_237042 = invoke(stypy.reporting.localization.Localization(__file__, 43, 9), zeros_237039, *[int_237040], **kwargs_237041)
    
    # Assigning a type to the variable 'df' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'df', zeros_call_result_237042)
    
    # Assigning a BinOp to a Subscript (line 44):
    
    # Assigning a BinOp to a Subscript (line 44):
    float_237043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'float')
    
    # Obtaining the type of the subscript
    int_237044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 18), 'int')
    # Getting the type of 'x' (line 44)
    x_237045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'x')
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___237046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), x_237045, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_237047 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), getitem___237046, int_237044)
    
    # Applying the binary operator '*' (line 44)
    result_mul_237048 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '*', float_237043, subscript_call_result_237047)
    
    float_237049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'float')
    
    # Obtaining the type of the subscript
    int_237050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 29), 'int')
    # Getting the type of 'x' (line 44)
    x_237051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'x')
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___237052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), x_237051, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_237053 = invoke(stypy.reporting.localization.Localization(__file__, 44, 27), getitem___237052, int_237050)
    
    # Applying the binary operator '*' (line 44)
    result_mul_237054 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 23), '*', float_237049, subscript_call_result_237053)
    
    # Applying the binary operator '+' (line 44)
    result_add_237055 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '+', result_mul_237048, result_mul_237054)
    
    float_237056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'float')
    # Applying the binary operator '-' (line 44)
    result_sub_237057 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 32), '-', result_add_237055, float_237056)
    
    # Getting the type of 'df' (line 44)
    df_237058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'df')
    int_237059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 7), 'int')
    # Storing an element on a container (line 44)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 4), df_237058, (int_237059, result_sub_237057))
    
    # Assigning a BinOp to a Subscript (line 45):
    
    # Assigning a BinOp to a Subscript (line 45):
    float_237060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'float')
    
    # Obtaining the type of the subscript
    int_237061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'int')
    # Getting the type of 'x' (line 45)
    x_237062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'x')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___237063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 16), x_237062, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_237064 = invoke(stypy.reporting.localization.Localization(__file__, 45, 16), getitem___237063, int_237061)
    
    # Applying the binary operator '*' (line 45)
    result_mul_237065 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 12), '*', float_237060, subscript_call_result_237064)
    
    float_237066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'float')
    
    # Obtaining the type of the subscript
    int_237067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 29), 'int')
    # Getting the type of 'x' (line 45)
    x_237068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'x')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___237069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 27), x_237068, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_237070 = invoke(stypy.reporting.localization.Localization(__file__, 45, 27), getitem___237069, int_237067)
    
    # Applying the binary operator '*' (line 45)
    result_mul_237071 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 23), '*', float_237066, subscript_call_result_237070)
    
    # Applying the binary operator '+' (line 45)
    result_add_237072 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 12), '+', result_mul_237065, result_mul_237071)
    
    # Getting the type of 'df' (line 45)
    df_237073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'df')
    int_237074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 7), 'int')
    # Storing an element on a container (line 45)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 4), df_237073, (int_237074, result_add_237072))
    
    # Obtaining an instance of the builtin type 'tuple' (line 47)
    tuple_237075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 47)
    # Adding element type (line 47)
    # Getting the type of 'f' (line 47)
    f_237076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 11), tuple_237075, f_237076)
    # Adding element type (line 47)
    # Getting the type of 'df' (line 47)
    df_237077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'df')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 11), tuple_237075, df_237077)
    
    # Assigning a type to the variable 'stypy_return_type' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type', tuple_237075)
    
    # ################# End of 'func2d_easyderiv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func2d_easyderiv' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_237078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_237078)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func2d_easyderiv'
    return stypy_return_type_237078

# Assigning a type to the variable 'func2d_easyderiv' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'func2d_easyderiv', func2d_easyderiv)
# Declaration of the 'MyTakeStep1' class
# Getting the type of 'RandomDisplacement' (line 49)
RandomDisplacement_237079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'RandomDisplacement')

class MyTakeStep1(RandomDisplacement_237079, ):
    str_237080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'str', "use a copy of displace, but have it set a special parameter to\n    make sure it's actually being used.")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyTakeStep1.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 53):
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'False' (line 53)
        False_237081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'False')
        # Getting the type of 'self' (line 53)
        self_237082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'been_called' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_237082, 'been_called', False_237081)
        
        # Call to __init__(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_237089 = {}
        
        # Call to super(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'MyTakeStep1' (line 54)
        MyTakeStep1_237084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'MyTakeStep1', False)
        # Getting the type of 'self' (line 54)
        self_237085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'self', False)
        # Processing the call keyword arguments (line 54)
        kwargs_237086 = {}
        # Getting the type of 'super' (line 54)
        super_237083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'super', False)
        # Calling super(args, kwargs) (line 54)
        super_call_result_237087 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), super_237083, *[MyTakeStep1_237084, self_237085], **kwargs_237086)
        
        # Obtaining the member '__init__' of a type (line 54)
        init___237088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), super_call_result_237087, '__init__')
        # Calling __init__(args, kwargs) (line 54)
        init___call_result_237090 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), init___237088, *[], **kwargs_237089)
        
        
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
        module_type_store = module_type_store.open_function_context('__call__', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_localization', localization)
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_function_name', 'MyTakeStep1.__call__')
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MyTakeStep1.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyTakeStep1.__call__', ['x'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'True' (line 57)
        True_237091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'True')
        # Getting the type of 'self' (line 57)
        self_237092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'been_called' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_237092, 'been_called', True_237091)
        
        # Call to __call__(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'x' (line 58)
        x_237099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 49), 'x', False)
        # Processing the call keyword arguments (line 58)
        kwargs_237100 = {}
        
        # Call to super(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'MyTakeStep1' (line 58)
        MyTakeStep1_237094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'MyTakeStep1', False)
        # Getting the type of 'self' (line 58)
        self_237095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'self', False)
        # Processing the call keyword arguments (line 58)
        kwargs_237096 = {}
        # Getting the type of 'super' (line 58)
        super_237093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'super', False)
        # Calling super(args, kwargs) (line 58)
        super_call_result_237097 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), super_237093, *[MyTakeStep1_237094, self_237095], **kwargs_237096)
        
        # Obtaining the member '__call__' of a type (line 58)
        call___237098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 15), super_call_result_237097, '__call__')
        # Calling __call__(args, kwargs) (line 58)
        call___call_result_237101 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), call___237098, *[x_237099], **kwargs_237100)
        
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', call___call_result_237101)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_237102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_237102


# Assigning a type to the variable 'MyTakeStep1' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'MyTakeStep1', MyTakeStep1)

@norecursion
def myTakeStep2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'myTakeStep2'
    module_type_store = module_type_store.open_function_context('myTakeStep2', 61, 0, False)
    
    # Passed parameters checking function
    myTakeStep2.stypy_localization = localization
    myTakeStep2.stypy_type_of_self = None
    myTakeStep2.stypy_type_store = module_type_store
    myTakeStep2.stypy_function_name = 'myTakeStep2'
    myTakeStep2.stypy_param_names_list = ['x']
    myTakeStep2.stypy_varargs_param_name = None
    myTakeStep2.stypy_kwargs_param_name = None
    myTakeStep2.stypy_call_defaults = defaults
    myTakeStep2.stypy_call_varargs = varargs
    myTakeStep2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'myTakeStep2', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'myTakeStep2', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'myTakeStep2(...)' code ##################

    str_237103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', 'redo RandomDisplacement in function form without the attribute stepsize\n    to make sure still everything works ok\n    ')
    
    # Assigning a Num to a Name (line 65):
    
    # Assigning a Num to a Name (line 65):
    float_237104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 8), 'float')
    # Assigning a type to the variable 's' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 's', float_237104)
    
    # Getting the type of 'x' (line 66)
    x_237105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'x')
    
    # Call to uniform(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Getting the type of 's' (line 66)
    s_237109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 's', False)
    # Applying the 'usub' unary operator (line 66)
    result___neg___237110 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 27), 'usub', s_237109)
    
    # Getting the type of 's' (line 66)
    s_237111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 31), 's', False)
    
    # Call to shape(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'x' (line 66)
    x_237114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 43), 'x', False)
    # Processing the call keyword arguments (line 66)
    kwargs_237115 = {}
    # Getting the type of 'np' (line 66)
    np_237112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'np', False)
    # Obtaining the member 'shape' of a type (line 66)
    shape_237113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 34), np_237112, 'shape')
    # Calling shape(args, kwargs) (line 66)
    shape_call_result_237116 = invoke(stypy.reporting.localization.Localization(__file__, 66, 34), shape_237113, *[x_237114], **kwargs_237115)
    
    # Processing the call keyword arguments (line 66)
    kwargs_237117 = {}
    # Getting the type of 'np' (line 66)
    np_237106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 66)
    random_237107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 9), np_237106, 'random')
    # Obtaining the member 'uniform' of a type (line 66)
    uniform_237108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 9), random_237107, 'uniform')
    # Calling uniform(args, kwargs) (line 66)
    uniform_call_result_237118 = invoke(stypy.reporting.localization.Localization(__file__, 66, 9), uniform_237108, *[result___neg___237110, s_237111, shape_call_result_237116], **kwargs_237117)
    
    # Applying the binary operator '+=' (line 66)
    result_iadd_237119 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 4), '+=', x_237105, uniform_call_result_237118)
    # Assigning a type to the variable 'x' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'x', result_iadd_237119)
    
    # Getting the type of 'x' (line 67)
    x_237120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type', x_237120)
    
    # ################# End of 'myTakeStep2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'myTakeStep2' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_237121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_237121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'myTakeStep2'
    return stypy_return_type_237121

# Assigning a type to the variable 'myTakeStep2' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'myTakeStep2', myTakeStep2)
# Declaration of the 'MyAcceptTest' class

class MyAcceptTest(object, ):
    str_237122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', "pass a custom accept test\n\n    This does nothing but make sure it's being used and ensure all the\n    possible return values are accepted\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyAcceptTest.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 77):
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of 'False' (line 77)
        False_237123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'False')
        # Getting the type of 'self' (line 77)
        self_237124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'been_called' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_237124, 'been_called', False_237123)
        
        # Assigning a Num to a Attribute (line 78):
        
        # Assigning a Num to a Attribute (line 78):
        int_237125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'int')
        # Getting the type of 'self' (line 78)
        self_237126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'ncalls' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_237126, 'ncalls', int_237125)
        
        # Assigning a List to a Attribute (line 79):
        
        # Assigning a List to a Attribute (line 79):
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_237127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        # Getting the type of 'False' (line 79)
        False_237128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, False_237128)
        # Adding element type (line 79)
        str_237129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 31), 'str', 'force accept')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, str_237129)
        # Adding element type (line 79)
        # Getting the type of 'True' (line 79)
        True_237130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 47), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, True_237130)
        # Adding element type (line 79)
        
        # Call to bool_(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'True' (line 79)
        True_237133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 62), 'True', False)
        # Processing the call keyword arguments (line 79)
        kwargs_237134 = {}
        # Getting the type of 'np' (line 79)
        np_237131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 53), 'np', False)
        # Obtaining the member 'bool_' of a type (line 79)
        bool__237132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 53), np_237131, 'bool_')
        # Calling bool_(args, kwargs) (line 79)
        bool__call_result_237135 = invoke(stypy.reporting.localization.Localization(__file__, 79, 53), bool__237132, *[True_237133], **kwargs_237134)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, bool__call_result_237135)
        # Adding element type (line 79)
        
        # Call to bool_(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'False' (line 80)
        False_237138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 'False', False)
        # Processing the call keyword arguments (line 80)
        kwargs_237139 = {}
        # Getting the type of 'np' (line 80)
        np_237136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'np', False)
        # Obtaining the member 'bool_' of a type (line 80)
        bool__237137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 24), np_237136, 'bool_')
        # Calling bool_(args, kwargs) (line 80)
        bool__call_result_237140 = invoke(stypy.reporting.localization.Localization(__file__, 80, 24), bool__237137, *[False_237138], **kwargs_237139)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, bool__call_result_237140)
        # Adding element type (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_237141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, list_237141)
        # Adding element type (line 79)
        
        # Obtaining an instance of the builtin type 'dict' (line 80)
        dict_237142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 45), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 80)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, dict_237142)
        # Adding element type (line 79)
        int_237143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, int_237143)
        # Adding element type (line 79)
        int_237144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 23), list_237127, int_237144)
        
        # Getting the type of 'self' (line 79)
        self_237145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'testres' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_237145, 'testres', list_237127)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_localization', localization)
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_function_name', 'MyAcceptTest.__call__')
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MyAcceptTest.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyAcceptTest.__call__', [], None, 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 83):
        
        # Assigning a Name to a Attribute (line 83):
        # Getting the type of 'True' (line 83)
        True_237146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'True')
        # Getting the type of 'self' (line 83)
        self_237147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
        # Setting the type of the member 'been_called' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_237147, 'been_called', True_237146)
        
        # Getting the type of 'self' (line 84)
        self_237148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Obtaining the member 'ncalls' of a type (line 84)
        ncalls_237149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_237148, 'ncalls')
        int_237150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'int')
        # Applying the binary operator '+=' (line 84)
        result_iadd_237151 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 8), '+=', ncalls_237149, int_237150)
        # Getting the type of 'self' (line 84)
        self_237152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member 'ncalls' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_237152, 'ncalls', result_iadd_237151)
        
        
        
        # Getting the type of 'self' (line 85)
        self_237153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'self')
        # Obtaining the member 'ncalls' of a type (line 85)
        ncalls_237154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 11), self_237153, 'ncalls')
        int_237155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'int')
        # Applying the binary operator '-' (line 85)
        result_sub_237156 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), '-', ncalls_237154, int_237155)
        
        
        # Call to len(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 85)
        self_237158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 33), 'self', False)
        # Obtaining the member 'testres' of a type (line 85)
        testres_237159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 33), self_237158, 'testres')
        # Processing the call keyword arguments (line 85)
        kwargs_237160 = {}
        # Getting the type of 'len' (line 85)
        len_237157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'len', False)
        # Calling len(args, kwargs) (line 85)
        len_call_result_237161 = invoke(stypy.reporting.localization.Localization(__file__, 85, 29), len_237157, *[testres_237159], **kwargs_237160)
        
        # Applying the binary operator '<' (line 85)
        result_lt_237162 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), '<', result_sub_237156, len_call_result_237161)
        
        # Testing the type of an if condition (line 85)
        if_condition_237163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), result_lt_237162)
        # Assigning a type to the variable 'if_condition_237163' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_237163', if_condition_237163)
        # SSA begins for if statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 86)
        self_237164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'self')
        # Obtaining the member 'ncalls' of a type (line 86)
        ncalls_237165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 32), self_237164, 'ncalls')
        int_237166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 46), 'int')
        # Applying the binary operator '-' (line 86)
        result_sub_237167 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 32), '-', ncalls_237165, int_237166)
        
        # Getting the type of 'self' (line 86)
        self_237168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'self')
        # Obtaining the member 'testres' of a type (line 86)
        testres_237169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), self_237168, 'testres')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___237170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), testres_237169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_237171 = invoke(stypy.reporting.localization.Localization(__file__, 86, 19), getitem___237170, result_sub_237167)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', subscript_call_result_237171)
        # SSA branch for the else part of an if statement (line 85)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'True' (line 88)
        True_237172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', True_237172)
        # SSA join for if statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_237173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_237173


# Assigning a type to the variable 'MyAcceptTest' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'MyAcceptTest', MyAcceptTest)
# Declaration of the 'MyCallBack' class

class MyCallBack(object, ):
    str_237174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', "pass a custom callback function\n\n    This makes sure it's being used.  It also returns True after 10\n    steps to ensure that it's stopping early.\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyCallBack.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 99):
        
        # Assigning a Name to a Attribute (line 99):
        # Getting the type of 'False' (line 99)
        False_237175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'False')
        # Getting the type of 'self' (line 99)
        self_237176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member 'been_called' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_237176, 'been_called', False_237175)
        
        # Assigning a Num to a Attribute (line 100):
        
        # Assigning a Num to a Attribute (line 100):
        int_237177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'int')
        # Getting the type of 'self' (line 100)
        self_237178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member 'ncalls' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_237178, 'ncalls', int_237177)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MyCallBack.__call__.__dict__.__setitem__('stypy_localization', localization)
        MyCallBack.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MyCallBack.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MyCallBack.__call__.__dict__.__setitem__('stypy_function_name', 'MyCallBack.__call__')
        MyCallBack.__call__.__dict__.__setitem__('stypy_param_names_list', ['x', 'f', 'accepted'])
        MyCallBack.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MyCallBack.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MyCallBack.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MyCallBack.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MyCallBack.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MyCallBack.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyCallBack.__call__', ['x', 'f', 'accepted'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x', 'f', 'accepted'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 103):
        
        # Assigning a Name to a Attribute (line 103):
        # Getting the type of 'True' (line 103)
        True_237179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'True')
        # Getting the type of 'self' (line 103)
        self_237180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self')
        # Setting the type of the member 'been_called' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_237180, 'been_called', True_237179)
        
        # Getting the type of 'self' (line 104)
        self_237181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self')
        # Obtaining the member 'ncalls' of a type (line 104)
        ncalls_237182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_237181, 'ncalls')
        int_237183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 23), 'int')
        # Applying the binary operator '+=' (line 104)
        result_iadd_237184 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 8), '+=', ncalls_237182, int_237183)
        # Getting the type of 'self' (line 104)
        self_237185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self')
        # Setting the type of the member 'ncalls' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_237185, 'ncalls', result_iadd_237184)
        
        
        
        # Getting the type of 'self' (line 105)
        self_237186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'self')
        # Obtaining the member 'ncalls' of a type (line 105)
        ncalls_237187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 11), self_237186, 'ncalls')
        int_237188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 26), 'int')
        # Applying the binary operator '==' (line 105)
        result_eq_237189 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), '==', ncalls_237187, int_237188)
        
        # Testing the type of an if condition (line 105)
        if_condition_237190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), result_eq_237189)
        # Assigning a type to the variable 'if_condition_237190' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_237190', if_condition_237190)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 106)
        True_237191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'stypy_return_type', True_237191)
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_237192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237192)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_237192


# Assigning a type to the variable 'MyCallBack' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'MyCallBack', MyCallBack)
# Declaration of the 'TestBasinHopping' class

class TestBasinHopping(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.setup_method')
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.setup_method', [], None, None, defaults, varargs, kwargs)

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

        str_237193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'str', ' Tests setup.\n\n        Run tests based on the 1-D and 2-D functions described above.\n        ')
        
        # Assigning a Tuple to a Attribute (line 116):
        
        # Assigning a Tuple to a Attribute (line 116):
        
        # Obtaining an instance of the builtin type 'tuple' (line 116)
        tuple_237194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 116)
        # Adding element type (line 116)
        float_237195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), tuple_237194, float_237195)
        # Adding element type (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_237196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        float_237197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 24), list_237196, float_237197)
        # Adding element type (line 116)
        float_237198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 24), list_237196, float_237198)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), tuple_237194, list_237196)
        
        # Getting the type of 'self' (line 116)
        self_237199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member 'x0' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_237199, 'x0', tuple_237194)
        
        # Assigning a Tuple to a Attribute (line 117):
        
        # Assigning a Tuple to a Attribute (line 117):
        
        # Obtaining an instance of the builtin type 'tuple' (line 117)
        tuple_237200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 117)
        # Adding element type (line 117)
        float_237201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), tuple_237200, float_237201)
        # Adding element type (line 117)
        
        # Call to array(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_237204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        float_237205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 37), list_237204, float_237205)
        # Adding element type (line 117)
        float_237206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 37), list_237204, float_237206)
        
        # Processing the call keyword arguments (line 117)
        kwargs_237207 = {}
        # Getting the type of 'np' (line 117)
        np_237202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'np', False)
        # Obtaining the member 'array' of a type (line 117)
        array_237203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 28), np_237202, 'array')
        # Calling array(args, kwargs) (line 117)
        array_call_result_237208 = invoke(stypy.reporting.localization.Localization(__file__, 117, 28), array_237203, *[list_237204], **kwargs_237207)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), tuple_237200, array_call_result_237208)
        
        # Getting the type of 'self' (line 117)
        self_237209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Setting the type of the member 'sol' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_237209, 'sol', tuple_237200)
        
        # Assigning a Num to a Attribute (line 119):
        
        # Assigning a Num to a Attribute (line 119):
        int_237210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 19), 'int')
        # Getting the type of 'self' (line 119)
        self_237211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'tol' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_237211, 'tol', int_237210)
        
        # Assigning a Num to a Attribute (line 121):
        
        # Assigning a Num to a Attribute (line 121):
        int_237212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 21), 'int')
        # Getting the type of 'self' (line 121)
        self_237213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 'niter' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_237213, 'niter', int_237212)
        
        # Assigning a Name to a Attribute (line 122):
        
        # Assigning a Name to a Attribute (line 122):
        # Getting the type of 'False' (line 122)
        False_237214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'False')
        # Getting the type of 'self' (line 122)
        self_237215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member 'disp' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_237215, 'disp', False_237214)
        
        # Call to seed(...): (line 125)
        # Processing the call arguments (line 125)
        int_237219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'int')
        # Processing the call keyword arguments (line 125)
        kwargs_237220 = {}
        # Getting the type of 'np' (line 125)
        np_237216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 125)
        random_237217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), np_237216, 'random')
        # Obtaining the member 'seed' of a type (line 125)
        seed_237218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), random_237217, 'seed')
        # Calling seed(args, kwargs) (line 125)
        seed_call_result_237221 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), seed_237218, *[int_237219], **kwargs_237220)
        
        
        # Assigning a Dict to a Attribute (line 127):
        
        # Assigning a Dict to a Attribute (line 127):
        
        # Obtaining an instance of the builtin type 'dict' (line 127)
        dict_237222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 127)
        # Adding element type (key, value) (line 127)
        str_237223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 23), 'str', 'method')
        str_237224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 33), 'str', 'L-BFGS-B')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), dict_237222, (str_237223, str_237224))
        # Adding element type (key, value) (line 127)
        str_237225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 45), 'str', 'jac')
        # Getting the type of 'True' (line 127)
        True_237226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 52), 'True')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), dict_237222, (str_237225, True_237226))
        
        # Getting the type of 'self' (line 127)
        self_237227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self')
        # Setting the type of the member 'kwargs' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_237227, 'kwargs', dict_237222)
        
        # Assigning a Dict to a Attribute (line 128):
        
        # Assigning a Dict to a Attribute (line 128):
        
        # Obtaining an instance of the builtin type 'dict' (line 128)
        dict_237228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 29), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 128)
        # Adding element type (key, value) (line 128)
        str_237229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'str', 'method')
        str_237230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 40), 'str', 'L-BFGS-B')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 29), dict_237228, (str_237229, str_237230))
        
        # Getting the type of 'self' (line 128)
        self_237231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'kwargs_nograd' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_237231, 'kwargs_nograd', dict_237228)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_237232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237232)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_237232


    @norecursion
    def test_TypeError(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_TypeError'
        module_type_store = module_type_store.open_function_context('test_TypeError', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_TypeError')
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_TypeError.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_TypeError', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_TypeError', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_TypeError(...)' code ##################

        
        # Assigning a Num to a Name (line 132):
        
        # Assigning a Num to a Name (line 132):
        int_237233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'int')
        # Assigning a type to the variable 'i' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'i', int_237233)
        
        # Call to assert_raises(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'TypeError' (line 134)
        TypeError_237235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'TypeError', False)
        # Getting the type of 'basinhopping' (line 134)
        basinhopping_237236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 33), 'basinhopping', False)
        # Getting the type of 'func2d' (line 134)
        func2d_237237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 47), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 134)
        i_237238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 63), 'i', False)
        # Getting the type of 'self' (line 134)
        self_237239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 55), 'self', False)
        # Obtaining the member 'x0' of a type (line 134)
        x0_237240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 55), self_237239, 'x0')
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___237241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 55), x0_237240, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_237242 = invoke(stypy.reporting.localization.Localization(__file__, 134, 55), getitem___237241, i_237238)
        
        # Processing the call keyword arguments (line 134)
        int_237243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 36), 'int')
        keyword_237244 = int_237243
        kwargs_237245 = {'take_step': keyword_237244}
        # Getting the type of 'assert_raises' (line 134)
        assert_raises_237234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 134)
        assert_raises_call_result_237246 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), assert_raises_237234, *[TypeError_237235, basinhopping_237236, func2d_237237, subscript_call_result_237242], **kwargs_237245)
        
        
        # Call to assert_raises(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'TypeError' (line 137)
        TypeError_237248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'TypeError', False)
        # Getting the type of 'basinhopping' (line 137)
        basinhopping_237249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'basinhopping', False)
        # Getting the type of 'func2d' (line 137)
        func2d_237250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 47), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 137)
        i_237251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 63), 'i', False)
        # Getting the type of 'self' (line 137)
        self_237252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 55), 'self', False)
        # Obtaining the member 'x0' of a type (line 137)
        x0_237253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 55), self_237252, 'x0')
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___237254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 55), x0_237253, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_237255 = invoke(stypy.reporting.localization.Localization(__file__, 137, 55), getitem___237254, i_237251)
        
        # Processing the call keyword arguments (line 137)
        int_237256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 38), 'int')
        keyword_237257 = int_237256
        kwargs_237258 = {'accept_test': keyword_237257}
        # Getting the type of 'assert_raises' (line 137)
        assert_raises_237247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 137)
        assert_raises_call_result_237259 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert_raises_237247, *[TypeError_237248, basinhopping_237249, func2d_237250, subscript_call_result_237255], **kwargs_237258)
        
        
        # ################# End of 'test_TypeError(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_TypeError' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_237260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237260)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_TypeError'
        return stypy_return_type_237260


    @norecursion
    def test_1d_grad(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1d_grad'
        module_type_store = module_type_store.open_function_context('test_1d_grad', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_1d_grad')
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_1d_grad.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_1d_grad', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1d_grad', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1d_grad(...)' code ##################

        
        # Assigning a Num to a Name (line 142):
        
        # Assigning a Num to a Name (line 142):
        int_237261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 12), 'int')
        # Assigning a type to the variable 'i' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'i', int_237261)
        
        # Assigning a Call to a Name (line 143):
        
        # Assigning a Call to a Name (line 143):
        
        # Call to basinhopping(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'func1d' (line 143)
        func1d_237263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'func1d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 143)
        i_237264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 43), 'i', False)
        # Getting the type of 'self' (line 143)
        self_237265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 35), 'self', False)
        # Obtaining the member 'x0' of a type (line 143)
        x0_237266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 35), self_237265, 'x0')
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___237267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 35), x0_237266, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_237268 = invoke(stypy.reporting.localization.Localization(__file__, 143, 35), getitem___237267, i_237264)
        
        # Processing the call keyword arguments (line 143)
        # Getting the type of 'self' (line 143)
        self_237269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 64), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 143)
        kwargs_237270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 64), self_237269, 'kwargs')
        keyword_237271 = kwargs_237270
        # Getting the type of 'self' (line 144)
        self_237272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'self', False)
        # Obtaining the member 'niter' of a type (line 144)
        niter_237273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 33), self_237272, 'niter')
        keyword_237274 = niter_237273
        # Getting the type of 'self' (line 144)
        self_237275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 50), 'self', False)
        # Obtaining the member 'disp' of a type (line 144)
        disp_237276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 50), self_237275, 'disp')
        keyword_237277 = disp_237276
        kwargs_237278 = {'disp': keyword_237277, 'niter': keyword_237274, 'minimizer_kwargs': keyword_237271}
        # Getting the type of 'basinhopping' (line 143)
        basinhopping_237262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 143)
        basinhopping_call_result_237279 = invoke(stypy.reporting.localization.Localization(__file__, 143, 14), basinhopping_237262, *[func1d_237263, subscript_call_result_237268], **kwargs_237278)
        
        # Assigning a type to the variable 'res' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'res', basinhopping_call_result_237279)
        
        # Call to assert_almost_equal(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'res' (line 145)
        res_237281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 145)
        x_237282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 28), res_237281, 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 145)
        i_237283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 44), 'i', False)
        # Getting the type of 'self' (line 145)
        self_237284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 35), 'self', False)
        # Obtaining the member 'sol' of a type (line 145)
        sol_237285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 35), self_237284, 'sol')
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___237286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 35), sol_237285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_237287 = invoke(stypy.reporting.localization.Localization(__file__, 145, 35), getitem___237286, i_237283)
        
        # Getting the type of 'self' (line 145)
        self_237288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 48), 'self', False)
        # Obtaining the member 'tol' of a type (line 145)
        tol_237289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 48), self_237288, 'tol')
        # Processing the call keyword arguments (line 145)
        kwargs_237290 = {}
        # Getting the type of 'assert_almost_equal' (line 145)
        assert_almost_equal_237280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 145)
        assert_almost_equal_call_result_237291 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), assert_almost_equal_237280, *[x_237282, subscript_call_result_237287, tol_237289], **kwargs_237290)
        
        
        # ################# End of 'test_1d_grad(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1d_grad' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_237292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1d_grad'
        return stypy_return_type_237292


    @norecursion
    def test_2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2d'
        module_type_store = module_type_store.open_function_context('test_2d', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_2d')
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2d(...)' code ##################

        
        # Assigning a Num to a Name (line 149):
        
        # Assigning a Num to a Name (line 149):
        int_237293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'int')
        # Assigning a type to the variable 'i' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'i', int_237293)
        
        # Assigning a Call to a Name (line 150):
        
        # Assigning a Call to a Name (line 150):
        
        # Call to basinhopping(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'func2d' (line 150)
        func2d_237295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 150)
        i_237296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 43), 'i', False)
        # Getting the type of 'self' (line 150)
        self_237297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 35), 'self', False)
        # Obtaining the member 'x0' of a type (line 150)
        x0_237298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 35), self_237297, 'x0')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___237299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 35), x0_237298, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_237300 = invoke(stypy.reporting.localization.Localization(__file__, 150, 35), getitem___237299, i_237296)
        
        # Processing the call keyword arguments (line 150)
        # Getting the type of 'self' (line 150)
        self_237301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 64), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 150)
        kwargs_237302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 64), self_237301, 'kwargs')
        keyword_237303 = kwargs_237302
        # Getting the type of 'self' (line 151)
        self_237304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 33), 'self', False)
        # Obtaining the member 'niter' of a type (line 151)
        niter_237305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 33), self_237304, 'niter')
        keyword_237306 = niter_237305
        # Getting the type of 'self' (line 151)
        self_237307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 50), 'self', False)
        # Obtaining the member 'disp' of a type (line 151)
        disp_237308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 50), self_237307, 'disp')
        keyword_237309 = disp_237308
        kwargs_237310 = {'disp': keyword_237309, 'niter': keyword_237306, 'minimizer_kwargs': keyword_237303}
        # Getting the type of 'basinhopping' (line 150)
        basinhopping_237294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 150)
        basinhopping_call_result_237311 = invoke(stypy.reporting.localization.Localization(__file__, 150, 14), basinhopping_237294, *[func2d_237295, subscript_call_result_237300], **kwargs_237310)
        
        # Assigning a type to the variable 'res' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'res', basinhopping_call_result_237311)
        
        # Call to assert_almost_equal(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'res' (line 152)
        res_237313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 152)
        x_237314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 28), res_237313, 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 152)
        i_237315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 44), 'i', False)
        # Getting the type of 'self' (line 152)
        self_237316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 35), 'self', False)
        # Obtaining the member 'sol' of a type (line 152)
        sol_237317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 35), self_237316, 'sol')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___237318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 35), sol_237317, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_237319 = invoke(stypy.reporting.localization.Localization(__file__, 152, 35), getitem___237318, i_237315)
        
        # Getting the type of 'self' (line 152)
        self_237320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 48), 'self', False)
        # Obtaining the member 'tol' of a type (line 152)
        tol_237321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 48), self_237320, 'tol')
        # Processing the call keyword arguments (line 152)
        kwargs_237322 = {}
        # Getting the type of 'assert_almost_equal' (line 152)
        assert_almost_equal_237312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 152)
        assert_almost_equal_call_result_237323 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), assert_almost_equal_237312, *[x_237314, subscript_call_result_237319, tol_237321], **kwargs_237322)
        
        
        # Call to assert_(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Getting the type of 'res' (line 153)
        res_237325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'res', False)
        # Obtaining the member 'nfev' of a type (line 153)
        nfev_237326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), res_237325, 'nfev')
        int_237327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 27), 'int')
        # Applying the binary operator '>' (line 153)
        result_gt_237328 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 16), '>', nfev_237326, int_237327)
        
        # Processing the call keyword arguments (line 153)
        kwargs_237329 = {}
        # Getting the type of 'assert_' (line 153)
        assert__237324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 153)
        assert__call_result_237330 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert__237324, *[result_gt_237328], **kwargs_237329)
        
        
        # ################# End of 'test_2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2d' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_237331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237331)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2d'
        return stypy_return_type_237331


    @norecursion
    def test_njev(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_njev'
        module_type_store = module_type_store.open_function_context('test_njev', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_njev')
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_njev.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_njev', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_njev', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_njev(...)' code ##################

        
        # Assigning a Num to a Name (line 157):
        
        # Assigning a Num to a Name (line 157):
        int_237332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 12), 'int')
        # Assigning a type to the variable 'i' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'i', int_237332)
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to copy(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_237336 = {}
        # Getting the type of 'self' (line 158)
        self_237333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 158)
        kwargs_237334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 27), self_237333, 'kwargs')
        # Obtaining the member 'copy' of a type (line 158)
        copy_237335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 27), kwargs_237334, 'copy')
        # Calling copy(args, kwargs) (line 158)
        copy_call_result_237337 = invoke(stypy.reporting.localization.Localization(__file__, 158, 27), copy_237335, *[], **kwargs_237336)
        
        # Assigning a type to the variable 'minimizer_kwargs' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'minimizer_kwargs', copy_call_result_237337)
        
        # Assigning a Str to a Subscript (line 160):
        
        # Assigning a Str to a Subscript (line 160):
        str_237338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 37), 'str', 'BFGS')
        # Getting the type of 'minimizer_kwargs' (line 160)
        minimizer_kwargs_237339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'minimizer_kwargs')
        str_237340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'str', 'method')
        # Storing an element on a container (line 160)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 8), minimizer_kwargs_237339, (str_237340, str_237338))
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to basinhopping(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'func2d' (line 161)
        func2d_237342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 161)
        i_237343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 43), 'i', False)
        # Getting the type of 'self' (line 161)
        self_237344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'self', False)
        # Obtaining the member 'x0' of a type (line 161)
        x0_237345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 35), self_237344, 'x0')
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___237346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 35), x0_237345, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_237347 = invoke(stypy.reporting.localization.Localization(__file__, 161, 35), getitem___237346, i_237343)
        
        # Processing the call keyword arguments (line 161)
        # Getting the type of 'minimizer_kwargs' (line 162)
        minimizer_kwargs_237348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 44), 'minimizer_kwargs', False)
        keyword_237349 = minimizer_kwargs_237348
        # Getting the type of 'self' (line 162)
        self_237350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 68), 'self', False)
        # Obtaining the member 'niter' of a type (line 162)
        niter_237351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 68), self_237350, 'niter')
        keyword_237352 = niter_237351
        # Getting the type of 'self' (line 163)
        self_237353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 32), 'self', False)
        # Obtaining the member 'disp' of a type (line 163)
        disp_237354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 32), self_237353, 'disp')
        keyword_237355 = disp_237354
        kwargs_237356 = {'disp': keyword_237355, 'niter': keyword_237352, 'minimizer_kwargs': keyword_237349}
        # Getting the type of 'basinhopping' (line 161)
        basinhopping_237341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 161)
        basinhopping_call_result_237357 = invoke(stypy.reporting.localization.Localization(__file__, 161, 14), basinhopping_237341, *[func2d_237342, subscript_call_result_237347], **kwargs_237356)
        
        # Assigning a type to the variable 'res' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'res', basinhopping_call_result_237357)
        
        # Call to assert_(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Getting the type of 'res' (line 164)
        res_237359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'res', False)
        # Obtaining the member 'nfev' of a type (line 164)
        nfev_237360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), res_237359, 'nfev')
        int_237361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 27), 'int')
        # Applying the binary operator '>' (line 164)
        result_gt_237362 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 16), '>', nfev_237360, int_237361)
        
        # Processing the call keyword arguments (line 164)
        kwargs_237363 = {}
        # Getting the type of 'assert_' (line 164)
        assert__237358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 164)
        assert__call_result_237364 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), assert__237358, *[result_gt_237362], **kwargs_237363)
        
        
        # Call to assert_equal(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'res' (line 165)
        res_237366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'res', False)
        # Obtaining the member 'nfev' of a type (line 165)
        nfev_237367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 21), res_237366, 'nfev')
        # Getting the type of 'res' (line 165)
        res_237368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 31), 'res', False)
        # Obtaining the member 'njev' of a type (line 165)
        njev_237369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 31), res_237368, 'njev')
        # Processing the call keyword arguments (line 165)
        kwargs_237370 = {}
        # Getting the type of 'assert_equal' (line 165)
        assert_equal_237365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 165)
        assert_equal_call_result_237371 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert_equal_237365, *[nfev_237367, njev_237369], **kwargs_237370)
        
        
        # ################# End of 'test_njev(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_njev' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_237372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237372)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_njev'
        return stypy_return_type_237372


    @norecursion
    def test_jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_jac'
        module_type_store = module_type_store.open_function_context('test_jac', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_jac')
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_jac.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_jac', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_jac', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_jac(...)' code ##################

        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to copy(...): (line 169)
        # Processing the call keyword arguments (line 169)
        kwargs_237376 = {}
        # Getting the type of 'self' (line 169)
        self_237373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 27), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 169)
        kwargs_237374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 27), self_237373, 'kwargs')
        # Obtaining the member 'copy' of a type (line 169)
        copy_237375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 27), kwargs_237374, 'copy')
        # Calling copy(args, kwargs) (line 169)
        copy_call_result_237377 = invoke(stypy.reporting.localization.Localization(__file__, 169, 27), copy_237375, *[], **kwargs_237376)
        
        # Assigning a type to the variable 'minimizer_kwargs' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'minimizer_kwargs', copy_call_result_237377)
        
        # Assigning a Str to a Subscript (line 171):
        
        # Assigning a Str to a Subscript (line 171):
        str_237378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 37), 'str', 'BFGS')
        # Getting the type of 'minimizer_kwargs' (line 171)
        minimizer_kwargs_237379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'minimizer_kwargs')
        str_237380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 25), 'str', 'method')
        # Storing an element on a container (line 171)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 8), minimizer_kwargs_237379, (str_237380, str_237378))
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to basinhopping(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'func2d_easyderiv' (line 173)
        func2d_easyderiv_237382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'func2d_easyderiv', False)
        
        # Obtaining an instance of the builtin type 'list' (line 173)
        list_237383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 173)
        # Adding element type (line 173)
        float_237384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 45), list_237383, float_237384)
        # Adding element type (line 173)
        float_237385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 45), list_237383, float_237385)
        
        # Processing the call keyword arguments (line 173)
        # Getting the type of 'minimizer_kwargs' (line 174)
        minimizer_kwargs_237386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 44), 'minimizer_kwargs', False)
        keyword_237387 = minimizer_kwargs_237386
        # Getting the type of 'self' (line 174)
        self_237388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 68), 'self', False)
        # Obtaining the member 'niter' of a type (line 174)
        niter_237389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 68), self_237388, 'niter')
        keyword_237390 = niter_237389
        # Getting the type of 'self' (line 175)
        self_237391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 32), 'self', False)
        # Obtaining the member 'disp' of a type (line 175)
        disp_237392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 32), self_237391, 'disp')
        keyword_237393 = disp_237392
        kwargs_237394 = {'disp': keyword_237393, 'niter': keyword_237390, 'minimizer_kwargs': keyword_237387}
        # Getting the type of 'basinhopping' (line 173)
        basinhopping_237381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 173)
        basinhopping_call_result_237395 = invoke(stypy.reporting.localization.Localization(__file__, 173, 14), basinhopping_237381, *[func2d_easyderiv_237382, list_237383], **kwargs_237394)
        
        # Assigning a type to the variable 'res' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'res', basinhopping_call_result_237395)
        
        # Call to assert_(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to hasattr(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'res' (line 177)
        res_237398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'res', False)
        # Obtaining the member 'lowest_optimization_result' of a type (line 177)
        lowest_optimization_result_237399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 24), res_237398, 'lowest_optimization_result')
        str_237400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 56), 'str', 'jac')
        # Processing the call keyword arguments (line 177)
        kwargs_237401 = {}
        # Getting the type of 'hasattr' (line 177)
        hasattr_237397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 177)
        hasattr_call_result_237402 = invoke(stypy.reporting.localization.Localization(__file__, 177, 16), hasattr_237397, *[lowest_optimization_result_237399, str_237400], **kwargs_237401)
        
        # Processing the call keyword arguments (line 177)
        kwargs_237403 = {}
        # Getting the type of 'assert_' (line 177)
        assert__237396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 177)
        assert__call_result_237404 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assert__237396, *[hasattr_call_result_237402], **kwargs_237403)
        
        
        # Assigning a Call to a Tuple (line 180):
        
        # Assigning a Subscript to a Name (line 180):
        
        # Obtaining the type of the subscript
        int_237405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'int')
        
        # Call to func2d_easyderiv(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'res' (line 180)
        res_237407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'res', False)
        # Obtaining the member 'x' of a type (line 180)
        x_237408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 39), res_237407, 'x')
        # Processing the call keyword arguments (line 180)
        kwargs_237409 = {}
        # Getting the type of 'func2d_easyderiv' (line 180)
        func2d_easyderiv_237406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'func2d_easyderiv', False)
        # Calling func2d_easyderiv(args, kwargs) (line 180)
        func2d_easyderiv_call_result_237410 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), func2d_easyderiv_237406, *[x_237408], **kwargs_237409)
        
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___237411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), func2d_easyderiv_call_result_237410, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_237412 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), getitem___237411, int_237405)
        
        # Assigning a type to the variable 'tuple_var_assignment_236796' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_236796', subscript_call_result_237412)
        
        # Assigning a Subscript to a Name (line 180):
        
        # Obtaining the type of the subscript
        int_237413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'int')
        
        # Call to func2d_easyderiv(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'res' (line 180)
        res_237415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'res', False)
        # Obtaining the member 'x' of a type (line 180)
        x_237416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 39), res_237415, 'x')
        # Processing the call keyword arguments (line 180)
        kwargs_237417 = {}
        # Getting the type of 'func2d_easyderiv' (line 180)
        func2d_easyderiv_237414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'func2d_easyderiv', False)
        # Calling func2d_easyderiv(args, kwargs) (line 180)
        func2d_easyderiv_call_result_237418 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), func2d_easyderiv_237414, *[x_237416], **kwargs_237417)
        
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___237419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), func2d_easyderiv_call_result_237418, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_237420 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), getitem___237419, int_237413)
        
        # Assigning a type to the variable 'tuple_var_assignment_236797' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_236797', subscript_call_result_237420)
        
        # Assigning a Name to a Name (line 180):
        # Getting the type of 'tuple_var_assignment_236796' (line 180)
        tuple_var_assignment_236796_237421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_236796')
        # Assigning a type to the variable '_' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), '_', tuple_var_assignment_236796_237421)
        
        # Assigning a Name to a Name (line 180):
        # Getting the type of 'tuple_var_assignment_236797' (line 180)
        tuple_var_assignment_236797_237422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_236797')
        # Assigning a type to the variable 'jacobian' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'jacobian', tuple_var_assignment_236797_237422)
        
        # Call to assert_almost_equal(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'res' (line 181)
        res_237424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'res', False)
        # Obtaining the member 'lowest_optimization_result' of a type (line 181)
        lowest_optimization_result_237425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), res_237424, 'lowest_optimization_result')
        # Obtaining the member 'jac' of a type (line 181)
        jac_237426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), lowest_optimization_result_237425, 'jac')
        # Getting the type of 'jacobian' (line 181)
        jacobian_237427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 64), 'jacobian', False)
        # Getting the type of 'self' (line 181)
        self_237428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 74), 'self', False)
        # Obtaining the member 'tol' of a type (line 181)
        tol_237429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 74), self_237428, 'tol')
        # Processing the call keyword arguments (line 181)
        kwargs_237430 = {}
        # Getting the type of 'assert_almost_equal' (line 181)
        assert_almost_equal_237423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 181)
        assert_almost_equal_call_result_237431 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assert_almost_equal_237423, *[jac_237426, jacobian_237427, tol_237429], **kwargs_237430)
        
        
        # ################# End of 'test_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_237432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237432)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_jac'
        return stypy_return_type_237432


    @norecursion
    def test_2d_nograd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2d_nograd'
        module_type_store = module_type_store.open_function_context('test_2d_nograd', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_2d_nograd')
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_2d_nograd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_2d_nograd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2d_nograd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2d_nograd(...)' code ##################

        
        # Assigning a Num to a Name (line 185):
        
        # Assigning a Num to a Name (line 185):
        int_237433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 12), 'int')
        # Assigning a type to the variable 'i' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'i', int_237433)
        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to basinhopping(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'func2d_nograd' (line 186)
        func2d_nograd_237435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'func2d_nograd', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 186)
        i_237436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 50), 'i', False)
        # Getting the type of 'self' (line 186)
        self_237437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 42), 'self', False)
        # Obtaining the member 'x0' of a type (line 186)
        x0_237438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 42), self_237437, 'x0')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___237439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 42), x0_237438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_237440 = invoke(stypy.reporting.localization.Localization(__file__, 186, 42), getitem___237439, i_237436)
        
        # Processing the call keyword arguments (line 186)
        # Getting the type of 'self' (line 187)
        self_237441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 44), 'self', False)
        # Obtaining the member 'kwargs_nograd' of a type (line 187)
        kwargs_nograd_237442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 44), self_237441, 'kwargs_nograd')
        keyword_237443 = kwargs_nograd_237442
        # Getting the type of 'self' (line 188)
        self_237444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'self', False)
        # Obtaining the member 'niter' of a type (line 188)
        niter_237445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 33), self_237444, 'niter')
        keyword_237446 = niter_237445
        # Getting the type of 'self' (line 188)
        self_237447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 50), 'self', False)
        # Obtaining the member 'disp' of a type (line 188)
        disp_237448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 50), self_237447, 'disp')
        keyword_237449 = disp_237448
        kwargs_237450 = {'disp': keyword_237449, 'niter': keyword_237446, 'minimizer_kwargs': keyword_237443}
        # Getting the type of 'basinhopping' (line 186)
        basinhopping_237434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 186)
        basinhopping_call_result_237451 = invoke(stypy.reporting.localization.Localization(__file__, 186, 14), basinhopping_237434, *[func2d_nograd_237435, subscript_call_result_237440], **kwargs_237450)
        
        # Assigning a type to the variable 'res' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'res', basinhopping_call_result_237451)
        
        # Call to assert_almost_equal(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'res' (line 189)
        res_237453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 189)
        x_237454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 28), res_237453, 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 189)
        i_237455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'i', False)
        # Getting the type of 'self' (line 189)
        self_237456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 35), 'self', False)
        # Obtaining the member 'sol' of a type (line 189)
        sol_237457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 35), self_237456, 'sol')
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___237458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 35), sol_237457, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 189)
        subscript_call_result_237459 = invoke(stypy.reporting.localization.Localization(__file__, 189, 35), getitem___237458, i_237455)
        
        # Getting the type of 'self' (line 189)
        self_237460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 48), 'self', False)
        # Obtaining the member 'tol' of a type (line 189)
        tol_237461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 48), self_237460, 'tol')
        # Processing the call keyword arguments (line 189)
        kwargs_237462 = {}
        # Getting the type of 'assert_almost_equal' (line 189)
        assert_almost_equal_237452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 189)
        assert_almost_equal_call_result_237463 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assert_almost_equal_237452, *[x_237454, subscript_call_result_237459, tol_237461], **kwargs_237462)
        
        
        # ################# End of 'test_2d_nograd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2d_nograd' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_237464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237464)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2d_nograd'
        return stypy_return_type_237464


    @norecursion
    def test_all_minimizers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_all_minimizers'
        module_type_store = module_type_store.open_function_context('test_all_minimizers', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_all_minimizers')
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_all_minimizers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_all_minimizers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_all_minimizers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_all_minimizers(...)' code ##################

        
        # Assigning a Num to a Name (line 194):
        
        # Assigning a Num to a Name (line 194):
        int_237465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 12), 'int')
        # Assigning a type to the variable 'i' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'i', int_237465)
        
        # Assigning a List to a Name (line 195):
        
        # Assigning a List to a Name (line 195):
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_237466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        str_237467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 19), 'str', 'CG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_237466, str_237467)
        # Adding element type (line 195)
        str_237468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'str', 'BFGS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_237466, str_237468)
        # Adding element type (line 195)
        str_237469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 33), 'str', 'Newton-CG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_237466, str_237469)
        # Adding element type (line 195)
        str_237470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 46), 'str', 'L-BFGS-B')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_237466, str_237470)
        # Adding element type (line 195)
        str_237471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 58), 'str', 'TNC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_237466, str_237471)
        # Adding element type (line 195)
        str_237472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 65), 'str', 'SLSQP')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_237466, str_237472)
        
        # Assigning a type to the variable 'methods' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'methods', list_237466)
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to copy(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'self' (line 196)
        self_237475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 37), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 196)
        kwargs_237476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 37), self_237475, 'kwargs')
        # Processing the call keyword arguments (line 196)
        kwargs_237477 = {}
        # Getting the type of 'copy' (line 196)
        copy_237473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'copy', False)
        # Obtaining the member 'copy' of a type (line 196)
        copy_237474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 27), copy_237473, 'copy')
        # Calling copy(args, kwargs) (line 196)
        copy_call_result_237478 = invoke(stypy.reporting.localization.Localization(__file__, 196, 27), copy_237474, *[kwargs_237476], **kwargs_237477)
        
        # Assigning a type to the variable 'minimizer_kwargs' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'minimizer_kwargs', copy_call_result_237478)
        
        # Getting the type of 'methods' (line 197)
        methods_237479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 22), 'methods')
        # Testing the type of a for loop iterable (line 197)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 197, 8), methods_237479)
        # Getting the type of the for loop variable (line 197)
        for_loop_var_237480 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 197, 8), methods_237479)
        # Assigning a type to the variable 'method' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'method', for_loop_var_237480)
        # SSA begins for a for statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 198):
        
        # Assigning a Name to a Subscript (line 198):
        # Getting the type of 'method' (line 198)
        method_237481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 41), 'method')
        # Getting the type of 'minimizer_kwargs' (line 198)
        minimizer_kwargs_237482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'minimizer_kwargs')
        str_237483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 29), 'str', 'method')
        # Storing an element on a container (line 198)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 12), minimizer_kwargs_237482, (str_237483, method_237481))
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to basinhopping(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'func2d' (line 199)
        func2d_237485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 199)
        i_237486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 47), 'i', False)
        # Getting the type of 'self' (line 199)
        self_237487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'self', False)
        # Obtaining the member 'x0' of a type (line 199)
        x0_237488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 39), self_237487, 'x0')
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___237489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 39), x0_237488, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_237490 = invoke(stypy.reporting.localization.Localization(__file__, 199, 39), getitem___237489, i_237486)
        
        # Processing the call keyword arguments (line 199)
        # Getting the type of 'minimizer_kwargs' (line 200)
        minimizer_kwargs_237491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 48), 'minimizer_kwargs', False)
        keyword_237492 = minimizer_kwargs_237491
        # Getting the type of 'self' (line 201)
        self_237493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 37), 'self', False)
        # Obtaining the member 'niter' of a type (line 201)
        niter_237494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 37), self_237493, 'niter')
        keyword_237495 = niter_237494
        # Getting the type of 'self' (line 201)
        self_237496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 54), 'self', False)
        # Obtaining the member 'disp' of a type (line 201)
        disp_237497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 54), self_237496, 'disp')
        keyword_237498 = disp_237497
        kwargs_237499 = {'disp': keyword_237498, 'niter': keyword_237495, 'minimizer_kwargs': keyword_237492}
        # Getting the type of 'basinhopping' (line 199)
        basinhopping_237484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 199)
        basinhopping_call_result_237500 = invoke(stypy.reporting.localization.Localization(__file__, 199, 18), basinhopping_237484, *[func2d_237485, subscript_call_result_237490], **kwargs_237499)
        
        # Assigning a type to the variable 'res' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'res', basinhopping_call_result_237500)
        
        # Call to assert_almost_equal(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'res' (line 202)
        res_237502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 32), 'res', False)
        # Obtaining the member 'x' of a type (line 202)
        x_237503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 32), res_237502, 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 202)
        i_237504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 48), 'i', False)
        # Getting the type of 'self' (line 202)
        self_237505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 39), 'self', False)
        # Obtaining the member 'sol' of a type (line 202)
        sol_237506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 39), self_237505, 'sol')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___237507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 39), sol_237506, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_237508 = invoke(stypy.reporting.localization.Localization(__file__, 202, 39), getitem___237507, i_237504)
        
        # Getting the type of 'self' (line 202)
        self_237509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 52), 'self', False)
        # Obtaining the member 'tol' of a type (line 202)
        tol_237510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 52), self_237509, 'tol')
        # Processing the call keyword arguments (line 202)
        kwargs_237511 = {}
        # Getting the type of 'assert_almost_equal' (line 202)
        assert_almost_equal_237501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 202)
        assert_almost_equal_call_result_237512 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), assert_almost_equal_237501, *[x_237503, subscript_call_result_237508, tol_237510], **kwargs_237511)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_all_minimizers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_all_minimizers' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_237513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_all_minimizers'
        return stypy_return_type_237513


    @norecursion
    def test_all_nograd_minimizers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_all_nograd_minimizers'
        module_type_store = module_type_store.open_function_context('test_all_nograd_minimizers', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_all_nograd_minimizers')
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_all_nograd_minimizers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_all_nograd_minimizers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_all_nograd_minimizers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_all_nograd_minimizers(...)' code ##################

        
        # Assigning a Num to a Name (line 207):
        
        # Assigning a Num to a Name (line 207):
        int_237514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 12), 'int')
        # Assigning a type to the variable 'i' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'i', int_237514)
        
        # Assigning a List to a Name (line 208):
        
        # Assigning a List to a Name (line 208):
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_237515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        str_237516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'str', 'CG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), list_237515, str_237516)
        # Adding element type (line 208)
        str_237517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 25), 'str', 'BFGS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), list_237515, str_237517)
        # Adding element type (line 208)
        str_237518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 33), 'str', 'L-BFGS-B')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), list_237515, str_237518)
        # Adding element type (line 208)
        str_237519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 45), 'str', 'TNC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), list_237515, str_237519)
        # Adding element type (line 208)
        str_237520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 52), 'str', 'SLSQP')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), list_237515, str_237520)
        # Adding element type (line 208)
        str_237521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 19), 'str', 'Nelder-Mead')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), list_237515, str_237521)
        # Adding element type (line 208)
        str_237522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 34), 'str', 'Powell')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), list_237515, str_237522)
        # Adding element type (line 208)
        str_237523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 44), 'str', 'COBYLA')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 18), list_237515, str_237523)
        
        # Assigning a type to the variable 'methods' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'methods', list_237515)
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to copy(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'self' (line 210)
        self_237526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 37), 'self', False)
        # Obtaining the member 'kwargs_nograd' of a type (line 210)
        kwargs_nograd_237527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 37), self_237526, 'kwargs_nograd')
        # Processing the call keyword arguments (line 210)
        kwargs_237528 = {}
        # Getting the type of 'copy' (line 210)
        copy_237524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'copy', False)
        # Obtaining the member 'copy' of a type (line 210)
        copy_237525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 27), copy_237524, 'copy')
        # Calling copy(args, kwargs) (line 210)
        copy_call_result_237529 = invoke(stypy.reporting.localization.Localization(__file__, 210, 27), copy_237525, *[kwargs_nograd_237527], **kwargs_237528)
        
        # Assigning a type to the variable 'minimizer_kwargs' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'minimizer_kwargs', copy_call_result_237529)
        
        # Getting the type of 'methods' (line 211)
        methods_237530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'methods')
        # Testing the type of a for loop iterable (line 211)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 211, 8), methods_237530)
        # Getting the type of the for loop variable (line 211)
        for_loop_var_237531 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 211, 8), methods_237530)
        # Assigning a type to the variable 'method' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'method', for_loop_var_237531)
        # SSA begins for a for statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 212):
        
        # Assigning a Name to a Subscript (line 212):
        # Getting the type of 'method' (line 212)
        method_237532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'method')
        # Getting the type of 'minimizer_kwargs' (line 212)
        minimizer_kwargs_237533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'minimizer_kwargs')
        str_237534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 29), 'str', 'method')
        # Storing an element on a container (line 212)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 12), minimizer_kwargs_237533, (str_237534, method_237532))
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to basinhopping(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'func2d_nograd' (line 213)
        func2d_nograd_237536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 31), 'func2d_nograd', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 213)
        i_237537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 54), 'i', False)
        # Getting the type of 'self' (line 213)
        self_237538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 46), 'self', False)
        # Obtaining the member 'x0' of a type (line 213)
        x0_237539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 46), self_237538, 'x0')
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___237540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 46), x0_237539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_237541 = invoke(stypy.reporting.localization.Localization(__file__, 213, 46), getitem___237540, i_237537)
        
        # Processing the call keyword arguments (line 213)
        # Getting the type of 'minimizer_kwargs' (line 214)
        minimizer_kwargs_237542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 48), 'minimizer_kwargs', False)
        keyword_237543 = minimizer_kwargs_237542
        # Getting the type of 'self' (line 215)
        self_237544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 37), 'self', False)
        # Obtaining the member 'niter' of a type (line 215)
        niter_237545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 37), self_237544, 'niter')
        keyword_237546 = niter_237545
        # Getting the type of 'self' (line 215)
        self_237547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 54), 'self', False)
        # Obtaining the member 'disp' of a type (line 215)
        disp_237548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 54), self_237547, 'disp')
        keyword_237549 = disp_237548
        kwargs_237550 = {'disp': keyword_237549, 'niter': keyword_237546, 'minimizer_kwargs': keyword_237543}
        # Getting the type of 'basinhopping' (line 213)
        basinhopping_237535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 213)
        basinhopping_call_result_237551 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), basinhopping_237535, *[func2d_nograd_237536, subscript_call_result_237541], **kwargs_237550)
        
        # Assigning a type to the variable 'res' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'res', basinhopping_call_result_237551)
        
        # Assigning a Attribute to a Name (line 216):
        
        # Assigning a Attribute to a Name (line 216):
        # Getting the type of 'self' (line 216)
        self_237552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'self')
        # Obtaining the member 'tol' of a type (line 216)
        tol_237553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 18), self_237552, 'tol')
        # Assigning a type to the variable 'tol' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'tol', tol_237553)
        
        
        # Getting the type of 'method' (line 217)
        method_237554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'method')
        str_237555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 25), 'str', 'COBYLA')
        # Applying the binary operator '==' (line 217)
        result_eq_237556 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 15), '==', method_237554, str_237555)
        
        # Testing the type of an if condition (line 217)
        if_condition_237557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), result_eq_237556)
        # Assigning a type to the variable 'if_condition_237557' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_237557', if_condition_237557)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 218):
        
        # Assigning a Num to a Name (line 218):
        int_237558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'int')
        # Assigning a type to the variable 'tol' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'tol', int_237558)
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_almost_equal(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'res' (line 219)
        res_237560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 32), 'res', False)
        # Obtaining the member 'x' of a type (line 219)
        x_237561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 32), res_237560, 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 219)
        i_237562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 48), 'i', False)
        # Getting the type of 'self' (line 219)
        self_237563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 39), 'self', False)
        # Obtaining the member 'sol' of a type (line 219)
        sol_237564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 39), self_237563, 'sol')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___237565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 39), sol_237564, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_237566 = invoke(stypy.reporting.localization.Localization(__file__, 219, 39), getitem___237565, i_237562)
        
        # Processing the call keyword arguments (line 219)
        # Getting the type of 'tol' (line 219)
        tol_237567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 60), 'tol', False)
        keyword_237568 = tol_237567
        kwargs_237569 = {'decimal': keyword_237568}
        # Getting the type of 'assert_almost_equal' (line 219)
        assert_almost_equal_237559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 219)
        assert_almost_equal_call_result_237570 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), assert_almost_equal_237559, *[x_237561, subscript_call_result_237566], **kwargs_237569)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_all_nograd_minimizers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_all_nograd_minimizers' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_237571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237571)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_all_nograd_minimizers'
        return stypy_return_type_237571


    @norecursion
    def test_pass_takestep(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pass_takestep'
        module_type_store = module_type_store.open_function_context('test_pass_takestep', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_pass_takestep')
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_pass_takestep.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_pass_takestep', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pass_takestep', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pass_takestep(...)' code ##################

        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to MyTakeStep1(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_237573 = {}
        # Getting the type of 'MyTakeStep1' (line 224)
        MyTakeStep1_237572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'MyTakeStep1', False)
        # Calling MyTakeStep1(args, kwargs) (line 224)
        MyTakeStep1_call_result_237574 = invoke(stypy.reporting.localization.Localization(__file__, 224, 19), MyTakeStep1_237572, *[], **kwargs_237573)
        
        # Assigning a type to the variable 'takestep' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'takestep', MyTakeStep1_call_result_237574)
        
        # Assigning a Attribute to a Name (line 225):
        
        # Assigning a Attribute to a Name (line 225):
        # Getting the type of 'takestep' (line 225)
        takestep_237575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'takestep')
        # Obtaining the member 'stepsize' of a type (line 225)
        stepsize_237576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), takestep_237575, 'stepsize')
        # Assigning a type to the variable 'initial_step_size' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'initial_step_size', stepsize_237576)
        
        # Assigning a Num to a Name (line 226):
        
        # Assigning a Num to a Name (line 226):
        int_237577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 12), 'int')
        # Assigning a type to the variable 'i' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'i', int_237577)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to basinhopping(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'func2d' (line 227)
        func2d_237579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 227)
        i_237580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 43), 'i', False)
        # Getting the type of 'self' (line 227)
        self_237581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 35), 'self', False)
        # Obtaining the member 'x0' of a type (line 227)
        x0_237582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 35), self_237581, 'x0')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___237583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 35), x0_237582, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_237584 = invoke(stypy.reporting.localization.Localization(__file__, 227, 35), getitem___237583, i_237580)
        
        # Processing the call keyword arguments (line 227)
        # Getting the type of 'self' (line 227)
        self_237585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 64), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 227)
        kwargs_237586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 64), self_237585, 'kwargs')
        keyword_237587 = kwargs_237586
        # Getting the type of 'self' (line 228)
        self_237588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 33), 'self', False)
        # Obtaining the member 'niter' of a type (line 228)
        niter_237589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 33), self_237588, 'niter')
        keyword_237590 = niter_237589
        # Getting the type of 'self' (line 228)
        self_237591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'self', False)
        # Obtaining the member 'disp' of a type (line 228)
        disp_237592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 50), self_237591, 'disp')
        keyword_237593 = disp_237592
        # Getting the type of 'takestep' (line 229)
        takestep_237594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 37), 'takestep', False)
        keyword_237595 = takestep_237594
        kwargs_237596 = {'disp': keyword_237593, 'niter': keyword_237590, 'take_step': keyword_237595, 'minimizer_kwargs': keyword_237587}
        # Getting the type of 'basinhopping' (line 227)
        basinhopping_237578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 227)
        basinhopping_call_result_237597 = invoke(stypy.reporting.localization.Localization(__file__, 227, 14), basinhopping_237578, *[func2d_237579, subscript_call_result_237584], **kwargs_237596)
        
        # Assigning a type to the variable 'res' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'res', basinhopping_call_result_237597)
        
        # Call to assert_almost_equal(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'res' (line 230)
        res_237599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 230)
        x_237600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 28), res_237599, 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 230)
        i_237601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 44), 'i', False)
        # Getting the type of 'self' (line 230)
        self_237602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'self', False)
        # Obtaining the member 'sol' of a type (line 230)
        sol_237603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 35), self_237602, 'sol')
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___237604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 35), sol_237603, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_237605 = invoke(stypy.reporting.localization.Localization(__file__, 230, 35), getitem___237604, i_237601)
        
        # Getting the type of 'self' (line 230)
        self_237606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'self', False)
        # Obtaining the member 'tol' of a type (line 230)
        tol_237607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 48), self_237606, 'tol')
        # Processing the call keyword arguments (line 230)
        kwargs_237608 = {}
        # Getting the type of 'assert_almost_equal' (line 230)
        assert_almost_equal_237598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 230)
        assert_almost_equal_call_result_237609 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assert_almost_equal_237598, *[x_237600, subscript_call_result_237605, tol_237607], **kwargs_237608)
        
        
        # Call to assert_(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'takestep' (line 231)
        takestep_237611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'takestep', False)
        # Obtaining the member 'been_called' of a type (line 231)
        been_called_237612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 16), takestep_237611, 'been_called')
        # Processing the call keyword arguments (line 231)
        kwargs_237613 = {}
        # Getting the type of 'assert_' (line 231)
        assert__237610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 231)
        assert__call_result_237614 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), assert__237610, *[been_called_237612], **kwargs_237613)
        
        
        # Call to assert_(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Getting the type of 'initial_step_size' (line 233)
        initial_step_size_237616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'initial_step_size', False)
        # Getting the type of 'takestep' (line 233)
        takestep_237617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 37), 'takestep', False)
        # Obtaining the member 'stepsize' of a type (line 233)
        stepsize_237618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 37), takestep_237617, 'stepsize')
        # Applying the binary operator '!=' (line 233)
        result_ne_237619 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 16), '!=', initial_step_size_237616, stepsize_237618)
        
        # Processing the call keyword arguments (line 233)
        kwargs_237620 = {}
        # Getting the type of 'assert_' (line 233)
        assert__237615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 233)
        assert__call_result_237621 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assert__237615, *[result_ne_237619], **kwargs_237620)
        
        
        # ################# End of 'test_pass_takestep(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pass_takestep' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_237622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pass_takestep'
        return stypy_return_type_237622


    @norecursion
    def test_pass_simple_takestep(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pass_simple_takestep'
        module_type_store = module_type_store.open_function_context('test_pass_simple_takestep', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_pass_simple_takestep')
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_pass_simple_takestep.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_pass_simple_takestep', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pass_simple_takestep', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pass_simple_takestep(...)' code ##################

        
        # Assigning a Name to a Name (line 237):
        
        # Assigning a Name to a Name (line 237):
        # Getting the type of 'myTakeStep2' (line 237)
        myTakeStep2_237623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'myTakeStep2')
        # Assigning a type to the variable 'takestep' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'takestep', myTakeStep2_237623)
        
        # Assigning a Num to a Name (line 238):
        
        # Assigning a Num to a Name (line 238):
        int_237624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'int')
        # Assigning a type to the variable 'i' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'i', int_237624)
        
        # Assigning a Call to a Name (line 239):
        
        # Assigning a Call to a Name (line 239):
        
        # Call to basinhopping(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'func2d_nograd' (line 239)
        func2d_nograd_237626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'func2d_nograd', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 239)
        i_237627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 50), 'i', False)
        # Getting the type of 'self' (line 239)
        self_237628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 42), 'self', False)
        # Obtaining the member 'x0' of a type (line 239)
        x0_237629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 42), self_237628, 'x0')
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___237630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 42), x0_237629, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_237631 = invoke(stypy.reporting.localization.Localization(__file__, 239, 42), getitem___237630, i_237627)
        
        # Processing the call keyword arguments (line 239)
        # Getting the type of 'self' (line 240)
        self_237632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'self', False)
        # Obtaining the member 'kwargs_nograd' of a type (line 240)
        kwargs_nograd_237633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 44), self_237632, 'kwargs_nograd')
        keyword_237634 = kwargs_nograd_237633
        # Getting the type of 'self' (line 241)
        self_237635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 33), 'self', False)
        # Obtaining the member 'niter' of a type (line 241)
        niter_237636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 33), self_237635, 'niter')
        keyword_237637 = niter_237636
        # Getting the type of 'self' (line 241)
        self_237638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 50), 'self', False)
        # Obtaining the member 'disp' of a type (line 241)
        disp_237639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 50), self_237638, 'disp')
        keyword_237640 = disp_237639
        # Getting the type of 'takestep' (line 242)
        takestep_237641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 37), 'takestep', False)
        keyword_237642 = takestep_237641
        kwargs_237643 = {'disp': keyword_237640, 'niter': keyword_237637, 'take_step': keyword_237642, 'minimizer_kwargs': keyword_237634}
        # Getting the type of 'basinhopping' (line 239)
        basinhopping_237625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 239)
        basinhopping_call_result_237644 = invoke(stypy.reporting.localization.Localization(__file__, 239, 14), basinhopping_237625, *[func2d_nograd_237626, subscript_call_result_237631], **kwargs_237643)
        
        # Assigning a type to the variable 'res' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'res', basinhopping_call_result_237644)
        
        # Call to assert_almost_equal(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'res' (line 243)
        res_237646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 243)
        x_237647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 28), res_237646, 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 243)
        i_237648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 44), 'i', False)
        # Getting the type of 'self' (line 243)
        self_237649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 35), 'self', False)
        # Obtaining the member 'sol' of a type (line 243)
        sol_237650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 35), self_237649, 'sol')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___237651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 35), sol_237650, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_237652 = invoke(stypy.reporting.localization.Localization(__file__, 243, 35), getitem___237651, i_237648)
        
        # Getting the type of 'self' (line 243)
        self_237653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 48), 'self', False)
        # Obtaining the member 'tol' of a type (line 243)
        tol_237654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 48), self_237653, 'tol')
        # Processing the call keyword arguments (line 243)
        kwargs_237655 = {}
        # Getting the type of 'assert_almost_equal' (line 243)
        assert_almost_equal_237645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 243)
        assert_almost_equal_call_result_237656 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), assert_almost_equal_237645, *[x_237647, subscript_call_result_237652, tol_237654], **kwargs_237655)
        
        
        # ################# End of 'test_pass_simple_takestep(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pass_simple_takestep' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_237657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237657)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pass_simple_takestep'
        return stypy_return_type_237657


    @norecursion
    def test_pass_accept_test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pass_accept_test'
        module_type_store = module_type_store.open_function_context('test_pass_accept_test', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_pass_accept_test')
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_pass_accept_test.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_pass_accept_test', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pass_accept_test', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pass_accept_test(...)' code ##################

        
        # Assigning a Call to a Name (line 249):
        
        # Assigning a Call to a Name (line 249):
        
        # Call to MyAcceptTest(...): (line 249)
        # Processing the call keyword arguments (line 249)
        kwargs_237659 = {}
        # Getting the type of 'MyAcceptTest' (line 249)
        MyAcceptTest_237658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 22), 'MyAcceptTest', False)
        # Calling MyAcceptTest(args, kwargs) (line 249)
        MyAcceptTest_call_result_237660 = invoke(stypy.reporting.localization.Localization(__file__, 249, 22), MyAcceptTest_237658, *[], **kwargs_237659)
        
        # Assigning a type to the variable 'accept_test' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'accept_test', MyAcceptTest_call_result_237660)
        
        # Assigning a Num to a Name (line 250):
        
        # Assigning a Num to a Name (line 250):
        int_237661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 12), 'int')
        # Assigning a type to the variable 'i' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'i', int_237661)
        
        # Call to basinhopping(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'func2d' (line 252)
        func2d_237663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 21), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 252)
        i_237664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 37), 'i', False)
        # Getting the type of 'self' (line 252)
        self_237665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'self', False)
        # Obtaining the member 'x0' of a type (line 252)
        x0_237666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 29), self_237665, 'x0')
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___237667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 29), x0_237666, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_237668 = invoke(stypy.reporting.localization.Localization(__file__, 252, 29), getitem___237667, i_237664)
        
        # Processing the call keyword arguments (line 252)
        # Getting the type of 'self' (line 252)
        self_237669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 58), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 252)
        kwargs_237670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 58), self_237669, 'kwargs')
        keyword_237671 = kwargs_237670
        int_237672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 27), 'int')
        keyword_237673 = int_237672
        # Getting the type of 'self' (line 253)
        self_237674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 36), 'self', False)
        # Obtaining the member 'disp' of a type (line 253)
        disp_237675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 36), self_237674, 'disp')
        keyword_237676 = disp_237675
        # Getting the type of 'accept_test' (line 253)
        accept_test_237677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 59), 'accept_test', False)
        keyword_237678 = accept_test_237677
        kwargs_237679 = {'disp': keyword_237676, 'niter': keyword_237673, 'accept_test': keyword_237678, 'minimizer_kwargs': keyword_237671}
        # Getting the type of 'basinhopping' (line 252)
        basinhopping_237662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 252)
        basinhopping_call_result_237680 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), basinhopping_237662, *[func2d_237663, subscript_call_result_237668], **kwargs_237679)
        
        
        # Call to assert_(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'accept_test' (line 254)
        accept_test_237682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'accept_test', False)
        # Obtaining the member 'been_called' of a type (line 254)
        been_called_237683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 16), accept_test_237682, 'been_called')
        # Processing the call keyword arguments (line 254)
        kwargs_237684 = {}
        # Getting the type of 'assert_' (line 254)
        assert__237681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 254)
        assert__call_result_237685 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), assert__237681, *[been_called_237683], **kwargs_237684)
        
        
        # ################# End of 'test_pass_accept_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pass_accept_test' in the type store
        # Getting the type of 'stypy_return_type' (line 245)
        stypy_return_type_237686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237686)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pass_accept_test'
        return stypy_return_type_237686


    @norecursion
    def test_pass_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pass_callback'
        module_type_store = module_type_store.open_function_context('test_pass_callback', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_pass_callback')
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_pass_callback.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_pass_callback', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pass_callback', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pass_callback(...)' code ##################

        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to MyCallBack(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_237688 = {}
        # Getting the type of 'MyCallBack' (line 260)
        MyCallBack_237687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 19), 'MyCallBack', False)
        # Calling MyCallBack(args, kwargs) (line 260)
        MyCallBack_call_result_237689 = invoke(stypy.reporting.localization.Localization(__file__, 260, 19), MyCallBack_237687, *[], **kwargs_237688)
        
        # Assigning a type to the variable 'callback' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'callback', MyCallBack_call_result_237689)
        
        # Assigning a Num to a Name (line 261):
        
        # Assigning a Num to a Name (line 261):
        int_237690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 12), 'int')
        # Assigning a type to the variable 'i' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'i', int_237690)
        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to basinhopping(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'func2d' (line 263)
        func2d_237692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 263)
        i_237693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 43), 'i', False)
        # Getting the type of 'self' (line 263)
        self_237694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 35), 'self', False)
        # Obtaining the member 'x0' of a type (line 263)
        x0_237695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 35), self_237694, 'x0')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___237696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 35), x0_237695, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_237697 = invoke(stypy.reporting.localization.Localization(__file__, 263, 35), getitem___237696, i_237693)
        
        # Processing the call keyword arguments (line 263)
        # Getting the type of 'self' (line 263)
        self_237698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 64), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 263)
        kwargs_237699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 64), self_237698, 'kwargs')
        keyword_237700 = kwargs_237699
        int_237701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 33), 'int')
        keyword_237702 = int_237701
        # Getting the type of 'self' (line 264)
        self_237703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 42), 'self', False)
        # Obtaining the member 'disp' of a type (line 264)
        disp_237704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 42), self_237703, 'disp')
        keyword_237705 = disp_237704
        # Getting the type of 'callback' (line 264)
        callback_237706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 62), 'callback', False)
        keyword_237707 = callback_237706
        kwargs_237708 = {'disp': keyword_237705, 'niter': keyword_237702, 'minimizer_kwargs': keyword_237700, 'callback': keyword_237707}
        # Getting the type of 'basinhopping' (line 263)
        basinhopping_237691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 263)
        basinhopping_call_result_237709 = invoke(stypy.reporting.localization.Localization(__file__, 263, 14), basinhopping_237691, *[func2d_237692, subscript_call_result_237697], **kwargs_237708)
        
        # Assigning a type to the variable 'res' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'res', basinhopping_call_result_237709)
        
        # Call to assert_(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'callback' (line 265)
        callback_237711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'callback', False)
        # Obtaining the member 'been_called' of a type (line 265)
        been_called_237712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 16), callback_237711, 'been_called')
        # Processing the call keyword arguments (line 265)
        kwargs_237713 = {}
        # Getting the type of 'assert_' (line 265)
        assert__237710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 265)
        assert__call_result_237714 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), assert__237710, *[been_called_237712], **kwargs_237713)
        
        
        # Call to assert_(...): (line 266)
        # Processing the call arguments (line 266)
        
        str_237716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'str', 'callback')
        
        # Obtaining the type of the subscript
        int_237717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 42), 'int')
        # Getting the type of 'res' (line 266)
        res_237718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 30), 'res', False)
        # Obtaining the member 'message' of a type (line 266)
        message_237719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 30), res_237718, 'message')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___237720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 30), message_237719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_237721 = invoke(stypy.reporting.localization.Localization(__file__, 266, 30), getitem___237720, int_237717)
        
        # Applying the binary operator 'in' (line 266)
        result_contains_237722 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 16), 'in', str_237716, subscript_call_result_237721)
        
        # Processing the call keyword arguments (line 266)
        kwargs_237723 = {}
        # Getting the type of 'assert_' (line 266)
        assert__237715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 266)
        assert__call_result_237724 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), assert__237715, *[result_contains_237722], **kwargs_237723)
        
        
        # Call to assert_equal(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'res' (line 267)
        res_237726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'res', False)
        # Obtaining the member 'nit' of a type (line 267)
        nit_237727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 21), res_237726, 'nit')
        int_237728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 30), 'int')
        # Processing the call keyword arguments (line 267)
        kwargs_237729 = {}
        # Getting the type of 'assert_equal' (line 267)
        assert_equal_237725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 267)
        assert_equal_call_result_237730 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), assert_equal_237725, *[nit_237727, int_237728], **kwargs_237729)
        
        
        # ################# End of 'test_pass_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pass_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_237731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237731)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pass_callback'
        return stypy_return_type_237731


    @norecursion
    def test_minimizer_fail(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimizer_fail'
        module_type_store = module_type_store.open_function_context('test_minimizer_fail', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_minimizer_fail')
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_minimizer_fail.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_minimizer_fail', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimizer_fail', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimizer_fail(...)' code ##################

        
        # Assigning a Num to a Name (line 271):
        
        # Assigning a Num to a Name (line 271):
        int_237732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 12), 'int')
        # Assigning a type to the variable 'i' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'i', int_237732)
        
        # Assigning a Call to a Subscript (line 272):
        
        # Assigning a Call to a Subscript (line 272):
        
        # Call to dict(...): (line 272)
        # Processing the call keyword arguments (line 272)
        int_237734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 46), 'int')
        keyword_237735 = int_237734
        kwargs_237736 = {'maxiter': keyword_237735}
        # Getting the type of 'dict' (line 272)
        dict_237733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 33), 'dict', False)
        # Calling dict(args, kwargs) (line 272)
        dict_call_result_237737 = invoke(stypy.reporting.localization.Localization(__file__, 272, 33), dict_237733, *[], **kwargs_237736)
        
        # Getting the type of 'self' (line 272)
        self_237738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Obtaining the member 'kwargs' of a type (line 272)
        kwargs_237739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_237738, 'kwargs')
        str_237740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 20), 'str', 'options')
        # Storing an element on a container (line 272)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 8), kwargs_237739, (str_237740, dict_call_result_237737))
        
        # Assigning a Num to a Attribute (line 273):
        
        # Assigning a Num to a Attribute (line 273):
        int_237741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 21), 'int')
        # Getting the type of 'self' (line 273)
        self_237742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self')
        # Setting the type of the member 'niter' of a type (line 273)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_237742, 'niter', int_237741)
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to basinhopping(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'func2d' (line 274)
        func2d_237744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 27), 'func2d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 274)
        i_237745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 43), 'i', False)
        # Getting the type of 'self' (line 274)
        self_237746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'self', False)
        # Obtaining the member 'x0' of a type (line 274)
        x0_237747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 35), self_237746, 'x0')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___237748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 35), x0_237747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_237749 = invoke(stypy.reporting.localization.Localization(__file__, 274, 35), getitem___237748, i_237745)
        
        # Processing the call keyword arguments (line 274)
        # Getting the type of 'self' (line 274)
        self_237750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 64), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 274)
        kwargs_237751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 64), self_237750, 'kwargs')
        keyword_237752 = kwargs_237751
        # Getting the type of 'self' (line 275)
        self_237753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 33), 'self', False)
        # Obtaining the member 'niter' of a type (line 275)
        niter_237754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 33), self_237753, 'niter')
        keyword_237755 = niter_237754
        # Getting the type of 'self' (line 275)
        self_237756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 50), 'self', False)
        # Obtaining the member 'disp' of a type (line 275)
        disp_237757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 50), self_237756, 'disp')
        keyword_237758 = disp_237757
        kwargs_237759 = {'disp': keyword_237758, 'niter': keyword_237755, 'minimizer_kwargs': keyword_237752}
        # Getting the type of 'basinhopping' (line 274)
        basinhopping_237743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 274)
        basinhopping_call_result_237760 = invoke(stypy.reporting.localization.Localization(__file__, 274, 14), basinhopping_237743, *[func2d_237744, subscript_call_result_237749], **kwargs_237759)
        
        # Assigning a type to the variable 'res' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'res', basinhopping_call_result_237760)
        
        # Call to assert_equal(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'res' (line 278)
        res_237762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'res', False)
        # Obtaining the member 'nit' of a type (line 278)
        nit_237763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 21), res_237762, 'nit')
        int_237764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'int')
        # Applying the binary operator '+' (line 278)
        result_add_237765 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 21), '+', nit_237763, int_237764)
        
        # Getting the type of 'res' (line 278)
        res_237766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 34), 'res', False)
        # Obtaining the member 'minimization_failures' of a type (line 278)
        minimization_failures_237767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 34), res_237766, 'minimization_failures')
        # Processing the call keyword arguments (line 278)
        kwargs_237768 = {}
        # Getting the type of 'assert_equal' (line 278)
        assert_equal_237761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 278)
        assert_equal_call_result_237769 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), assert_equal_237761, *[result_add_237765, minimization_failures_237767], **kwargs_237768)
        
        
        # ################# End of 'test_minimizer_fail(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimizer_fail' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_237770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimizer_fail'
        return stypy_return_type_237770


    @norecursion
    def test_niter_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_niter_zero'
        module_type_store = module_type_store.open_function_context('test_niter_zero', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_niter_zero')
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_niter_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_niter_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_niter_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_niter_zero(...)' code ##################

        
        # Assigning a Num to a Name (line 282):
        
        # Assigning a Num to a Name (line 282):
        int_237771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 12), 'int')
        # Assigning a type to the variable 'i' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'i', int_237771)
        
        # Assigning a Call to a Name (line 283):
        
        # Assigning a Call to a Name (line 283):
        
        # Call to basinhopping(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'func1d' (line 283)
        func1d_237773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'func1d', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 283)
        i_237774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 43), 'i', False)
        # Getting the type of 'self' (line 283)
        self_237775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 35), 'self', False)
        # Obtaining the member 'x0' of a type (line 283)
        x0_237776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 35), self_237775, 'x0')
        # Obtaining the member '__getitem__' of a type (line 283)
        getitem___237777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 35), x0_237776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 283)
        subscript_call_result_237778 = invoke(stypy.reporting.localization.Localization(__file__, 283, 35), getitem___237777, i_237774)
        
        # Processing the call keyword arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_237779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 64), 'self', False)
        # Obtaining the member 'kwargs' of a type (line 283)
        kwargs_237780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 64), self_237779, 'kwargs')
        keyword_237781 = kwargs_237780
        int_237782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 33), 'int')
        keyword_237783 = int_237782
        # Getting the type of 'self' (line 284)
        self_237784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 41), 'self', False)
        # Obtaining the member 'disp' of a type (line 284)
        disp_237785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 41), self_237784, 'disp')
        keyword_237786 = disp_237785
        kwargs_237787 = {'disp': keyword_237786, 'niter': keyword_237783, 'minimizer_kwargs': keyword_237781}
        # Getting the type of 'basinhopping' (line 283)
        basinhopping_237772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 14), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 283)
        basinhopping_call_result_237788 = invoke(stypy.reporting.localization.Localization(__file__, 283, 14), basinhopping_237772, *[func1d_237773, subscript_call_result_237778], **kwargs_237787)
        
        # Assigning a type to the variable 'res' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'res', basinhopping_call_result_237788)
        
        # ################# End of 'test_niter_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_niter_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_237789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_niter_zero'
        return stypy_return_type_237789


    @norecursion
    def test_seed_reproducibility(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_seed_reproducibility'
        module_type_store = module_type_store.open_function_context('test_seed_reproducibility', 286, 4, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_localization', localization)
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_function_name', 'TestBasinHopping.test_seed_reproducibility')
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_param_names_list', [])
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBasinHopping.test_seed_reproducibility.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.test_seed_reproducibility', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_seed_reproducibility', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_seed_reproducibility(...)' code ##################

        
        # Assigning a Dict to a Name (line 288):
        
        # Assigning a Dict to a Name (line 288):
        
        # Obtaining an instance of the builtin type 'dict' (line 288)
        dict_237790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 288)
        # Adding element type (key, value) (line 288)
        str_237791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 28), 'str', 'method')
        str_237792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 38), 'str', 'L-BFGS-B')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 27), dict_237790, (str_237791, str_237792))
        # Adding element type (key, value) (line 288)
        str_237793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 50), 'str', 'jac')
        # Getting the type of 'True' (line 288)
        True_237794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 57), 'True')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 27), dict_237790, (str_237793, True_237794))
        
        # Assigning a type to the variable 'minimizer_kwargs' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'minimizer_kwargs', dict_237790)
        
        # Assigning a List to a Name (line 290):
        
        # Assigning a List to a Name (line 290):
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_237795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        
        # Assigning a type to the variable 'f_1' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'f_1', list_237795)

        @norecursion
        def callback(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'callback'
            module_type_store = module_type_store.open_function_context('callback', 292, 8, False)
            
            # Passed parameters checking function
            callback.stypy_localization = localization
            callback.stypy_type_of_self = None
            callback.stypy_type_store = module_type_store
            callback.stypy_function_name = 'callback'
            callback.stypy_param_names_list = ['x', 'f', 'accepted']
            callback.stypy_varargs_param_name = None
            callback.stypy_kwargs_param_name = None
            callback.stypy_call_defaults = defaults
            callback.stypy_call_varargs = varargs
            callback.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'callback', ['x', 'f', 'accepted'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'callback', localization, ['x', 'f', 'accepted'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'callback(...)' code ##################

            
            # Call to append(...): (line 293)
            # Processing the call arguments (line 293)
            # Getting the type of 'f' (line 293)
            f_237798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 23), 'f', False)
            # Processing the call keyword arguments (line 293)
            kwargs_237799 = {}
            # Getting the type of 'f_1' (line 293)
            f_1_237796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'f_1', False)
            # Obtaining the member 'append' of a type (line 293)
            append_237797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), f_1_237796, 'append')
            # Calling append(args, kwargs) (line 293)
            append_call_result_237800 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), append_237797, *[f_237798], **kwargs_237799)
            
            
            # ################# End of 'callback(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'callback' in the type store
            # Getting the type of 'stypy_return_type' (line 292)
            stypy_return_type_237801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_237801)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'callback'
            return stypy_return_type_237801

        # Assigning a type to the variable 'callback' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'callback', callback)
        
        # Call to basinhopping(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'func2d' (line 295)
        func2d_237803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'func2d', False)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_237804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        float_237805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 29), list_237804, float_237805)
        # Adding element type (line 295)
        float_237806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 29), list_237804, float_237806)
        
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'minimizer_kwargs' (line 295)
        minimizer_kwargs_237807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 58), 'minimizer_kwargs', False)
        keyword_237808 = minimizer_kwargs_237807
        int_237809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 27), 'int')
        keyword_237810 = int_237809
        # Getting the type of 'callback' (line 296)
        callback_237811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 40), 'callback', False)
        keyword_237812 = callback_237811
        int_237813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 55), 'int')
        keyword_237814 = int_237813
        kwargs_237815 = {'niter': keyword_237810, 'seed': keyword_237814, 'minimizer_kwargs': keyword_237808, 'callback': keyword_237812}
        # Getting the type of 'basinhopping' (line 295)
        basinhopping_237802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 295)
        basinhopping_call_result_237816 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), basinhopping_237802, *[func2d_237803, list_237804], **kwargs_237815)
        
        
        # Assigning a List to a Name (line 298):
        
        # Assigning a List to a Name (line 298):
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_237817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        
        # Assigning a type to the variable 'f_2' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'f_2', list_237817)

        @norecursion
        def callback2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'callback2'
            module_type_store = module_type_store.open_function_context('callback2', 300, 8, False)
            
            # Passed parameters checking function
            callback2.stypy_localization = localization
            callback2.stypy_type_of_self = None
            callback2.stypy_type_store = module_type_store
            callback2.stypy_function_name = 'callback2'
            callback2.stypy_param_names_list = ['x', 'f', 'accepted']
            callback2.stypy_varargs_param_name = None
            callback2.stypy_kwargs_param_name = None
            callback2.stypy_call_defaults = defaults
            callback2.stypy_call_varargs = varargs
            callback2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'callback2', ['x', 'f', 'accepted'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'callback2', localization, ['x', 'f', 'accepted'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'callback2(...)' code ##################

            
            # Call to append(...): (line 301)
            # Processing the call arguments (line 301)
            # Getting the type of 'f' (line 301)
            f_237820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'f', False)
            # Processing the call keyword arguments (line 301)
            kwargs_237821 = {}
            # Getting the type of 'f_2' (line 301)
            f_2_237818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'f_2', False)
            # Obtaining the member 'append' of a type (line 301)
            append_237819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), f_2_237818, 'append')
            # Calling append(args, kwargs) (line 301)
            append_call_result_237822 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), append_237819, *[f_237820], **kwargs_237821)
            
            
            # ################# End of 'callback2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'callback2' in the type store
            # Getting the type of 'stypy_return_type' (line 300)
            stypy_return_type_237823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_237823)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'callback2'
            return stypy_return_type_237823

        # Assigning a type to the variable 'callback2' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'callback2', callback2)
        
        # Call to basinhopping(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'func2d' (line 303)
        func2d_237825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 21), 'func2d', False)
        
        # Obtaining an instance of the builtin type 'list' (line 303)
        list_237826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 303)
        # Adding element type (line 303)
        float_237827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 29), list_237826, float_237827)
        # Adding element type (line 303)
        float_237828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 29), list_237826, float_237828)
        
        # Processing the call keyword arguments (line 303)
        # Getting the type of 'minimizer_kwargs' (line 303)
        minimizer_kwargs_237829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 58), 'minimizer_kwargs', False)
        keyword_237830 = minimizer_kwargs_237829
        int_237831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 27), 'int')
        keyword_237832 = int_237831
        # Getting the type of 'callback2' (line 304)
        callback2_237833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 40), 'callback2', False)
        keyword_237834 = callback2_237833
        int_237835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 56), 'int')
        keyword_237836 = int_237835
        kwargs_237837 = {'niter': keyword_237832, 'seed': keyword_237836, 'minimizer_kwargs': keyword_237830, 'callback': keyword_237834}
        # Getting the type of 'basinhopping' (line 303)
        basinhopping_237824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'basinhopping', False)
        # Calling basinhopping(args, kwargs) (line 303)
        basinhopping_call_result_237838 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), basinhopping_237824, *[func2d_237825, list_237826], **kwargs_237837)
        
        
        # Call to assert_equal(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Call to array(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'f_1' (line 305)
        f_1_237842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 30), 'f_1', False)
        # Processing the call keyword arguments (line 305)
        kwargs_237843 = {}
        # Getting the type of 'np' (line 305)
        np_237840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 305)
        array_237841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 21), np_237840, 'array')
        # Calling array(args, kwargs) (line 305)
        array_call_result_237844 = invoke(stypy.reporting.localization.Localization(__file__, 305, 21), array_237841, *[f_1_237842], **kwargs_237843)
        
        
        # Call to array(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'f_2' (line 305)
        f_2_237847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 45), 'f_2', False)
        # Processing the call keyword arguments (line 305)
        kwargs_237848 = {}
        # Getting the type of 'np' (line 305)
        np_237845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 36), 'np', False)
        # Obtaining the member 'array' of a type (line 305)
        array_237846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 36), np_237845, 'array')
        # Calling array(args, kwargs) (line 305)
        array_call_result_237849 = invoke(stypy.reporting.localization.Localization(__file__, 305, 36), array_237846, *[f_2_237847], **kwargs_237848)
        
        # Processing the call keyword arguments (line 305)
        kwargs_237850 = {}
        # Getting the type of 'assert_equal' (line 305)
        assert_equal_237839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 305)
        assert_equal_call_result_237851 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), assert_equal_237839, *[array_call_result_237844, array_call_result_237849], **kwargs_237850)
        
        
        # ################# End of 'test_seed_reproducibility(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_seed_reproducibility' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_237852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237852)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_seed_reproducibility'
        return stypy_return_type_237852


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 109, 0, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBasinHopping.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBasinHopping' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'TestBasinHopping', TestBasinHopping)
# Declaration of the 'Test_Storage' class

class Test_Storage(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Storage.setup_method.__dict__.__setitem__('stypy_localization', localization)
        Test_Storage.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Storage.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Storage.setup_method.__dict__.__setitem__('stypy_function_name', 'Test_Storage.setup_method')
        Test_Storage.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Storage.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Storage.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Storage.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Storage.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Storage.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Storage.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Storage.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 310):
        
        # Assigning a Call to a Attribute (line 310):
        
        # Call to array(...): (line 310)
        # Processing the call arguments (line 310)
        int_237855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 27), 'int')
        # Processing the call keyword arguments (line 310)
        kwargs_237856 = {}
        # Getting the type of 'np' (line 310)
        np_237853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 310)
        array_237854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 18), np_237853, 'array')
        # Calling array(args, kwargs) (line 310)
        array_call_result_237857 = invoke(stypy.reporting.localization.Localization(__file__, 310, 18), array_237854, *[int_237855], **kwargs_237856)
        
        # Getting the type of 'self' (line 310)
        self_237858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self')
        # Setting the type of the member 'x0' of a type (line 310)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_237858, 'x0', array_call_result_237857)
        
        # Assigning a Num to a Attribute (line 311):
        
        # Assigning a Num to a Attribute (line 311):
        int_237859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 18), 'int')
        # Getting the type of 'self' (line 311)
        self_237860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'self')
        # Setting the type of the member 'f0' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), self_237860, 'f0', int_237859)
        
        # Assigning a Call to a Name (line 313):
        
        # Assigning a Call to a Name (line 313):
        
        # Call to OptimizeResult(...): (line 313)
        # Processing the call keyword arguments (line 313)
        kwargs_237862 = {}
        # Getting the type of 'OptimizeResult' (line 313)
        OptimizeResult_237861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 17), 'OptimizeResult', False)
        # Calling OptimizeResult(args, kwargs) (line 313)
        OptimizeResult_call_result_237863 = invoke(stypy.reporting.localization.Localization(__file__, 313, 17), OptimizeResult_237861, *[], **kwargs_237862)
        
        # Assigning a type to the variable 'minres' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'minres', OptimizeResult_call_result_237863)
        
        # Assigning a Attribute to a Attribute (line 314):
        
        # Assigning a Attribute to a Attribute (line 314):
        # Getting the type of 'self' (line 314)
        self_237864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'self')
        # Obtaining the member 'x0' of a type (line 314)
        x0_237865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 19), self_237864, 'x0')
        # Getting the type of 'minres' (line 314)
        minres_237866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'minres')
        # Setting the type of the member 'x' of a type (line 314)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), minres_237866, 'x', x0_237865)
        
        # Assigning a Attribute to a Attribute (line 315):
        
        # Assigning a Attribute to a Attribute (line 315):
        # Getting the type of 'self' (line 315)
        self_237867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 21), 'self')
        # Obtaining the member 'f0' of a type (line 315)
        f0_237868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 21), self_237867, 'f0')
        # Getting the type of 'minres' (line 315)
        minres_237869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'minres')
        # Setting the type of the member 'fun' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), minres_237869, 'fun', f0_237868)
        
        # Assigning a Call to a Attribute (line 317):
        
        # Assigning a Call to a Attribute (line 317):
        
        # Call to Storage(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'minres' (line 317)
        minres_237871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 31), 'minres', False)
        # Processing the call keyword arguments (line 317)
        kwargs_237872 = {}
        # Getting the type of 'Storage' (line 317)
        Storage_237870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 23), 'Storage', False)
        # Calling Storage(args, kwargs) (line 317)
        Storage_call_result_237873 = invoke(stypy.reporting.localization.Localization(__file__, 317, 23), Storage_237870, *[minres_237871], **kwargs_237872)
        
        # Getting the type of 'self' (line 317)
        self_237874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self')
        # Setting the type of the member 'storage' of a type (line 317)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_237874, 'storage', Storage_call_result_237873)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_237875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_237875


    @norecursion
    def test_higher_f_rejected(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_higher_f_rejected'
        module_type_store = module_type_store.open_function_context('test_higher_f_rejected', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_localization', localization)
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_function_name', 'Test_Storage.test_higher_f_rejected')
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Storage.test_higher_f_rejected.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Storage.test_higher_f_rejected', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_higher_f_rejected', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_higher_f_rejected(...)' code ##################

        
        # Assigning a Call to a Name (line 320):
        
        # Assigning a Call to a Name (line 320):
        
        # Call to OptimizeResult(...): (line 320)
        # Processing the call keyword arguments (line 320)
        kwargs_237877 = {}
        # Getting the type of 'OptimizeResult' (line 320)
        OptimizeResult_237876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 21), 'OptimizeResult', False)
        # Calling OptimizeResult(args, kwargs) (line 320)
        OptimizeResult_call_result_237878 = invoke(stypy.reporting.localization.Localization(__file__, 320, 21), OptimizeResult_237876, *[], **kwargs_237877)
        
        # Assigning a type to the variable 'new_minres' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'new_minres', OptimizeResult_call_result_237878)
        
        # Assigning a BinOp to a Attribute (line 321):
        
        # Assigning a BinOp to a Attribute (line 321):
        # Getting the type of 'self' (line 321)
        self_237879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'self')
        # Obtaining the member 'x0' of a type (line 321)
        x0_237880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 23), self_237879, 'x0')
        int_237881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 33), 'int')
        # Applying the binary operator '+' (line 321)
        result_add_237882 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 23), '+', x0_237880, int_237881)
        
        # Getting the type of 'new_minres' (line 321)
        new_minres_237883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'new_minres')
        # Setting the type of the member 'x' of a type (line 321)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), new_minres_237883, 'x', result_add_237882)
        
        # Assigning a BinOp to a Attribute (line 322):
        
        # Assigning a BinOp to a Attribute (line 322):
        # Getting the type of 'self' (line 322)
        self_237884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'self')
        # Obtaining the member 'f0' of a type (line 322)
        f0_237885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 25), self_237884, 'f0')
        int_237886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 35), 'int')
        # Applying the binary operator '+' (line 322)
        result_add_237887 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 25), '+', f0_237885, int_237886)
        
        # Getting the type of 'new_minres' (line 322)
        new_minres_237888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'new_minres')
        # Setting the type of the member 'fun' of a type (line 322)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), new_minres_237888, 'fun', result_add_237887)
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to update(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'new_minres' (line 324)
        new_minres_237892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'new_minres', False)
        # Processing the call keyword arguments (line 324)
        kwargs_237893 = {}
        # Getting the type of 'self' (line 324)
        self_237889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 14), 'self', False)
        # Obtaining the member 'storage' of a type (line 324)
        storage_237890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 14), self_237889, 'storage')
        # Obtaining the member 'update' of a type (line 324)
        update_237891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 14), storage_237890, 'update')
        # Calling update(args, kwargs) (line 324)
        update_call_result_237894 = invoke(stypy.reporting.localization.Localization(__file__, 324, 14), update_237891, *[new_minres_237892], **kwargs_237893)
        
        # Assigning a type to the variable 'ret' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'ret', update_call_result_237894)
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to get_lowest(...): (line 325)
        # Processing the call keyword arguments (line 325)
        kwargs_237898 = {}
        # Getting the type of 'self' (line 325)
        self_237895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 17), 'self', False)
        # Obtaining the member 'storage' of a type (line 325)
        storage_237896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 17), self_237895, 'storage')
        # Obtaining the member 'get_lowest' of a type (line 325)
        get_lowest_237897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 17), storage_237896, 'get_lowest')
        # Calling get_lowest(args, kwargs) (line 325)
        get_lowest_call_result_237899 = invoke(stypy.reporting.localization.Localization(__file__, 325, 17), get_lowest_237897, *[], **kwargs_237898)
        
        # Assigning a type to the variable 'minres' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'minres', get_lowest_call_result_237899)
        
        # Call to assert_equal(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'self' (line 326)
        self_237901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 21), 'self', False)
        # Obtaining the member 'x0' of a type (line 326)
        x0_237902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 21), self_237901, 'x0')
        # Getting the type of 'minres' (line 326)
        minres_237903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 30), 'minres', False)
        # Obtaining the member 'x' of a type (line 326)
        x_237904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 30), minres_237903, 'x')
        # Processing the call keyword arguments (line 326)
        kwargs_237905 = {}
        # Getting the type of 'assert_equal' (line 326)
        assert_equal_237900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 326)
        assert_equal_call_result_237906 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), assert_equal_237900, *[x0_237902, x_237904], **kwargs_237905)
        
        
        # Call to assert_equal(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'self' (line 327)
        self_237908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'self', False)
        # Obtaining the member 'f0' of a type (line 327)
        f0_237909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 21), self_237908, 'f0')
        # Getting the type of 'minres' (line 327)
        minres_237910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 30), 'minres', False)
        # Obtaining the member 'fun' of a type (line 327)
        fun_237911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 30), minres_237910, 'fun')
        # Processing the call keyword arguments (line 327)
        kwargs_237912 = {}
        # Getting the type of 'assert_equal' (line 327)
        assert_equal_237907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 327)
        assert_equal_call_result_237913 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), assert_equal_237907, *[f0_237909, fun_237911], **kwargs_237912)
        
        
        # Call to assert_(...): (line 328)
        # Processing the call arguments (line 328)
        
        # Getting the type of 'ret' (line 328)
        ret_237915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 20), 'ret', False)
        # Applying the 'not' unary operator (line 328)
        result_not__237916 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 16), 'not', ret_237915)
        
        # Processing the call keyword arguments (line 328)
        kwargs_237917 = {}
        # Getting the type of 'assert_' (line 328)
        assert__237914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 328)
        assert__call_result_237918 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), assert__237914, *[result_not__237916], **kwargs_237917)
        
        
        # ################# End of 'test_higher_f_rejected(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_higher_f_rejected' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_237919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237919)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_higher_f_rejected'
        return stypy_return_type_237919


    @norecursion
    def test_lower_f_accepted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lower_f_accepted'
        module_type_store = module_type_store.open_function_context('test_lower_f_accepted', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_localization', localization)
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_function_name', 'Test_Storage.test_lower_f_accepted')
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Storage.test_lower_f_accepted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Storage.test_lower_f_accepted', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lower_f_accepted', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lower_f_accepted(...)' code ##################

        
        # Assigning a Call to a Name (line 331):
        
        # Assigning a Call to a Name (line 331):
        
        # Call to OptimizeResult(...): (line 331)
        # Processing the call keyword arguments (line 331)
        kwargs_237921 = {}
        # Getting the type of 'OptimizeResult' (line 331)
        OptimizeResult_237920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'OptimizeResult', False)
        # Calling OptimizeResult(args, kwargs) (line 331)
        OptimizeResult_call_result_237922 = invoke(stypy.reporting.localization.Localization(__file__, 331, 21), OptimizeResult_237920, *[], **kwargs_237921)
        
        # Assigning a type to the variable 'new_minres' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'new_minres', OptimizeResult_call_result_237922)
        
        # Assigning a BinOp to a Attribute (line 332):
        
        # Assigning a BinOp to a Attribute (line 332):
        # Getting the type of 'self' (line 332)
        self_237923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 23), 'self')
        # Obtaining the member 'x0' of a type (line 332)
        x0_237924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 23), self_237923, 'x0')
        int_237925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 33), 'int')
        # Applying the binary operator '+' (line 332)
        result_add_237926 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 23), '+', x0_237924, int_237925)
        
        # Getting the type of 'new_minres' (line 332)
        new_minres_237927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'new_minres')
        # Setting the type of the member 'x' of a type (line 332)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), new_minres_237927, 'x', result_add_237926)
        
        # Assigning a BinOp to a Attribute (line 333):
        
        # Assigning a BinOp to a Attribute (line 333):
        # Getting the type of 'self' (line 333)
        self_237928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 25), 'self')
        # Obtaining the member 'f0' of a type (line 333)
        f0_237929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 25), self_237928, 'f0')
        int_237930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 35), 'int')
        # Applying the binary operator '-' (line 333)
        result_sub_237931 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 25), '-', f0_237929, int_237930)
        
        # Getting the type of 'new_minres' (line 333)
        new_minres_237932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'new_minres')
        # Setting the type of the member 'fun' of a type (line 333)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), new_minres_237932, 'fun', result_sub_237931)
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to update(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'new_minres' (line 335)
        new_minres_237936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 34), 'new_minres', False)
        # Processing the call keyword arguments (line 335)
        kwargs_237937 = {}
        # Getting the type of 'self' (line 335)
        self_237933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'self', False)
        # Obtaining the member 'storage' of a type (line 335)
        storage_237934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 14), self_237933, 'storage')
        # Obtaining the member 'update' of a type (line 335)
        update_237935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 14), storage_237934, 'update')
        # Calling update(args, kwargs) (line 335)
        update_call_result_237938 = invoke(stypy.reporting.localization.Localization(__file__, 335, 14), update_237935, *[new_minres_237936], **kwargs_237937)
        
        # Assigning a type to the variable 'ret' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'ret', update_call_result_237938)
        
        # Assigning a Call to a Name (line 336):
        
        # Assigning a Call to a Name (line 336):
        
        # Call to get_lowest(...): (line 336)
        # Processing the call keyword arguments (line 336)
        kwargs_237942 = {}
        # Getting the type of 'self' (line 336)
        self_237939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 17), 'self', False)
        # Obtaining the member 'storage' of a type (line 336)
        storage_237940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 17), self_237939, 'storage')
        # Obtaining the member 'get_lowest' of a type (line 336)
        get_lowest_237941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 17), storage_237940, 'get_lowest')
        # Calling get_lowest(args, kwargs) (line 336)
        get_lowest_call_result_237943 = invoke(stypy.reporting.localization.Localization(__file__, 336, 17), get_lowest_237941, *[], **kwargs_237942)
        
        # Assigning a type to the variable 'minres' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'minres', get_lowest_call_result_237943)
        
        # Call to assert_(...): (line 337)
        # Processing the call arguments (line 337)
        
        # Getting the type of 'self' (line 337)
        self_237945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'self', False)
        # Obtaining the member 'x0' of a type (line 337)
        x0_237946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 16), self_237945, 'x0')
        # Getting the type of 'minres' (line 337)
        minres_237947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'minres', False)
        # Obtaining the member 'x' of a type (line 337)
        x_237948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 27), minres_237947, 'x')
        # Applying the binary operator '!=' (line 337)
        result_ne_237949 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 16), '!=', x0_237946, x_237948)
        
        # Processing the call keyword arguments (line 337)
        kwargs_237950 = {}
        # Getting the type of 'assert_' (line 337)
        assert__237944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 337)
        assert__call_result_237951 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), assert__237944, *[result_ne_237949], **kwargs_237950)
        
        
        # Call to assert_(...): (line 338)
        # Processing the call arguments (line 338)
        
        # Getting the type of 'self' (line 338)
        self_237953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'self', False)
        # Obtaining the member 'f0' of a type (line 338)
        f0_237954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 16), self_237953, 'f0')
        # Getting the type of 'minres' (line 338)
        minres_237955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 27), 'minres', False)
        # Obtaining the member 'fun' of a type (line 338)
        fun_237956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 27), minres_237955, 'fun')
        # Applying the binary operator '!=' (line 338)
        result_ne_237957 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 16), '!=', f0_237954, fun_237956)
        
        # Processing the call keyword arguments (line 338)
        kwargs_237958 = {}
        # Getting the type of 'assert_' (line 338)
        assert__237952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 338)
        assert__call_result_237959 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), assert__237952, *[result_ne_237957], **kwargs_237958)
        
        
        # Call to assert_(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'ret' (line 339)
        ret_237961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'ret', False)
        # Processing the call keyword arguments (line 339)
        kwargs_237962 = {}
        # Getting the type of 'assert_' (line 339)
        assert__237960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 339)
        assert__call_result_237963 = invoke(stypy.reporting.localization.Localization(__file__, 339, 8), assert__237960, *[ret_237961], **kwargs_237962)
        
        
        # ################# End of 'test_lower_f_accepted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lower_f_accepted' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_237964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237964)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lower_f_accepted'
        return stypy_return_type_237964


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 308, 0, False)
        # Assigning a type to the variable 'self' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Storage.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_Storage' (line 308)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'Test_Storage', Test_Storage)
# Declaration of the 'Test_RandomDisplacement' class

class Test_RandomDisplacement(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_localization', localization)
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_function_name', 'Test_RandomDisplacement.setup_method')
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_RandomDisplacement.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_RandomDisplacement.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 344):
        
        # Assigning a Num to a Attribute (line 344):
        float_237965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 24), 'float')
        # Getting the type of 'self' (line 344)
        self_237966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'self')
        # Setting the type of the member 'stepsize' of a type (line 344)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), self_237966, 'stepsize', float_237965)
        
        # Assigning a Call to a Attribute (line 345):
        
        # Assigning a Call to a Attribute (line 345):
        
        # Call to RandomDisplacement(...): (line 345)
        # Processing the call keyword arguments (line 345)
        # Getting the type of 'self' (line 345)
        self_237968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 52), 'self', False)
        # Obtaining the member 'stepsize' of a type (line 345)
        stepsize_237969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 52), self_237968, 'stepsize')
        keyword_237970 = stepsize_237969
        kwargs_237971 = {'stepsize': keyword_237970}
        # Getting the type of 'RandomDisplacement' (line 345)
        RandomDisplacement_237967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'RandomDisplacement', False)
        # Calling RandomDisplacement(args, kwargs) (line 345)
        RandomDisplacement_call_result_237972 = invoke(stypy.reporting.localization.Localization(__file__, 345, 24), RandomDisplacement_237967, *[], **kwargs_237971)
        
        # Getting the type of 'self' (line 345)
        self_237973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self')
        # Setting the type of the member 'displace' of a type (line 345)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_237973, 'displace', RandomDisplacement_call_result_237972)
        
        # Assigning a Num to a Attribute (line 346):
        
        # Assigning a Num to a Attribute (line 346):
        int_237974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 17), 'int')
        # Getting the type of 'self' (line 346)
        self_237975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'self')
        # Setting the type of the member 'N' of a type (line 346)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), self_237975, 'N', int_237974)
        
        # Assigning a Call to a Attribute (line 347):
        
        # Assigning a Call to a Attribute (line 347):
        
        # Call to zeros(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Obtaining an instance of the builtin type 'list' (line 347)
        list_237978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 347)
        # Adding element type (line 347)
        # Getting the type of 'self' (line 347)
        self_237979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 'self', False)
        # Obtaining the member 'N' of a type (line 347)
        N_237980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 28), self_237979, 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 27), list_237978, N_237980)
        
        # Processing the call keyword arguments (line 347)
        kwargs_237981 = {}
        # Getting the type of 'np' (line 347)
        np_237976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'np', False)
        # Obtaining the member 'zeros' of a type (line 347)
        zeros_237977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 18), np_237976, 'zeros')
        # Calling zeros(args, kwargs) (line 347)
        zeros_call_result_237982 = invoke(stypy.reporting.localization.Localization(__file__, 347, 18), zeros_237977, *[list_237978], **kwargs_237981)
        
        # Getting the type of 'self' (line 347)
        self_237983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self')
        # Setting the type of the member 'x0' of a type (line 347)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_237983, 'x0', zeros_call_result_237982)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_237984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237984)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_237984


    @norecursion
    def test_random(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random'
        module_type_store = module_type_store.open_function_context('test_random', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_localization', localization)
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_function_name', 'Test_RandomDisplacement.test_random')
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_param_names_list', [])
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_RandomDisplacement.test_random.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_RandomDisplacement.test_random', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random(...)' code ##################

        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Call to displace(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'self' (line 353)
        self_237987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 26), 'self', False)
        # Obtaining the member 'x0' of a type (line 353)
        x0_237988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 26), self_237987, 'x0')
        # Processing the call keyword arguments (line 353)
        kwargs_237989 = {}
        # Getting the type of 'self' (line 353)
        self_237985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'self', False)
        # Obtaining the member 'displace' of a type (line 353)
        displace_237986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 12), self_237985, 'displace')
        # Calling displace(args, kwargs) (line 353)
        displace_call_result_237990 = invoke(stypy.reporting.localization.Localization(__file__, 353, 12), displace_237986, *[x0_237988], **kwargs_237989)
        
        # Assigning a type to the variable 'x' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'x', displace_call_result_237990)
        
        # Assigning a BinOp to a Name (line 354):
        
        # Assigning a BinOp to a Name (line 354):
        float_237991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 13), 'float')
        # Getting the type of 'self' (line 354)
        self_237992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'self')
        # Obtaining the member 'stepsize' of a type (line 354)
        stepsize_237993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 18), self_237992, 'stepsize')
        # Applying the binary operator '*' (line 354)
        result_mul_237994 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 13), '*', float_237991, stepsize_237993)
        
        int_237995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 36), 'int')
        # Applying the binary operator '**' (line 354)
        result_pow_237996 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 12), '**', result_mul_237994, int_237995)
        
        int_237997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 40), 'int')
        # Applying the binary operator 'div' (line 354)
        result_div_237998 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 12), 'div', result_pow_237996, int_237997)
        
        # Assigning a type to the variable 'v' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'v', result_div_237998)
        
        # Call to assert_almost_equal(...): (line 355)
        # Processing the call arguments (line 355)
        
        # Call to mean(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'x' (line 355)
        x_238002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 36), 'x', False)
        # Processing the call keyword arguments (line 355)
        kwargs_238003 = {}
        # Getting the type of 'np' (line 355)
        np_238000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 28), 'np', False)
        # Obtaining the member 'mean' of a type (line 355)
        mean_238001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 28), np_238000, 'mean')
        # Calling mean(args, kwargs) (line 355)
        mean_call_result_238004 = invoke(stypy.reporting.localization.Localization(__file__, 355, 28), mean_238001, *[x_238002], **kwargs_238003)
        
        float_238005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 40), 'float')
        int_238006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 44), 'int')
        # Processing the call keyword arguments (line 355)
        kwargs_238007 = {}
        # Getting the type of 'assert_almost_equal' (line 355)
        assert_almost_equal_237999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 355)
        assert_almost_equal_call_result_238008 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), assert_almost_equal_237999, *[mean_call_result_238004, float_238005, int_238006], **kwargs_238007)
        
        
        # Call to assert_almost_equal(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Call to var(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'x' (line 356)
        x_238012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 35), 'x', False)
        # Processing the call keyword arguments (line 356)
        kwargs_238013 = {}
        # Getting the type of 'np' (line 356)
        np_238010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 28), 'np', False)
        # Obtaining the member 'var' of a type (line 356)
        var_238011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 28), np_238010, 'var')
        # Calling var(args, kwargs) (line 356)
        var_call_result_238014 = invoke(stypy.reporting.localization.Localization(__file__, 356, 28), var_238011, *[x_238012], **kwargs_238013)
        
        # Getting the type of 'v' (line 356)
        v_238015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 39), 'v', False)
        int_238016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 42), 'int')
        # Processing the call keyword arguments (line 356)
        kwargs_238017 = {}
        # Getting the type of 'assert_almost_equal' (line 356)
        assert_almost_equal_238009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 356)
        assert_almost_equal_call_result_238018 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), assert_almost_equal_238009, *[var_call_result_238014, v_238015, int_238016], **kwargs_238017)
        
        
        # ################# End of 'test_random(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random' in the type store
        # Getting the type of 'stypy_return_type' (line 349)
        stypy_return_type_238019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238019)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random'
        return stypy_return_type_238019


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 342, 0, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_RandomDisplacement.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_RandomDisplacement' (line 342)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'Test_RandomDisplacement', Test_RandomDisplacement)
# Declaration of the 'Test_Metropolis' class

class Test_Metropolis(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_localization', localization)
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_function_name', 'Test_Metropolis.setup_method')
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Metropolis.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Metropolis.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 361):
        
        # Assigning a Num to a Attribute (line 361):
        float_238020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 17), 'float')
        # Getting the type of 'self' (line 361)
        self_238021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self')
        # Setting the type of the member 'T' of a type (line 361)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_238021, 'T', float_238020)
        
        # Assigning a Call to a Attribute (line 362):
        
        # Assigning a Call to a Attribute (line 362):
        
        # Call to Metropolis(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'self' (line 362)
        self_238023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 30), 'self', False)
        # Obtaining the member 'T' of a type (line 362)
        T_238024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 30), self_238023, 'T')
        # Processing the call keyword arguments (line 362)
        kwargs_238025 = {}
        # Getting the type of 'Metropolis' (line 362)
        Metropolis_238022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'Metropolis', False)
        # Calling Metropolis(args, kwargs) (line 362)
        Metropolis_call_result_238026 = invoke(stypy.reporting.localization.Localization(__file__, 362, 19), Metropolis_238022, *[T_238024], **kwargs_238025)
        
        # Getting the type of 'self' (line 362)
        self_238027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'self')
        # Setting the type of the member 'met' of a type (line 362)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), self_238027, 'met', Metropolis_call_result_238026)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_238028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_238028


    @norecursion
    def test_boolean_return(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_boolean_return'
        module_type_store = module_type_store.open_function_context('test_boolean_return', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_localization', localization)
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_function_name', 'Test_Metropolis.test_boolean_return')
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Metropolis.test_boolean_return.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Metropolis.test_boolean_return', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_boolean_return', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_boolean_return(...)' code ##################

        
        # Assigning a Call to a Name (line 367):
        
        # Assigning a Call to a Name (line 367):
        
        # Call to met(...): (line 367)
        # Processing the call keyword arguments (line 367)
        float_238031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 29), 'float')
        keyword_238032 = float_238031
        float_238033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 39), 'float')
        keyword_238034 = float_238033
        kwargs_238035 = {'f_new': keyword_238032, 'f_old': keyword_238034}
        # Getting the type of 'self' (line 367)
        self_238029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 14), 'self', False)
        # Obtaining the member 'met' of a type (line 367)
        met_238030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 14), self_238029, 'met')
        # Calling met(args, kwargs) (line 367)
        met_call_result_238036 = invoke(stypy.reporting.localization.Localization(__file__, 367, 14), met_238030, *[], **kwargs_238035)
        
        # Assigning a type to the variable 'ret' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'ret', met_call_result_238036)
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'ret' (line 368)
        ret_238038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'ret', False)
        # Getting the type of 'bool' (line 368)
        bool_238039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 31), 'bool', False)
        # Processing the call keyword arguments (line 368)
        kwargs_238040 = {}
        # Getting the type of 'isinstance' (line 368)
        isinstance_238037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 368)
        isinstance_call_result_238041 = invoke(stypy.reporting.localization.Localization(__file__, 368, 15), isinstance_238037, *[ret_238038, bool_238039], **kwargs_238040)
        
        
        # ################# End of 'test_boolean_return(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_boolean_return' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_238042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_boolean_return'
        return stypy_return_type_238042


    @norecursion
    def test_lower_f_accepted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lower_f_accepted'
        module_type_store = module_type_store.open_function_context('test_lower_f_accepted', 370, 4, False)
        # Assigning a type to the variable 'self' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_localization', localization)
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_function_name', 'Test_Metropolis.test_lower_f_accepted')
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Metropolis.test_lower_f_accepted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Metropolis.test_lower_f_accepted', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lower_f_accepted', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lower_f_accepted(...)' code ##################

        
        # Call to assert_(...): (line 371)
        # Processing the call arguments (line 371)
        
        # Call to met(...): (line 371)
        # Processing the call keyword arguments (line 371)
        float_238046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 31), 'float')
        keyword_238047 = float_238046
        float_238048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 41), 'float')
        keyword_238049 = float_238048
        kwargs_238050 = {'f_new': keyword_238047, 'f_old': keyword_238049}
        # Getting the type of 'self' (line 371)
        self_238044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'self', False)
        # Obtaining the member 'met' of a type (line 371)
        met_238045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 16), self_238044, 'met')
        # Calling met(args, kwargs) (line 371)
        met_call_result_238051 = invoke(stypy.reporting.localization.Localization(__file__, 371, 16), met_238045, *[], **kwargs_238050)
        
        # Processing the call keyword arguments (line 371)
        kwargs_238052 = {}
        # Getting the type of 'assert_' (line 371)
        assert__238043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 371)
        assert__call_result_238053 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), assert__238043, *[met_call_result_238051], **kwargs_238052)
        
        
        # ################# End of 'test_lower_f_accepted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lower_f_accepted' in the type store
        # Getting the type of 'stypy_return_type' (line 370)
        stypy_return_type_238054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lower_f_accepted'
        return stypy_return_type_238054


    @norecursion
    def test_KeyError(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_KeyError'
        module_type_store = module_type_store.open_function_context('test_KeyError', 373, 4, False)
        # Assigning a type to the variable 'self' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_localization', localization)
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_function_name', 'Test_Metropolis.test_KeyError')
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Metropolis.test_KeyError.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Metropolis.test_KeyError', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_KeyError', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_KeyError(...)' code ##################

        
        # Call to assert_raises(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'KeyError' (line 375)
        KeyError_238056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 22), 'KeyError', False)
        # Getting the type of 'self' (line 375)
        self_238057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 32), 'self', False)
        # Obtaining the member 'met' of a type (line 375)
        met_238058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 32), self_238057, 'met')
        # Processing the call keyword arguments (line 375)
        float_238059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 48), 'float')
        keyword_238060 = float_238059
        kwargs_238061 = {'f_old': keyword_238060}
        # Getting the type of 'assert_raises' (line 375)
        assert_raises_238055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 375)
        assert_raises_call_result_238062 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), assert_raises_238055, *[KeyError_238056, met_238058], **kwargs_238061)
        
        
        # Call to assert_raises(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'KeyError' (line 376)
        KeyError_238064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 22), 'KeyError', False)
        # Getting the type of 'self' (line 376)
        self_238065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 32), 'self', False)
        # Obtaining the member 'met' of a type (line 376)
        met_238066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 32), self_238065, 'met')
        # Processing the call keyword arguments (line 376)
        float_238067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 48), 'float')
        keyword_238068 = float_238067
        kwargs_238069 = {'f_new': keyword_238068}
        # Getting the type of 'assert_raises' (line 376)
        assert_raises_238063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 376)
        assert_raises_call_result_238070 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), assert_raises_238063, *[KeyError_238064, met_238066], **kwargs_238069)
        
        
        # ################# End of 'test_KeyError(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_KeyError' in the type store
        # Getting the type of 'stypy_return_type' (line 373)
        stypy_return_type_238071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238071)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_KeyError'
        return stypy_return_type_238071


    @norecursion
    def test_accept(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_accept'
        module_type_store = module_type_store.open_function_context('test_accept', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_localization', localization)
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_function_name', 'Test_Metropolis.test_accept')
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Metropolis.test_accept.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Metropolis.test_accept', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_accept', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_accept(...)' code ##################

        
        # Assigning a Name to a Name (line 380):
        
        # Assigning a Name to a Name (line 380):
        # Getting the type of 'False' (line 380)
        False_238072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'False')
        # Assigning a type to the variable 'one_accept' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'one_accept', False_238072)
        
        # Assigning a Name to a Name (line 381):
        
        # Assigning a Name to a Name (line 381):
        # Getting the type of 'False' (line 381)
        False_238073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'False')
        # Assigning a type to the variable 'one_reject' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'one_reject', False_238073)
        
        
        # Call to range(...): (line 382)
        # Processing the call arguments (line 382)
        int_238075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 23), 'int')
        # Processing the call keyword arguments (line 382)
        kwargs_238076 = {}
        # Getting the type of 'range' (line 382)
        range_238074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 17), 'range', False)
        # Calling range(args, kwargs) (line 382)
        range_call_result_238077 = invoke(stypy.reporting.localization.Localization(__file__, 382, 17), range_238074, *[int_238075], **kwargs_238076)
        
        # Testing the type of a for loop iterable (line 382)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 382, 8), range_call_result_238077)
        # Getting the type of the for loop variable (line 382)
        for_loop_var_238078 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 382, 8), range_call_result_238077)
        # Assigning a type to the variable 'i' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'i', for_loop_var_238078)
        # SSA begins for a for statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'one_accept' (line 383)
        one_accept_238079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'one_accept')
        # Getting the type of 'one_reject' (line 383)
        one_reject_238080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 30), 'one_reject')
        # Applying the binary operator 'and' (line 383)
        result_and_keyword_238081 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 15), 'and', one_accept_238079, one_reject_238080)
        
        # Testing the type of an if condition (line 383)
        if_condition_238082 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 12), result_and_keyword_238081)
        # Assigning a type to the variable 'if_condition_238082' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'if_condition_238082', if_condition_238082)
        # SSA begins for if statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 383)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to met(...): (line 385)
        # Processing the call keyword arguments (line 385)
        float_238085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 33), 'float')
        keyword_238086 = float_238085
        float_238087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 43), 'float')
        keyword_238088 = float_238087
        kwargs_238089 = {'f_new': keyword_238086, 'f_old': keyword_238088}
        # Getting the type of 'self' (line 385)
        self_238083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'self', False)
        # Obtaining the member 'met' of a type (line 385)
        met_238084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), self_238083, 'met')
        # Calling met(args, kwargs) (line 385)
        met_call_result_238090 = invoke(stypy.reporting.localization.Localization(__file__, 385, 18), met_238084, *[], **kwargs_238089)
        
        # Assigning a type to the variable 'ret' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'ret', met_call_result_238090)
        
        # Getting the type of 'ret' (line 386)
        ret_238091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'ret')
        # Testing the type of an if condition (line 386)
        if_condition_238092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 12), ret_238091)
        # Assigning a type to the variable 'if_condition_238092' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'if_condition_238092', if_condition_238092)
        # SSA begins for if statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 387):
        
        # Assigning a Name to a Name (line 387):
        # Getting the type of 'True' (line 387)
        True_238093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 29), 'True')
        # Assigning a type to the variable 'one_accept' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'one_accept', True_238093)
        # SSA branch for the else part of an if statement (line 386)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 389):
        
        # Assigning a Name to a Name (line 389):
        # Getting the type of 'True' (line 389)
        True_238094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 29), 'True')
        # Assigning a type to the variable 'one_reject' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'one_reject', True_238094)
        # SSA join for if statement (line 386)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'one_accept' (line 390)
        one_accept_238096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'one_accept', False)
        # Processing the call keyword arguments (line 390)
        kwargs_238097 = {}
        # Getting the type of 'assert_' (line 390)
        assert__238095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 390)
        assert__call_result_238098 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), assert__238095, *[one_accept_238096], **kwargs_238097)
        
        
        # Call to assert_(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'one_reject' (line 391)
        one_reject_238100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'one_reject', False)
        # Processing the call keyword arguments (line 391)
        kwargs_238101 = {}
        # Getting the type of 'assert_' (line 391)
        assert__238099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 391)
        assert__call_result_238102 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), assert__238099, *[one_reject_238100], **kwargs_238101)
        
        
        # ################# End of 'test_accept(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_accept' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_238103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238103)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_accept'
        return stypy_return_type_238103


    @norecursion
    def test_GH7495(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_GH7495'
        module_type_store = module_type_store.open_function_context('test_GH7495', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_localization', localization)
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_function_name', 'Test_Metropolis.test_GH7495')
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Metropolis.test_GH7495.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Metropolis.test_GH7495', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_GH7495', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_GH7495(...)' code ##################

        
        # Assigning a Call to a Name (line 396):
        
        # Assigning a Call to a Name (line 396):
        
        # Call to Metropolis(...): (line 396)
        # Processing the call arguments (line 396)
        int_238105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 25), 'int')
        # Processing the call keyword arguments (line 396)
        kwargs_238106 = {}
        # Getting the type of 'Metropolis' (line 396)
        Metropolis_238104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 14), 'Metropolis', False)
        # Calling Metropolis(args, kwargs) (line 396)
        Metropolis_call_result_238107 = invoke(stypy.reporting.localization.Localization(__file__, 396, 14), Metropolis_238104, *[int_238105], **kwargs_238106)
        
        # Assigning a type to the variable 'met' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'met', Metropolis_call_result_238107)
        
        # Call to errstate(...): (line 397)
        # Processing the call keyword arguments (line 397)
        str_238110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 30), 'str', 'raise')
        keyword_238111 = str_238110
        kwargs_238112 = {'over': keyword_238111}
        # Getting the type of 'np' (line 397)
        np_238108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 13), 'np', False)
        # Obtaining the member 'errstate' of a type (line 397)
        errstate_238109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 13), np_238108, 'errstate')
        # Calling errstate(args, kwargs) (line 397)
        errstate_call_result_238113 = invoke(stypy.reporting.localization.Localization(__file__, 397, 13), errstate_238109, *[], **kwargs_238112)
        
        with_238114 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 397, 13), errstate_call_result_238113, 'with parameter', '__enter__', '__exit__')

        if with_238114:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 397)
            enter___238115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 13), errstate_call_result_238113, '__enter__')
            with_enter_238116 = invoke(stypy.reporting.localization.Localization(__file__, 397, 13), enter___238115)
            
            # Call to accept_reject(...): (line 398)
            # Processing the call arguments (line 398)
            int_238119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 30), 'int')
            int_238120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 33), 'int')
            # Processing the call keyword arguments (line 398)
            kwargs_238121 = {}
            # Getting the type of 'met' (line 398)
            met_238117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'met', False)
            # Obtaining the member 'accept_reject' of a type (line 398)
            accept_reject_238118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), met_238117, 'accept_reject')
            # Calling accept_reject(args, kwargs) (line 398)
            accept_reject_call_result_238122 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), accept_reject_238118, *[int_238119, int_238120], **kwargs_238121)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 397)
            exit___238123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 13), errstate_call_result_238113, '__exit__')
            with_exit_238124 = invoke(stypy.reporting.localization.Localization(__file__, 397, 13), exit___238123, None, None, None)

        
        # ################# End of 'test_GH7495(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_GH7495' in the type store
        # Getting the type of 'stypy_return_type' (line 393)
        stypy_return_type_238125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_GH7495'
        return stypy_return_type_238125


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 359, 0, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Metropolis.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_Metropolis' (line 359)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 0), 'Test_Metropolis', Test_Metropolis)
# Declaration of the 'Test_AdaptiveStepsize' class

class Test_AdaptiveStepsize(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_localization', localization)
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_function_name', 'Test_AdaptiveStepsize.setup_method')
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_AdaptiveStepsize.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_AdaptiveStepsize.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 403):
        
        # Assigning a Num to a Attribute (line 403):
        float_238126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 24), 'float')
        # Getting the type of 'self' (line 403)
        self_238127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self')
        # Setting the type of the member 'stepsize' of a type (line 403)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_238127, 'stepsize', float_238126)
        
        # Assigning a Call to a Attribute (line 404):
        
        # Assigning a Call to a Attribute (line 404):
        
        # Call to RandomDisplacement(...): (line 404)
        # Processing the call keyword arguments (line 404)
        # Getting the type of 'self' (line 404)
        self_238129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 46), 'self', False)
        # Obtaining the member 'stepsize' of a type (line 404)
        stepsize_238130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 46), self_238129, 'stepsize')
        keyword_238131 = stepsize_238130
        kwargs_238132 = {'stepsize': keyword_238131}
        # Getting the type of 'RandomDisplacement' (line 404)
        RandomDisplacement_238128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 18), 'RandomDisplacement', False)
        # Calling RandomDisplacement(args, kwargs) (line 404)
        RandomDisplacement_call_result_238133 = invoke(stypy.reporting.localization.Localization(__file__, 404, 18), RandomDisplacement_238128, *[], **kwargs_238132)
        
        # Getting the type of 'self' (line 404)
        self_238134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'self')
        # Setting the type of the member 'ts' of a type (line 404)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), self_238134, 'ts', RandomDisplacement_call_result_238133)
        
        # Assigning a Num to a Attribute (line 405):
        
        # Assigning a Num to a Attribute (line 405):
        float_238135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 34), 'float')
        # Getting the type of 'self' (line 405)
        self_238136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'self')
        # Setting the type of the member 'target_accept_rate' of a type (line 405)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), self_238136, 'target_accept_rate', float_238135)
        
        # Assigning a Call to a Attribute (line 406):
        
        # Assigning a Call to a Attribute (line 406):
        
        # Call to AdaptiveStepsize(...): (line 406)
        # Processing the call keyword arguments (line 406)
        # Getting the type of 'self' (line 406)
        self_238138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 50), 'self', False)
        # Obtaining the member 'ts' of a type (line 406)
        ts_238139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 50), self_238138, 'ts')
        keyword_238140 = ts_238139
        # Getting the type of 'False' (line 406)
        False_238141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 67), 'False', False)
        keyword_238142 = False_238141
        # Getting the type of 'self' (line 407)
        self_238143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 54), 'self', False)
        # Obtaining the member 'target_accept_rate' of a type (line 407)
        target_accept_rate_238144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 54), self_238143, 'target_accept_rate')
        keyword_238145 = target_accept_rate_238144
        kwargs_238146 = {'takestep': keyword_238140, 'verbose': keyword_238142, 'accept_rate': keyword_238145}
        # Getting the type of 'AdaptiveStepsize' (line 406)
        AdaptiveStepsize_238137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 24), 'AdaptiveStepsize', False)
        # Calling AdaptiveStepsize(args, kwargs) (line 406)
        AdaptiveStepsize_call_result_238147 = invoke(stypy.reporting.localization.Localization(__file__, 406, 24), AdaptiveStepsize_238137, *[], **kwargs_238146)
        
        # Getting the type of 'self' (line 406)
        self_238148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'self')
        # Setting the type of the member 'takestep' of a type (line 406)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_238148, 'takestep', AdaptiveStepsize_call_result_238147)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_238149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_238149


    @norecursion
    def test_adaptive_increase(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_adaptive_increase'
        module_type_store = module_type_store.open_function_context('test_adaptive_increase', 409, 4, False)
        # Assigning a type to the variable 'self' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_localization', localization)
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_function_name', 'Test_AdaptiveStepsize.test_adaptive_increase')
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_param_names_list', [])
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_AdaptiveStepsize.test_adaptive_increase.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_AdaptiveStepsize.test_adaptive_increase', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_adaptive_increase', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_adaptive_increase(...)' code ##################

        
        # Assigning a Num to a Name (line 411):
        
        # Assigning a Num to a Name (line 411):
        float_238150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 12), 'float')
        # Assigning a type to the variable 'x' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'x', float_238150)
        
        # Call to takestep(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'x' (line 412)
        x_238153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 22), 'x', False)
        # Processing the call keyword arguments (line 412)
        kwargs_238154 = {}
        # Getting the type of 'self' (line 412)
        self_238151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'self', False)
        # Obtaining the member 'takestep' of a type (line 412)
        takestep_238152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), self_238151, 'takestep')
        # Calling takestep(args, kwargs) (line 412)
        takestep_call_result_238155 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), takestep_238152, *[x_238153], **kwargs_238154)
        
        
        # Call to report(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'False' (line 413)
        False_238159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 29), 'False', False)
        # Processing the call keyword arguments (line 413)
        kwargs_238160 = {}
        # Getting the type of 'self' (line 413)
        self_238156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'self', False)
        # Obtaining the member 'takestep' of a type (line 413)
        takestep_238157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 8), self_238156, 'takestep')
        # Obtaining the member 'report' of a type (line 413)
        report_238158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 8), takestep_238157, 'report')
        # Calling report(args, kwargs) (line 413)
        report_call_result_238161 = invoke(stypy.reporting.localization.Localization(__file__, 413, 8), report_238158, *[False_238159], **kwargs_238160)
        
        
        
        # Call to range(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'self' (line 414)
        self_238163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 23), 'self', False)
        # Obtaining the member 'takestep' of a type (line 414)
        takestep_238164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 23), self_238163, 'takestep')
        # Obtaining the member 'interval' of a type (line 414)
        interval_238165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 23), takestep_238164, 'interval')
        # Processing the call keyword arguments (line 414)
        kwargs_238166 = {}
        # Getting the type of 'range' (line 414)
        range_238162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 17), 'range', False)
        # Calling range(args, kwargs) (line 414)
        range_call_result_238167 = invoke(stypy.reporting.localization.Localization(__file__, 414, 17), range_238162, *[interval_238165], **kwargs_238166)
        
        # Testing the type of a for loop iterable (line 414)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 414, 8), range_call_result_238167)
        # Getting the type of the for loop variable (line 414)
        for_loop_var_238168 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 414, 8), range_call_result_238167)
        # Assigning a type to the variable 'i' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'i', for_loop_var_238168)
        # SSA begins for a for statement (line 414)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to takestep(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'x' (line 415)
        x_238171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 26), 'x', False)
        # Processing the call keyword arguments (line 415)
        kwargs_238172 = {}
        # Getting the type of 'self' (line 415)
        self_238169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'self', False)
        # Obtaining the member 'takestep' of a type (line 415)
        takestep_238170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 12), self_238169, 'takestep')
        # Calling takestep(args, kwargs) (line 415)
        takestep_call_result_238173 = invoke(stypy.reporting.localization.Localization(__file__, 415, 12), takestep_238170, *[x_238171], **kwargs_238172)
        
        
        # Call to report(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'True' (line 416)
        True_238177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 33), 'True', False)
        # Processing the call keyword arguments (line 416)
        kwargs_238178 = {}
        # Getting the type of 'self' (line 416)
        self_238174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'self', False)
        # Obtaining the member 'takestep' of a type (line 416)
        takestep_238175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), self_238174, 'takestep')
        # Obtaining the member 'report' of a type (line 416)
        report_238176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), takestep_238175, 'report')
        # Calling report(args, kwargs) (line 416)
        report_call_result_238179 = invoke(stypy.reporting.localization.Localization(__file__, 416, 12), report_238176, *[True_238177], **kwargs_238178)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 417)
        # Processing the call arguments (line 417)
        
        # Getting the type of 'self' (line 417)
        self_238181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 16), 'self', False)
        # Obtaining the member 'ts' of a type (line 417)
        ts_238182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 16), self_238181, 'ts')
        # Obtaining the member 'stepsize' of a type (line 417)
        stepsize_238183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 16), ts_238182, 'stepsize')
        # Getting the type of 'self' (line 417)
        self_238184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 35), 'self', False)
        # Obtaining the member 'stepsize' of a type (line 417)
        stepsize_238185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 35), self_238184, 'stepsize')
        # Applying the binary operator '>' (line 417)
        result_gt_238186 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 16), '>', stepsize_238183, stepsize_238185)
        
        # Processing the call keyword arguments (line 417)
        kwargs_238187 = {}
        # Getting the type of 'assert_' (line 417)
        assert__238180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 417)
        assert__call_result_238188 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), assert__238180, *[result_gt_238186], **kwargs_238187)
        
        
        # ################# End of 'test_adaptive_increase(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_adaptive_increase' in the type store
        # Getting the type of 'stypy_return_type' (line 409)
        stypy_return_type_238189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_adaptive_increase'
        return stypy_return_type_238189


    @norecursion
    def test_adaptive_decrease(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_adaptive_decrease'
        module_type_store = module_type_store.open_function_context('test_adaptive_decrease', 419, 4, False)
        # Assigning a type to the variable 'self' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_localization', localization)
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_function_name', 'Test_AdaptiveStepsize.test_adaptive_decrease')
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_param_names_list', [])
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_AdaptiveStepsize.test_adaptive_decrease.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_AdaptiveStepsize.test_adaptive_decrease', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_adaptive_decrease', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_adaptive_decrease(...)' code ##################

        
        # Assigning a Num to a Name (line 421):
        
        # Assigning a Num to a Name (line 421):
        float_238190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 12), 'float')
        # Assigning a type to the variable 'x' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'x', float_238190)
        
        # Call to takestep(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'x' (line 422)
        x_238193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 22), 'x', False)
        # Processing the call keyword arguments (line 422)
        kwargs_238194 = {}
        # Getting the type of 'self' (line 422)
        self_238191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'self', False)
        # Obtaining the member 'takestep' of a type (line 422)
        takestep_238192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), self_238191, 'takestep')
        # Calling takestep(args, kwargs) (line 422)
        takestep_call_result_238195 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), takestep_238192, *[x_238193], **kwargs_238194)
        
        
        # Call to report(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'True' (line 423)
        True_238199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 29), 'True', False)
        # Processing the call keyword arguments (line 423)
        kwargs_238200 = {}
        # Getting the type of 'self' (line 423)
        self_238196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'self', False)
        # Obtaining the member 'takestep' of a type (line 423)
        takestep_238197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), self_238196, 'takestep')
        # Obtaining the member 'report' of a type (line 423)
        report_238198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), takestep_238197, 'report')
        # Calling report(args, kwargs) (line 423)
        report_call_result_238201 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), report_238198, *[True_238199], **kwargs_238200)
        
        
        
        # Call to range(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'self' (line 424)
        self_238203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 23), 'self', False)
        # Obtaining the member 'takestep' of a type (line 424)
        takestep_238204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 23), self_238203, 'takestep')
        # Obtaining the member 'interval' of a type (line 424)
        interval_238205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 23), takestep_238204, 'interval')
        # Processing the call keyword arguments (line 424)
        kwargs_238206 = {}
        # Getting the type of 'range' (line 424)
        range_238202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 17), 'range', False)
        # Calling range(args, kwargs) (line 424)
        range_call_result_238207 = invoke(stypy.reporting.localization.Localization(__file__, 424, 17), range_238202, *[interval_238205], **kwargs_238206)
        
        # Testing the type of a for loop iterable (line 424)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 424, 8), range_call_result_238207)
        # Getting the type of the for loop variable (line 424)
        for_loop_var_238208 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 424, 8), range_call_result_238207)
        # Assigning a type to the variable 'i' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'i', for_loop_var_238208)
        # SSA begins for a for statement (line 424)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to takestep(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'x' (line 425)
        x_238211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'x', False)
        # Processing the call keyword arguments (line 425)
        kwargs_238212 = {}
        # Getting the type of 'self' (line 425)
        self_238209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'self', False)
        # Obtaining the member 'takestep' of a type (line 425)
        takestep_238210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), self_238209, 'takestep')
        # Calling takestep(args, kwargs) (line 425)
        takestep_call_result_238213 = invoke(stypy.reporting.localization.Localization(__file__, 425, 12), takestep_238210, *[x_238211], **kwargs_238212)
        
        
        # Call to report(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'False' (line 426)
        False_238217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 33), 'False', False)
        # Processing the call keyword arguments (line 426)
        kwargs_238218 = {}
        # Getting the type of 'self' (line 426)
        self_238214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'self', False)
        # Obtaining the member 'takestep' of a type (line 426)
        takestep_238215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), self_238214, 'takestep')
        # Obtaining the member 'report' of a type (line 426)
        report_238216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), takestep_238215, 'report')
        # Calling report(args, kwargs) (line 426)
        report_call_result_238219 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), report_238216, *[False_238217], **kwargs_238218)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 427)
        # Processing the call arguments (line 427)
        
        # Getting the type of 'self' (line 427)
        self_238221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'self', False)
        # Obtaining the member 'ts' of a type (line 427)
        ts_238222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 16), self_238221, 'ts')
        # Obtaining the member 'stepsize' of a type (line 427)
        stepsize_238223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 16), ts_238222, 'stepsize')
        # Getting the type of 'self' (line 427)
        self_238224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 35), 'self', False)
        # Obtaining the member 'stepsize' of a type (line 427)
        stepsize_238225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 35), self_238224, 'stepsize')
        # Applying the binary operator '<' (line 427)
        result_lt_238226 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 16), '<', stepsize_238223, stepsize_238225)
        
        # Processing the call keyword arguments (line 427)
        kwargs_238227 = {}
        # Getting the type of 'assert_' (line 427)
        assert__238220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 427)
        assert__call_result_238228 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), assert__238220, *[result_lt_238226], **kwargs_238227)
        
        
        # ################# End of 'test_adaptive_decrease(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_adaptive_decrease' in the type store
        # Getting the type of 'stypy_return_type' (line 419)
        stypy_return_type_238229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238229)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_adaptive_decrease'
        return stypy_return_type_238229


    @norecursion
    def test_all_accepted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_all_accepted'
        module_type_store = module_type_store.open_function_context('test_all_accepted', 429, 4, False)
        # Assigning a type to the variable 'self' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_localization', localization)
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_function_name', 'Test_AdaptiveStepsize.test_all_accepted')
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_param_names_list', [])
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_AdaptiveStepsize.test_all_accepted.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_AdaptiveStepsize.test_all_accepted', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_all_accepted', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_all_accepted(...)' code ##################

        
        # Assigning a Num to a Name (line 431):
        
        # Assigning a Num to a Name (line 431):
        float_238230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 12), 'float')
        # Assigning a type to the variable 'x' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'x', float_238230)
        
        
        # Call to range(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'self' (line 432)
        self_238232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 23), 'self', False)
        # Obtaining the member 'takestep' of a type (line 432)
        takestep_238233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 23), self_238232, 'takestep')
        # Obtaining the member 'interval' of a type (line 432)
        interval_238234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 23), takestep_238233, 'interval')
        int_238235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 48), 'int')
        # Applying the binary operator '+' (line 432)
        result_add_238236 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 23), '+', interval_238234, int_238235)
        
        # Processing the call keyword arguments (line 432)
        kwargs_238237 = {}
        # Getting the type of 'range' (line 432)
        range_238231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 17), 'range', False)
        # Calling range(args, kwargs) (line 432)
        range_call_result_238238 = invoke(stypy.reporting.localization.Localization(__file__, 432, 17), range_238231, *[result_add_238236], **kwargs_238237)
        
        # Testing the type of a for loop iterable (line 432)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 432, 8), range_call_result_238238)
        # Getting the type of the for loop variable (line 432)
        for_loop_var_238239 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 432, 8), range_call_result_238238)
        # Assigning a type to the variable 'i' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'i', for_loop_var_238239)
        # SSA begins for a for statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to takestep(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'x' (line 433)
        x_238242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 26), 'x', False)
        # Processing the call keyword arguments (line 433)
        kwargs_238243 = {}
        # Getting the type of 'self' (line 433)
        self_238240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'self', False)
        # Obtaining the member 'takestep' of a type (line 433)
        takestep_238241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 12), self_238240, 'takestep')
        # Calling takestep(args, kwargs) (line 433)
        takestep_call_result_238244 = invoke(stypy.reporting.localization.Localization(__file__, 433, 12), takestep_238241, *[x_238242], **kwargs_238243)
        
        
        # Call to report(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'True' (line 434)
        True_238248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 33), 'True', False)
        # Processing the call keyword arguments (line 434)
        kwargs_238249 = {}
        # Getting the type of 'self' (line 434)
        self_238245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'self', False)
        # Obtaining the member 'takestep' of a type (line 434)
        takestep_238246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), self_238245, 'takestep')
        # Obtaining the member 'report' of a type (line 434)
        report_238247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), takestep_238246, 'report')
        # Calling report(args, kwargs) (line 434)
        report_call_result_238250 = invoke(stypy.reporting.localization.Localization(__file__, 434, 12), report_238247, *[True_238248], **kwargs_238249)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 435)
        # Processing the call arguments (line 435)
        
        # Getting the type of 'self' (line 435)
        self_238252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 16), 'self', False)
        # Obtaining the member 'ts' of a type (line 435)
        ts_238253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 16), self_238252, 'ts')
        # Obtaining the member 'stepsize' of a type (line 435)
        stepsize_238254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 16), ts_238253, 'stepsize')
        # Getting the type of 'self' (line 435)
        self_238255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 35), 'self', False)
        # Obtaining the member 'stepsize' of a type (line 435)
        stepsize_238256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 35), self_238255, 'stepsize')
        # Applying the binary operator '>' (line 435)
        result_gt_238257 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 16), '>', stepsize_238254, stepsize_238256)
        
        # Processing the call keyword arguments (line 435)
        kwargs_238258 = {}
        # Getting the type of 'assert_' (line 435)
        assert__238251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 435)
        assert__call_result_238259 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), assert__238251, *[result_gt_238257], **kwargs_238258)
        
        
        # ################# End of 'test_all_accepted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_all_accepted' in the type store
        # Getting the type of 'stypy_return_type' (line 429)
        stypy_return_type_238260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238260)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_all_accepted'
        return stypy_return_type_238260


    @norecursion
    def test_all_rejected(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_all_rejected'
        module_type_store = module_type_store.open_function_context('test_all_rejected', 437, 4, False)
        # Assigning a type to the variable 'self' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_localization', localization)
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_function_name', 'Test_AdaptiveStepsize.test_all_rejected')
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_param_names_list', [])
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_AdaptiveStepsize.test_all_rejected.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_AdaptiveStepsize.test_all_rejected', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_all_rejected', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_all_rejected(...)' code ##################

        
        # Assigning a Num to a Name (line 439):
        
        # Assigning a Num to a Name (line 439):
        float_238261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 12), 'float')
        # Assigning a type to the variable 'x' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'x', float_238261)
        
        
        # Call to range(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'self' (line 440)
        self_238263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'self', False)
        # Obtaining the member 'takestep' of a type (line 440)
        takestep_238264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), self_238263, 'takestep')
        # Obtaining the member 'interval' of a type (line 440)
        interval_238265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), takestep_238264, 'interval')
        int_238266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 48), 'int')
        # Applying the binary operator '+' (line 440)
        result_add_238267 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 23), '+', interval_238265, int_238266)
        
        # Processing the call keyword arguments (line 440)
        kwargs_238268 = {}
        # Getting the type of 'range' (line 440)
        range_238262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 17), 'range', False)
        # Calling range(args, kwargs) (line 440)
        range_call_result_238269 = invoke(stypy.reporting.localization.Localization(__file__, 440, 17), range_238262, *[result_add_238267], **kwargs_238268)
        
        # Testing the type of a for loop iterable (line 440)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 8), range_call_result_238269)
        # Getting the type of the for loop variable (line 440)
        for_loop_var_238270 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 8), range_call_result_238269)
        # Assigning a type to the variable 'i' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'i', for_loop_var_238270)
        # SSA begins for a for statement (line 440)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to takestep(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'x' (line 441)
        x_238273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 26), 'x', False)
        # Processing the call keyword arguments (line 441)
        kwargs_238274 = {}
        # Getting the type of 'self' (line 441)
        self_238271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'self', False)
        # Obtaining the member 'takestep' of a type (line 441)
        takestep_238272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), self_238271, 'takestep')
        # Calling takestep(args, kwargs) (line 441)
        takestep_call_result_238275 = invoke(stypy.reporting.localization.Localization(__file__, 441, 12), takestep_238272, *[x_238273], **kwargs_238274)
        
        
        # Call to report(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'False' (line 442)
        False_238279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 33), 'False', False)
        # Processing the call keyword arguments (line 442)
        kwargs_238280 = {}
        # Getting the type of 'self' (line 442)
        self_238276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'self', False)
        # Obtaining the member 'takestep' of a type (line 442)
        takestep_238277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), self_238276, 'takestep')
        # Obtaining the member 'report' of a type (line 442)
        report_238278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), takestep_238277, 'report')
        # Calling report(args, kwargs) (line 442)
        report_call_result_238281 = invoke(stypy.reporting.localization.Localization(__file__, 442, 12), report_238278, *[False_238279], **kwargs_238280)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 443)
        # Processing the call arguments (line 443)
        
        # Getting the type of 'self' (line 443)
        self_238283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 16), 'self', False)
        # Obtaining the member 'ts' of a type (line 443)
        ts_238284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 16), self_238283, 'ts')
        # Obtaining the member 'stepsize' of a type (line 443)
        stepsize_238285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 16), ts_238284, 'stepsize')
        # Getting the type of 'self' (line 443)
        self_238286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 35), 'self', False)
        # Obtaining the member 'stepsize' of a type (line 443)
        stepsize_238287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 35), self_238286, 'stepsize')
        # Applying the binary operator '<' (line 443)
        result_lt_238288 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 16), '<', stepsize_238285, stepsize_238287)
        
        # Processing the call keyword arguments (line 443)
        kwargs_238289 = {}
        # Getting the type of 'assert_' (line 443)
        assert__238282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 443)
        assert__call_result_238290 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), assert__238282, *[result_lt_238288], **kwargs_238289)
        
        
        # ################# End of 'test_all_rejected(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_all_rejected' in the type store
        # Getting the type of 'stypy_return_type' (line 437)
        stypy_return_type_238291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_238291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_all_rejected'
        return stypy_return_type_238291


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 401, 0, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_AdaptiveStepsize.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_AdaptiveStepsize' (line 401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'Test_AdaptiveStepsize', Test_AdaptiveStepsize)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
