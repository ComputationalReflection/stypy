
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit test for SLSQP optimization.
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import pytest
7: from numpy.testing import (assert_, assert_array_almost_equal,
8:                            assert_allclose, assert_equal)
9: from pytest import raises as assert_raises
10: import numpy as np
11: 
12: from scipy.optimize import fmin_slsqp, minimize
13: 
14: 
15: class MyCallBack(object):
16:     '''pass a custom callback function
17: 
18:     This makes sure it's being used.
19:     '''
20:     def __init__(self):
21:         self.been_called = False
22:         self.ncalls = 0
23: 
24:     def __call__(self, x):
25:         self.been_called = True
26:         self.ncalls += 1
27: 
28: 
29: class TestSLSQP(object):
30:     '''
31:     Test SLSQP algorithm using Example 14.4 from Numerical Methods for
32:     Engineers by Steven Chapra and Raymond Canale.
33:     This example maximizes the function f(x) = 2*x*y + 2*x - x**2 - 2*y**2,
34:     which has a maximum at x=2, y=1.
35:     '''
36:     def setup_method(self):
37:         self.opts = {'disp': False}
38: 
39:     def fun(self, d, sign=1.0):
40:         '''
41:         Arguments:
42:         d     - A list of two elements, where d[0] represents x and d[1] represents y
43:                  in the following equation.
44:         sign - A multiplier for f.  Since we want to optimize it, and the scipy
45:                optimizers can only minimize functions, we need to multiply it by
46:                -1 to achieve the desired solution
47:         Returns:
48:         2*x*y + 2*x - x**2 - 2*y**2
49: 
50:         '''
51:         x = d[0]
52:         y = d[1]
53:         return sign*(2*x*y + 2*x - x**2 - 2*y**2)
54: 
55:     def jac(self, d, sign=1.0):
56:         '''
57:         This is the derivative of fun, returning a numpy array
58:         representing df/dx and df/dy.
59: 
60:         '''
61:         x = d[0]
62:         y = d[1]
63:         dfdx = sign*(-2*x + 2*y + 2)
64:         dfdy = sign*(2*x - 4*y)
65:         return np.array([dfdx, dfdy], float)
66: 
67:     def fun_and_jac(self, d, sign=1.0):
68:         return self.fun(d, sign), self.jac(d, sign)
69: 
70:     def f_eqcon(self, x, sign=1.0):
71:         ''' Equality constraint '''
72:         return np.array([x[0] - x[1]])
73: 
74:     def fprime_eqcon(self, x, sign=1.0):
75:         ''' Equality constraint, derivative '''
76:         return np.array([[1, -1]])
77: 
78:     def f_eqcon_scalar(self, x, sign=1.0):
79:         ''' Scalar equality constraint '''
80:         return self.f_eqcon(x, sign)[0]
81: 
82:     def fprime_eqcon_scalar(self, x, sign=1.0):
83:         ''' Scalar equality constraint, derivative '''
84:         return self.fprime_eqcon(x, sign)[0].tolist()
85: 
86:     def f_ieqcon(self, x, sign=1.0):
87:         ''' Inequality constraint '''
88:         return np.array([x[0] - x[1] - 1.0])
89: 
90:     def fprime_ieqcon(self, x, sign=1.0):
91:         ''' Inequality constraint, derivative '''
92:         return np.array([[1, -1]])
93: 
94:     def f_ieqcon2(self, x):
95:         ''' Vector inequality constraint '''
96:         return np.asarray(x)
97: 
98:     def fprime_ieqcon2(self, x):
99:         ''' Vector inequality constraint, derivative '''
100:         return np.identity(x.shape[0])
101: 
102:     # minimize
103:     def test_minimize_unbounded_approximated(self):
104:         # Minimize, method='SLSQP': unbounded, approximated jacobian.
105:         res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
106:                        method='SLSQP', options=self.opts)
107:         assert_(res['success'], res['message'])
108:         assert_allclose(res.x, [2, 1])
109: 
110:     def test_minimize_unbounded_given(self):
111:         # Minimize, method='SLSQP': unbounded, given jacobian.
112:         res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
113:                        jac=self.jac, method='SLSQP', options=self.opts)
114:         assert_(res['success'], res['message'])
115:         assert_allclose(res.x, [2, 1])
116: 
117:     def test_minimize_bounded_approximated(self):
118:         # Minimize, method='SLSQP': bounded, approximated jacobian.
119:         with np.errstate(invalid='ignore'):
120:             res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
121:                            bounds=((2.5, None), (None, 0.5)),
122:                            method='SLSQP', options=self.opts)
123:         assert_(res['success'], res['message'])
124:         assert_allclose(res.x, [2.5, 0.5])
125:         assert_(2.5 <= res.x[0])
126:         assert_(res.x[1] <= 0.5)
127: 
128:     def test_minimize_unbounded_combined(self):
129:         # Minimize, method='SLSQP': unbounded, combined function and jacobian.
130:         res = minimize(self.fun_and_jac, [-1.0, 1.0], args=(-1.0, ),
131:                        jac=True, method='SLSQP', options=self.opts)
132:         assert_(res['success'], res['message'])
133:         assert_allclose(res.x, [2, 1])
134: 
135:     def test_minimize_equality_approximated(self):
136:         # Minimize with method='SLSQP': equality constraint, approx. jacobian.
137:         res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
138:                        constraints={'type': 'eq',
139:                                     'fun': self.f_eqcon,
140:                                     'args': (-1.0, )},
141:                        method='SLSQP', options=self.opts)
142:         assert_(res['success'], res['message'])
143:         assert_allclose(res.x, [1, 1])
144: 
145:     def test_minimize_equality_given(self):
146:         # Minimize with method='SLSQP': equality constraint, given jacobian.
147:         res = minimize(self.fun, [-1.0, 1.0], jac=self.jac,
148:                        method='SLSQP', args=(-1.0,),
149:                        constraints={'type': 'eq', 'fun':self.f_eqcon,
150:                                     'args': (-1.0, )},
151:                        options=self.opts)
152:         assert_(res['success'], res['message'])
153:         assert_allclose(res.x, [1, 1])
154: 
155:     def test_minimize_equality_given2(self):
156:         # Minimize with method='SLSQP': equality constraint, given jacobian
157:         # for fun and const.
158:         res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
159:                        jac=self.jac, args=(-1.0,),
160:                        constraints={'type': 'eq',
161:                                     'fun': self.f_eqcon,
162:                                     'args': (-1.0, ),
163:                                     'jac': self.fprime_eqcon},
164:                        options=self.opts)
165:         assert_(res['success'], res['message'])
166:         assert_allclose(res.x, [1, 1])
167: 
168:     def test_minimize_equality_given_cons_scalar(self):
169:         # Minimize with method='SLSQP': scalar equality constraint, given
170:         # jacobian for fun and const.
171:         res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
172:                        jac=self.jac, args=(-1.0,),
173:                        constraints={'type': 'eq',
174:                                     'fun': self.f_eqcon_scalar,
175:                                     'args': (-1.0, ),
176:                                     'jac': self.fprime_eqcon_scalar},
177:                        options=self.opts)
178:         assert_(res['success'], res['message'])
179:         assert_allclose(res.x, [1, 1])
180: 
181:     def test_minimize_inequality_given(self):
182:         # Minimize with method='SLSQP': inequality constraint, given jacobian.
183:         res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
184:                        jac=self.jac, args=(-1.0, ),
185:                        constraints={'type': 'ineq',
186:                                     'fun': self.f_ieqcon,
187:                                     'args': (-1.0, )},
188:                        options=self.opts)
189:         assert_(res['success'], res['message'])
190:         assert_allclose(res.x, [2, 1], atol=1e-3)
191: 
192:     def test_minimize_inequality_given_vector_constraints(self):
193:         # Minimize with method='SLSQP': vector inequality constraint, given
194:         # jacobian.
195:         res = minimize(self.fun, [-1.0, 1.0], jac=self.jac,
196:                        method='SLSQP', args=(-1.0,),
197:                        constraints={'type': 'ineq',
198:                                     'fun': self.f_ieqcon2,
199:                                     'jac': self.fprime_ieqcon2},
200:                        options=self.opts)
201:         assert_(res['success'], res['message'])
202:         assert_allclose(res.x, [2, 1])
203: 
204:     def test_minimize_bound_equality_given2(self):
205:         # Minimize with method='SLSQP': bounds, eq. const., given jac. for
206:         # fun. and const.
207:         res = minimize(self.fun, [-1.0, 1.0], method='SLSQP',
208:                        jac=self.jac, args=(-1.0, ),
209:                        bounds=[(-0.8, 1.), (-1, 0.8)],
210:                        constraints={'type': 'eq',
211:                                     'fun': self.f_eqcon,
212:                                     'args': (-1.0, ),
213:                                     'jac': self.fprime_eqcon},
214:                        options=self.opts)
215:         assert_(res['success'], res['message'])
216:         assert_allclose(res.x, [0.8, 0.8], atol=1e-3)
217:         assert_(-0.8 <= res.x[0] <= 1)
218:         assert_(-1 <= res.x[1] <= 0.8)
219: 
220:     # fmin_slsqp
221:     def test_unbounded_approximated(self):
222:         # SLSQP: unbounded, approximated jacobian.
223:         res = fmin_slsqp(self.fun, [-1.0, 1.0], args=(-1.0, ),
224:                          iprint = 0, full_output = 1)
225:         x, fx, its, imode, smode = res
226:         assert_(imode == 0, imode)
227:         assert_array_almost_equal(x, [2, 1])
228: 
229:     def test_unbounded_given(self):
230:         # SLSQP: unbounded, given jacobian.
231:         res = fmin_slsqp(self.fun, [-1.0, 1.0], args=(-1.0, ),
232:                          fprime = self.jac, iprint = 0,
233:                          full_output = 1)
234:         x, fx, its, imode, smode = res
235:         assert_(imode == 0, imode)
236:         assert_array_almost_equal(x, [2, 1])
237: 
238:     def test_equality_approximated(self):
239:         # SLSQP: equality constraint, approximated jacobian.
240:         res = fmin_slsqp(self.fun,[-1.0,1.0], args=(-1.0,),
241:                          eqcons = [self.f_eqcon],
242:                          iprint = 0, full_output = 1)
243:         x, fx, its, imode, smode = res
244:         assert_(imode == 0, imode)
245:         assert_array_almost_equal(x, [1, 1])
246: 
247:     def test_equality_given(self):
248:         # SLSQP: equality constraint, given jacobian.
249:         res = fmin_slsqp(self.fun, [-1.0, 1.0],
250:                          fprime=self.jac, args=(-1.0,),
251:                          eqcons = [self.f_eqcon], iprint = 0,
252:                          full_output = 1)
253:         x, fx, its, imode, smode = res
254:         assert_(imode == 0, imode)
255:         assert_array_almost_equal(x, [1, 1])
256: 
257:     def test_equality_given2(self):
258:         # SLSQP: equality constraint, given jacobian for fun and const.
259:         res = fmin_slsqp(self.fun, [-1.0, 1.0],
260:                          fprime=self.jac, args=(-1.0,),
261:                          f_eqcons = self.f_eqcon,
262:                          fprime_eqcons = self.fprime_eqcon,
263:                          iprint = 0,
264:                          full_output = 1)
265:         x, fx, its, imode, smode = res
266:         assert_(imode == 0, imode)
267:         assert_array_almost_equal(x, [1, 1])
268: 
269:     def test_inequality_given(self):
270:         # SLSQP: inequality constraint, given jacobian.
271:         res = fmin_slsqp(self.fun, [-1.0, 1.0],
272:                          fprime=self.jac, args=(-1.0, ),
273:                          ieqcons = [self.f_ieqcon],
274:                          iprint = 0, full_output = 1)
275:         x, fx, its, imode, smode = res
276:         assert_(imode == 0, imode)
277:         assert_array_almost_equal(x, [2, 1], decimal=3)
278: 
279:     def test_bound_equality_given2(self):
280:         # SLSQP: bounds, eq. const., given jac. for fun. and const.
281:         res = fmin_slsqp(self.fun, [-1.0, 1.0],
282:                          fprime=self.jac, args=(-1.0, ),
283:                          bounds = [(-0.8, 1.), (-1, 0.8)],
284:                          f_eqcons = self.f_eqcon,
285:                          fprime_eqcons = self.fprime_eqcon,
286:                          iprint = 0, full_output = 1)
287:         x, fx, its, imode, smode = res
288:         assert_(imode == 0, imode)
289:         assert_array_almost_equal(x, [0.8, 0.8], decimal=3)
290:         assert_(-0.8 <= x[0] <= 1)
291:         assert_(-1 <= x[1] <= 0.8)
292: 
293:     def test_scalar_constraints(self):
294:         # Regression test for gh-2182
295:         x = fmin_slsqp(lambda z: z**2, [3.],
296:                        ieqcons=[lambda z: z[0] - 1],
297:                        iprint=0)
298:         assert_array_almost_equal(x, [1.])
299: 
300:         x = fmin_slsqp(lambda z: z**2, [3.],
301:                        f_ieqcons=lambda z: [z[0] - 1],
302:                        iprint=0)
303:         assert_array_almost_equal(x, [1.])
304: 
305:     def test_integer_bounds(self):
306:         # This should not raise an exception
307:         fmin_slsqp(lambda z: z**2 - 1, [0], bounds=[[0, 1]], iprint=0)
308: 
309:     def test_obj_must_return_scalar(self):
310:         # Regression test for Github Issue #5433
311:         # If objective function does not return a scalar, raises ValueError
312:         with assert_raises(ValueError):
313:             fmin_slsqp(lambda x: [0, 1], [1, 2, 3])
314: 
315:     def test_obj_returns_scalar_in_list(self):
316:         # Test for Github Issue #5433 and PR #6691
317:         # Objective function should be able to return length-1 Python list
318:         #  containing the scalar
319:         fmin_slsqp(lambda x: [0], [1, 2, 3], iprint=0)
320: 
321:     def test_callback(self):
322:         # Minimize, method='SLSQP': unbounded, approximated jacobian. Check for callback
323:         callback = MyCallBack()
324:         res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
325:                        method='SLSQP', callback=callback, options=self.opts)
326:         assert_(res['success'], res['message'])
327:         assert_(callback.been_called)
328:         assert_equal(callback.ncalls, res['nit'])
329: 
330:     def test_inconsistent_linearization(self):
331:         # SLSQP must be able to solve this problem, even if the
332:         # linearized problem at the starting point is infeasible.
333: 
334:         # Linearized constraints are
335:         #
336:         #    2*x0[0]*x[0] >= 1
337:         #
338:         # At x0 = [0, 1], the second constraint is clearly infeasible.
339:         # This triggers a call with n2==1 in the LSQ subroutine.
340:         x = [0, 1]
341:         f1 = lambda x: x[0] + x[1] - 2
342:         f2 = lambda x: x[0]**2 - 1
343:         sol = minimize(
344:             lambda x: x[0]**2 + x[1]**2,
345:             x,
346:             constraints=({'type':'eq','fun': f1},
347:                          {'type':'ineq','fun': f2}),
348:             bounds=((0,None), (0,None)),
349:             method='SLSQP')
350:         x = sol.x
351: 
352:         assert_allclose(f1(x), 0, atol=1e-8)
353:         assert_(f2(x) >= -1e-8)
354:         assert_(sol.success, sol)
355: 
356:     @pytest.mark.xfail(reason="This bug is not fixed")
357:     def test_regression_5743(self):
358:         # SLSQP must not indicate success for this problem,
359:         # which is infeasible.
360:         x = [1, 2]
361:         sol = minimize(
362:             lambda x: x[0]**2 + x[1]**2,
363:             x,
364:             constraints=({'type':'eq','fun': lambda x: x[0]+x[1]-1},
365:                          {'type':'ineq','fun': lambda x: x[0]-2}),
366:             bounds=((0,None), (0,None)),
367:             method='SLSQP')
368:         assert_(not sol.success, sol)
369: 
370:     def test_gh_6676(self):
371:         def func(x):
372:             return (x[0] - 1)**2 + 2*(x[1] - 1)**2 + 0.5*(x[2] - 1)**2
373: 
374:         sol = minimize(func, [0, 0, 0], method='SLSQP')
375:         assert_(sol.jac.shape == (3,))
376: 
377:     def test_invalid_bounds(self):
378:         # Raise correct error when lower bound is greater than upper bound.
379:         # See Github issue 6875.
380:         bounds_list = [
381:             ((1, 2), (2, 1)),
382:             ((2, 1), (1, 2)),
383:             ((2, 1), (2, 1)),
384:             ((np.inf, 0), (np.inf, 0)),
385:             ((1, -np.inf), (0, 1)),
386:         ]
387:         for bounds in bounds_list:
388:             with assert_raises(ValueError):
389:                 minimize(self.fun, [-1.0, 1.0], bounds=bounds, method='SLSQP')
390: 
391:     def test_bounds_clipping(self):
392:         #
393:         # SLSQP returns bogus results for initial guess out of bounds, gh-6859
394:         #
395:         def f(x):
396:             return (x[0] - 1)**2
397: 
398:         sol = minimize(f, [10], method='slsqp', bounds=[(None, 0)])
399:         assert_(sol.success)
400:         assert_allclose(sol.x, 0, atol=1e-10)
401: 
402:         sol = minimize(f, [-10], method='slsqp', bounds=[(2, None)])
403:         assert_(sol.success)
404:         assert_allclose(sol.x, 2, atol=1e-10)
405: 
406:         sol = minimize(f, [-10], method='slsqp', bounds=[(None, 0)])
407:         assert_(sol.success)
408:         assert_allclose(sol.x, 0, atol=1e-10)
409: 
410:         sol = minimize(f, [10], method='slsqp', bounds=[(2, None)])
411:         assert_(sol.success)
412:         assert_allclose(sol.x, 2, atol=1e-10)
413: 
414:         sol = minimize(f, [-0.5], method='slsqp', bounds=[(-1, 0)])
415:         assert_(sol.success)
416:         assert_allclose(sol.x, 0, atol=1e-10)
417: 
418:         sol = minimize(f, [10], method='slsqp', bounds=[(-1, 0)])
419:         assert_(sol.success)
420:         assert_allclose(sol.x, 0, atol=1e-10)
421: 
422:     def test_infeasible_initial(self):
423:         # Check SLSQP behavior with infeasible initial point
424:         def f(x):
425:             x, = x
426:             return x*x - 2*x + 1
427: 
428:         cons_u = [{'type': 'ineq', 'fun': lambda x: 0 - x}]
429:         cons_l = [{'type': 'ineq', 'fun': lambda x: x - 2}]
430:         cons_ul = [{'type': 'ineq', 'fun': lambda x: 0 - x},
431:                    {'type': 'ineq', 'fun': lambda x: x + 1}]
432: 
433:         sol = minimize(f, [10], method='slsqp', constraints=cons_u)
434:         assert_(sol.success)
435:         assert_allclose(sol.x, 0, atol=1e-10)
436: 
437:         sol = minimize(f, [-10], method='slsqp', constraints=cons_l)
438:         assert_(sol.success)
439:         assert_allclose(sol.x, 2, atol=1e-10)
440: 
441:         sol = minimize(f, [-10], method='slsqp', constraints=cons_u)
442:         assert_(sol.success)
443:         assert_allclose(sol.x, 0, atol=1e-10)
444: 
445:         sol = minimize(f, [10], method='slsqp', constraints=cons_l)
446:         assert_(sol.success)
447:         assert_allclose(sol.x, 2, atol=1e-10)
448: 
449:         sol = minimize(f, [-0.5], method='slsqp', constraints=cons_ul)
450:         assert_(sol.success)
451:         assert_allclose(sol.x, 0, atol=1e-10)
452: 
453:         sol = minimize(f, [10], method='slsqp', constraints=cons_ul)
454:         assert_(sol.success)
455:         assert_allclose(sol.x, 0, atol=1e-10)
456: 
457: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_229622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nUnit test for SLSQP optimization.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import pytest' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229623 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_229623) is not StypyTypeError):

    if (import_229623 != 'pyd_module'):
        __import__(import_229623)
        sys_modules_229624 = sys.modules[import_229623]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_229624.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_229623)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_, assert_array_almost_equal, assert_allclose, assert_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229625 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_229625) is not StypyTypeError):

    if (import_229625 != 'pyd_module'):
        __import__(import_229625)
        sys_modules_229626 = sys.modules[import_229625]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_229626.module_type_store, module_type_store, ['assert_', 'assert_array_almost_equal', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_229626, sys_modules_229626.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_array_almost_equal, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_array_almost_equal', 'assert_allclose', 'assert_equal'], [assert_, assert_array_almost_equal, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_229625)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from pytest import assert_raises' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229627 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_229627) is not StypyTypeError):

    if (import_229627 != 'pyd_module'):
        __import__(import_229627)
        sys_modules_229628 = sys.modules[import_229627]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_229628.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_229628, sys_modules_229628.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_229627)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229629 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_229629) is not StypyTypeError):

    if (import_229629 != 'pyd_module'):
        __import__(import_229629)
        sys_modules_229630 = sys.modules[import_229629]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_229630.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_229629)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.optimize import fmin_slsqp, minimize' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_229631 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize')

if (type(import_229631) is not StypyTypeError):

    if (import_229631 != 'pyd_module'):
        __import__(import_229631)
        sys_modules_229632 = sys.modules[import_229631]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize', sys_modules_229632.module_type_store, module_type_store, ['fmin_slsqp', 'minimize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_229632, sys_modules_229632.module_type_store, module_type_store)
    else:
        from scipy.optimize import fmin_slsqp, minimize

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize', None, module_type_store, ['fmin_slsqp', 'minimize'], [fmin_slsqp, minimize])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.optimize', import_229631)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'MyCallBack' class

class MyCallBack(object, ):
    str_229633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', "pass a custom callback function\n\n    This makes sure it's being used.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
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

        
        # Assigning a Name to a Attribute (line 21):
        
        # Assigning a Name to a Attribute (line 21):
        # Getting the type of 'False' (line 21)
        False_229634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'False')
        # Getting the type of 'self' (line 21)
        self_229635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member 'been_called' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_229635, 'been_called', False_229634)
        
        # Assigning a Num to a Attribute (line 22):
        
        # Assigning a Num to a Attribute (line 22):
        int_229636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'int')
        # Getting the type of 'self' (line 22)
        self_229637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'ncalls' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_229637, 'ncalls', int_229636)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MyCallBack.__call__.__dict__.__setitem__('stypy_localization', localization)
        MyCallBack.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MyCallBack.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MyCallBack.__call__.__dict__.__setitem__('stypy_function_name', 'MyCallBack.__call__')
        MyCallBack.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        MyCallBack.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MyCallBack.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MyCallBack.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MyCallBack.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MyCallBack.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MyCallBack.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MyCallBack.__call__', ['x'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 25):
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'True' (line 25)
        True_229638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'True')
        # Getting the type of 'self' (line 25)
        self_229639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'been_called' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_229639, 'been_called', True_229638)
        
        # Getting the type of 'self' (line 26)
        self_229640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Obtaining the member 'ncalls' of a type (line 26)
        ncalls_229641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_229640, 'ncalls')
        int_229642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'int')
        # Applying the binary operator '+=' (line 26)
        result_iadd_229643 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 8), '+=', ncalls_229641, int_229642)
        # Getting the type of 'self' (line 26)
        self_229644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'ncalls' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_229644, 'ncalls', result_iadd_229643)
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_229645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229645)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_229645


# Assigning a type to the variable 'MyCallBack' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'MyCallBack', MyCallBack)
# Declaration of the 'TestSLSQP' class

class TestSLSQP(object, ):
    str_229646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '\n    Test SLSQP algorithm using Example 14.4 from Numerical Methods for\n    Engineers by Steven Chapra and Raymond Canale.\n    This example maximizes the function f(x) = 2*x*y + 2*x - x**2 - 2*y**2,\n    which has a maximum at x=2, y=1.\n    ')

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.setup_method')
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Dict to a Attribute (line 37):
        
        # Assigning a Dict to a Attribute (line 37):
        
        # Obtaining an instance of the builtin type 'dict' (line 37)
        dict_229647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 37)
        # Adding element type (key, value) (line 37)
        str_229648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'str', 'disp')
        # Getting the type of 'False' (line 37)
        False_229649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'False')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 20), dict_229647, (str_229648, False_229649))
        
        # Getting the type of 'self' (line 37)
        self_229650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'opts' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_229650, 'opts', dict_229647)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_229651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229651)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_229651


    @norecursion
    def fun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'float')
        defaults = [float_229652]
        # Create a new context for function 'fun'
        module_type_store = module_type_store.open_function_context('fun', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.fun.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.fun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.fun.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.fun.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.fun')
        TestSLSQP.fun.__dict__.__setitem__('stypy_param_names_list', ['d', 'sign'])
        TestSLSQP.fun.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.fun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.fun.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.fun.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.fun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.fun.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.fun', ['d', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun', localization, ['d', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun(...)' code ##################

        str_229653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n        Arguments:\n        d     - A list of two elements, where d[0] represents x and d[1] represents y\n                 in the following equation.\n        sign - A multiplier for f.  Since we want to optimize it, and the scipy\n               optimizers can only minimize functions, we need to multiply it by\n               -1 to achieve the desired solution\n        Returns:\n        2*x*y + 2*x - x**2 - 2*y**2\n\n        ')
        
        # Assigning a Subscript to a Name (line 51):
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_229654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'int')
        # Getting the type of 'd' (line 51)
        d_229655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'd')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___229656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), d_229655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_229657 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), getitem___229656, int_229654)
        
        # Assigning a type to the variable 'x' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'x', subscript_call_result_229657)
        
        # Assigning a Subscript to a Name (line 52):
        
        # Assigning a Subscript to a Name (line 52):
        
        # Obtaining the type of the subscript
        int_229658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 14), 'int')
        # Getting the type of 'd' (line 52)
        d_229659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'd')
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___229660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), d_229659, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_229661 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), getitem___229660, int_229658)
        
        # Assigning a type to the variable 'y' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'y', subscript_call_result_229661)
        # Getting the type of 'sign' (line 53)
        sign_229662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'sign')
        int_229663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 21), 'int')
        # Getting the type of 'x' (line 53)
        x_229664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'x')
        # Applying the binary operator '*' (line 53)
        result_mul_229665 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 21), '*', int_229663, x_229664)
        
        # Getting the type of 'y' (line 53)
        y_229666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'y')
        # Applying the binary operator '*' (line 53)
        result_mul_229667 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 24), '*', result_mul_229665, y_229666)
        
        int_229668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'int')
        # Getting the type of 'x' (line 53)
        x_229669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'x')
        # Applying the binary operator '*' (line 53)
        result_mul_229670 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 29), '*', int_229668, x_229669)
        
        # Applying the binary operator '+' (line 53)
        result_add_229671 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 21), '+', result_mul_229667, result_mul_229670)
        
        # Getting the type of 'x' (line 53)
        x_229672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 35), 'x')
        int_229673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'int')
        # Applying the binary operator '**' (line 53)
        result_pow_229674 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 35), '**', x_229672, int_229673)
        
        # Applying the binary operator '-' (line 53)
        result_sub_229675 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 33), '-', result_add_229671, result_pow_229674)
        
        int_229676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 42), 'int')
        # Getting the type of 'y' (line 53)
        y_229677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 44), 'y')
        int_229678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 47), 'int')
        # Applying the binary operator '**' (line 53)
        result_pow_229679 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 44), '**', y_229677, int_229678)
        
        # Applying the binary operator '*' (line 53)
        result_mul_229680 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 42), '*', int_229676, result_pow_229679)
        
        # Applying the binary operator '-' (line 53)
        result_sub_229681 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 40), '-', result_sub_229675, result_mul_229680)
        
        # Applying the binary operator '*' (line 53)
        result_mul_229682 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 15), '*', sign_229662, result_sub_229681)
        
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', result_mul_229682)
        
        # ################# End of 'fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_229683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229683)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun'
        return stypy_return_type_229683


    @norecursion
    def jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'float')
        defaults = [float_229684]
        # Create a new context for function 'jac'
        module_type_store = module_type_store.open_function_context('jac', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.jac.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.jac.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.jac')
        TestSLSQP.jac.__dict__.__setitem__('stypy_param_names_list', ['d', 'sign'])
        TestSLSQP.jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.jac.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.jac', ['d', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jac', localization, ['d', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jac(...)' code ##################

        str_229685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', '\n        This is the derivative of fun, returning a numpy array\n        representing df/dx and df/dy.\n\n        ')
        
        # Assigning a Subscript to a Name (line 61):
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_229686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 14), 'int')
        # Getting the type of 'd' (line 61)
        d_229687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'd')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___229688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), d_229687, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_229689 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), getitem___229688, int_229686)
        
        # Assigning a type to the variable 'x' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'x', subscript_call_result_229689)
        
        # Assigning a Subscript to a Name (line 62):
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        int_229690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 14), 'int')
        # Getting the type of 'd' (line 62)
        d_229691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'd')
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___229692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), d_229691, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_229693 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), getitem___229692, int_229690)
        
        # Assigning a type to the variable 'y' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'y', subscript_call_result_229693)
        
        # Assigning a BinOp to a Name (line 63):
        
        # Assigning a BinOp to a Name (line 63):
        # Getting the type of 'sign' (line 63)
        sign_229694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'sign')
        int_229695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
        # Getting the type of 'x' (line 63)
        x_229696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'x')
        # Applying the binary operator '*' (line 63)
        result_mul_229697 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 21), '*', int_229695, x_229696)
        
        int_229698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'int')
        # Getting the type of 'y' (line 63)
        y_229699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'y')
        # Applying the binary operator '*' (line 63)
        result_mul_229700 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 28), '*', int_229698, y_229699)
        
        # Applying the binary operator '+' (line 63)
        result_add_229701 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 21), '+', result_mul_229697, result_mul_229700)
        
        int_229702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'int')
        # Applying the binary operator '+' (line 63)
        result_add_229703 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 32), '+', result_add_229701, int_229702)
        
        # Applying the binary operator '*' (line 63)
        result_mul_229704 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 15), '*', sign_229694, result_add_229703)
        
        # Assigning a type to the variable 'dfdx' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'dfdx', result_mul_229704)
        
        # Assigning a BinOp to a Name (line 64):
        
        # Assigning a BinOp to a Name (line 64):
        # Getting the type of 'sign' (line 64)
        sign_229705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'sign')
        int_229706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 21), 'int')
        # Getting the type of 'x' (line 64)
        x_229707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'x')
        # Applying the binary operator '*' (line 64)
        result_mul_229708 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 21), '*', int_229706, x_229707)
        
        int_229709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 27), 'int')
        # Getting the type of 'y' (line 64)
        y_229710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'y')
        # Applying the binary operator '*' (line 64)
        result_mul_229711 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 27), '*', int_229709, y_229710)
        
        # Applying the binary operator '-' (line 64)
        result_sub_229712 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 21), '-', result_mul_229708, result_mul_229711)
        
        # Applying the binary operator '*' (line 64)
        result_mul_229713 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 15), '*', sign_229705, result_sub_229712)
        
        # Assigning a type to the variable 'dfdy' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'dfdy', result_mul_229713)
        
        # Call to array(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_229716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        # Getting the type of 'dfdx' (line 65)
        dfdx_229717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'dfdx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_229716, dfdx_229717)
        # Adding element type (line 65)
        # Getting the type of 'dfdy' (line 65)
        dfdy_229718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'dfdy', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_229716, dfdy_229718)
        
        # Getting the type of 'float' (line 65)
        float_229719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 38), 'float', False)
        # Processing the call keyword arguments (line 65)
        kwargs_229720 = {}
        # Getting the type of 'np' (line 65)
        np_229714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 65)
        array_229715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), np_229714, 'array')
        # Calling array(args, kwargs) (line 65)
        array_call_result_229721 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), array_229715, *[list_229716, float_229719], **kwargs_229720)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', array_call_result_229721)
        
        # ################# End of 'jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jac' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_229722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229722)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jac'
        return stypy_return_type_229722


    @norecursion
    def fun_and_jac(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'float')
        defaults = [float_229723]
        # Create a new context for function 'fun_and_jac'
        module_type_store = module_type_store.open_function_context('fun_and_jac', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.fun_and_jac')
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_param_names_list', ['d', 'sign'])
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.fun_and_jac.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.fun_and_jac', ['d', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun_and_jac', localization, ['d', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun_and_jac(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_229724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        
        # Call to fun(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'd' (line 68)
        d_229727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'd', False)
        # Getting the type of 'sign' (line 68)
        sign_229728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'sign', False)
        # Processing the call keyword arguments (line 68)
        kwargs_229729 = {}
        # Getting the type of 'self' (line 68)
        self_229725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'self', False)
        # Obtaining the member 'fun' of a type (line 68)
        fun_229726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), self_229725, 'fun')
        # Calling fun(args, kwargs) (line 68)
        fun_call_result_229730 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), fun_229726, *[d_229727, sign_229728], **kwargs_229729)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 15), tuple_229724, fun_call_result_229730)
        # Adding element type (line 68)
        
        # Call to jac(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'd' (line 68)
        d_229733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 43), 'd', False)
        # Getting the type of 'sign' (line 68)
        sign_229734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 46), 'sign', False)
        # Processing the call keyword arguments (line 68)
        kwargs_229735 = {}
        # Getting the type of 'self' (line 68)
        self_229731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'self', False)
        # Obtaining the member 'jac' of a type (line 68)
        jac_229732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 34), self_229731, 'jac')
        # Calling jac(args, kwargs) (line 68)
        jac_call_result_229736 = invoke(stypy.reporting.localization.Localization(__file__, 68, 34), jac_229732, *[d_229733, sign_229734], **kwargs_229735)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 15), tuple_229724, jac_call_result_229736)
        
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type', tuple_229724)
        
        # ################# End of 'fun_and_jac(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun_and_jac' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_229737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229737)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun_and_jac'
        return stypy_return_type_229737


    @norecursion
    def f_eqcon(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'float')
        defaults = [float_229738]
        # Create a new context for function 'f_eqcon'
        module_type_store = module_type_store.open_function_context('f_eqcon', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.f_eqcon')
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_param_names_list', ['x', 'sign'])
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.f_eqcon.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.f_eqcon', ['x', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f_eqcon', localization, ['x', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f_eqcon(...)' code ##################

        str_229739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'str', ' Equality constraint ')
        
        # Call to array(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_229742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        
        # Obtaining the type of the subscript
        int_229743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 27), 'int')
        # Getting the type of 'x' (line 72)
        x_229744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___229745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), x_229744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_229746 = invoke(stypy.reporting.localization.Localization(__file__, 72, 25), getitem___229745, int_229743)
        
        
        # Obtaining the type of the subscript
        int_229747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 34), 'int')
        # Getting the type of 'x' (line 72)
        x_229748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___229749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), x_229748, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_229750 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), getitem___229749, int_229747)
        
        # Applying the binary operator '-' (line 72)
        result_sub_229751 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 25), '-', subscript_call_result_229746, subscript_call_result_229750)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_229742, result_sub_229751)
        
        # Processing the call keyword arguments (line 72)
        kwargs_229752 = {}
        # Getting the type of 'np' (line 72)
        np_229740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 72)
        array_229741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), np_229740, 'array')
        # Calling array(args, kwargs) (line 72)
        array_call_result_229753 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), array_229741, *[list_229742], **kwargs_229752)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', array_call_result_229753)
        
        # ################# End of 'f_eqcon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f_eqcon' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_229754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f_eqcon'
        return stypy_return_type_229754


    @norecursion
    def fprime_eqcon(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 35), 'float')
        defaults = [float_229755]
        # Create a new context for function 'fprime_eqcon'
        module_type_store = module_type_store.open_function_context('fprime_eqcon', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.fprime_eqcon')
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_param_names_list', ['x', 'sign'])
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.fprime_eqcon.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.fprime_eqcon', ['x', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fprime_eqcon', localization, ['x', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fprime_eqcon(...)' code ##################

        str_229756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'str', ' Equality constraint, derivative ')
        
        # Call to array(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_229759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_229760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_229761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 25), list_229760, int_229761)
        # Adding element type (line 76)
        int_229762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 25), list_229760, int_229762)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 24), list_229759, list_229760)
        
        # Processing the call keyword arguments (line 76)
        kwargs_229763 = {}
        # Getting the type of 'np' (line 76)
        np_229757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 76)
        array_229758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 15), np_229757, 'array')
        # Calling array(args, kwargs) (line 76)
        array_call_result_229764 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), array_229758, *[list_229759], **kwargs_229763)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', array_call_result_229764)
        
        # ################# End of 'fprime_eqcon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fprime_eqcon' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_229765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fprime_eqcon'
        return stypy_return_type_229765


    @norecursion
    def f_eqcon_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 37), 'float')
        defaults = [float_229766]
        # Create a new context for function 'f_eqcon_scalar'
        module_type_store = module_type_store.open_function_context('f_eqcon_scalar', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.f_eqcon_scalar')
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_param_names_list', ['x', 'sign'])
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.f_eqcon_scalar.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.f_eqcon_scalar', ['x', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f_eqcon_scalar', localization, ['x', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f_eqcon_scalar(...)' code ##################

        str_229767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'str', ' Scalar equality constraint ')
        
        # Obtaining the type of the subscript
        int_229768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 37), 'int')
        
        # Call to f_eqcon(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'x' (line 80)
        x_229771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'x', False)
        # Getting the type of 'sign' (line 80)
        sign_229772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'sign', False)
        # Processing the call keyword arguments (line 80)
        kwargs_229773 = {}
        # Getting the type of 'self' (line 80)
        self_229769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 80)
        f_eqcon_229770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), self_229769, 'f_eqcon')
        # Calling f_eqcon(args, kwargs) (line 80)
        f_eqcon_call_result_229774 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), f_eqcon_229770, *[x_229771, sign_229772], **kwargs_229773)
        
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___229775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), f_eqcon_call_result_229774, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_229776 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), getitem___229775, int_229768)
        
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', subscript_call_result_229776)
        
        # ################# End of 'f_eqcon_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f_eqcon_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_229777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229777)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f_eqcon_scalar'
        return stypy_return_type_229777


    @norecursion
    def fprime_eqcon_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 42), 'float')
        defaults = [float_229778]
        # Create a new context for function 'fprime_eqcon_scalar'
        module_type_store = module_type_store.open_function_context('fprime_eqcon_scalar', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.fprime_eqcon_scalar')
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_param_names_list', ['x', 'sign'])
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.fprime_eqcon_scalar.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.fprime_eqcon_scalar', ['x', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fprime_eqcon_scalar', localization, ['x', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fprime_eqcon_scalar(...)' code ##################

        str_229779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'str', ' Scalar equality constraint, derivative ')
        
        # Call to tolist(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_229790 = {}
        
        # Obtaining the type of the subscript
        int_229780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 42), 'int')
        
        # Call to fprime_eqcon(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'x' (line 84)
        x_229783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 33), 'x', False)
        # Getting the type of 'sign' (line 84)
        sign_229784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'sign', False)
        # Processing the call keyword arguments (line 84)
        kwargs_229785 = {}
        # Getting the type of 'self' (line 84)
        self_229781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'self', False)
        # Obtaining the member 'fprime_eqcon' of a type (line 84)
        fprime_eqcon_229782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 15), self_229781, 'fprime_eqcon')
        # Calling fprime_eqcon(args, kwargs) (line 84)
        fprime_eqcon_call_result_229786 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), fprime_eqcon_229782, *[x_229783, sign_229784], **kwargs_229785)
        
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___229787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 15), fprime_eqcon_call_result_229786, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_229788 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), getitem___229787, int_229780)
        
        # Obtaining the member 'tolist' of a type (line 84)
        tolist_229789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 15), subscript_call_result_229788, 'tolist')
        # Calling tolist(args, kwargs) (line 84)
        tolist_call_result_229791 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), tolist_229789, *[], **kwargs_229790)
        
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', tolist_call_result_229791)
        
        # ################# End of 'fprime_eqcon_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fprime_eqcon_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_229792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229792)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fprime_eqcon_scalar'
        return stypy_return_type_229792


    @norecursion
    def f_ieqcon(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'float')
        defaults = [float_229793]
        # Create a new context for function 'f_ieqcon'
        module_type_store = module_type_store.open_function_context('f_ieqcon', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.f_ieqcon')
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_param_names_list', ['x', 'sign'])
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.f_ieqcon.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.f_ieqcon', ['x', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f_ieqcon', localization, ['x', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f_ieqcon(...)' code ##################

        str_229794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'str', ' Inequality constraint ')
        
        # Call to array(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_229797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        
        # Obtaining the type of the subscript
        int_229798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 27), 'int')
        # Getting the type of 'x' (line 88)
        x_229799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___229800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), x_229799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_229801 = invoke(stypy.reporting.localization.Localization(__file__, 88, 25), getitem___229800, int_229798)
        
        
        # Obtaining the type of the subscript
        int_229802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 34), 'int')
        # Getting the type of 'x' (line 88)
        x_229803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___229804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 32), x_229803, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_229805 = invoke(stypy.reporting.localization.Localization(__file__, 88, 32), getitem___229804, int_229802)
        
        # Applying the binary operator '-' (line 88)
        result_sub_229806 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 25), '-', subscript_call_result_229801, subscript_call_result_229805)
        
        float_229807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 39), 'float')
        # Applying the binary operator '-' (line 88)
        result_sub_229808 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 37), '-', result_sub_229806, float_229807)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 24), list_229797, result_sub_229808)
        
        # Processing the call keyword arguments (line 88)
        kwargs_229809 = {}
        # Getting the type of 'np' (line 88)
        np_229795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 88)
        array_229796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), np_229795, 'array')
        # Calling array(args, kwargs) (line 88)
        array_call_result_229810 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), array_229796, *[list_229797], **kwargs_229809)
        
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', array_call_result_229810)
        
        # ################# End of 'f_ieqcon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f_ieqcon' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_229811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229811)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f_ieqcon'
        return stypy_return_type_229811


    @norecursion
    def fprime_ieqcon(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_229812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 36), 'float')
        defaults = [float_229812]
        # Create a new context for function 'fprime_ieqcon'
        module_type_store = module_type_store.open_function_context('fprime_ieqcon', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.fprime_ieqcon')
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_param_names_list', ['x', 'sign'])
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.fprime_ieqcon.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.fprime_ieqcon', ['x', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fprime_ieqcon', localization, ['x', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fprime_ieqcon(...)' code ##################

        str_229813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'str', ' Inequality constraint, derivative ')
        
        # Call to array(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_229816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_229817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        int_229818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 25), list_229817, int_229818)
        # Adding element type (line 92)
        int_229819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 25), list_229817, int_229819)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 24), list_229816, list_229817)
        
        # Processing the call keyword arguments (line 92)
        kwargs_229820 = {}
        # Getting the type of 'np' (line 92)
        np_229814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 92)
        array_229815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 15), np_229814, 'array')
        # Calling array(args, kwargs) (line 92)
        array_call_result_229821 = invoke(stypy.reporting.localization.Localization(__file__, 92, 15), array_229815, *[list_229816], **kwargs_229820)
        
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', array_call_result_229821)
        
        # ################# End of 'fprime_ieqcon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fprime_ieqcon' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_229822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fprime_ieqcon'
        return stypy_return_type_229822


    @norecursion
    def f_ieqcon2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f_ieqcon2'
        module_type_store = module_type_store.open_function_context('f_ieqcon2', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.f_ieqcon2')
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.f_ieqcon2.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.f_ieqcon2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f_ieqcon2', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f_ieqcon2(...)' code ##################

        str_229823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'str', ' Vector inequality constraint ')
        
        # Call to asarray(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'x' (line 96)
        x_229826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'x', False)
        # Processing the call keyword arguments (line 96)
        kwargs_229827 = {}
        # Getting the type of 'np' (line 96)
        np_229824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'np', False)
        # Obtaining the member 'asarray' of a type (line 96)
        asarray_229825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), np_229824, 'asarray')
        # Calling asarray(args, kwargs) (line 96)
        asarray_call_result_229828 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), asarray_229825, *[x_229826], **kwargs_229827)
        
        # Assigning a type to the variable 'stypy_return_type' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'stypy_return_type', asarray_call_result_229828)
        
        # ################# End of 'f_ieqcon2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f_ieqcon2' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_229829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229829)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f_ieqcon2'
        return stypy_return_type_229829


    @norecursion
    def fprime_ieqcon2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fprime_ieqcon2'
        module_type_store = module_type_store.open_function_context('fprime_ieqcon2', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.fprime_ieqcon2')
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.fprime_ieqcon2.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.fprime_ieqcon2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fprime_ieqcon2', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fprime_ieqcon2(...)' code ##################

        str_229830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'str', ' Vector inequality constraint, derivative ')
        
        # Call to identity(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Obtaining the type of the subscript
        int_229833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 35), 'int')
        # Getting the type of 'x' (line 100)
        x_229834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'x', False)
        # Obtaining the member 'shape' of a type (line 100)
        shape_229835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 27), x_229834, 'shape')
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___229836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 27), shape_229835, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_229837 = invoke(stypy.reporting.localization.Localization(__file__, 100, 27), getitem___229836, int_229833)
        
        # Processing the call keyword arguments (line 100)
        kwargs_229838 = {}
        # Getting the type of 'np' (line 100)
        np_229831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'np', False)
        # Obtaining the member 'identity' of a type (line 100)
        identity_229832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), np_229831, 'identity')
        # Calling identity(args, kwargs) (line 100)
        identity_call_result_229839 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), identity_229832, *[subscript_call_result_229837], **kwargs_229838)
        
        # Assigning a type to the variable 'stypy_return_type' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'stypy_return_type', identity_call_result_229839)
        
        # ################# End of 'fprime_ieqcon2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fprime_ieqcon2' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_229840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fprime_ieqcon2'
        return stypy_return_type_229840


    @norecursion
    def test_minimize_unbounded_approximated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_unbounded_approximated'
        module_type_store = module_type_store.open_function_context('test_minimize_unbounded_approximated', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_unbounded_approximated')
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_unbounded_approximated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_unbounded_approximated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_unbounded_approximated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_unbounded_approximated(...)' code ##################

        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to minimize(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'self' (line 105)
        self_229842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 105)
        fun_229843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 23), self_229842, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_229844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        float_229845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 33), list_229844, float_229845)
        # Adding element type (line 105)
        float_229846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 33), list_229844, float_229846)
        
        # Processing the call keyword arguments (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_229847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        float_229848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 52), tuple_229847, float_229848)
        
        keyword_229849 = tuple_229847
        str_229850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 30), 'str', 'SLSQP')
        keyword_229851 = str_229850
        # Getting the type of 'self' (line 106)
        self_229852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 47), 'self', False)
        # Obtaining the member 'opts' of a type (line 106)
        opts_229853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 47), self_229852, 'opts')
        keyword_229854 = opts_229853
        kwargs_229855 = {'args': keyword_229849, 'method': keyword_229851, 'options': keyword_229854}
        # Getting the type of 'minimize' (line 105)
        minimize_229841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 105)
        minimize_call_result_229856 = invoke(stypy.reporting.localization.Localization(__file__, 105, 14), minimize_229841, *[fun_229843, list_229844], **kwargs_229855)
        
        # Assigning a type to the variable 'res' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'res', minimize_call_result_229856)
        
        # Call to assert_(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Obtaining the type of the subscript
        str_229858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 20), 'str', 'success')
        # Getting the type of 'res' (line 107)
        res_229859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___229860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), res_229859, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_229861 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), getitem___229860, str_229858)
        
        
        # Obtaining the type of the subscript
        str_229862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 36), 'str', 'message')
        # Getting the type of 'res' (line 107)
        res_229863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___229864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 32), res_229863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_229865 = invoke(stypy.reporting.localization.Localization(__file__, 107, 32), getitem___229864, str_229862)
        
        # Processing the call keyword arguments (line 107)
        kwargs_229866 = {}
        # Getting the type of 'assert_' (line 107)
        assert__229857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 107)
        assert__call_result_229867 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert__229857, *[subscript_call_result_229861, subscript_call_result_229865], **kwargs_229866)
        
        
        # Call to assert_allclose(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'res' (line 108)
        res_229869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 108)
        x_229870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 24), res_229869, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_229871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        int_229872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), list_229871, int_229872)
        # Adding element type (line 108)
        int_229873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), list_229871, int_229873)
        
        # Processing the call keyword arguments (line 108)
        kwargs_229874 = {}
        # Getting the type of 'assert_allclose' (line 108)
        assert_allclose_229868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 108)
        assert_allclose_call_result_229875 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assert_allclose_229868, *[x_229870, list_229871], **kwargs_229874)
        
        
        # ################# End of 'test_minimize_unbounded_approximated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_unbounded_approximated' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_229876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229876)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_unbounded_approximated'
        return stypy_return_type_229876


    @norecursion
    def test_minimize_unbounded_given(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_unbounded_given'
        module_type_store = module_type_store.open_function_context('test_minimize_unbounded_given', 110, 4, False)
        # Assigning a type to the variable 'self' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_unbounded_given')
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_unbounded_given.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_unbounded_given', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_unbounded_given', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_unbounded_given(...)' code ##################

        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to minimize(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_229878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 112)
        fun_229879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 23), self_229878, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_229880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        float_229881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 33), list_229880, float_229881)
        # Adding element type (line 112)
        float_229882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 33), list_229880, float_229882)
        
        # Processing the call keyword arguments (line 112)
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_229883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        float_229884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 52), tuple_229883, float_229884)
        
        keyword_229885 = tuple_229883
        # Getting the type of 'self' (line 113)
        self_229886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'self', False)
        # Obtaining the member 'jac' of a type (line 113)
        jac_229887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 27), self_229886, 'jac')
        keyword_229888 = jac_229887
        str_229889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'str', 'SLSQP')
        keyword_229890 = str_229889
        # Getting the type of 'self' (line 113)
        self_229891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 61), 'self', False)
        # Obtaining the member 'opts' of a type (line 113)
        opts_229892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 61), self_229891, 'opts')
        keyword_229893 = opts_229892
        kwargs_229894 = {'args': keyword_229885, 'options': keyword_229893, 'jac': keyword_229888, 'method': keyword_229890}
        # Getting the type of 'minimize' (line 112)
        minimize_229877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 112)
        minimize_call_result_229895 = invoke(stypy.reporting.localization.Localization(__file__, 112, 14), minimize_229877, *[fun_229879, list_229880], **kwargs_229894)
        
        # Assigning a type to the variable 'res' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'res', minimize_call_result_229895)
        
        # Call to assert_(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Obtaining the type of the subscript
        str_229897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 20), 'str', 'success')
        # Getting the type of 'res' (line 114)
        res_229898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___229899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), res_229898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_229900 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), getitem___229899, str_229897)
        
        
        # Obtaining the type of the subscript
        str_229901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 36), 'str', 'message')
        # Getting the type of 'res' (line 114)
        res_229902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___229903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 32), res_229902, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_229904 = invoke(stypy.reporting.localization.Localization(__file__, 114, 32), getitem___229903, str_229901)
        
        # Processing the call keyword arguments (line 114)
        kwargs_229905 = {}
        # Getting the type of 'assert_' (line 114)
        assert__229896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 114)
        assert__call_result_229906 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assert__229896, *[subscript_call_result_229900, subscript_call_result_229904], **kwargs_229905)
        
        
        # Call to assert_allclose(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'res' (line 115)
        res_229908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 115)
        x_229909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 24), res_229908, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_229910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        int_229911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 31), list_229910, int_229911)
        # Adding element type (line 115)
        int_229912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 31), list_229910, int_229912)
        
        # Processing the call keyword arguments (line 115)
        kwargs_229913 = {}
        # Getting the type of 'assert_allclose' (line 115)
        assert_allclose_229907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 115)
        assert_allclose_call_result_229914 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), assert_allclose_229907, *[x_229909, list_229910], **kwargs_229913)
        
        
        # ################# End of 'test_minimize_unbounded_given(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_unbounded_given' in the type store
        # Getting the type of 'stypy_return_type' (line 110)
        stypy_return_type_229915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229915)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_unbounded_given'
        return stypy_return_type_229915


    @norecursion
    def test_minimize_bounded_approximated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_bounded_approximated'
        module_type_store = module_type_store.open_function_context('test_minimize_bounded_approximated', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_bounded_approximated')
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_bounded_approximated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_bounded_approximated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_bounded_approximated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_bounded_approximated(...)' code ##################

        
        # Call to errstate(...): (line 119)
        # Processing the call keyword arguments (line 119)
        str_229918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 33), 'str', 'ignore')
        keyword_229919 = str_229918
        kwargs_229920 = {'invalid': keyword_229919}
        # Getting the type of 'np' (line 119)
        np_229916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 13), 'np', False)
        # Obtaining the member 'errstate' of a type (line 119)
        errstate_229917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 13), np_229916, 'errstate')
        # Calling errstate(args, kwargs) (line 119)
        errstate_call_result_229921 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), errstate_229917, *[], **kwargs_229920)
        
        with_229922 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 119, 13), errstate_call_result_229921, 'with parameter', '__enter__', '__exit__')

        if with_229922:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 119)
            enter___229923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 13), errstate_call_result_229921, '__enter__')
            with_enter_229924 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), enter___229923)
            
            # Assigning a Call to a Name (line 120):
            
            # Assigning a Call to a Name (line 120):
            
            # Call to minimize(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'self' (line 120)
            self_229926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'self', False)
            # Obtaining the member 'fun' of a type (line 120)
            fun_229927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), self_229926, 'fun')
            
            # Obtaining an instance of the builtin type 'list' (line 120)
            list_229928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 37), 'list')
            # Adding type elements to the builtin type 'list' instance (line 120)
            # Adding element type (line 120)
            float_229929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 38), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 37), list_229928, float_229929)
            # Adding element type (line 120)
            float_229930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 44), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 37), list_229928, float_229930)
            
            # Processing the call keyword arguments (line 120)
            
            # Obtaining an instance of the builtin type 'tuple' (line 120)
            tuple_229931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 120)
            # Adding element type (line 120)
            float_229932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 56), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 56), tuple_229931, float_229932)
            
            keyword_229933 = tuple_229931
            
            # Obtaining an instance of the builtin type 'tuple' (line 121)
            tuple_229934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 121)
            # Adding element type (line 121)
            
            # Obtaining an instance of the builtin type 'tuple' (line 121)
            tuple_229935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 36), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 121)
            # Adding element type (line 121)
            float_229936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 36), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 36), tuple_229935, float_229936)
            # Adding element type (line 121)
            # Getting the type of 'None' (line 121)
            None_229937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 41), 'None', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 36), tuple_229935, None_229937)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 35), tuple_229934, tuple_229935)
            # Adding element type (line 121)
            
            # Obtaining an instance of the builtin type 'tuple' (line 121)
            tuple_229938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 121)
            # Adding element type (line 121)
            # Getting the type of 'None' (line 121)
            None_229939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 49), 'None', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 49), tuple_229938, None_229939)
            # Adding element type (line 121)
            float_229940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 55), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 49), tuple_229938, float_229940)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 35), tuple_229934, tuple_229938)
            
            keyword_229941 = tuple_229934
            str_229942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'str', 'SLSQP')
            keyword_229943 = str_229942
            # Getting the type of 'self' (line 122)
            self_229944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 51), 'self', False)
            # Obtaining the member 'opts' of a type (line 122)
            opts_229945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 51), self_229944, 'opts')
            keyword_229946 = opts_229945
            kwargs_229947 = {'args': keyword_229933, 'options': keyword_229946, 'bounds': keyword_229941, 'method': keyword_229943}
            # Getting the type of 'minimize' (line 120)
            minimize_229925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'minimize', False)
            # Calling minimize(args, kwargs) (line 120)
            minimize_call_result_229948 = invoke(stypy.reporting.localization.Localization(__file__, 120, 18), minimize_229925, *[fun_229927, list_229928], **kwargs_229947)
            
            # Assigning a type to the variable 'res' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'res', minimize_call_result_229948)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 119)
            exit___229949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 13), errstate_call_result_229921, '__exit__')
            with_exit_229950 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), exit___229949, None, None, None)

        
        # Call to assert_(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Obtaining the type of the subscript
        str_229952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'str', 'success')
        # Getting the type of 'res' (line 123)
        res_229953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___229954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), res_229953, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_229955 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), getitem___229954, str_229952)
        
        
        # Obtaining the type of the subscript
        str_229956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 36), 'str', 'message')
        # Getting the type of 'res' (line 123)
        res_229957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___229958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 32), res_229957, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_229959 = invoke(stypy.reporting.localization.Localization(__file__, 123, 32), getitem___229958, str_229956)
        
        # Processing the call keyword arguments (line 123)
        kwargs_229960 = {}
        # Getting the type of 'assert_' (line 123)
        assert__229951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 123)
        assert__call_result_229961 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), assert__229951, *[subscript_call_result_229955, subscript_call_result_229959], **kwargs_229960)
        
        
        # Call to assert_allclose(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'res' (line 124)
        res_229963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 124)
        x_229964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 24), res_229963, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_229965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        float_229966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 31), list_229965, float_229966)
        # Adding element type (line 124)
        float_229967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 31), list_229965, float_229967)
        
        # Processing the call keyword arguments (line 124)
        kwargs_229968 = {}
        # Getting the type of 'assert_allclose' (line 124)
        assert_allclose_229962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 124)
        assert_allclose_call_result_229969 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assert_allclose_229962, *[x_229964, list_229965], **kwargs_229968)
        
        
        # Call to assert_(...): (line 125)
        # Processing the call arguments (line 125)
        
        float_229971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 16), 'float')
        
        # Obtaining the type of the subscript
        int_229972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 29), 'int')
        # Getting the type of 'res' (line 125)
        res_229973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'res', False)
        # Obtaining the member 'x' of a type (line 125)
        x_229974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 23), res_229973, 'x')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___229975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 23), x_229974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_229976 = invoke(stypy.reporting.localization.Localization(__file__, 125, 23), getitem___229975, int_229972)
        
        # Applying the binary operator '<=' (line 125)
        result_le_229977 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 16), '<=', float_229971, subscript_call_result_229976)
        
        # Processing the call keyword arguments (line 125)
        kwargs_229978 = {}
        # Getting the type of 'assert_' (line 125)
        assert__229970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 125)
        assert__call_result_229979 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assert__229970, *[result_le_229977], **kwargs_229978)
        
        
        # Call to assert_(...): (line 126)
        # Processing the call arguments (line 126)
        
        
        # Obtaining the type of the subscript
        int_229981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 22), 'int')
        # Getting the type of 'res' (line 126)
        res_229982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'res', False)
        # Obtaining the member 'x' of a type (line 126)
        x_229983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 16), res_229982, 'x')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___229984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 16), x_229983, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_229985 = invoke(stypy.reporting.localization.Localization(__file__, 126, 16), getitem___229984, int_229981)
        
        float_229986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'float')
        # Applying the binary operator '<=' (line 126)
        result_le_229987 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 16), '<=', subscript_call_result_229985, float_229986)
        
        # Processing the call keyword arguments (line 126)
        kwargs_229988 = {}
        # Getting the type of 'assert_' (line 126)
        assert__229980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 126)
        assert__call_result_229989 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assert__229980, *[result_le_229987], **kwargs_229988)
        
        
        # ################# End of 'test_minimize_bounded_approximated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_bounded_approximated' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_229990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229990)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_bounded_approximated'
        return stypy_return_type_229990


    @norecursion
    def test_minimize_unbounded_combined(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_unbounded_combined'
        module_type_store = module_type_store.open_function_context('test_minimize_unbounded_combined', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_unbounded_combined')
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_unbounded_combined.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_unbounded_combined', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_unbounded_combined', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_unbounded_combined(...)' code ##################

        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to minimize(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'self' (line 130)
        self_229992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 23), 'self', False)
        # Obtaining the member 'fun_and_jac' of a type (line 130)
        fun_and_jac_229993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 23), self_229992, 'fun_and_jac')
        
        # Obtaining an instance of the builtin type 'list' (line 130)
        list_229994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 130)
        # Adding element type (line 130)
        float_229995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 41), list_229994, float_229995)
        # Adding element type (line 130)
        float_229996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 41), list_229994, float_229996)
        
        # Processing the call keyword arguments (line 130)
        
        # Obtaining an instance of the builtin type 'tuple' (line 130)
        tuple_229997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 130)
        # Adding element type (line 130)
        float_229998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 60), tuple_229997, float_229998)
        
        keyword_229999 = tuple_229997
        # Getting the type of 'True' (line 131)
        True_230000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'True', False)
        keyword_230001 = True_230000
        str_230002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 40), 'str', 'SLSQP')
        keyword_230003 = str_230002
        # Getting the type of 'self' (line 131)
        self_230004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 57), 'self', False)
        # Obtaining the member 'opts' of a type (line 131)
        opts_230005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 57), self_230004, 'opts')
        keyword_230006 = opts_230005
        kwargs_230007 = {'args': keyword_229999, 'options': keyword_230006, 'jac': keyword_230001, 'method': keyword_230003}
        # Getting the type of 'minimize' (line 130)
        minimize_229991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 130)
        minimize_call_result_230008 = invoke(stypy.reporting.localization.Localization(__file__, 130, 14), minimize_229991, *[fun_and_jac_229993, list_229994], **kwargs_230007)
        
        # Assigning a type to the variable 'res' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'res', minimize_call_result_230008)
        
        # Call to assert_(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Obtaining the type of the subscript
        str_230010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 20), 'str', 'success')
        # Getting the type of 'res' (line 132)
        res_230011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___230012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), res_230011, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_230013 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), getitem___230012, str_230010)
        
        
        # Obtaining the type of the subscript
        str_230014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 36), 'str', 'message')
        # Getting the type of 'res' (line 132)
        res_230015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___230016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 32), res_230015, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_230017 = invoke(stypy.reporting.localization.Localization(__file__, 132, 32), getitem___230016, str_230014)
        
        # Processing the call keyword arguments (line 132)
        kwargs_230018 = {}
        # Getting the type of 'assert_' (line 132)
        assert__230009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 132)
        assert__call_result_230019 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), assert__230009, *[subscript_call_result_230013, subscript_call_result_230017], **kwargs_230018)
        
        
        # Call to assert_allclose(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'res' (line 133)
        res_230021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 133)
        x_230022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), res_230021, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_230023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        int_230024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 31), list_230023, int_230024)
        # Adding element type (line 133)
        int_230025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 31), list_230023, int_230025)
        
        # Processing the call keyword arguments (line 133)
        kwargs_230026 = {}
        # Getting the type of 'assert_allclose' (line 133)
        assert_allclose_230020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 133)
        assert_allclose_call_result_230027 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assert_allclose_230020, *[x_230022, list_230023], **kwargs_230026)
        
        
        # ################# End of 'test_minimize_unbounded_combined(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_unbounded_combined' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_230028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_unbounded_combined'
        return stypy_return_type_230028


    @norecursion
    def test_minimize_equality_approximated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_equality_approximated'
        module_type_store = module_type_store.open_function_context('test_minimize_equality_approximated', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_equality_approximated')
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_equality_approximated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_equality_approximated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_equality_approximated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_equality_approximated(...)' code ##################

        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to minimize(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_230030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 137)
        fun_230031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 23), self_230030, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_230032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        float_230033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 33), list_230032, float_230033)
        # Adding element type (line 137)
        float_230034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 33), list_230032, float_230034)
        
        # Processing the call keyword arguments (line 137)
        
        # Obtaining an instance of the builtin type 'tuple' (line 137)
        tuple_230035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 137)
        # Adding element type (line 137)
        float_230036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 52), tuple_230035, float_230036)
        
        keyword_230037 = tuple_230035
        
        # Obtaining an instance of the builtin type 'dict' (line 138)
        dict_230038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 138)
        # Adding element type (key, value) (line 138)
        str_230039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'str', 'type')
        str_230040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 44), 'str', 'eq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), dict_230038, (str_230039, str_230040))
        # Adding element type (key, value) (line 138)
        str_230041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 36), 'str', 'fun')
        # Getting the type of 'self' (line 139)
        self_230042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 43), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 139)
        f_eqcon_230043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 43), self_230042, 'f_eqcon')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), dict_230038, (str_230041, f_eqcon_230043))
        # Adding element type (key, value) (line 138)
        str_230044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 36), 'str', 'args')
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_230045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        float_230046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 45), tuple_230045, float_230046)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), dict_230038, (str_230044, tuple_230045))
        
        keyword_230047 = dict_230038
        str_230048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'str', 'SLSQP')
        keyword_230049 = str_230048
        # Getting the type of 'self' (line 141)
        self_230050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 47), 'self', False)
        # Obtaining the member 'opts' of a type (line 141)
        opts_230051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 47), self_230050, 'opts')
        keyword_230052 = opts_230051
        kwargs_230053 = {'args': keyword_230037, 'options': keyword_230052, 'method': keyword_230049, 'constraints': keyword_230047}
        # Getting the type of 'minimize' (line 137)
        minimize_230029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 137)
        minimize_call_result_230054 = invoke(stypy.reporting.localization.Localization(__file__, 137, 14), minimize_230029, *[fun_230031, list_230032], **kwargs_230053)
        
        # Assigning a type to the variable 'res' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'res', minimize_call_result_230054)
        
        # Call to assert_(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Obtaining the type of the subscript
        str_230056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 20), 'str', 'success')
        # Getting the type of 'res' (line 142)
        res_230057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___230058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), res_230057, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_230059 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), getitem___230058, str_230056)
        
        
        # Obtaining the type of the subscript
        str_230060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 36), 'str', 'message')
        # Getting the type of 'res' (line 142)
        res_230061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___230062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 32), res_230061, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_230063 = invoke(stypy.reporting.localization.Localization(__file__, 142, 32), getitem___230062, str_230060)
        
        # Processing the call keyword arguments (line 142)
        kwargs_230064 = {}
        # Getting the type of 'assert_' (line 142)
        assert__230055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 142)
        assert__call_result_230065 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assert__230055, *[subscript_call_result_230059, subscript_call_result_230063], **kwargs_230064)
        
        
        # Call to assert_allclose(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'res' (line 143)
        res_230067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 143)
        x_230068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 24), res_230067, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_230069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        int_230070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 31), list_230069, int_230070)
        # Adding element type (line 143)
        int_230071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 31), list_230069, int_230071)
        
        # Processing the call keyword arguments (line 143)
        kwargs_230072 = {}
        # Getting the type of 'assert_allclose' (line 143)
        assert_allclose_230066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 143)
        assert_allclose_call_result_230073 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assert_allclose_230066, *[x_230068, list_230069], **kwargs_230072)
        
        
        # ################# End of 'test_minimize_equality_approximated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_equality_approximated' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_230074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230074)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_equality_approximated'
        return stypy_return_type_230074


    @norecursion
    def test_minimize_equality_given(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_equality_given'
        module_type_store = module_type_store.open_function_context('test_minimize_equality_given', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_equality_given')
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_equality_given.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_equality_given', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_equality_given', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_equality_given(...)' code ##################

        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to minimize(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'self' (line 147)
        self_230076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 147)
        fun_230077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 23), self_230076, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_230078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        float_230079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 33), list_230078, float_230079)
        # Adding element type (line 147)
        float_230080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 33), list_230078, float_230080)
        
        # Processing the call keyword arguments (line 147)
        # Getting the type of 'self' (line 147)
        self_230081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 50), 'self', False)
        # Obtaining the member 'jac' of a type (line 147)
        jac_230082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 50), self_230081, 'jac')
        keyword_230083 = jac_230082
        str_230084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 30), 'str', 'SLSQP')
        keyword_230085 = str_230084
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_230086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        float_230087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 45), tuple_230086, float_230087)
        
        keyword_230088 = tuple_230086
        
        # Obtaining an instance of the builtin type 'dict' (line 149)
        dict_230089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 149)
        # Adding element type (key, value) (line 149)
        str_230090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 36), 'str', 'type')
        str_230091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 44), 'str', 'eq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), dict_230089, (str_230090, str_230091))
        # Adding element type (key, value) (line 149)
        str_230092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 50), 'str', 'fun')
        # Getting the type of 'self' (line 149)
        self_230093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 56), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 149)
        f_eqcon_230094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 56), self_230093, 'f_eqcon')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), dict_230089, (str_230092, f_eqcon_230094))
        # Adding element type (key, value) (line 149)
        str_230095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 36), 'str', 'args')
        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_230096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        float_230097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 45), tuple_230096, float_230097)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), dict_230089, (str_230095, tuple_230096))
        
        keyword_230098 = dict_230089
        # Getting the type of 'self' (line 151)
        self_230099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'self', False)
        # Obtaining the member 'opts' of a type (line 151)
        opts_230100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 31), self_230099, 'opts')
        keyword_230101 = opts_230100
        kwargs_230102 = {'args': keyword_230088, 'options': keyword_230101, 'jac': keyword_230083, 'constraints': keyword_230098, 'method': keyword_230085}
        # Getting the type of 'minimize' (line 147)
        minimize_230075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 147)
        minimize_call_result_230103 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), minimize_230075, *[fun_230077, list_230078], **kwargs_230102)
        
        # Assigning a type to the variable 'res' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'res', minimize_call_result_230103)
        
        # Call to assert_(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Obtaining the type of the subscript
        str_230105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'str', 'success')
        # Getting the type of 'res' (line 152)
        res_230106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___230107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), res_230106, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_230108 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), getitem___230107, str_230105)
        
        
        # Obtaining the type of the subscript
        str_230109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 36), 'str', 'message')
        # Getting the type of 'res' (line 152)
        res_230110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___230111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 32), res_230110, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_230112 = invoke(stypy.reporting.localization.Localization(__file__, 152, 32), getitem___230111, str_230109)
        
        # Processing the call keyword arguments (line 152)
        kwargs_230113 = {}
        # Getting the type of 'assert_' (line 152)
        assert__230104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 152)
        assert__call_result_230114 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), assert__230104, *[subscript_call_result_230108, subscript_call_result_230112], **kwargs_230113)
        
        
        # Call to assert_allclose(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'res' (line 153)
        res_230116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 153)
        x_230117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 24), res_230116, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_230118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        # Adding element type (line 153)
        int_230119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 31), list_230118, int_230119)
        # Adding element type (line 153)
        int_230120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 31), list_230118, int_230120)
        
        # Processing the call keyword arguments (line 153)
        kwargs_230121 = {}
        # Getting the type of 'assert_allclose' (line 153)
        assert_allclose_230115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 153)
        assert_allclose_call_result_230122 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert_allclose_230115, *[x_230117, list_230118], **kwargs_230121)
        
        
        # ################# End of 'test_minimize_equality_given(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_equality_given' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_230123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_equality_given'
        return stypy_return_type_230123


    @norecursion
    def test_minimize_equality_given2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_equality_given2'
        module_type_store = module_type_store.open_function_context('test_minimize_equality_given2', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_equality_given2')
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_equality_given2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_equality_given2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_equality_given2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_equality_given2(...)' code ##################

        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to minimize(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_230125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 158)
        fun_230126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 23), self_230125, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_230127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        float_230128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 33), list_230127, float_230128)
        # Adding element type (line 158)
        float_230129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 33), list_230127, float_230129)
        
        # Processing the call keyword arguments (line 158)
        str_230130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 53), 'str', 'SLSQP')
        keyword_230131 = str_230130
        # Getting the type of 'self' (line 159)
        self_230132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), 'self', False)
        # Obtaining the member 'jac' of a type (line 159)
        jac_230133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 27), self_230132, 'jac')
        keyword_230134 = jac_230133
        
        # Obtaining an instance of the builtin type 'tuple' (line 159)
        tuple_230135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 159)
        # Adding element type (line 159)
        float_230136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 43), tuple_230135, float_230136)
        
        keyword_230137 = tuple_230135
        
        # Obtaining an instance of the builtin type 'dict' (line 160)
        dict_230138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 160)
        # Adding element type (key, value) (line 160)
        str_230139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'str', 'type')
        str_230140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 44), 'str', 'eq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 35), dict_230138, (str_230139, str_230140))
        # Adding element type (key, value) (line 160)
        str_230141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 36), 'str', 'fun')
        # Getting the type of 'self' (line 161)
        self_230142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 43), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 161)
        f_eqcon_230143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 43), self_230142, 'f_eqcon')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 35), dict_230138, (str_230141, f_eqcon_230143))
        # Adding element type (key, value) (line 160)
        str_230144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 36), 'str', 'args')
        
        # Obtaining an instance of the builtin type 'tuple' (line 162)
        tuple_230145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 162)
        # Adding element type (line 162)
        float_230146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 45), tuple_230145, float_230146)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 35), dict_230138, (str_230144, tuple_230145))
        # Adding element type (key, value) (line 160)
        str_230147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 36), 'str', 'jac')
        # Getting the type of 'self' (line 163)
        self_230148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 43), 'self', False)
        # Obtaining the member 'fprime_eqcon' of a type (line 163)
        fprime_eqcon_230149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 43), self_230148, 'fprime_eqcon')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 35), dict_230138, (str_230147, fprime_eqcon_230149))
        
        keyword_230150 = dict_230138
        # Getting the type of 'self' (line 164)
        self_230151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 31), 'self', False)
        # Obtaining the member 'opts' of a type (line 164)
        opts_230152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 31), self_230151, 'opts')
        keyword_230153 = opts_230152
        kwargs_230154 = {'args': keyword_230137, 'options': keyword_230153, 'method': keyword_230131, 'constraints': keyword_230150, 'jac': keyword_230134}
        # Getting the type of 'minimize' (line 158)
        minimize_230124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 158)
        minimize_call_result_230155 = invoke(stypy.reporting.localization.Localization(__file__, 158, 14), minimize_230124, *[fun_230126, list_230127], **kwargs_230154)
        
        # Assigning a type to the variable 'res' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'res', minimize_call_result_230155)
        
        # Call to assert_(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Obtaining the type of the subscript
        str_230157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'str', 'success')
        # Getting the type of 'res' (line 165)
        res_230158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___230159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), res_230158, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_230160 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), getitem___230159, str_230157)
        
        
        # Obtaining the type of the subscript
        str_230161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'str', 'message')
        # Getting the type of 'res' (line 165)
        res_230162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___230163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), res_230162, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_230164 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), getitem___230163, str_230161)
        
        # Processing the call keyword arguments (line 165)
        kwargs_230165 = {}
        # Getting the type of 'assert_' (line 165)
        assert__230156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 165)
        assert__call_result_230166 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert__230156, *[subscript_call_result_230160, subscript_call_result_230164], **kwargs_230165)
        
        
        # Call to assert_allclose(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'res' (line 166)
        res_230168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 166)
        x_230169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 24), res_230168, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 166)
        list_230170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 166)
        # Adding element type (line 166)
        int_230171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 31), list_230170, int_230171)
        # Adding element type (line 166)
        int_230172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 31), list_230170, int_230172)
        
        # Processing the call keyword arguments (line 166)
        kwargs_230173 = {}
        # Getting the type of 'assert_allclose' (line 166)
        assert_allclose_230167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 166)
        assert_allclose_call_result_230174 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), assert_allclose_230167, *[x_230169, list_230170], **kwargs_230173)
        
        
        # ################# End of 'test_minimize_equality_given2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_equality_given2' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_230175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_equality_given2'
        return stypy_return_type_230175


    @norecursion
    def test_minimize_equality_given_cons_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_equality_given_cons_scalar'
        module_type_store = module_type_store.open_function_context('test_minimize_equality_given_cons_scalar', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_equality_given_cons_scalar')
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_equality_given_cons_scalar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_equality_given_cons_scalar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_equality_given_cons_scalar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_equality_given_cons_scalar(...)' code ##################

        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to minimize(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'self' (line 171)
        self_230177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 171)
        fun_230178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 23), self_230177, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_230179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        float_230180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 33), list_230179, float_230180)
        # Adding element type (line 171)
        float_230181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 33), list_230179, float_230181)
        
        # Processing the call keyword arguments (line 171)
        str_230182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 53), 'str', 'SLSQP')
        keyword_230183 = str_230182
        # Getting the type of 'self' (line 172)
        self_230184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'self', False)
        # Obtaining the member 'jac' of a type (line 172)
        jac_230185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 27), self_230184, 'jac')
        keyword_230186 = jac_230185
        
        # Obtaining an instance of the builtin type 'tuple' (line 172)
        tuple_230187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 172)
        # Adding element type (line 172)
        float_230188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 43), tuple_230187, float_230188)
        
        keyword_230189 = tuple_230187
        
        # Obtaining an instance of the builtin type 'dict' (line 173)
        dict_230190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 173)
        # Adding element type (key, value) (line 173)
        str_230191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 36), 'str', 'type')
        str_230192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 44), 'str', 'eq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 35), dict_230190, (str_230191, str_230192))
        # Adding element type (key, value) (line 173)
        str_230193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 36), 'str', 'fun')
        # Getting the type of 'self' (line 174)
        self_230194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'self', False)
        # Obtaining the member 'f_eqcon_scalar' of a type (line 174)
        f_eqcon_scalar_230195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 43), self_230194, 'f_eqcon_scalar')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 35), dict_230190, (str_230193, f_eqcon_scalar_230195))
        # Adding element type (key, value) (line 173)
        str_230196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 36), 'str', 'args')
        
        # Obtaining an instance of the builtin type 'tuple' (line 175)
        tuple_230197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 175)
        # Adding element type (line 175)
        float_230198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 45), tuple_230197, float_230198)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 35), dict_230190, (str_230196, tuple_230197))
        # Adding element type (key, value) (line 173)
        str_230199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 36), 'str', 'jac')
        # Getting the type of 'self' (line 176)
        self_230200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'self', False)
        # Obtaining the member 'fprime_eqcon_scalar' of a type (line 176)
        fprime_eqcon_scalar_230201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 43), self_230200, 'fprime_eqcon_scalar')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 35), dict_230190, (str_230199, fprime_eqcon_scalar_230201))
        
        keyword_230202 = dict_230190
        # Getting the type of 'self' (line 177)
        self_230203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'self', False)
        # Obtaining the member 'opts' of a type (line 177)
        opts_230204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 31), self_230203, 'opts')
        keyword_230205 = opts_230204
        kwargs_230206 = {'args': keyword_230189, 'options': keyword_230205, 'method': keyword_230183, 'constraints': keyword_230202, 'jac': keyword_230186}
        # Getting the type of 'minimize' (line 171)
        minimize_230176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 171)
        minimize_call_result_230207 = invoke(stypy.reporting.localization.Localization(__file__, 171, 14), minimize_230176, *[fun_230178, list_230179], **kwargs_230206)
        
        # Assigning a type to the variable 'res' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'res', minimize_call_result_230207)
        
        # Call to assert_(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining the type of the subscript
        str_230209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 20), 'str', 'success')
        # Getting the type of 'res' (line 178)
        res_230210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___230211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), res_230210, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_230212 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), getitem___230211, str_230209)
        
        
        # Obtaining the type of the subscript
        str_230213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'str', 'message')
        # Getting the type of 'res' (line 178)
        res_230214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___230215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 32), res_230214, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_230216 = invoke(stypy.reporting.localization.Localization(__file__, 178, 32), getitem___230215, str_230213)
        
        # Processing the call keyword arguments (line 178)
        kwargs_230217 = {}
        # Getting the type of 'assert_' (line 178)
        assert__230208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 178)
        assert__call_result_230218 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), assert__230208, *[subscript_call_result_230212, subscript_call_result_230216], **kwargs_230217)
        
        
        # Call to assert_allclose(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'res' (line 179)
        res_230220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 179)
        x_230221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 24), res_230220, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_230222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        int_230223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 31), list_230222, int_230223)
        # Adding element type (line 179)
        int_230224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 31), list_230222, int_230224)
        
        # Processing the call keyword arguments (line 179)
        kwargs_230225 = {}
        # Getting the type of 'assert_allclose' (line 179)
        assert_allclose_230219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 179)
        assert_allclose_call_result_230226 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), assert_allclose_230219, *[x_230221, list_230222], **kwargs_230225)
        
        
        # ################# End of 'test_minimize_equality_given_cons_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_equality_given_cons_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_230227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_equality_given_cons_scalar'
        return stypy_return_type_230227


    @norecursion
    def test_minimize_inequality_given(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_inequality_given'
        module_type_store = module_type_store.open_function_context('test_minimize_inequality_given', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_inequality_given')
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_inequality_given.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_inequality_given', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_inequality_given', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_inequality_given(...)' code ##################

        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to minimize(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'self' (line 183)
        self_230229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 183)
        fun_230230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 23), self_230229, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_230231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        float_230232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 33), list_230231, float_230232)
        # Adding element type (line 183)
        float_230233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 33), list_230231, float_230233)
        
        # Processing the call keyword arguments (line 183)
        str_230234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 53), 'str', 'SLSQP')
        keyword_230235 = str_230234
        # Getting the type of 'self' (line 184)
        self_230236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'self', False)
        # Obtaining the member 'jac' of a type (line 184)
        jac_230237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 27), self_230236, 'jac')
        keyword_230238 = jac_230237
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_230239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        float_230240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 43), tuple_230239, float_230240)
        
        keyword_230241 = tuple_230239
        
        # Obtaining an instance of the builtin type 'dict' (line 185)
        dict_230242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 185)
        # Adding element type (key, value) (line 185)
        str_230243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 36), 'str', 'type')
        str_230244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 44), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 35), dict_230242, (str_230243, str_230244))
        # Adding element type (key, value) (line 185)
        str_230245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 36), 'str', 'fun')
        # Getting the type of 'self' (line 186)
        self_230246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 43), 'self', False)
        # Obtaining the member 'f_ieqcon' of a type (line 186)
        f_ieqcon_230247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 43), self_230246, 'f_ieqcon')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 35), dict_230242, (str_230245, f_ieqcon_230247))
        # Adding element type (key, value) (line 185)
        str_230248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 36), 'str', 'args')
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_230249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        float_230250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 45), tuple_230249, float_230250)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 35), dict_230242, (str_230248, tuple_230249))
        
        keyword_230251 = dict_230242
        # Getting the type of 'self' (line 188)
        self_230252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'self', False)
        # Obtaining the member 'opts' of a type (line 188)
        opts_230253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 31), self_230252, 'opts')
        keyword_230254 = opts_230253
        kwargs_230255 = {'args': keyword_230241, 'options': keyword_230254, 'method': keyword_230235, 'constraints': keyword_230251, 'jac': keyword_230238}
        # Getting the type of 'minimize' (line 183)
        minimize_230228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 183)
        minimize_call_result_230256 = invoke(stypy.reporting.localization.Localization(__file__, 183, 14), minimize_230228, *[fun_230230, list_230231], **kwargs_230255)
        
        # Assigning a type to the variable 'res' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'res', minimize_call_result_230256)
        
        # Call to assert_(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Obtaining the type of the subscript
        str_230258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 20), 'str', 'success')
        # Getting the type of 'res' (line 189)
        res_230259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___230260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 16), res_230259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 189)
        subscript_call_result_230261 = invoke(stypy.reporting.localization.Localization(__file__, 189, 16), getitem___230260, str_230258)
        
        
        # Obtaining the type of the subscript
        str_230262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 36), 'str', 'message')
        # Getting the type of 'res' (line 189)
        res_230263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___230264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 32), res_230263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 189)
        subscript_call_result_230265 = invoke(stypy.reporting.localization.Localization(__file__, 189, 32), getitem___230264, str_230262)
        
        # Processing the call keyword arguments (line 189)
        kwargs_230266 = {}
        # Getting the type of 'assert_' (line 189)
        assert__230257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 189)
        assert__call_result_230267 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assert__230257, *[subscript_call_result_230261, subscript_call_result_230265], **kwargs_230266)
        
        
        # Call to assert_allclose(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'res' (line 190)
        res_230269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 190)
        x_230270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 24), res_230269, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 190)
        list_230271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 190)
        # Adding element type (line 190)
        int_230272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 31), list_230271, int_230272)
        # Adding element type (line 190)
        int_230273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 31), list_230271, int_230273)
        
        # Processing the call keyword arguments (line 190)
        float_230274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 44), 'float')
        keyword_230275 = float_230274
        kwargs_230276 = {'atol': keyword_230275}
        # Getting the type of 'assert_allclose' (line 190)
        assert_allclose_230268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 190)
        assert_allclose_call_result_230277 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), assert_allclose_230268, *[x_230270, list_230271], **kwargs_230276)
        
        
        # ################# End of 'test_minimize_inequality_given(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_inequality_given' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_230278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230278)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_inequality_given'
        return stypy_return_type_230278


    @norecursion
    def test_minimize_inequality_given_vector_constraints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_inequality_given_vector_constraints'
        module_type_store = module_type_store.open_function_context('test_minimize_inequality_given_vector_constraints', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_inequality_given_vector_constraints')
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_inequality_given_vector_constraints.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_inequality_given_vector_constraints', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_inequality_given_vector_constraints', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_inequality_given_vector_constraints(...)' code ##################

        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to minimize(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'self' (line 195)
        self_230280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 195)
        fun_230281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 23), self_230280, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_230282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        float_230283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 33), list_230282, float_230283)
        # Adding element type (line 195)
        float_230284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 33), list_230282, float_230284)
        
        # Processing the call keyword arguments (line 195)
        # Getting the type of 'self' (line 195)
        self_230285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 50), 'self', False)
        # Obtaining the member 'jac' of a type (line 195)
        jac_230286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 50), self_230285, 'jac')
        keyword_230287 = jac_230286
        str_230288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 30), 'str', 'SLSQP')
        keyword_230289 = str_230288
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_230290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        float_230291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 45), tuple_230290, float_230291)
        
        keyword_230292 = tuple_230290
        
        # Obtaining an instance of the builtin type 'dict' (line 197)
        dict_230293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 197)
        # Adding element type (key, value) (line 197)
        str_230294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 36), 'str', 'type')
        str_230295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 44), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 35), dict_230293, (str_230294, str_230295))
        # Adding element type (key, value) (line 197)
        str_230296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 36), 'str', 'fun')
        # Getting the type of 'self' (line 198)
        self_230297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 43), 'self', False)
        # Obtaining the member 'f_ieqcon2' of a type (line 198)
        f_ieqcon2_230298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 43), self_230297, 'f_ieqcon2')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 35), dict_230293, (str_230296, f_ieqcon2_230298))
        # Adding element type (key, value) (line 197)
        str_230299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 36), 'str', 'jac')
        # Getting the type of 'self' (line 199)
        self_230300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 43), 'self', False)
        # Obtaining the member 'fprime_ieqcon2' of a type (line 199)
        fprime_ieqcon2_230301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 43), self_230300, 'fprime_ieqcon2')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 35), dict_230293, (str_230299, fprime_ieqcon2_230301))
        
        keyword_230302 = dict_230293
        # Getting the type of 'self' (line 200)
        self_230303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 31), 'self', False)
        # Obtaining the member 'opts' of a type (line 200)
        opts_230304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 31), self_230303, 'opts')
        keyword_230305 = opts_230304
        kwargs_230306 = {'args': keyword_230292, 'options': keyword_230305, 'jac': keyword_230287, 'constraints': keyword_230302, 'method': keyword_230289}
        # Getting the type of 'minimize' (line 195)
        minimize_230279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 195)
        minimize_call_result_230307 = invoke(stypy.reporting.localization.Localization(__file__, 195, 14), minimize_230279, *[fun_230281, list_230282], **kwargs_230306)
        
        # Assigning a type to the variable 'res' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'res', minimize_call_result_230307)
        
        # Call to assert_(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Obtaining the type of the subscript
        str_230309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 20), 'str', 'success')
        # Getting the type of 'res' (line 201)
        res_230310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___230311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), res_230310, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_230312 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), getitem___230311, str_230309)
        
        
        # Obtaining the type of the subscript
        str_230313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 36), 'str', 'message')
        # Getting the type of 'res' (line 201)
        res_230314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___230315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 32), res_230314, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_230316 = invoke(stypy.reporting.localization.Localization(__file__, 201, 32), getitem___230315, str_230313)
        
        # Processing the call keyword arguments (line 201)
        kwargs_230317 = {}
        # Getting the type of 'assert_' (line 201)
        assert__230308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 201)
        assert__call_result_230318 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), assert__230308, *[subscript_call_result_230312, subscript_call_result_230316], **kwargs_230317)
        
        
        # Call to assert_allclose(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'res' (line 202)
        res_230320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 202)
        x_230321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 24), res_230320, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_230322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        int_230323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 31), list_230322, int_230323)
        # Adding element type (line 202)
        int_230324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 31), list_230322, int_230324)
        
        # Processing the call keyword arguments (line 202)
        kwargs_230325 = {}
        # Getting the type of 'assert_allclose' (line 202)
        assert_allclose_230319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 202)
        assert_allclose_call_result_230326 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assert_allclose_230319, *[x_230321, list_230322], **kwargs_230325)
        
        
        # ################# End of 'test_minimize_inequality_given_vector_constraints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_inequality_given_vector_constraints' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_230327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230327)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_inequality_given_vector_constraints'
        return stypy_return_type_230327


    @norecursion
    def test_minimize_bound_equality_given2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_bound_equality_given2'
        module_type_store = module_type_store.open_function_context('test_minimize_bound_equality_given2', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_minimize_bound_equality_given2')
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_minimize_bound_equality_given2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_minimize_bound_equality_given2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_bound_equality_given2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_bound_equality_given2(...)' code ##################

        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to minimize(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'self' (line 207)
        self_230329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 207)
        fun_230330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 23), self_230329, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_230331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        float_230332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 33), list_230331, float_230332)
        # Adding element type (line 207)
        float_230333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 33), list_230331, float_230333)
        
        # Processing the call keyword arguments (line 207)
        str_230334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 53), 'str', 'SLSQP')
        keyword_230335 = str_230334
        # Getting the type of 'self' (line 208)
        self_230336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'self', False)
        # Obtaining the member 'jac' of a type (line 208)
        jac_230337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 27), self_230336, 'jac')
        keyword_230338 = jac_230337
        
        # Obtaining an instance of the builtin type 'tuple' (line 208)
        tuple_230339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 208)
        # Adding element type (line 208)
        float_230340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 43), tuple_230339, float_230340)
        
        keyword_230341 = tuple_230339
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_230342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        
        # Obtaining an instance of the builtin type 'tuple' (line 209)
        tuple_230343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 209)
        # Adding element type (line 209)
        float_230344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 32), tuple_230343, float_230344)
        # Adding element type (line 209)
        float_230345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 32), tuple_230343, float_230345)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 30), list_230342, tuple_230343)
        # Adding element type (line 209)
        
        # Obtaining an instance of the builtin type 'tuple' (line 209)
        tuple_230346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 209)
        # Adding element type (line 209)
        int_230347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 44), tuple_230346, int_230347)
        # Adding element type (line 209)
        float_230348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 44), tuple_230346, float_230348)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 30), list_230342, tuple_230346)
        
        keyword_230349 = list_230342
        
        # Obtaining an instance of the builtin type 'dict' (line 210)
        dict_230350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 210)
        # Adding element type (key, value) (line 210)
        str_230351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 36), 'str', 'type')
        str_230352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 44), 'str', 'eq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 35), dict_230350, (str_230351, str_230352))
        # Adding element type (key, value) (line 210)
        str_230353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 36), 'str', 'fun')
        # Getting the type of 'self' (line 211)
        self_230354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 43), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 211)
        f_eqcon_230355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 43), self_230354, 'f_eqcon')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 35), dict_230350, (str_230353, f_eqcon_230355))
        # Adding element type (key, value) (line 210)
        str_230356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 36), 'str', 'args')
        
        # Obtaining an instance of the builtin type 'tuple' (line 212)
        tuple_230357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 212)
        # Adding element type (line 212)
        float_230358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 45), tuple_230357, float_230358)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 35), dict_230350, (str_230356, tuple_230357))
        # Adding element type (key, value) (line 210)
        str_230359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 36), 'str', 'jac')
        # Getting the type of 'self' (line 213)
        self_230360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'self', False)
        # Obtaining the member 'fprime_eqcon' of a type (line 213)
        fprime_eqcon_230361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 43), self_230360, 'fprime_eqcon')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 35), dict_230350, (str_230359, fprime_eqcon_230361))
        
        keyword_230362 = dict_230350
        # Getting the type of 'self' (line 214)
        self_230363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 31), 'self', False)
        # Obtaining the member 'opts' of a type (line 214)
        opts_230364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 31), self_230363, 'opts')
        keyword_230365 = opts_230364
        kwargs_230366 = {'jac': keyword_230338, 'args': keyword_230341, 'bounds': keyword_230349, 'method': keyword_230335, 'options': keyword_230365, 'constraints': keyword_230362}
        # Getting the type of 'minimize' (line 207)
        minimize_230328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 207)
        minimize_call_result_230367 = invoke(stypy.reporting.localization.Localization(__file__, 207, 14), minimize_230328, *[fun_230330, list_230331], **kwargs_230366)
        
        # Assigning a type to the variable 'res' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'res', minimize_call_result_230367)
        
        # Call to assert_(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Obtaining the type of the subscript
        str_230369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 20), 'str', 'success')
        # Getting the type of 'res' (line 215)
        res_230370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___230371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 16), res_230370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_230372 = invoke(stypy.reporting.localization.Localization(__file__, 215, 16), getitem___230371, str_230369)
        
        
        # Obtaining the type of the subscript
        str_230373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 36), 'str', 'message')
        # Getting the type of 'res' (line 215)
        res_230374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___230375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 32), res_230374, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_230376 = invoke(stypy.reporting.localization.Localization(__file__, 215, 32), getitem___230375, str_230373)
        
        # Processing the call keyword arguments (line 215)
        kwargs_230377 = {}
        # Getting the type of 'assert_' (line 215)
        assert__230368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 215)
        assert__call_result_230378 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), assert__230368, *[subscript_call_result_230372, subscript_call_result_230376], **kwargs_230377)
        
        
        # Call to assert_allclose(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'res' (line 216)
        res_230380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 216)
        x_230381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 24), res_230380, 'x')
        
        # Obtaining an instance of the builtin type 'list' (line 216)
        list_230382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 216)
        # Adding element type (line 216)
        float_230383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 31), list_230382, float_230383)
        # Adding element type (line 216)
        float_230384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 31), list_230382, float_230384)
        
        # Processing the call keyword arguments (line 216)
        float_230385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 48), 'float')
        keyword_230386 = float_230385
        kwargs_230387 = {'atol': keyword_230386}
        # Getting the type of 'assert_allclose' (line 216)
        assert_allclose_230379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 216)
        assert_allclose_call_result_230388 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), assert_allclose_230379, *[x_230381, list_230382], **kwargs_230387)
        
        
        # Call to assert_(...): (line 217)
        # Processing the call arguments (line 217)
        
        float_230390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 16), 'float')
        
        # Obtaining the type of the subscript
        int_230391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 30), 'int')
        # Getting the type of 'res' (line 217)
        res_230392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'res', False)
        # Obtaining the member 'x' of a type (line 217)
        x_230393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 24), res_230392, 'x')
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___230394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 24), x_230393, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_230395 = invoke(stypy.reporting.localization.Localization(__file__, 217, 24), getitem___230394, int_230391)
        
        # Applying the binary operator '<=' (line 217)
        result_le_230396 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 16), '<=', float_230390, subscript_call_result_230395)
        int_230397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 36), 'int')
        # Applying the binary operator '<=' (line 217)
        result_le_230398 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 16), '<=', subscript_call_result_230395, int_230397)
        # Applying the binary operator '&' (line 217)
        result_and__230399 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 16), '&', result_le_230396, result_le_230398)
        
        # Processing the call keyword arguments (line 217)
        kwargs_230400 = {}
        # Getting the type of 'assert_' (line 217)
        assert__230389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 217)
        assert__call_result_230401 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), assert__230389, *[result_and__230399], **kwargs_230400)
        
        
        # Call to assert_(...): (line 218)
        # Processing the call arguments (line 218)
        
        int_230403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 16), 'int')
        
        # Obtaining the type of the subscript
        int_230404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 28), 'int')
        # Getting the type of 'res' (line 218)
        res_230405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'res', False)
        # Obtaining the member 'x' of a type (line 218)
        x_230406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 22), res_230405, 'x')
        # Obtaining the member '__getitem__' of a type (line 218)
        getitem___230407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 22), x_230406, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 218)
        subscript_call_result_230408 = invoke(stypy.reporting.localization.Localization(__file__, 218, 22), getitem___230407, int_230404)
        
        # Applying the binary operator '<=' (line 218)
        result_le_230409 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 16), '<=', int_230403, subscript_call_result_230408)
        float_230410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 34), 'float')
        # Applying the binary operator '<=' (line 218)
        result_le_230411 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 16), '<=', subscript_call_result_230408, float_230410)
        # Applying the binary operator '&' (line 218)
        result_and__230412 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 16), '&', result_le_230409, result_le_230411)
        
        # Processing the call keyword arguments (line 218)
        kwargs_230413 = {}
        # Getting the type of 'assert_' (line 218)
        assert__230402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 218)
        assert__call_result_230414 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), assert__230402, *[result_and__230412], **kwargs_230413)
        
        
        # ################# End of 'test_minimize_bound_equality_given2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_bound_equality_given2' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_230415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230415)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_bound_equality_given2'
        return stypy_return_type_230415


    @norecursion
    def test_unbounded_approximated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_unbounded_approximated'
        module_type_store = module_type_store.open_function_context('test_unbounded_approximated', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_unbounded_approximated')
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_unbounded_approximated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_unbounded_approximated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_unbounded_approximated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_unbounded_approximated(...)' code ##################

        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to fmin_slsqp(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_230417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'self', False)
        # Obtaining the member 'fun' of a type (line 223)
        fun_230418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), self_230417, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_230419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        float_230420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 35), list_230419, float_230420)
        # Adding element type (line 223)
        float_230421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 35), list_230419, float_230421)
        
        # Processing the call keyword arguments (line 223)
        
        # Obtaining an instance of the builtin type 'tuple' (line 223)
        tuple_230422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 223)
        # Adding element type (line 223)
        float_230423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 54), tuple_230422, float_230423)
        
        keyword_230424 = tuple_230422
        int_230425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 34), 'int')
        keyword_230426 = int_230425
        int_230427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 51), 'int')
        keyword_230428 = int_230427
        kwargs_230429 = {'iprint': keyword_230426, 'args': keyword_230424, 'full_output': keyword_230428}
        # Getting the type of 'fmin_slsqp' (line 223)
        fmin_slsqp_230416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 223)
        fmin_slsqp_call_result_230430 = invoke(stypy.reporting.localization.Localization(__file__, 223, 14), fmin_slsqp_230416, *[fun_230418, list_230419], **kwargs_230429)
        
        # Assigning a type to the variable 'res' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'res', fmin_slsqp_call_result_230430)
        
        # Assigning a Name to a Tuple (line 225):
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_230431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        # Getting the type of 'res' (line 225)
        res_230432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___230433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), res_230432, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_230434 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___230433, int_230431)
        
        # Assigning a type to the variable 'tuple_var_assignment_229586' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229586', subscript_call_result_230434)
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_230435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        # Getting the type of 'res' (line 225)
        res_230436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___230437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), res_230436, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_230438 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___230437, int_230435)
        
        # Assigning a type to the variable 'tuple_var_assignment_229587' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229587', subscript_call_result_230438)
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_230439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        # Getting the type of 'res' (line 225)
        res_230440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___230441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), res_230440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_230442 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___230441, int_230439)
        
        # Assigning a type to the variable 'tuple_var_assignment_229588' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229588', subscript_call_result_230442)
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_230443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        # Getting the type of 'res' (line 225)
        res_230444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___230445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), res_230444, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_230446 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___230445, int_230443)
        
        # Assigning a type to the variable 'tuple_var_assignment_229589' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229589', subscript_call_result_230446)
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_230447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        # Getting the type of 'res' (line 225)
        res_230448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___230449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), res_230448, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_230450 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___230449, int_230447)
        
        # Assigning a type to the variable 'tuple_var_assignment_229590' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229590', subscript_call_result_230450)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_229586' (line 225)
        tuple_var_assignment_229586_230451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229586')
        # Assigning a type to the variable 'x' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'x', tuple_var_assignment_229586_230451)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_229587' (line 225)
        tuple_var_assignment_229587_230452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229587')
        # Assigning a type to the variable 'fx' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'fx', tuple_var_assignment_229587_230452)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_229588' (line 225)
        tuple_var_assignment_229588_230453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229588')
        # Assigning a type to the variable 'its' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'its', tuple_var_assignment_229588_230453)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_229589' (line 225)
        tuple_var_assignment_229589_230454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229589')
        # Assigning a type to the variable 'imode' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'imode', tuple_var_assignment_229589_230454)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_229590' (line 225)
        tuple_var_assignment_229590_230455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_229590')
        # Assigning a type to the variable 'smode' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'smode', tuple_var_assignment_229590_230455)
        
        # Call to assert_(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Getting the type of 'imode' (line 226)
        imode_230457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'imode', False)
        int_230458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 25), 'int')
        # Applying the binary operator '==' (line 226)
        result_eq_230459 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 16), '==', imode_230457, int_230458)
        
        # Getting the type of 'imode' (line 226)
        imode_230460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'imode', False)
        # Processing the call keyword arguments (line 226)
        kwargs_230461 = {}
        # Getting the type of 'assert_' (line 226)
        assert__230456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 226)
        assert__call_result_230462 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), assert__230456, *[result_eq_230459, imode_230460], **kwargs_230461)
        
        
        # Call to assert_array_almost_equal(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'x' (line 227)
        x_230464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_230465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        int_230466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 37), list_230465, int_230466)
        # Adding element type (line 227)
        int_230467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 37), list_230465, int_230467)
        
        # Processing the call keyword arguments (line 227)
        kwargs_230468 = {}
        # Getting the type of 'assert_array_almost_equal' (line 227)
        assert_array_almost_equal_230463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 227)
        assert_array_almost_equal_call_result_230469 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assert_array_almost_equal_230463, *[x_230464, list_230465], **kwargs_230468)
        
        
        # ################# End of 'test_unbounded_approximated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_unbounded_approximated' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_230470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_unbounded_approximated'
        return stypy_return_type_230470


    @norecursion
    def test_unbounded_given(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_unbounded_given'
        module_type_store = module_type_store.open_function_context('test_unbounded_given', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_unbounded_given')
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_unbounded_given.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_unbounded_given', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_unbounded_given', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_unbounded_given(...)' code ##################

        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to fmin_slsqp(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'self' (line 231)
        self_230472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 25), 'self', False)
        # Obtaining the member 'fun' of a type (line 231)
        fun_230473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 25), self_230472, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_230474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        float_230475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 35), list_230474, float_230475)
        # Adding element type (line 231)
        float_230476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 35), list_230474, float_230476)
        
        # Processing the call keyword arguments (line 231)
        
        # Obtaining an instance of the builtin type 'tuple' (line 231)
        tuple_230477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 231)
        # Adding element type (line 231)
        float_230478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 54), tuple_230477, float_230478)
        
        keyword_230479 = tuple_230477
        # Getting the type of 'self' (line 232)
        self_230480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'self', False)
        # Obtaining the member 'jac' of a type (line 232)
        jac_230481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 34), self_230480, 'jac')
        keyword_230482 = jac_230481
        int_230483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 53), 'int')
        keyword_230484 = int_230483
        int_230485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 39), 'int')
        keyword_230486 = int_230485
        kwargs_230487 = {'iprint': keyword_230484, 'args': keyword_230479, 'fprime': keyword_230482, 'full_output': keyword_230486}
        # Getting the type of 'fmin_slsqp' (line 231)
        fmin_slsqp_230471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 14), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 231)
        fmin_slsqp_call_result_230488 = invoke(stypy.reporting.localization.Localization(__file__, 231, 14), fmin_slsqp_230471, *[fun_230473, list_230474], **kwargs_230487)
        
        # Assigning a type to the variable 'res' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'res', fmin_slsqp_call_result_230488)
        
        # Assigning a Name to a Tuple (line 234):
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_230489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
        # Getting the type of 'res' (line 234)
        res_230490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___230491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), res_230490, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_230492 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), getitem___230491, int_230489)
        
        # Assigning a type to the variable 'tuple_var_assignment_229591' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229591', subscript_call_result_230492)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_230493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
        # Getting the type of 'res' (line 234)
        res_230494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___230495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), res_230494, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_230496 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), getitem___230495, int_230493)
        
        # Assigning a type to the variable 'tuple_var_assignment_229592' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229592', subscript_call_result_230496)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_230497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
        # Getting the type of 'res' (line 234)
        res_230498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___230499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), res_230498, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_230500 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), getitem___230499, int_230497)
        
        # Assigning a type to the variable 'tuple_var_assignment_229593' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229593', subscript_call_result_230500)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_230501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
        # Getting the type of 'res' (line 234)
        res_230502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___230503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), res_230502, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_230504 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), getitem___230503, int_230501)
        
        # Assigning a type to the variable 'tuple_var_assignment_229594' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229594', subscript_call_result_230504)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_230505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
        # Getting the type of 'res' (line 234)
        res_230506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___230507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), res_230506, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_230508 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), getitem___230507, int_230505)
        
        # Assigning a type to the variable 'tuple_var_assignment_229595' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229595', subscript_call_result_230508)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_229591' (line 234)
        tuple_var_assignment_229591_230509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229591')
        # Assigning a type to the variable 'x' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'x', tuple_var_assignment_229591_230509)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_229592' (line 234)
        tuple_var_assignment_229592_230510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229592')
        # Assigning a type to the variable 'fx' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'fx', tuple_var_assignment_229592_230510)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_229593' (line 234)
        tuple_var_assignment_229593_230511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229593')
        # Assigning a type to the variable 'its' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'its', tuple_var_assignment_229593_230511)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_229594' (line 234)
        tuple_var_assignment_229594_230512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229594')
        # Assigning a type to the variable 'imode' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'imode', tuple_var_assignment_229594_230512)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_229595' (line 234)
        tuple_var_assignment_229595_230513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_229595')
        # Assigning a type to the variable 'smode' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 27), 'smode', tuple_var_assignment_229595_230513)
        
        # Call to assert_(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Getting the type of 'imode' (line 235)
        imode_230515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'imode', False)
        int_230516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 25), 'int')
        # Applying the binary operator '==' (line 235)
        result_eq_230517 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 16), '==', imode_230515, int_230516)
        
        # Getting the type of 'imode' (line 235)
        imode_230518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'imode', False)
        # Processing the call keyword arguments (line 235)
        kwargs_230519 = {}
        # Getting the type of 'assert_' (line 235)
        assert__230514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 235)
        assert__call_result_230520 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), assert__230514, *[result_eq_230517, imode_230518], **kwargs_230519)
        
        
        # Call to assert_array_almost_equal(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'x' (line 236)
        x_230522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_230523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        int_230524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 37), list_230523, int_230524)
        # Adding element type (line 236)
        int_230525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 37), list_230523, int_230525)
        
        # Processing the call keyword arguments (line 236)
        kwargs_230526 = {}
        # Getting the type of 'assert_array_almost_equal' (line 236)
        assert_array_almost_equal_230521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 236)
        assert_array_almost_equal_call_result_230527 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), assert_array_almost_equal_230521, *[x_230522, list_230523], **kwargs_230526)
        
        
        # ################# End of 'test_unbounded_given(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_unbounded_given' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_230528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_unbounded_given'
        return stypy_return_type_230528


    @norecursion
    def test_equality_approximated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_equality_approximated'
        module_type_store = module_type_store.open_function_context('test_equality_approximated', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_equality_approximated')
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_equality_approximated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_equality_approximated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_equality_approximated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_equality_approximated(...)' code ##################

        
        # Assigning a Call to a Name (line 240):
        
        # Assigning a Call to a Name (line 240):
        
        # Call to fmin_slsqp(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'self' (line 240)
        self_230530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'self', False)
        # Obtaining the member 'fun' of a type (line 240)
        fun_230531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 25), self_230530, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 240)
        list_230532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 240)
        # Adding element type (line 240)
        float_230533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 34), list_230532, float_230533)
        # Adding element type (line 240)
        float_230534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 34), list_230532, float_230534)
        
        # Processing the call keyword arguments (line 240)
        
        # Obtaining an instance of the builtin type 'tuple' (line 240)
        tuple_230535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 240)
        # Adding element type (line 240)
        float_230536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 52), tuple_230535, float_230536)
        
        keyword_230537 = tuple_230535
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_230538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        # Getting the type of 'self' (line 241)
        self_230539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 35), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 241)
        f_eqcon_230540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 35), self_230539, 'f_eqcon')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 34), list_230538, f_eqcon_230540)
        
        keyword_230541 = list_230538
        int_230542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 34), 'int')
        keyword_230543 = int_230542
        int_230544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 51), 'int')
        keyword_230545 = int_230544
        kwargs_230546 = {'iprint': keyword_230543, 'eqcons': keyword_230541, 'args': keyword_230537, 'full_output': keyword_230545}
        # Getting the type of 'fmin_slsqp' (line 240)
        fmin_slsqp_230529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 240)
        fmin_slsqp_call_result_230547 = invoke(stypy.reporting.localization.Localization(__file__, 240, 14), fmin_slsqp_230529, *[fun_230531, list_230532], **kwargs_230546)
        
        # Assigning a type to the variable 'res' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'res', fmin_slsqp_call_result_230547)
        
        # Assigning a Name to a Tuple (line 243):
        
        # Assigning a Subscript to a Name (line 243):
        
        # Obtaining the type of the subscript
        int_230548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
        # Getting the type of 'res' (line 243)
        res_230549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___230550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), res_230549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_230551 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), getitem___230550, int_230548)
        
        # Assigning a type to the variable 'tuple_var_assignment_229596' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229596', subscript_call_result_230551)
        
        # Assigning a Subscript to a Name (line 243):
        
        # Obtaining the type of the subscript
        int_230552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
        # Getting the type of 'res' (line 243)
        res_230553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___230554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), res_230553, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_230555 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), getitem___230554, int_230552)
        
        # Assigning a type to the variable 'tuple_var_assignment_229597' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229597', subscript_call_result_230555)
        
        # Assigning a Subscript to a Name (line 243):
        
        # Obtaining the type of the subscript
        int_230556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
        # Getting the type of 'res' (line 243)
        res_230557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___230558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), res_230557, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_230559 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), getitem___230558, int_230556)
        
        # Assigning a type to the variable 'tuple_var_assignment_229598' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229598', subscript_call_result_230559)
        
        # Assigning a Subscript to a Name (line 243):
        
        # Obtaining the type of the subscript
        int_230560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
        # Getting the type of 'res' (line 243)
        res_230561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___230562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), res_230561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_230563 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), getitem___230562, int_230560)
        
        # Assigning a type to the variable 'tuple_var_assignment_229599' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229599', subscript_call_result_230563)
        
        # Assigning a Subscript to a Name (line 243):
        
        # Obtaining the type of the subscript
        int_230564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
        # Getting the type of 'res' (line 243)
        res_230565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___230566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), res_230565, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_230567 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), getitem___230566, int_230564)
        
        # Assigning a type to the variable 'tuple_var_assignment_229600' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229600', subscript_call_result_230567)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'tuple_var_assignment_229596' (line 243)
        tuple_var_assignment_229596_230568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229596')
        # Assigning a type to the variable 'x' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'x', tuple_var_assignment_229596_230568)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'tuple_var_assignment_229597' (line 243)
        tuple_var_assignment_229597_230569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229597')
        # Assigning a type to the variable 'fx' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'fx', tuple_var_assignment_229597_230569)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'tuple_var_assignment_229598' (line 243)
        tuple_var_assignment_229598_230570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229598')
        # Assigning a type to the variable 'its' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'its', tuple_var_assignment_229598_230570)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'tuple_var_assignment_229599' (line 243)
        tuple_var_assignment_229599_230571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229599')
        # Assigning a type to the variable 'imode' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'imode', tuple_var_assignment_229599_230571)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'tuple_var_assignment_229600' (line 243)
        tuple_var_assignment_229600_230572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_229600')
        # Assigning a type to the variable 'smode' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 27), 'smode', tuple_var_assignment_229600_230572)
        
        # Call to assert_(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Getting the type of 'imode' (line 244)
        imode_230574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'imode', False)
        int_230575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 25), 'int')
        # Applying the binary operator '==' (line 244)
        result_eq_230576 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 16), '==', imode_230574, int_230575)
        
        # Getting the type of 'imode' (line 244)
        imode_230577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 28), 'imode', False)
        # Processing the call keyword arguments (line 244)
        kwargs_230578 = {}
        # Getting the type of 'assert_' (line 244)
        assert__230573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 244)
        assert__call_result_230579 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), assert__230573, *[result_eq_230576, imode_230577], **kwargs_230578)
        
        
        # Call to assert_array_almost_equal(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'x' (line 245)
        x_230581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 245)
        list_230582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 245)
        # Adding element type (line 245)
        int_230583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 37), list_230582, int_230583)
        # Adding element type (line 245)
        int_230584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 37), list_230582, int_230584)
        
        # Processing the call keyword arguments (line 245)
        kwargs_230585 = {}
        # Getting the type of 'assert_array_almost_equal' (line 245)
        assert_array_almost_equal_230580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 245)
        assert_array_almost_equal_call_result_230586 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), assert_array_almost_equal_230580, *[x_230581, list_230582], **kwargs_230585)
        
        
        # ################# End of 'test_equality_approximated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_equality_approximated' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_230587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230587)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_equality_approximated'
        return stypy_return_type_230587


    @norecursion
    def test_equality_given(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_equality_given'
        module_type_store = module_type_store.open_function_context('test_equality_given', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_equality_given')
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_equality_given.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_equality_given', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_equality_given', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_equality_given(...)' code ##################

        
        # Assigning a Call to a Name (line 249):
        
        # Assigning a Call to a Name (line 249):
        
        # Call to fmin_slsqp(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'self' (line 249)
        self_230589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'self', False)
        # Obtaining the member 'fun' of a type (line 249)
        fun_230590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 25), self_230589, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_230591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        float_230592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 35), list_230591, float_230592)
        # Adding element type (line 249)
        float_230593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 35), list_230591, float_230593)
        
        # Processing the call keyword arguments (line 249)
        # Getting the type of 'self' (line 250)
        self_230594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 32), 'self', False)
        # Obtaining the member 'jac' of a type (line 250)
        jac_230595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 32), self_230594, 'jac')
        keyword_230596 = jac_230595
        
        # Obtaining an instance of the builtin type 'tuple' (line 250)
        tuple_230597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 250)
        # Adding element type (line 250)
        float_230598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 48), tuple_230597, float_230598)
        
        keyword_230599 = tuple_230597
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_230600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        # Getting the type of 'self' (line 251)
        self_230601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 35), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 251)
        f_eqcon_230602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 35), self_230601, 'f_eqcon')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 34), list_230600, f_eqcon_230602)
        
        keyword_230603 = list_230600
        int_230604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 59), 'int')
        keyword_230605 = int_230604
        int_230606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 39), 'int')
        keyword_230607 = int_230606
        kwargs_230608 = {'iprint': keyword_230605, 'eqcons': keyword_230603, 'args': keyword_230599, 'fprime': keyword_230596, 'full_output': keyword_230607}
        # Getting the type of 'fmin_slsqp' (line 249)
        fmin_slsqp_230588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 14), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 249)
        fmin_slsqp_call_result_230609 = invoke(stypy.reporting.localization.Localization(__file__, 249, 14), fmin_slsqp_230588, *[fun_230590, list_230591], **kwargs_230608)
        
        # Assigning a type to the variable 'res' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'res', fmin_slsqp_call_result_230609)
        
        # Assigning a Name to a Tuple (line 253):
        
        # Assigning a Subscript to a Name (line 253):
        
        # Obtaining the type of the subscript
        int_230610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 8), 'int')
        # Getting the type of 'res' (line 253)
        res_230611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___230612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), res_230611, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_230613 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), getitem___230612, int_230610)
        
        # Assigning a type to the variable 'tuple_var_assignment_229601' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229601', subscript_call_result_230613)
        
        # Assigning a Subscript to a Name (line 253):
        
        # Obtaining the type of the subscript
        int_230614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 8), 'int')
        # Getting the type of 'res' (line 253)
        res_230615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___230616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), res_230615, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_230617 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), getitem___230616, int_230614)
        
        # Assigning a type to the variable 'tuple_var_assignment_229602' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229602', subscript_call_result_230617)
        
        # Assigning a Subscript to a Name (line 253):
        
        # Obtaining the type of the subscript
        int_230618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 8), 'int')
        # Getting the type of 'res' (line 253)
        res_230619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___230620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), res_230619, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_230621 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), getitem___230620, int_230618)
        
        # Assigning a type to the variable 'tuple_var_assignment_229603' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229603', subscript_call_result_230621)
        
        # Assigning a Subscript to a Name (line 253):
        
        # Obtaining the type of the subscript
        int_230622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 8), 'int')
        # Getting the type of 'res' (line 253)
        res_230623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___230624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), res_230623, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_230625 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), getitem___230624, int_230622)
        
        # Assigning a type to the variable 'tuple_var_assignment_229604' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229604', subscript_call_result_230625)
        
        # Assigning a Subscript to a Name (line 253):
        
        # Obtaining the type of the subscript
        int_230626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 8), 'int')
        # Getting the type of 'res' (line 253)
        res_230627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___230628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), res_230627, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_230629 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), getitem___230628, int_230626)
        
        # Assigning a type to the variable 'tuple_var_assignment_229605' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229605', subscript_call_result_230629)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'tuple_var_assignment_229601' (line 253)
        tuple_var_assignment_229601_230630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229601')
        # Assigning a type to the variable 'x' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'x', tuple_var_assignment_229601_230630)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'tuple_var_assignment_229602' (line 253)
        tuple_var_assignment_229602_230631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229602')
        # Assigning a type to the variable 'fx' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'fx', tuple_var_assignment_229602_230631)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'tuple_var_assignment_229603' (line 253)
        tuple_var_assignment_229603_230632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229603')
        # Assigning a type to the variable 'its' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'its', tuple_var_assignment_229603_230632)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'tuple_var_assignment_229604' (line 253)
        tuple_var_assignment_229604_230633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229604')
        # Assigning a type to the variable 'imode' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'imode', tuple_var_assignment_229604_230633)
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'tuple_var_assignment_229605' (line 253)
        tuple_var_assignment_229605_230634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tuple_var_assignment_229605')
        # Assigning a type to the variable 'smode' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 27), 'smode', tuple_var_assignment_229605_230634)
        
        # Call to assert_(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Getting the type of 'imode' (line 254)
        imode_230636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'imode', False)
        int_230637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'int')
        # Applying the binary operator '==' (line 254)
        result_eq_230638 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 16), '==', imode_230636, int_230637)
        
        # Getting the type of 'imode' (line 254)
        imode_230639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 'imode', False)
        # Processing the call keyword arguments (line 254)
        kwargs_230640 = {}
        # Getting the type of 'assert_' (line 254)
        assert__230635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 254)
        assert__call_result_230641 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), assert__230635, *[result_eq_230638, imode_230639], **kwargs_230640)
        
        
        # Call to assert_array_almost_equal(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'x' (line 255)
        x_230643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_230644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        int_230645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 37), list_230644, int_230645)
        # Adding element type (line 255)
        int_230646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 37), list_230644, int_230646)
        
        # Processing the call keyword arguments (line 255)
        kwargs_230647 = {}
        # Getting the type of 'assert_array_almost_equal' (line 255)
        assert_array_almost_equal_230642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 255)
        assert_array_almost_equal_call_result_230648 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assert_array_almost_equal_230642, *[x_230643, list_230644], **kwargs_230647)
        
        
        # ################# End of 'test_equality_given(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_equality_given' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_230649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_equality_given'
        return stypy_return_type_230649


    @norecursion
    def test_equality_given2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_equality_given2'
        module_type_store = module_type_store.open_function_context('test_equality_given2', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_equality_given2')
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_equality_given2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_equality_given2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_equality_given2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_equality_given2(...)' code ##################

        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to fmin_slsqp(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'self' (line 259)
        self_230651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 25), 'self', False)
        # Obtaining the member 'fun' of a type (line 259)
        fun_230652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 25), self_230651, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_230653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        float_230654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 35), list_230653, float_230654)
        # Adding element type (line 259)
        float_230655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 35), list_230653, float_230655)
        
        # Processing the call keyword arguments (line 259)
        # Getting the type of 'self' (line 260)
        self_230656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 32), 'self', False)
        # Obtaining the member 'jac' of a type (line 260)
        jac_230657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 32), self_230656, 'jac')
        keyword_230658 = jac_230657
        
        # Obtaining an instance of the builtin type 'tuple' (line 260)
        tuple_230659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 260)
        # Adding element type (line 260)
        float_230660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 48), tuple_230659, float_230660)
        
        keyword_230661 = tuple_230659
        # Getting the type of 'self' (line 261)
        self_230662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 36), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 261)
        f_eqcon_230663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 36), self_230662, 'f_eqcon')
        keyword_230664 = f_eqcon_230663
        # Getting the type of 'self' (line 262)
        self_230665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 41), 'self', False)
        # Obtaining the member 'fprime_eqcon' of a type (line 262)
        fprime_eqcon_230666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 41), self_230665, 'fprime_eqcon')
        keyword_230667 = fprime_eqcon_230666
        int_230668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 34), 'int')
        keyword_230669 = int_230668
        int_230670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 39), 'int')
        keyword_230671 = int_230670
        kwargs_230672 = {'full_output': keyword_230671, 'iprint': keyword_230669, 'args': keyword_230661, 'f_eqcons': keyword_230664, 'fprime': keyword_230658, 'fprime_eqcons': keyword_230667}
        # Getting the type of 'fmin_slsqp' (line 259)
        fmin_slsqp_230650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 14), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 259)
        fmin_slsqp_call_result_230673 = invoke(stypy.reporting.localization.Localization(__file__, 259, 14), fmin_slsqp_230650, *[fun_230652, list_230653], **kwargs_230672)
        
        # Assigning a type to the variable 'res' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'res', fmin_slsqp_call_result_230673)
        
        # Assigning a Name to a Tuple (line 265):
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_230674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        # Getting the type of 'res' (line 265)
        res_230675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___230676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), res_230675, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_230677 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___230676, int_230674)
        
        # Assigning a type to the variable 'tuple_var_assignment_229606' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229606', subscript_call_result_230677)
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_230678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        # Getting the type of 'res' (line 265)
        res_230679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___230680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), res_230679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_230681 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___230680, int_230678)
        
        # Assigning a type to the variable 'tuple_var_assignment_229607' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229607', subscript_call_result_230681)
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_230682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        # Getting the type of 'res' (line 265)
        res_230683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___230684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), res_230683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_230685 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___230684, int_230682)
        
        # Assigning a type to the variable 'tuple_var_assignment_229608' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229608', subscript_call_result_230685)
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_230686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        # Getting the type of 'res' (line 265)
        res_230687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___230688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), res_230687, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_230689 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___230688, int_230686)
        
        # Assigning a type to the variable 'tuple_var_assignment_229609' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229609', subscript_call_result_230689)
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_230690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        # Getting the type of 'res' (line 265)
        res_230691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___230692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), res_230691, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_230693 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___230692, int_230690)
        
        # Assigning a type to the variable 'tuple_var_assignment_229610' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229610', subscript_call_result_230693)
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'tuple_var_assignment_229606' (line 265)
        tuple_var_assignment_229606_230694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229606')
        # Assigning a type to the variable 'x' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'x', tuple_var_assignment_229606_230694)
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'tuple_var_assignment_229607' (line 265)
        tuple_var_assignment_229607_230695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229607')
        # Assigning a type to the variable 'fx' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'fx', tuple_var_assignment_229607_230695)
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'tuple_var_assignment_229608' (line 265)
        tuple_var_assignment_229608_230696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229608')
        # Assigning a type to the variable 'its' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'its', tuple_var_assignment_229608_230696)
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'tuple_var_assignment_229609' (line 265)
        tuple_var_assignment_229609_230697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229609')
        # Assigning a type to the variable 'imode' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'imode', tuple_var_assignment_229609_230697)
        
        # Assigning a Name to a Name (line 265):
        # Getting the type of 'tuple_var_assignment_229610' (line 265)
        tuple_var_assignment_229610_230698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_229610')
        # Assigning a type to the variable 'smode' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 27), 'smode', tuple_var_assignment_229610_230698)
        
        # Call to assert_(...): (line 266)
        # Processing the call arguments (line 266)
        
        # Getting the type of 'imode' (line 266)
        imode_230700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'imode', False)
        int_230701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 25), 'int')
        # Applying the binary operator '==' (line 266)
        result_eq_230702 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 16), '==', imode_230700, int_230701)
        
        # Getting the type of 'imode' (line 266)
        imode_230703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'imode', False)
        # Processing the call keyword arguments (line 266)
        kwargs_230704 = {}
        # Getting the type of 'assert_' (line 266)
        assert__230699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 266)
        assert__call_result_230705 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), assert__230699, *[result_eq_230702, imode_230703], **kwargs_230704)
        
        
        # Call to assert_array_almost_equal(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'x' (line 267)
        x_230707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_230708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_230709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 37), list_230708, int_230709)
        # Adding element type (line 267)
        int_230710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 37), list_230708, int_230710)
        
        # Processing the call keyword arguments (line 267)
        kwargs_230711 = {}
        # Getting the type of 'assert_array_almost_equal' (line 267)
        assert_array_almost_equal_230706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 267)
        assert_array_almost_equal_call_result_230712 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), assert_array_almost_equal_230706, *[x_230707, list_230708], **kwargs_230711)
        
        
        # ################# End of 'test_equality_given2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_equality_given2' in the type store
        # Getting the type of 'stypy_return_type' (line 257)
        stypy_return_type_230713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_equality_given2'
        return stypy_return_type_230713


    @norecursion
    def test_inequality_given(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_inequality_given'
        module_type_store = module_type_store.open_function_context('test_inequality_given', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_inequality_given')
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_inequality_given.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_inequality_given', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_inequality_given', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_inequality_given(...)' code ##################

        
        # Assigning a Call to a Name (line 271):
        
        # Assigning a Call to a Name (line 271):
        
        # Call to fmin_slsqp(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'self' (line 271)
        self_230715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 25), 'self', False)
        # Obtaining the member 'fun' of a type (line 271)
        fun_230716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 25), self_230715, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 271)
        list_230717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 271)
        # Adding element type (line 271)
        float_230718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 35), list_230717, float_230718)
        # Adding element type (line 271)
        float_230719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 35), list_230717, float_230719)
        
        # Processing the call keyword arguments (line 271)
        # Getting the type of 'self' (line 272)
        self_230720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 32), 'self', False)
        # Obtaining the member 'jac' of a type (line 272)
        jac_230721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 32), self_230720, 'jac')
        keyword_230722 = jac_230721
        
        # Obtaining an instance of the builtin type 'tuple' (line 272)
        tuple_230723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 272)
        # Adding element type (line 272)
        float_230724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 48), tuple_230723, float_230724)
        
        keyword_230725 = tuple_230723
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_230726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        # Adding element type (line 273)
        # Getting the type of 'self' (line 273)
        self_230727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 36), 'self', False)
        # Obtaining the member 'f_ieqcon' of a type (line 273)
        f_ieqcon_230728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 36), self_230727, 'f_ieqcon')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 35), list_230726, f_ieqcon_230728)
        
        keyword_230729 = list_230726
        int_230730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 34), 'int')
        keyword_230731 = int_230730
        int_230732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 51), 'int')
        keyword_230733 = int_230732
        kwargs_230734 = {'iprint': keyword_230731, 'ieqcons': keyword_230729, 'args': keyword_230725, 'fprime': keyword_230722, 'full_output': keyword_230733}
        # Getting the type of 'fmin_slsqp' (line 271)
        fmin_slsqp_230714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 271)
        fmin_slsqp_call_result_230735 = invoke(stypy.reporting.localization.Localization(__file__, 271, 14), fmin_slsqp_230714, *[fun_230716, list_230717], **kwargs_230734)
        
        # Assigning a type to the variable 'res' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'res', fmin_slsqp_call_result_230735)
        
        # Assigning a Name to a Tuple (line 275):
        
        # Assigning a Subscript to a Name (line 275):
        
        # Obtaining the type of the subscript
        int_230736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 8), 'int')
        # Getting the type of 'res' (line 275)
        res_230737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___230738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), res_230737, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_230739 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), getitem___230738, int_230736)
        
        # Assigning a type to the variable 'tuple_var_assignment_229611' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229611', subscript_call_result_230739)
        
        # Assigning a Subscript to a Name (line 275):
        
        # Obtaining the type of the subscript
        int_230740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 8), 'int')
        # Getting the type of 'res' (line 275)
        res_230741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___230742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), res_230741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_230743 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), getitem___230742, int_230740)
        
        # Assigning a type to the variable 'tuple_var_assignment_229612' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229612', subscript_call_result_230743)
        
        # Assigning a Subscript to a Name (line 275):
        
        # Obtaining the type of the subscript
        int_230744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 8), 'int')
        # Getting the type of 'res' (line 275)
        res_230745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___230746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), res_230745, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_230747 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), getitem___230746, int_230744)
        
        # Assigning a type to the variable 'tuple_var_assignment_229613' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229613', subscript_call_result_230747)
        
        # Assigning a Subscript to a Name (line 275):
        
        # Obtaining the type of the subscript
        int_230748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 8), 'int')
        # Getting the type of 'res' (line 275)
        res_230749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___230750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), res_230749, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_230751 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), getitem___230750, int_230748)
        
        # Assigning a type to the variable 'tuple_var_assignment_229614' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229614', subscript_call_result_230751)
        
        # Assigning a Subscript to a Name (line 275):
        
        # Obtaining the type of the subscript
        int_230752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 8), 'int')
        # Getting the type of 'res' (line 275)
        res_230753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___230754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), res_230753, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_230755 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), getitem___230754, int_230752)
        
        # Assigning a type to the variable 'tuple_var_assignment_229615' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229615', subscript_call_result_230755)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'tuple_var_assignment_229611' (line 275)
        tuple_var_assignment_229611_230756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229611')
        # Assigning a type to the variable 'x' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'x', tuple_var_assignment_229611_230756)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'tuple_var_assignment_229612' (line 275)
        tuple_var_assignment_229612_230757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229612')
        # Assigning a type to the variable 'fx' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'fx', tuple_var_assignment_229612_230757)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'tuple_var_assignment_229613' (line 275)
        tuple_var_assignment_229613_230758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229613')
        # Assigning a type to the variable 'its' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'its', tuple_var_assignment_229613_230758)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'tuple_var_assignment_229614' (line 275)
        tuple_var_assignment_229614_230759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229614')
        # Assigning a type to the variable 'imode' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 20), 'imode', tuple_var_assignment_229614_230759)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'tuple_var_assignment_229615' (line 275)
        tuple_var_assignment_229615_230760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tuple_var_assignment_229615')
        # Assigning a type to the variable 'smode' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'smode', tuple_var_assignment_229615_230760)
        
        # Call to assert_(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Getting the type of 'imode' (line 276)
        imode_230762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'imode', False)
        int_230763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 25), 'int')
        # Applying the binary operator '==' (line 276)
        result_eq_230764 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 16), '==', imode_230762, int_230763)
        
        # Getting the type of 'imode' (line 276)
        imode_230765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'imode', False)
        # Processing the call keyword arguments (line 276)
        kwargs_230766 = {}
        # Getting the type of 'assert_' (line 276)
        assert__230761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 276)
        assert__call_result_230767 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), assert__230761, *[result_eq_230764, imode_230765], **kwargs_230766)
        
        
        # Call to assert_array_almost_equal(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'x' (line 277)
        x_230769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_230770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        int_230771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 37), list_230770, int_230771)
        # Adding element type (line 277)
        int_230772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 37), list_230770, int_230772)
        
        # Processing the call keyword arguments (line 277)
        int_230773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 53), 'int')
        keyword_230774 = int_230773
        kwargs_230775 = {'decimal': keyword_230774}
        # Getting the type of 'assert_array_almost_equal' (line 277)
        assert_array_almost_equal_230768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 277)
        assert_array_almost_equal_call_result_230776 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), assert_array_almost_equal_230768, *[x_230769, list_230770], **kwargs_230775)
        
        
        # ################# End of 'test_inequality_given(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_inequality_given' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_230777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230777)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_inequality_given'
        return stypy_return_type_230777


    @norecursion
    def test_bound_equality_given2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bound_equality_given2'
        module_type_store = module_type_store.open_function_context('test_bound_equality_given2', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_bound_equality_given2')
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_bound_equality_given2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_bound_equality_given2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bound_equality_given2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bound_equality_given2(...)' code ##################

        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to fmin_slsqp(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'self' (line 281)
        self_230779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 25), 'self', False)
        # Obtaining the member 'fun' of a type (line 281)
        fun_230780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 25), self_230779, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 281)
        list_230781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 281)
        # Adding element type (line 281)
        float_230782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 35), list_230781, float_230782)
        # Adding element type (line 281)
        float_230783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 35), list_230781, float_230783)
        
        # Processing the call keyword arguments (line 281)
        # Getting the type of 'self' (line 282)
        self_230784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'self', False)
        # Obtaining the member 'jac' of a type (line 282)
        jac_230785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 32), self_230784, 'jac')
        keyword_230786 = jac_230785
        
        # Obtaining an instance of the builtin type 'tuple' (line 282)
        tuple_230787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 282)
        # Adding element type (line 282)
        float_230788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 48), tuple_230787, float_230788)
        
        keyword_230789 = tuple_230787
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_230790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        
        # Obtaining an instance of the builtin type 'tuple' (line 283)
        tuple_230791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 283)
        # Adding element type (line 283)
        float_230792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 36), tuple_230791, float_230792)
        # Adding element type (line 283)
        float_230793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 36), tuple_230791, float_230793)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 34), list_230790, tuple_230791)
        # Adding element type (line 283)
        
        # Obtaining an instance of the builtin type 'tuple' (line 283)
        tuple_230794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 283)
        # Adding element type (line 283)
        int_230795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 48), tuple_230794, int_230795)
        # Adding element type (line 283)
        float_230796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 48), tuple_230794, float_230796)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 34), list_230790, tuple_230794)
        
        keyword_230797 = list_230790
        # Getting the type of 'self' (line 284)
        self_230798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 36), 'self', False)
        # Obtaining the member 'f_eqcon' of a type (line 284)
        f_eqcon_230799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 36), self_230798, 'f_eqcon')
        keyword_230800 = f_eqcon_230799
        # Getting the type of 'self' (line 285)
        self_230801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 41), 'self', False)
        # Obtaining the member 'fprime_eqcon' of a type (line 285)
        fprime_eqcon_230802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 41), self_230801, 'fprime_eqcon')
        keyword_230803 = fprime_eqcon_230802
        int_230804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 34), 'int')
        keyword_230805 = int_230804
        int_230806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 51), 'int')
        keyword_230807 = int_230806
        kwargs_230808 = {'full_output': keyword_230807, 'iprint': keyword_230805, 'args': keyword_230789, 'bounds': keyword_230797, 'f_eqcons': keyword_230800, 'fprime': keyword_230786, 'fprime_eqcons': keyword_230803}
        # Getting the type of 'fmin_slsqp' (line 281)
        fmin_slsqp_230778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 14), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 281)
        fmin_slsqp_call_result_230809 = invoke(stypy.reporting.localization.Localization(__file__, 281, 14), fmin_slsqp_230778, *[fun_230780, list_230781], **kwargs_230808)
        
        # Assigning a type to the variable 'res' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'res', fmin_slsqp_call_result_230809)
        
        # Assigning a Name to a Tuple (line 287):
        
        # Assigning a Subscript to a Name (line 287):
        
        # Obtaining the type of the subscript
        int_230810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
        # Getting the type of 'res' (line 287)
        res_230811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___230812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), res_230811, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_230813 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___230812, int_230810)
        
        # Assigning a type to the variable 'tuple_var_assignment_229616' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229616', subscript_call_result_230813)
        
        # Assigning a Subscript to a Name (line 287):
        
        # Obtaining the type of the subscript
        int_230814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
        # Getting the type of 'res' (line 287)
        res_230815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___230816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), res_230815, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_230817 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___230816, int_230814)
        
        # Assigning a type to the variable 'tuple_var_assignment_229617' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229617', subscript_call_result_230817)
        
        # Assigning a Subscript to a Name (line 287):
        
        # Obtaining the type of the subscript
        int_230818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
        # Getting the type of 'res' (line 287)
        res_230819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___230820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), res_230819, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_230821 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___230820, int_230818)
        
        # Assigning a type to the variable 'tuple_var_assignment_229618' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229618', subscript_call_result_230821)
        
        # Assigning a Subscript to a Name (line 287):
        
        # Obtaining the type of the subscript
        int_230822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
        # Getting the type of 'res' (line 287)
        res_230823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___230824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), res_230823, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_230825 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___230824, int_230822)
        
        # Assigning a type to the variable 'tuple_var_assignment_229619' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229619', subscript_call_result_230825)
        
        # Assigning a Subscript to a Name (line 287):
        
        # Obtaining the type of the subscript
        int_230826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 8), 'int')
        # Getting the type of 'res' (line 287)
        res_230827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'res')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___230828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), res_230827, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_230829 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), getitem___230828, int_230826)
        
        # Assigning a type to the variable 'tuple_var_assignment_229620' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229620', subscript_call_result_230829)
        
        # Assigning a Name to a Name (line 287):
        # Getting the type of 'tuple_var_assignment_229616' (line 287)
        tuple_var_assignment_229616_230830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229616')
        # Assigning a type to the variable 'x' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'x', tuple_var_assignment_229616_230830)
        
        # Assigning a Name to a Name (line 287):
        # Getting the type of 'tuple_var_assignment_229617' (line 287)
        tuple_var_assignment_229617_230831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229617')
        # Assigning a type to the variable 'fx' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'fx', tuple_var_assignment_229617_230831)
        
        # Assigning a Name to a Name (line 287):
        # Getting the type of 'tuple_var_assignment_229618' (line 287)
        tuple_var_assignment_229618_230832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229618')
        # Assigning a type to the variable 'its' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'its', tuple_var_assignment_229618_230832)
        
        # Assigning a Name to a Name (line 287):
        # Getting the type of 'tuple_var_assignment_229619' (line 287)
        tuple_var_assignment_229619_230833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229619')
        # Assigning a type to the variable 'imode' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'imode', tuple_var_assignment_229619_230833)
        
        # Assigning a Name to a Name (line 287):
        # Getting the type of 'tuple_var_assignment_229620' (line 287)
        tuple_var_assignment_229620_230834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'tuple_var_assignment_229620')
        # Assigning a type to the variable 'smode' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 27), 'smode', tuple_var_assignment_229620_230834)
        
        # Call to assert_(...): (line 288)
        # Processing the call arguments (line 288)
        
        # Getting the type of 'imode' (line 288)
        imode_230836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'imode', False)
        int_230837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 25), 'int')
        # Applying the binary operator '==' (line 288)
        result_eq_230838 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 16), '==', imode_230836, int_230837)
        
        # Getting the type of 'imode' (line 288)
        imode_230839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 28), 'imode', False)
        # Processing the call keyword arguments (line 288)
        kwargs_230840 = {}
        # Getting the type of 'assert_' (line 288)
        assert__230835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 288)
        assert__call_result_230841 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), assert__230835, *[result_eq_230838, imode_230839], **kwargs_230840)
        
        
        # Call to assert_array_almost_equal(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'x' (line 289)
        x_230843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 289)
        list_230844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 289)
        # Adding element type (line 289)
        float_230845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 37), list_230844, float_230845)
        # Adding element type (line 289)
        float_230846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 37), list_230844, float_230846)
        
        # Processing the call keyword arguments (line 289)
        int_230847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 57), 'int')
        keyword_230848 = int_230847
        kwargs_230849 = {'decimal': keyword_230848}
        # Getting the type of 'assert_array_almost_equal' (line 289)
        assert_array_almost_equal_230842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 289)
        assert_array_almost_equal_call_result_230850 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), assert_array_almost_equal_230842, *[x_230843, list_230844], **kwargs_230849)
        
        
        # Call to assert_(...): (line 290)
        # Processing the call arguments (line 290)
        
        float_230852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 16), 'float')
        
        # Obtaining the type of the subscript
        int_230853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 26), 'int')
        # Getting the type of 'x' (line 290)
        x_230854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___230855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 24), x_230854, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_230856 = invoke(stypy.reporting.localization.Localization(__file__, 290, 24), getitem___230855, int_230853)
        
        # Applying the binary operator '<=' (line 290)
        result_le_230857 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 16), '<=', float_230852, subscript_call_result_230856)
        int_230858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 32), 'int')
        # Applying the binary operator '<=' (line 290)
        result_le_230859 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 16), '<=', subscript_call_result_230856, int_230858)
        # Applying the binary operator '&' (line 290)
        result_and__230860 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 16), '&', result_le_230857, result_le_230859)
        
        # Processing the call keyword arguments (line 290)
        kwargs_230861 = {}
        # Getting the type of 'assert_' (line 290)
        assert__230851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 290)
        assert__call_result_230862 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), assert__230851, *[result_and__230860], **kwargs_230861)
        
        
        # Call to assert_(...): (line 291)
        # Processing the call arguments (line 291)
        
        int_230864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 16), 'int')
        
        # Obtaining the type of the subscript
        int_230865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 24), 'int')
        # Getting the type of 'x' (line 291)
        x_230866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 22), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___230867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 22), x_230866, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_230868 = invoke(stypy.reporting.localization.Localization(__file__, 291, 22), getitem___230867, int_230865)
        
        # Applying the binary operator '<=' (line 291)
        result_le_230869 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 16), '<=', int_230864, subscript_call_result_230868)
        float_230870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 30), 'float')
        # Applying the binary operator '<=' (line 291)
        result_le_230871 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 16), '<=', subscript_call_result_230868, float_230870)
        # Applying the binary operator '&' (line 291)
        result_and__230872 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 16), '&', result_le_230869, result_le_230871)
        
        # Processing the call keyword arguments (line 291)
        kwargs_230873 = {}
        # Getting the type of 'assert_' (line 291)
        assert__230863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 291)
        assert__call_result_230874 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), assert__230863, *[result_and__230872], **kwargs_230873)
        
        
        # ################# End of 'test_bound_equality_given2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bound_equality_given2' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_230875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bound_equality_given2'
        return stypy_return_type_230875


    @norecursion
    def test_scalar_constraints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_constraints'
        module_type_store = module_type_store.open_function_context('test_scalar_constraints', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_scalar_constraints')
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_scalar_constraints.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_scalar_constraints', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_constraints', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_constraints(...)' code ##################

        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to fmin_slsqp(...): (line 295)
        # Processing the call arguments (line 295)

        @norecursion
        def _stypy_temp_lambda_99(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_99'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_99', 295, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_99.stypy_localization = localization
            _stypy_temp_lambda_99.stypy_type_of_self = None
            _stypy_temp_lambda_99.stypy_type_store = module_type_store
            _stypy_temp_lambda_99.stypy_function_name = '_stypy_temp_lambda_99'
            _stypy_temp_lambda_99.stypy_param_names_list = ['z']
            _stypy_temp_lambda_99.stypy_varargs_param_name = None
            _stypy_temp_lambda_99.stypy_kwargs_param_name = None
            _stypy_temp_lambda_99.stypy_call_defaults = defaults
            _stypy_temp_lambda_99.stypy_call_varargs = varargs
            _stypy_temp_lambda_99.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_99', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_99', ['z'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'z' (line 295)
            z_230877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 33), 'z', False)
            int_230878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 36), 'int')
            # Applying the binary operator '**' (line 295)
            result_pow_230879 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 33), '**', z_230877, int_230878)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'stypy_return_type', result_pow_230879)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_99' in the type store
            # Getting the type of 'stypy_return_type' (line 295)
            stypy_return_type_230880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_230880)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_99'
            return stypy_return_type_230880

        # Assigning a type to the variable '_stypy_temp_lambda_99' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), '_stypy_temp_lambda_99', _stypy_temp_lambda_99)
        # Getting the type of '_stypy_temp_lambda_99' (line 295)
        _stypy_temp_lambda_99_230881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), '_stypy_temp_lambda_99')
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_230882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        float_230883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 39), list_230882, float_230883)
        
        # Processing the call keyword arguments (line 295)
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_230884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)

        @norecursion
        def _stypy_temp_lambda_100(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_100'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_100', 296, 32, True)
            # Passed parameters checking function
            _stypy_temp_lambda_100.stypy_localization = localization
            _stypy_temp_lambda_100.stypy_type_of_self = None
            _stypy_temp_lambda_100.stypy_type_store = module_type_store
            _stypy_temp_lambda_100.stypy_function_name = '_stypy_temp_lambda_100'
            _stypy_temp_lambda_100.stypy_param_names_list = ['z']
            _stypy_temp_lambda_100.stypy_varargs_param_name = None
            _stypy_temp_lambda_100.stypy_kwargs_param_name = None
            _stypy_temp_lambda_100.stypy_call_defaults = defaults
            _stypy_temp_lambda_100.stypy_call_varargs = varargs
            _stypy_temp_lambda_100.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_100', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_100', ['z'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_230885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 44), 'int')
            # Getting the type of 'z' (line 296)
            z_230886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 42), 'z', False)
            # Obtaining the member '__getitem__' of a type (line 296)
            getitem___230887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 42), z_230886, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 296)
            subscript_call_result_230888 = invoke(stypy.reporting.localization.Localization(__file__, 296, 42), getitem___230887, int_230885)
            
            int_230889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 49), 'int')
            # Applying the binary operator '-' (line 296)
            result_sub_230890 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 42), '-', subscript_call_result_230888, int_230889)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), 'stypy_return_type', result_sub_230890)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_100' in the type store
            # Getting the type of 'stypy_return_type' (line 296)
            stypy_return_type_230891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_230891)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_100'
            return stypy_return_type_230891

        # Assigning a type to the variable '_stypy_temp_lambda_100' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), '_stypy_temp_lambda_100', _stypy_temp_lambda_100)
        # Getting the type of '_stypy_temp_lambda_100' (line 296)
        _stypy_temp_lambda_100_230892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), '_stypy_temp_lambda_100')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 31), list_230884, _stypy_temp_lambda_100_230892)
        
        keyword_230893 = list_230884
        int_230894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 30), 'int')
        keyword_230895 = int_230894
        kwargs_230896 = {'iprint': keyword_230895, 'ieqcons': keyword_230893}
        # Getting the type of 'fmin_slsqp' (line 295)
        fmin_slsqp_230876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 295)
        fmin_slsqp_call_result_230897 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), fmin_slsqp_230876, *[_stypy_temp_lambda_99_230881, list_230882], **kwargs_230896)
        
        # Assigning a type to the variable 'x' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'x', fmin_slsqp_call_result_230897)
        
        # Call to assert_array_almost_equal(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'x' (line 298)
        x_230899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_230900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        # Adding element type (line 298)
        float_230901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 37), list_230900, float_230901)
        
        # Processing the call keyword arguments (line 298)
        kwargs_230902 = {}
        # Getting the type of 'assert_array_almost_equal' (line 298)
        assert_array_almost_equal_230898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 298)
        assert_array_almost_equal_call_result_230903 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), assert_array_almost_equal_230898, *[x_230899, list_230900], **kwargs_230902)
        
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to fmin_slsqp(...): (line 300)
        # Processing the call arguments (line 300)

        @norecursion
        def _stypy_temp_lambda_101(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_101'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_101', 300, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_101.stypy_localization = localization
            _stypy_temp_lambda_101.stypy_type_of_self = None
            _stypy_temp_lambda_101.stypy_type_store = module_type_store
            _stypy_temp_lambda_101.stypy_function_name = '_stypy_temp_lambda_101'
            _stypy_temp_lambda_101.stypy_param_names_list = ['z']
            _stypy_temp_lambda_101.stypy_varargs_param_name = None
            _stypy_temp_lambda_101.stypy_kwargs_param_name = None
            _stypy_temp_lambda_101.stypy_call_defaults = defaults
            _stypy_temp_lambda_101.stypy_call_varargs = varargs
            _stypy_temp_lambda_101.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_101', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_101', ['z'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'z' (line 300)
            z_230905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 33), 'z', False)
            int_230906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 36), 'int')
            # Applying the binary operator '**' (line 300)
            result_pow_230907 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 33), '**', z_230905, int_230906)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 'stypy_return_type', result_pow_230907)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_101' in the type store
            # Getting the type of 'stypy_return_type' (line 300)
            stypy_return_type_230908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_230908)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_101'
            return stypy_return_type_230908

        # Assigning a type to the variable '_stypy_temp_lambda_101' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), '_stypy_temp_lambda_101', _stypy_temp_lambda_101)
        # Getting the type of '_stypy_temp_lambda_101' (line 300)
        _stypy_temp_lambda_101_230909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), '_stypy_temp_lambda_101')
        
        # Obtaining an instance of the builtin type 'list' (line 300)
        list_230910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 300)
        # Adding element type (line 300)
        float_230911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 39), list_230910, float_230911)
        
        # Processing the call keyword arguments (line 300)

        @norecursion
        def _stypy_temp_lambda_102(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_102'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_102', 301, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_102.stypy_localization = localization
            _stypy_temp_lambda_102.stypy_type_of_self = None
            _stypy_temp_lambda_102.stypy_type_store = module_type_store
            _stypy_temp_lambda_102.stypy_function_name = '_stypy_temp_lambda_102'
            _stypy_temp_lambda_102.stypy_param_names_list = ['z']
            _stypy_temp_lambda_102.stypy_varargs_param_name = None
            _stypy_temp_lambda_102.stypy_kwargs_param_name = None
            _stypy_temp_lambda_102.stypy_call_defaults = defaults
            _stypy_temp_lambda_102.stypy_call_varargs = varargs
            _stypy_temp_lambda_102.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_102', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_102', ['z'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 301)
            list_230912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 301)
            # Adding element type (line 301)
            
            # Obtaining the type of the subscript
            int_230913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 46), 'int')
            # Getting the type of 'z' (line 301)
            z_230914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 44), 'z', False)
            # Obtaining the member '__getitem__' of a type (line 301)
            getitem___230915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 44), z_230914, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 301)
            subscript_call_result_230916 = invoke(stypy.reporting.localization.Localization(__file__, 301, 44), getitem___230915, int_230913)
            
            int_230917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 51), 'int')
            # Applying the binary operator '-' (line 301)
            result_sub_230918 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 44), '-', subscript_call_result_230916, int_230917)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 43), list_230912, result_sub_230918)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 33), 'stypy_return_type', list_230912)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_102' in the type store
            # Getting the type of 'stypy_return_type' (line 301)
            stypy_return_type_230919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_230919)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_102'
            return stypy_return_type_230919

        # Assigning a type to the variable '_stypy_temp_lambda_102' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 33), '_stypy_temp_lambda_102', _stypy_temp_lambda_102)
        # Getting the type of '_stypy_temp_lambda_102' (line 301)
        _stypy_temp_lambda_102_230920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 33), '_stypy_temp_lambda_102')
        keyword_230921 = _stypy_temp_lambda_102_230920
        int_230922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 30), 'int')
        keyword_230923 = int_230922
        kwargs_230924 = {'iprint': keyword_230923, 'f_ieqcons': keyword_230921}
        # Getting the type of 'fmin_slsqp' (line 300)
        fmin_slsqp_230904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 300)
        fmin_slsqp_call_result_230925 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), fmin_slsqp_230904, *[_stypy_temp_lambda_101_230909, list_230910], **kwargs_230924)
        
        # Assigning a type to the variable 'x' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'x', fmin_slsqp_call_result_230925)
        
        # Call to assert_array_almost_equal(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'x' (line 303)
        x_230927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 303)
        list_230928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 303)
        # Adding element type (line 303)
        float_230929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 37), list_230928, float_230929)
        
        # Processing the call keyword arguments (line 303)
        kwargs_230930 = {}
        # Getting the type of 'assert_array_almost_equal' (line 303)
        assert_array_almost_equal_230926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 303)
        assert_array_almost_equal_call_result_230931 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), assert_array_almost_equal_230926, *[x_230927, list_230928], **kwargs_230930)
        
        
        # ################# End of 'test_scalar_constraints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_constraints' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_230932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230932)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_constraints'
        return stypy_return_type_230932


    @norecursion
    def test_integer_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_bounds'
        module_type_store = module_type_store.open_function_context('test_integer_bounds', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_integer_bounds')
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_integer_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_integer_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_bounds(...)' code ##################

        
        # Call to fmin_slsqp(...): (line 307)
        # Processing the call arguments (line 307)

        @norecursion
        def _stypy_temp_lambda_103(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_103'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_103', 307, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_103.stypy_localization = localization
            _stypy_temp_lambda_103.stypy_type_of_self = None
            _stypy_temp_lambda_103.stypy_type_store = module_type_store
            _stypy_temp_lambda_103.stypy_function_name = '_stypy_temp_lambda_103'
            _stypy_temp_lambda_103.stypy_param_names_list = ['z']
            _stypy_temp_lambda_103.stypy_varargs_param_name = None
            _stypy_temp_lambda_103.stypy_kwargs_param_name = None
            _stypy_temp_lambda_103.stypy_call_defaults = defaults
            _stypy_temp_lambda_103.stypy_call_varargs = varargs
            _stypy_temp_lambda_103.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_103', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_103', ['z'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'z' (line 307)
            z_230934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 29), 'z', False)
            int_230935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 32), 'int')
            # Applying the binary operator '**' (line 307)
            result_pow_230936 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 29), '**', z_230934, int_230935)
            
            int_230937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 36), 'int')
            # Applying the binary operator '-' (line 307)
            result_sub_230938 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 29), '-', result_pow_230936, int_230937)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'stypy_return_type', result_sub_230938)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_103' in the type store
            # Getting the type of 'stypy_return_type' (line 307)
            stypy_return_type_230939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_230939)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_103'
            return stypy_return_type_230939

        # Assigning a type to the variable '_stypy_temp_lambda_103' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), '_stypy_temp_lambda_103', _stypy_temp_lambda_103)
        # Getting the type of '_stypy_temp_lambda_103' (line 307)
        _stypy_temp_lambda_103_230940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), '_stypy_temp_lambda_103')
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_230941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_230942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 39), list_230941, int_230942)
        
        # Processing the call keyword arguments (line 307)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_230943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_230944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_230945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 52), list_230944, int_230945)
        # Adding element type (line 307)
        int_230946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 52), list_230944, int_230946)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 51), list_230943, list_230944)
        
        keyword_230947 = list_230943
        int_230948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 68), 'int')
        keyword_230949 = int_230948
        kwargs_230950 = {'iprint': keyword_230949, 'bounds': keyword_230947}
        # Getting the type of 'fmin_slsqp' (line 307)
        fmin_slsqp_230933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 307)
        fmin_slsqp_call_result_230951 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), fmin_slsqp_230933, *[_stypy_temp_lambda_103_230940, list_230941], **kwargs_230950)
        
        
        # ################# End of 'test_integer_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_230952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_bounds'
        return stypy_return_type_230952


    @norecursion
    def test_obj_must_return_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_obj_must_return_scalar'
        module_type_store = module_type_store.open_function_context('test_obj_must_return_scalar', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_obj_must_return_scalar')
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_obj_must_return_scalar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_obj_must_return_scalar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_obj_must_return_scalar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_obj_must_return_scalar(...)' code ##################

        
        # Call to assert_raises(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'ValueError' (line 312)
        ValueError_230954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 312)
        kwargs_230955 = {}
        # Getting the type of 'assert_raises' (line 312)
        assert_raises_230953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 312)
        assert_raises_call_result_230956 = invoke(stypy.reporting.localization.Localization(__file__, 312, 13), assert_raises_230953, *[ValueError_230954], **kwargs_230955)
        
        with_230957 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 312, 13), assert_raises_call_result_230956, 'with parameter', '__enter__', '__exit__')

        if with_230957:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 312)
            enter___230958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 13), assert_raises_call_result_230956, '__enter__')
            with_enter_230959 = invoke(stypy.reporting.localization.Localization(__file__, 312, 13), enter___230958)
            
            # Call to fmin_slsqp(...): (line 313)
            # Processing the call arguments (line 313)

            @norecursion
            def _stypy_temp_lambda_104(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_104'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_104', 313, 23, True)
                # Passed parameters checking function
                _stypy_temp_lambda_104.stypy_localization = localization
                _stypy_temp_lambda_104.stypy_type_of_self = None
                _stypy_temp_lambda_104.stypy_type_store = module_type_store
                _stypy_temp_lambda_104.stypy_function_name = '_stypy_temp_lambda_104'
                _stypy_temp_lambda_104.stypy_param_names_list = ['x']
                _stypy_temp_lambda_104.stypy_varargs_param_name = None
                _stypy_temp_lambda_104.stypy_kwargs_param_name = None
                _stypy_temp_lambda_104.stypy_call_defaults = defaults
                _stypy_temp_lambda_104.stypy_call_varargs = varargs
                _stypy_temp_lambda_104.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_104', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_104', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Obtaining an instance of the builtin type 'list' (line 313)
                list_230961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 33), 'list')
                # Adding type elements to the builtin type 'list' instance (line 313)
                # Adding element type (line 313)
                int_230962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 34), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 33), list_230961, int_230962)
                # Adding element type (line 313)
                int_230963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 37), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 33), list_230961, int_230963)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 313)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), 'stypy_return_type', list_230961)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_104' in the type store
                # Getting the type of 'stypy_return_type' (line 313)
                stypy_return_type_230964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_230964)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_104'
                return stypy_return_type_230964

            # Assigning a type to the variable '_stypy_temp_lambda_104' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), '_stypy_temp_lambda_104', _stypy_temp_lambda_104)
            # Getting the type of '_stypy_temp_lambda_104' (line 313)
            _stypy_temp_lambda_104_230965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), '_stypy_temp_lambda_104')
            
            # Obtaining an instance of the builtin type 'list' (line 313)
            list_230966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 41), 'list')
            # Adding type elements to the builtin type 'list' instance (line 313)
            # Adding element type (line 313)
            int_230967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 42), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 41), list_230966, int_230967)
            # Adding element type (line 313)
            int_230968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 45), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 41), list_230966, int_230968)
            # Adding element type (line 313)
            int_230969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 48), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 41), list_230966, int_230969)
            
            # Processing the call keyword arguments (line 313)
            kwargs_230970 = {}
            # Getting the type of 'fmin_slsqp' (line 313)
            fmin_slsqp_230960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'fmin_slsqp', False)
            # Calling fmin_slsqp(args, kwargs) (line 313)
            fmin_slsqp_call_result_230971 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), fmin_slsqp_230960, *[_stypy_temp_lambda_104_230965, list_230966], **kwargs_230970)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 312)
            exit___230972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 13), assert_raises_call_result_230956, '__exit__')
            with_exit_230973 = invoke(stypy.reporting.localization.Localization(__file__, 312, 13), exit___230972, None, None, None)

        
        # ################# End of 'test_obj_must_return_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_obj_must_return_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_230974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230974)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_obj_must_return_scalar'
        return stypy_return_type_230974


    @norecursion
    def test_obj_returns_scalar_in_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_obj_returns_scalar_in_list'
        module_type_store = module_type_store.open_function_context('test_obj_returns_scalar_in_list', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_obj_returns_scalar_in_list')
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_obj_returns_scalar_in_list.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_obj_returns_scalar_in_list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_obj_returns_scalar_in_list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_obj_returns_scalar_in_list(...)' code ##################

        
        # Call to fmin_slsqp(...): (line 319)
        # Processing the call arguments (line 319)

        @norecursion
        def _stypy_temp_lambda_105(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_105'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_105', 319, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_105.stypy_localization = localization
            _stypy_temp_lambda_105.stypy_type_of_self = None
            _stypy_temp_lambda_105.stypy_type_store = module_type_store
            _stypy_temp_lambda_105.stypy_function_name = '_stypy_temp_lambda_105'
            _stypy_temp_lambda_105.stypy_param_names_list = ['x']
            _stypy_temp_lambda_105.stypy_varargs_param_name = None
            _stypy_temp_lambda_105.stypy_kwargs_param_name = None
            _stypy_temp_lambda_105.stypy_call_defaults = defaults
            _stypy_temp_lambda_105.stypy_call_varargs = varargs
            _stypy_temp_lambda_105.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_105', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_105', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 319)
            list_230976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 319)
            # Adding element type (line 319)
            int_230977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 30), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 29), list_230976, int_230977)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'stypy_return_type', list_230976)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_105' in the type store
            # Getting the type of 'stypy_return_type' (line 319)
            stypy_return_type_230978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_230978)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_105'
            return stypy_return_type_230978

        # Assigning a type to the variable '_stypy_temp_lambda_105' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), '_stypy_temp_lambda_105', _stypy_temp_lambda_105)
        # Getting the type of '_stypy_temp_lambda_105' (line 319)
        _stypy_temp_lambda_105_230979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), '_stypy_temp_lambda_105')
        
        # Obtaining an instance of the builtin type 'list' (line 319)
        list_230980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 319)
        # Adding element type (line 319)
        int_230981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 34), list_230980, int_230981)
        # Adding element type (line 319)
        int_230982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 34), list_230980, int_230982)
        # Adding element type (line 319)
        int_230983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 34), list_230980, int_230983)
        
        # Processing the call keyword arguments (line 319)
        int_230984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 52), 'int')
        keyword_230985 = int_230984
        kwargs_230986 = {'iprint': keyword_230985}
        # Getting the type of 'fmin_slsqp' (line 319)
        fmin_slsqp_230975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'fmin_slsqp', False)
        # Calling fmin_slsqp(args, kwargs) (line 319)
        fmin_slsqp_call_result_230987 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), fmin_slsqp_230975, *[_stypy_temp_lambda_105_230979, list_230980], **kwargs_230986)
        
        
        # ################# End of 'test_obj_returns_scalar_in_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_obj_returns_scalar_in_list' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_230988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230988)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_obj_returns_scalar_in_list'
        return stypy_return_type_230988


    @norecursion
    def test_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_callback'
        module_type_store = module_type_store.open_function_context('test_callback', 321, 4, False)
        # Assigning a type to the variable 'self' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_callback')
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_callback.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_callback', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_callback', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_callback(...)' code ##################

        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to MyCallBack(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_230990 = {}
        # Getting the type of 'MyCallBack' (line 323)
        MyCallBack_230989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'MyCallBack', False)
        # Calling MyCallBack(args, kwargs) (line 323)
        MyCallBack_call_result_230991 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), MyCallBack_230989, *[], **kwargs_230990)
        
        # Assigning a type to the variable 'callback' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'callback', MyCallBack_call_result_230991)
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to minimize(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'self' (line 324)
        self_230993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'self', False)
        # Obtaining the member 'fun' of a type (line 324)
        fun_230994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 23), self_230993, 'fun')
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_230995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        float_230996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 33), list_230995, float_230996)
        # Adding element type (line 324)
        float_230997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 33), list_230995, float_230997)
        
        # Processing the call keyword arguments (line 324)
        
        # Obtaining an instance of the builtin type 'tuple' (line 324)
        tuple_230998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 324)
        # Adding element type (line 324)
        float_230999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 52), tuple_230998, float_230999)
        
        keyword_231000 = tuple_230998
        str_231001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 30), 'str', 'SLSQP')
        keyword_231002 = str_231001
        # Getting the type of 'callback' (line 325)
        callback_231003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 48), 'callback', False)
        keyword_231004 = callback_231003
        # Getting the type of 'self' (line 325)
        self_231005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 66), 'self', False)
        # Obtaining the member 'opts' of a type (line 325)
        opts_231006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 66), self_231005, 'opts')
        keyword_231007 = opts_231006
        kwargs_231008 = {'callback': keyword_231004, 'args': keyword_231000, 'method': keyword_231002, 'options': keyword_231007}
        # Getting the type of 'minimize' (line 324)
        minimize_230992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 324)
        minimize_call_result_231009 = invoke(stypy.reporting.localization.Localization(__file__, 324, 14), minimize_230992, *[fun_230994, list_230995], **kwargs_231008)
        
        # Assigning a type to the variable 'res' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'res', minimize_call_result_231009)
        
        # Call to assert_(...): (line 326)
        # Processing the call arguments (line 326)
        
        # Obtaining the type of the subscript
        str_231011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 20), 'str', 'success')
        # Getting the type of 'res' (line 326)
        res_231012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 326)
        getitem___231013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), res_231012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 326)
        subscript_call_result_231014 = invoke(stypy.reporting.localization.Localization(__file__, 326, 16), getitem___231013, str_231011)
        
        
        # Obtaining the type of the subscript
        str_231015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 36), 'str', 'message')
        # Getting the type of 'res' (line 326)
        res_231016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 326)
        getitem___231017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), res_231016, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 326)
        subscript_call_result_231018 = invoke(stypy.reporting.localization.Localization(__file__, 326, 32), getitem___231017, str_231015)
        
        # Processing the call keyword arguments (line 326)
        kwargs_231019 = {}
        # Getting the type of 'assert_' (line 326)
        assert__231010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 326)
        assert__call_result_231020 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), assert__231010, *[subscript_call_result_231014, subscript_call_result_231018], **kwargs_231019)
        
        
        # Call to assert_(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'callback' (line 327)
        callback_231022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'callback', False)
        # Obtaining the member 'been_called' of a type (line 327)
        been_called_231023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), callback_231022, 'been_called')
        # Processing the call keyword arguments (line 327)
        kwargs_231024 = {}
        # Getting the type of 'assert_' (line 327)
        assert__231021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 327)
        assert__call_result_231025 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), assert__231021, *[been_called_231023], **kwargs_231024)
        
        
        # Call to assert_equal(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'callback' (line 328)
        callback_231027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 'callback', False)
        # Obtaining the member 'ncalls' of a type (line 328)
        ncalls_231028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 21), callback_231027, 'ncalls')
        
        # Obtaining the type of the subscript
        str_231029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 42), 'str', 'nit')
        # Getting the type of 'res' (line 328)
        res_231030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 38), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___231031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 38), res_231030, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_231032 = invoke(stypy.reporting.localization.Localization(__file__, 328, 38), getitem___231031, str_231029)
        
        # Processing the call keyword arguments (line 328)
        kwargs_231033 = {}
        # Getting the type of 'assert_equal' (line 328)
        assert_equal_231026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 328)
        assert_equal_call_result_231034 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), assert_equal_231026, *[ncalls_231028, subscript_call_result_231032], **kwargs_231033)
        
        
        # ################# End of 'test_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 321)
        stypy_return_type_231035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231035)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_callback'
        return stypy_return_type_231035


    @norecursion
    def test_inconsistent_linearization(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_inconsistent_linearization'
        module_type_store = module_type_store.open_function_context('test_inconsistent_linearization', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_inconsistent_linearization')
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_inconsistent_linearization.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_inconsistent_linearization', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_inconsistent_linearization', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_inconsistent_linearization(...)' code ##################

        
        # Assigning a List to a Name (line 340):
        
        # Assigning a List to a Name (line 340):
        
        # Obtaining an instance of the builtin type 'list' (line 340)
        list_231036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 340)
        # Adding element type (line 340)
        int_231037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 12), list_231036, int_231037)
        # Adding element type (line 340)
        int_231038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 12), list_231036, int_231038)
        
        # Assigning a type to the variable 'x' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'x', list_231036)
        
        # Assigning a Lambda to a Name (line 341):
        
        # Assigning a Lambda to a Name (line 341):

        @norecursion
        def _stypy_temp_lambda_106(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_106'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_106', 341, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_106.stypy_localization = localization
            _stypy_temp_lambda_106.stypy_type_of_self = None
            _stypy_temp_lambda_106.stypy_type_store = module_type_store
            _stypy_temp_lambda_106.stypy_function_name = '_stypy_temp_lambda_106'
            _stypy_temp_lambda_106.stypy_param_names_list = ['x']
            _stypy_temp_lambda_106.stypy_varargs_param_name = None
            _stypy_temp_lambda_106.stypy_kwargs_param_name = None
            _stypy_temp_lambda_106.stypy_call_defaults = defaults
            _stypy_temp_lambda_106.stypy_call_varargs = varargs
            _stypy_temp_lambda_106.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_106', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_106', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_231039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 25), 'int')
            # Getting the type of 'x' (line 341)
            x_231040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 23), 'x')
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___231041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 23), x_231040, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_231042 = invoke(stypy.reporting.localization.Localization(__file__, 341, 23), getitem___231041, int_231039)
            
            
            # Obtaining the type of the subscript
            int_231043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 32), 'int')
            # Getting the type of 'x' (line 341)
            x_231044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'x')
            # Obtaining the member '__getitem__' of a type (line 341)
            getitem___231045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 30), x_231044, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 341)
            subscript_call_result_231046 = invoke(stypy.reporting.localization.Localization(__file__, 341, 30), getitem___231045, int_231043)
            
            # Applying the binary operator '+' (line 341)
            result_add_231047 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 23), '+', subscript_call_result_231042, subscript_call_result_231046)
            
            int_231048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 37), 'int')
            # Applying the binary operator '-' (line 341)
            result_sub_231049 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 35), '-', result_add_231047, int_231048)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 13), 'stypy_return_type', result_sub_231049)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_106' in the type store
            # Getting the type of 'stypy_return_type' (line 341)
            stypy_return_type_231050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231050)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_106'
            return stypy_return_type_231050

        # Assigning a type to the variable '_stypy_temp_lambda_106' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 13), '_stypy_temp_lambda_106', _stypy_temp_lambda_106)
        # Getting the type of '_stypy_temp_lambda_106' (line 341)
        _stypy_temp_lambda_106_231051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 13), '_stypy_temp_lambda_106')
        # Assigning a type to the variable 'f1' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'f1', _stypy_temp_lambda_106_231051)
        
        # Assigning a Lambda to a Name (line 342):
        
        # Assigning a Lambda to a Name (line 342):

        @norecursion
        def _stypy_temp_lambda_107(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_107'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_107', 342, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_107.stypy_localization = localization
            _stypy_temp_lambda_107.stypy_type_of_self = None
            _stypy_temp_lambda_107.stypy_type_store = module_type_store
            _stypy_temp_lambda_107.stypy_function_name = '_stypy_temp_lambda_107'
            _stypy_temp_lambda_107.stypy_param_names_list = ['x']
            _stypy_temp_lambda_107.stypy_varargs_param_name = None
            _stypy_temp_lambda_107.stypy_kwargs_param_name = None
            _stypy_temp_lambda_107.stypy_call_defaults = defaults
            _stypy_temp_lambda_107.stypy_call_varargs = varargs
            _stypy_temp_lambda_107.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_107', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_107', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_231052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 25), 'int')
            # Getting the type of 'x' (line 342)
            x_231053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 23), 'x')
            # Obtaining the member '__getitem__' of a type (line 342)
            getitem___231054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 23), x_231053, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 342)
            subscript_call_result_231055 = invoke(stypy.reporting.localization.Localization(__file__, 342, 23), getitem___231054, int_231052)
            
            int_231056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 29), 'int')
            # Applying the binary operator '**' (line 342)
            result_pow_231057 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 23), '**', subscript_call_result_231055, int_231056)
            
            int_231058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 33), 'int')
            # Applying the binary operator '-' (line 342)
            result_sub_231059 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 23), '-', result_pow_231057, int_231058)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 342)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'stypy_return_type', result_sub_231059)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_107' in the type store
            # Getting the type of 'stypy_return_type' (line 342)
            stypy_return_type_231060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231060)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_107'
            return stypy_return_type_231060

        # Assigning a type to the variable '_stypy_temp_lambda_107' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), '_stypy_temp_lambda_107', _stypy_temp_lambda_107)
        # Getting the type of '_stypy_temp_lambda_107' (line 342)
        _stypy_temp_lambda_107_231061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), '_stypy_temp_lambda_107')
        # Assigning a type to the variable 'f2' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'f2', _stypy_temp_lambda_107_231061)
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to minimize(...): (line 343)
        # Processing the call arguments (line 343)

        @norecursion
        def _stypy_temp_lambda_108(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_108'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_108', 344, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_108.stypy_localization = localization
            _stypy_temp_lambda_108.stypy_type_of_self = None
            _stypy_temp_lambda_108.stypy_type_store = module_type_store
            _stypy_temp_lambda_108.stypy_function_name = '_stypy_temp_lambda_108'
            _stypy_temp_lambda_108.stypy_param_names_list = ['x']
            _stypy_temp_lambda_108.stypy_varargs_param_name = None
            _stypy_temp_lambda_108.stypy_kwargs_param_name = None
            _stypy_temp_lambda_108.stypy_call_defaults = defaults
            _stypy_temp_lambda_108.stypy_call_varargs = varargs
            _stypy_temp_lambda_108.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_108', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_108', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_231063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 24), 'int')
            # Getting the type of 'x' (line 344)
            x_231064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 22), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 344)
            getitem___231065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 22), x_231064, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 344)
            subscript_call_result_231066 = invoke(stypy.reporting.localization.Localization(__file__, 344, 22), getitem___231065, int_231063)
            
            int_231067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 28), 'int')
            # Applying the binary operator '**' (line 344)
            result_pow_231068 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 22), '**', subscript_call_result_231066, int_231067)
            
            
            # Obtaining the type of the subscript
            int_231069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 34), 'int')
            # Getting the type of 'x' (line 344)
            x_231070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 344)
            getitem___231071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 32), x_231070, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 344)
            subscript_call_result_231072 = invoke(stypy.reporting.localization.Localization(__file__, 344, 32), getitem___231071, int_231069)
            
            int_231073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 38), 'int')
            # Applying the binary operator '**' (line 344)
            result_pow_231074 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 32), '**', subscript_call_result_231072, int_231073)
            
            # Applying the binary operator '+' (line 344)
            result_add_231075 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 22), '+', result_pow_231068, result_pow_231074)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'stypy_return_type', result_add_231075)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_108' in the type store
            # Getting the type of 'stypy_return_type' (line 344)
            stypy_return_type_231076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231076)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_108'
            return stypy_return_type_231076

        # Assigning a type to the variable '_stypy_temp_lambda_108' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), '_stypy_temp_lambda_108', _stypy_temp_lambda_108)
        # Getting the type of '_stypy_temp_lambda_108' (line 344)
        _stypy_temp_lambda_108_231077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), '_stypy_temp_lambda_108')
        # Getting the type of 'x' (line 345)
        x_231078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'x', False)
        # Processing the call keyword arguments (line 343)
        
        # Obtaining an instance of the builtin type 'tuple' (line 346)
        tuple_231079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 346)
        # Adding element type (line 346)
        
        # Obtaining an instance of the builtin type 'dict' (line 346)
        dict_231080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 346)
        # Adding element type (key, value) (line 346)
        str_231081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 26), 'str', 'type')
        str_231082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 33), 'str', 'eq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 25), dict_231080, (str_231081, str_231082))
        # Adding element type (key, value) (line 346)
        str_231083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 38), 'str', 'fun')
        # Getting the type of 'f1' (line 346)
        f1_231084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 45), 'f1', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 25), dict_231080, (str_231083, f1_231084))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 25), tuple_231079, dict_231080)
        # Adding element type (line 346)
        
        # Obtaining an instance of the builtin type 'dict' (line 347)
        dict_231085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 347)
        # Adding element type (key, value) (line 347)
        str_231086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 26), 'str', 'type')
        str_231087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 33), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), dict_231085, (str_231086, str_231087))
        # Adding element type (key, value) (line 347)
        str_231088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 40), 'str', 'fun')
        # Getting the type of 'f2' (line 347)
        f2_231089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 47), 'f2', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), dict_231085, (str_231088, f2_231089))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 25), tuple_231079, dict_231085)
        
        keyword_231090 = tuple_231079
        
        # Obtaining an instance of the builtin type 'tuple' (line 348)
        tuple_231091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 348)
        # Adding element type (line 348)
        
        # Obtaining an instance of the builtin type 'tuple' (line 348)
        tuple_231092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 348)
        # Adding element type (line 348)
        int_231093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 21), tuple_231092, int_231093)
        # Adding element type (line 348)
        # Getting the type of 'None' (line 348)
        None_231094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 23), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 21), tuple_231092, None_231094)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_231091, tuple_231092)
        # Adding element type (line 348)
        
        # Obtaining an instance of the builtin type 'tuple' (line 348)
        tuple_231095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 348)
        # Adding element type (line 348)
        int_231096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 31), tuple_231095, int_231096)
        # Adding element type (line 348)
        # Getting the type of 'None' (line 348)
        None_231097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 33), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 31), tuple_231095, None_231097)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_231091, tuple_231095)
        
        keyword_231098 = tuple_231091
        str_231099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 19), 'str', 'SLSQP')
        keyword_231100 = str_231099
        kwargs_231101 = {'method': keyword_231100, 'bounds': keyword_231098, 'constraints': keyword_231090}
        # Getting the type of 'minimize' (line 343)
        minimize_231062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 343)
        minimize_call_result_231102 = invoke(stypy.reporting.localization.Localization(__file__, 343, 14), minimize_231062, *[_stypy_temp_lambda_108_231077, x_231078], **kwargs_231101)
        
        # Assigning a type to the variable 'sol' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'sol', minimize_call_result_231102)
        
        # Assigning a Attribute to a Name (line 350):
        
        # Assigning a Attribute to a Name (line 350):
        # Getting the type of 'sol' (line 350)
        sol_231103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'sol')
        # Obtaining the member 'x' of a type (line 350)
        x_231104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 12), sol_231103, 'x')
        # Assigning a type to the variable 'x' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'x', x_231104)
        
        # Call to assert_allclose(...): (line 352)
        # Processing the call arguments (line 352)
        
        # Call to f1(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'x' (line 352)
        x_231107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'x', False)
        # Processing the call keyword arguments (line 352)
        kwargs_231108 = {}
        # Getting the type of 'f1' (line 352)
        f1_231106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 24), 'f1', False)
        # Calling f1(args, kwargs) (line 352)
        f1_call_result_231109 = invoke(stypy.reporting.localization.Localization(__file__, 352, 24), f1_231106, *[x_231107], **kwargs_231108)
        
        int_231110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 31), 'int')
        # Processing the call keyword arguments (line 352)
        float_231111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 39), 'float')
        keyword_231112 = float_231111
        kwargs_231113 = {'atol': keyword_231112}
        # Getting the type of 'assert_allclose' (line 352)
        assert_allclose_231105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 352)
        assert_allclose_call_result_231114 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), assert_allclose_231105, *[f1_call_result_231109, int_231110], **kwargs_231113)
        
        
        # Call to assert_(...): (line 353)
        # Processing the call arguments (line 353)
        
        
        # Call to f2(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'x' (line 353)
        x_231117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'x', False)
        # Processing the call keyword arguments (line 353)
        kwargs_231118 = {}
        # Getting the type of 'f2' (line 353)
        f2_231116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'f2', False)
        # Calling f2(args, kwargs) (line 353)
        f2_call_result_231119 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), f2_231116, *[x_231117], **kwargs_231118)
        
        float_231120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 25), 'float')
        # Applying the binary operator '>=' (line 353)
        result_ge_231121 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 16), '>=', f2_call_result_231119, float_231120)
        
        # Processing the call keyword arguments (line 353)
        kwargs_231122 = {}
        # Getting the type of 'assert_' (line 353)
        assert__231115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 353)
        assert__call_result_231123 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), assert__231115, *[result_ge_231121], **kwargs_231122)
        
        
        # Call to assert_(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'sol' (line 354)
        sol_231125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 354)
        success_231126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 16), sol_231125, 'success')
        # Getting the type of 'sol' (line 354)
        sol_231127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 29), 'sol', False)
        # Processing the call keyword arguments (line 354)
        kwargs_231128 = {}
        # Getting the type of 'assert_' (line 354)
        assert__231124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 354)
        assert__call_result_231129 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), assert__231124, *[success_231126, sol_231127], **kwargs_231128)
        
        
        # ################# End of 'test_inconsistent_linearization(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_inconsistent_linearization' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_231130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231130)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_inconsistent_linearization'
        return stypy_return_type_231130


    @norecursion
    def test_regression_5743(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_regression_5743'
        module_type_store = module_type_store.open_function_context('test_regression_5743', 356, 4, False)
        # Assigning a type to the variable 'self' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_regression_5743')
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_regression_5743.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_regression_5743', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_regression_5743', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_regression_5743(...)' code ##################

        
        # Assigning a List to a Name (line 360):
        
        # Assigning a List to a Name (line 360):
        
        # Obtaining an instance of the builtin type 'list' (line 360)
        list_231131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 360)
        # Adding element type (line 360)
        int_231132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 12), list_231131, int_231132)
        # Adding element type (line 360)
        int_231133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 12), list_231131, int_231133)
        
        # Assigning a type to the variable 'x' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'x', list_231131)
        
        # Assigning a Call to a Name (line 361):
        
        # Assigning a Call to a Name (line 361):
        
        # Call to minimize(...): (line 361)
        # Processing the call arguments (line 361)

        @norecursion
        def _stypy_temp_lambda_109(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_109'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_109', 362, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_109.stypy_localization = localization
            _stypy_temp_lambda_109.stypy_type_of_self = None
            _stypy_temp_lambda_109.stypy_type_store = module_type_store
            _stypy_temp_lambda_109.stypy_function_name = '_stypy_temp_lambda_109'
            _stypy_temp_lambda_109.stypy_param_names_list = ['x']
            _stypy_temp_lambda_109.stypy_varargs_param_name = None
            _stypy_temp_lambda_109.stypy_kwargs_param_name = None
            _stypy_temp_lambda_109.stypy_call_defaults = defaults
            _stypy_temp_lambda_109.stypy_call_varargs = varargs
            _stypy_temp_lambda_109.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_109', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_109', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_231135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 24), 'int')
            # Getting the type of 'x' (line 362)
            x_231136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 362)
            getitem___231137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 22), x_231136, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 362)
            subscript_call_result_231138 = invoke(stypy.reporting.localization.Localization(__file__, 362, 22), getitem___231137, int_231135)
            
            int_231139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 28), 'int')
            # Applying the binary operator '**' (line 362)
            result_pow_231140 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 22), '**', subscript_call_result_231138, int_231139)
            
            
            # Obtaining the type of the subscript
            int_231141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 34), 'int')
            # Getting the type of 'x' (line 362)
            x_231142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 32), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 362)
            getitem___231143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 32), x_231142, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 362)
            subscript_call_result_231144 = invoke(stypy.reporting.localization.Localization(__file__, 362, 32), getitem___231143, int_231141)
            
            int_231145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 38), 'int')
            # Applying the binary operator '**' (line 362)
            result_pow_231146 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 32), '**', subscript_call_result_231144, int_231145)
            
            # Applying the binary operator '+' (line 362)
            result_add_231147 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 22), '+', result_pow_231140, result_pow_231146)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'stypy_return_type', result_add_231147)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_109' in the type store
            # Getting the type of 'stypy_return_type' (line 362)
            stypy_return_type_231148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231148)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_109'
            return stypy_return_type_231148

        # Assigning a type to the variable '_stypy_temp_lambda_109' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), '_stypy_temp_lambda_109', _stypy_temp_lambda_109)
        # Getting the type of '_stypy_temp_lambda_109' (line 362)
        _stypy_temp_lambda_109_231149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), '_stypy_temp_lambda_109')
        # Getting the type of 'x' (line 363)
        x_231150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'x', False)
        # Processing the call keyword arguments (line 361)
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_231151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        
        # Obtaining an instance of the builtin type 'dict' (line 364)
        dict_231152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 364)
        # Adding element type (key, value) (line 364)
        str_231153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 26), 'str', 'type')
        str_231154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 33), 'str', 'eq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 25), dict_231152, (str_231153, str_231154))
        # Adding element type (key, value) (line 364)
        str_231155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 38), 'str', 'fun')

        @norecursion
        def _stypy_temp_lambda_110(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_110'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_110', 364, 45, True)
            # Passed parameters checking function
            _stypy_temp_lambda_110.stypy_localization = localization
            _stypy_temp_lambda_110.stypy_type_of_self = None
            _stypy_temp_lambda_110.stypy_type_store = module_type_store
            _stypy_temp_lambda_110.stypy_function_name = '_stypy_temp_lambda_110'
            _stypy_temp_lambda_110.stypy_param_names_list = ['x']
            _stypy_temp_lambda_110.stypy_varargs_param_name = None
            _stypy_temp_lambda_110.stypy_kwargs_param_name = None
            _stypy_temp_lambda_110.stypy_call_defaults = defaults
            _stypy_temp_lambda_110.stypy_call_varargs = varargs
            _stypy_temp_lambda_110.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_110', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_110', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_231156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 57), 'int')
            # Getting the type of 'x' (line 364)
            x_231157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 55), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 364)
            getitem___231158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 55), x_231157, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 364)
            subscript_call_result_231159 = invoke(stypy.reporting.localization.Localization(__file__, 364, 55), getitem___231158, int_231156)
            
            
            # Obtaining the type of the subscript
            int_231160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 62), 'int')
            # Getting the type of 'x' (line 364)
            x_231161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 60), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 364)
            getitem___231162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 60), x_231161, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 364)
            subscript_call_result_231163 = invoke(stypy.reporting.localization.Localization(__file__, 364, 60), getitem___231162, int_231160)
            
            # Applying the binary operator '+' (line 364)
            result_add_231164 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 55), '+', subscript_call_result_231159, subscript_call_result_231163)
            
            int_231165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 65), 'int')
            # Applying the binary operator '-' (line 364)
            result_sub_231166 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 64), '-', result_add_231164, int_231165)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 364)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 45), 'stypy_return_type', result_sub_231166)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_110' in the type store
            # Getting the type of 'stypy_return_type' (line 364)
            stypy_return_type_231167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 45), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231167)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_110'
            return stypy_return_type_231167

        # Assigning a type to the variable '_stypy_temp_lambda_110' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 45), '_stypy_temp_lambda_110', _stypy_temp_lambda_110)
        # Getting the type of '_stypy_temp_lambda_110' (line 364)
        _stypy_temp_lambda_110_231168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 45), '_stypy_temp_lambda_110')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 25), dict_231152, (str_231155, _stypy_temp_lambda_110_231168))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 25), tuple_231151, dict_231152)
        # Adding element type (line 364)
        
        # Obtaining an instance of the builtin type 'dict' (line 365)
        dict_231169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 365)
        # Adding element type (key, value) (line 365)
        str_231170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 26), 'str', 'type')
        str_231171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 33), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 25), dict_231169, (str_231170, str_231171))
        # Adding element type (key, value) (line 365)
        str_231172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 40), 'str', 'fun')

        @norecursion
        def _stypy_temp_lambda_111(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_111'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_111', 365, 47, True)
            # Passed parameters checking function
            _stypy_temp_lambda_111.stypy_localization = localization
            _stypy_temp_lambda_111.stypy_type_of_self = None
            _stypy_temp_lambda_111.stypy_type_store = module_type_store
            _stypy_temp_lambda_111.stypy_function_name = '_stypy_temp_lambda_111'
            _stypy_temp_lambda_111.stypy_param_names_list = ['x']
            _stypy_temp_lambda_111.stypy_varargs_param_name = None
            _stypy_temp_lambda_111.stypy_kwargs_param_name = None
            _stypy_temp_lambda_111.stypy_call_defaults = defaults
            _stypy_temp_lambda_111.stypy_call_varargs = varargs
            _stypy_temp_lambda_111.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_111', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_111', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_231173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 59), 'int')
            # Getting the type of 'x' (line 365)
            x_231174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 57), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 365)
            getitem___231175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 57), x_231174, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 365)
            subscript_call_result_231176 = invoke(stypy.reporting.localization.Localization(__file__, 365, 57), getitem___231175, int_231173)
            
            int_231177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 62), 'int')
            # Applying the binary operator '-' (line 365)
            result_sub_231178 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 57), '-', subscript_call_result_231176, int_231177)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 365)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 47), 'stypy_return_type', result_sub_231178)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_111' in the type store
            # Getting the type of 'stypy_return_type' (line 365)
            stypy_return_type_231179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 47), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231179)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_111'
            return stypy_return_type_231179

        # Assigning a type to the variable '_stypy_temp_lambda_111' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 47), '_stypy_temp_lambda_111', _stypy_temp_lambda_111)
        # Getting the type of '_stypy_temp_lambda_111' (line 365)
        _stypy_temp_lambda_111_231180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 47), '_stypy_temp_lambda_111')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 25), dict_231169, (str_231172, _stypy_temp_lambda_111_231180))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 25), tuple_231151, dict_231169)
        
        keyword_231181 = tuple_231151
        
        # Obtaining an instance of the builtin type 'tuple' (line 366)
        tuple_231182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 366)
        # Adding element type (line 366)
        
        # Obtaining an instance of the builtin type 'tuple' (line 366)
        tuple_231183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 366)
        # Adding element type (line 366)
        int_231184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 21), tuple_231183, int_231184)
        # Adding element type (line 366)
        # Getting the type of 'None' (line 366)
        None_231185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 23), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 21), tuple_231183, None_231185)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 20), tuple_231182, tuple_231183)
        # Adding element type (line 366)
        
        # Obtaining an instance of the builtin type 'tuple' (line 366)
        tuple_231186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 366)
        # Adding element type (line 366)
        int_231187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 31), tuple_231186, int_231187)
        # Adding element type (line 366)
        # Getting the type of 'None' (line 366)
        None_231188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 33), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 31), tuple_231186, None_231188)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 20), tuple_231182, tuple_231186)
        
        keyword_231189 = tuple_231182
        str_231190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 19), 'str', 'SLSQP')
        keyword_231191 = str_231190
        kwargs_231192 = {'method': keyword_231191, 'bounds': keyword_231189, 'constraints': keyword_231181}
        # Getting the type of 'minimize' (line 361)
        minimize_231134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 361)
        minimize_call_result_231193 = invoke(stypy.reporting.localization.Localization(__file__, 361, 14), minimize_231134, *[_stypy_temp_lambda_109_231149, x_231150], **kwargs_231192)
        
        # Assigning a type to the variable 'sol' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'sol', minimize_call_result_231193)
        
        # Call to assert_(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Getting the type of 'sol' (line 368)
        sol_231195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 20), 'sol', False)
        # Obtaining the member 'success' of a type (line 368)
        success_231196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 20), sol_231195, 'success')
        # Applying the 'not' unary operator (line 368)
        result_not__231197 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 16), 'not', success_231196)
        
        # Getting the type of 'sol' (line 368)
        sol_231198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 33), 'sol', False)
        # Processing the call keyword arguments (line 368)
        kwargs_231199 = {}
        # Getting the type of 'assert_' (line 368)
        assert__231194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 368)
        assert__call_result_231200 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), assert__231194, *[result_not__231197, sol_231198], **kwargs_231199)
        
        
        # ################# End of 'test_regression_5743(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_regression_5743' in the type store
        # Getting the type of 'stypy_return_type' (line 356)
        stypy_return_type_231201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_regression_5743'
        return stypy_return_type_231201


    @norecursion
    def test_gh_6676(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gh_6676'
        module_type_store = module_type_store.open_function_context('test_gh_6676', 370, 4, False)
        # Assigning a type to the variable 'self' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_gh_6676')
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_gh_6676.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_gh_6676', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gh_6676', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gh_6676(...)' code ##################


        @norecursion
        def func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 371, 8, False)
            
            # Passed parameters checking function
            func.stypy_localization = localization
            func.stypy_type_of_self = None
            func.stypy_type_store = module_type_store
            func.stypy_function_name = 'func'
            func.stypy_param_names_list = ['x']
            func.stypy_varargs_param_name = None
            func.stypy_kwargs_param_name = None
            func.stypy_call_defaults = defaults
            func.stypy_call_varargs = varargs
            func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func(...)' code ##################

            
            # Obtaining the type of the subscript
            int_231202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 22), 'int')
            # Getting the type of 'x' (line 372)
            x_231203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'x')
            # Obtaining the member '__getitem__' of a type (line 372)
            getitem___231204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), x_231203, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 372)
            subscript_call_result_231205 = invoke(stypy.reporting.localization.Localization(__file__, 372, 20), getitem___231204, int_231202)
            
            int_231206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 27), 'int')
            # Applying the binary operator '-' (line 372)
            result_sub_231207 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 20), '-', subscript_call_result_231205, int_231206)
            
            int_231208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 31), 'int')
            # Applying the binary operator '**' (line 372)
            result_pow_231209 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 19), '**', result_sub_231207, int_231208)
            
            int_231210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 35), 'int')
            
            # Obtaining the type of the subscript
            int_231211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 40), 'int')
            # Getting the type of 'x' (line 372)
            x_231212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 38), 'x')
            # Obtaining the member '__getitem__' of a type (line 372)
            getitem___231213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 38), x_231212, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 372)
            subscript_call_result_231214 = invoke(stypy.reporting.localization.Localization(__file__, 372, 38), getitem___231213, int_231211)
            
            int_231215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 45), 'int')
            # Applying the binary operator '-' (line 372)
            result_sub_231216 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 38), '-', subscript_call_result_231214, int_231215)
            
            int_231217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 49), 'int')
            # Applying the binary operator '**' (line 372)
            result_pow_231218 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 37), '**', result_sub_231216, int_231217)
            
            # Applying the binary operator '*' (line 372)
            result_mul_231219 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 35), '*', int_231210, result_pow_231218)
            
            # Applying the binary operator '+' (line 372)
            result_add_231220 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 19), '+', result_pow_231209, result_mul_231219)
            
            float_231221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 53), 'float')
            
            # Obtaining the type of the subscript
            int_231222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 60), 'int')
            # Getting the type of 'x' (line 372)
            x_231223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 58), 'x')
            # Obtaining the member '__getitem__' of a type (line 372)
            getitem___231224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 58), x_231223, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 372)
            subscript_call_result_231225 = invoke(stypy.reporting.localization.Localization(__file__, 372, 58), getitem___231224, int_231222)
            
            int_231226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 65), 'int')
            # Applying the binary operator '-' (line 372)
            result_sub_231227 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 58), '-', subscript_call_result_231225, int_231226)
            
            int_231228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 69), 'int')
            # Applying the binary operator '**' (line 372)
            result_pow_231229 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 57), '**', result_sub_231227, int_231228)
            
            # Applying the binary operator '*' (line 372)
            result_mul_231230 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 53), '*', float_231221, result_pow_231229)
            
            # Applying the binary operator '+' (line 372)
            result_add_231231 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 51), '+', result_add_231220, result_mul_231230)
            
            # Assigning a type to the variable 'stypy_return_type' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'stypy_return_type', result_add_231231)
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 371)
            stypy_return_type_231232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231232)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_231232

        # Assigning a type to the variable 'func' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'func', func)
        
        # Assigning a Call to a Name (line 374):
        
        # Assigning a Call to a Name (line 374):
        
        # Call to minimize(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'func' (line 374)
        func_231234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'func', False)
        
        # Obtaining an instance of the builtin type 'list' (line 374)
        list_231235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 374)
        # Adding element type (line 374)
        int_231236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 29), list_231235, int_231236)
        # Adding element type (line 374)
        int_231237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 29), list_231235, int_231237)
        # Adding element type (line 374)
        int_231238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 29), list_231235, int_231238)
        
        # Processing the call keyword arguments (line 374)
        str_231239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 47), 'str', 'SLSQP')
        keyword_231240 = str_231239
        kwargs_231241 = {'method': keyword_231240}
        # Getting the type of 'minimize' (line 374)
        minimize_231233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 374)
        minimize_call_result_231242 = invoke(stypy.reporting.localization.Localization(__file__, 374, 14), minimize_231233, *[func_231234, list_231235], **kwargs_231241)
        
        # Assigning a type to the variable 'sol' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'sol', minimize_call_result_231242)
        
        # Call to assert_(...): (line 375)
        # Processing the call arguments (line 375)
        
        # Getting the type of 'sol' (line 375)
        sol_231244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'sol', False)
        # Obtaining the member 'jac' of a type (line 375)
        jac_231245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 16), sol_231244, 'jac')
        # Obtaining the member 'shape' of a type (line 375)
        shape_231246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 16), jac_231245, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 375)
        tuple_231247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 375)
        # Adding element type (line 375)
        int_231248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 34), tuple_231247, int_231248)
        
        # Applying the binary operator '==' (line 375)
        result_eq_231249 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 16), '==', shape_231246, tuple_231247)
        
        # Processing the call keyword arguments (line 375)
        kwargs_231250 = {}
        # Getting the type of 'assert_' (line 375)
        assert__231243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 375)
        assert__call_result_231251 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), assert__231243, *[result_eq_231249], **kwargs_231250)
        
        
        # ################# End of 'test_gh_6676(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gh_6676' in the type store
        # Getting the type of 'stypy_return_type' (line 370)
        stypy_return_type_231252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gh_6676'
        return stypy_return_type_231252


    @norecursion
    def test_invalid_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_invalid_bounds'
        module_type_store = module_type_store.open_function_context('test_invalid_bounds', 377, 4, False)
        # Assigning a type to the variable 'self' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_invalid_bounds')
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_invalid_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_invalid_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_invalid_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_invalid_bounds(...)' code ##################

        
        # Assigning a List to a Name (line 380):
        
        # Assigning a List to a Name (line 380):
        
        # Obtaining an instance of the builtin type 'list' (line 380)
        list_231253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 380)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_231254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_231255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        int_231256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 14), tuple_231255, int_231256)
        # Adding element type (line 381)
        int_231257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 14), tuple_231255, int_231257)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 13), tuple_231254, tuple_231255)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_231258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        int_231259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 22), tuple_231258, int_231259)
        # Adding element type (line 381)
        int_231260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 22), tuple_231258, int_231260)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 13), tuple_231254, tuple_231258)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 22), list_231253, tuple_231254)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'tuple' (line 382)
        tuple_231261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 382)
        # Adding element type (line 382)
        
        # Obtaining an instance of the builtin type 'tuple' (line 382)
        tuple_231262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 382)
        # Adding element type (line 382)
        int_231263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 14), tuple_231262, int_231263)
        # Adding element type (line 382)
        int_231264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 14), tuple_231262, int_231264)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), tuple_231261, tuple_231262)
        # Adding element type (line 382)
        
        # Obtaining an instance of the builtin type 'tuple' (line 382)
        tuple_231265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 382)
        # Adding element type (line 382)
        int_231266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 22), tuple_231265, int_231266)
        # Adding element type (line 382)
        int_231267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 22), tuple_231265, int_231267)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 13), tuple_231261, tuple_231265)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 22), list_231253, tuple_231261)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_231268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_231269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        int_231270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 14), tuple_231269, int_231270)
        # Adding element type (line 383)
        int_231271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 14), tuple_231269, int_231271)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 13), tuple_231268, tuple_231269)
        # Adding element type (line 383)
        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_231272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        int_231273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 22), tuple_231272, int_231273)
        # Adding element type (line 383)
        int_231274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 22), tuple_231272, int_231274)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 13), tuple_231268, tuple_231272)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 22), list_231253, tuple_231268)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'tuple' (line 384)
        tuple_231275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 384)
        # Adding element type (line 384)
        
        # Obtaining an instance of the builtin type 'tuple' (line 384)
        tuple_231276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 384)
        # Adding element type (line 384)
        # Getting the type of 'np' (line 384)
        np_231277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 14), 'np')
        # Obtaining the member 'inf' of a type (line 384)
        inf_231278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 14), np_231277, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 14), tuple_231276, inf_231278)
        # Adding element type (line 384)
        int_231279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 14), tuple_231276, int_231279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 13), tuple_231275, tuple_231276)
        # Adding element type (line 384)
        
        # Obtaining an instance of the builtin type 'tuple' (line 384)
        tuple_231280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 384)
        # Adding element type (line 384)
        # Getting the type of 'np' (line 384)
        np_231281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'np')
        # Obtaining the member 'inf' of a type (line 384)
        inf_231282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 27), np_231281, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 27), tuple_231280, inf_231282)
        # Adding element type (line 384)
        int_231283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 27), tuple_231280, int_231283)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 13), tuple_231275, tuple_231280)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 22), list_231253, tuple_231275)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'tuple' (line 385)
        tuple_231284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 385)
        # Adding element type (line 385)
        
        # Obtaining an instance of the builtin type 'tuple' (line 385)
        tuple_231285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 385)
        # Adding element type (line 385)
        int_231286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 14), tuple_231285, int_231286)
        # Adding element type (line 385)
        
        # Getting the type of 'np' (line 385)
        np_231287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'np')
        # Obtaining the member 'inf' of a type (line 385)
        inf_231288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), np_231287, 'inf')
        # Applying the 'usub' unary operator (line 385)
        result___neg___231289 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 17), 'usub', inf_231288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 14), tuple_231285, result___neg___231289)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 13), tuple_231284, tuple_231285)
        # Adding element type (line 385)
        
        # Obtaining an instance of the builtin type 'tuple' (line 385)
        tuple_231290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 385)
        # Adding element type (line 385)
        int_231291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 28), tuple_231290, int_231291)
        # Adding element type (line 385)
        int_231292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 28), tuple_231290, int_231292)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 13), tuple_231284, tuple_231290)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 22), list_231253, tuple_231284)
        
        # Assigning a type to the variable 'bounds_list' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'bounds_list', list_231253)
        
        # Getting the type of 'bounds_list' (line 387)
        bounds_list_231293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 22), 'bounds_list')
        # Testing the type of a for loop iterable (line 387)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 387, 8), bounds_list_231293)
        # Getting the type of the for loop variable (line 387)
        for_loop_var_231294 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 387, 8), bounds_list_231293)
        # Assigning a type to the variable 'bounds' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'bounds', for_loop_var_231294)
        # SSA begins for a for statement (line 387)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_raises(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'ValueError' (line 388)
        ValueError_231296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 31), 'ValueError', False)
        # Processing the call keyword arguments (line 388)
        kwargs_231297 = {}
        # Getting the type of 'assert_raises' (line 388)
        assert_raises_231295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 17), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 388)
        assert_raises_call_result_231298 = invoke(stypy.reporting.localization.Localization(__file__, 388, 17), assert_raises_231295, *[ValueError_231296], **kwargs_231297)
        
        with_231299 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 388, 17), assert_raises_call_result_231298, 'with parameter', '__enter__', '__exit__')

        if with_231299:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 388)
            enter___231300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 17), assert_raises_call_result_231298, '__enter__')
            with_enter_231301 = invoke(stypy.reporting.localization.Localization(__file__, 388, 17), enter___231300)
            
            # Call to minimize(...): (line 389)
            # Processing the call arguments (line 389)
            # Getting the type of 'self' (line 389)
            self_231303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 25), 'self', False)
            # Obtaining the member 'fun' of a type (line 389)
            fun_231304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 25), self_231303, 'fun')
            
            # Obtaining an instance of the builtin type 'list' (line 389)
            list_231305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 35), 'list')
            # Adding type elements to the builtin type 'list' instance (line 389)
            # Adding element type (line 389)
            float_231306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 36), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_231305, float_231306)
            # Adding element type (line 389)
            float_231307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 42), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_231305, float_231307)
            
            # Processing the call keyword arguments (line 389)
            # Getting the type of 'bounds' (line 389)
            bounds_231308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 55), 'bounds', False)
            keyword_231309 = bounds_231308
            str_231310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 70), 'str', 'SLSQP')
            keyword_231311 = str_231310
            kwargs_231312 = {'bounds': keyword_231309, 'method': keyword_231311}
            # Getting the type of 'minimize' (line 389)
            minimize_231302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'minimize', False)
            # Calling minimize(args, kwargs) (line 389)
            minimize_call_result_231313 = invoke(stypy.reporting.localization.Localization(__file__, 389, 16), minimize_231302, *[fun_231304, list_231305], **kwargs_231312)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 388)
            exit___231314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 17), assert_raises_call_result_231298, '__exit__')
            with_exit_231315 = invoke(stypy.reporting.localization.Localization(__file__, 388, 17), exit___231314, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_invalid_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_invalid_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 377)
        stypy_return_type_231316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231316)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_invalid_bounds'
        return stypy_return_type_231316


    @norecursion
    def test_bounds_clipping(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bounds_clipping'
        module_type_store = module_type_store.open_function_context('test_bounds_clipping', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_bounds_clipping')
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_bounds_clipping.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_bounds_clipping', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bounds_clipping', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bounds_clipping(...)' code ##################


        @norecursion
        def f(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'f'
            module_type_store = module_type_store.open_function_context('f', 395, 8, False)
            
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
            int_231317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 22), 'int')
            # Getting the type of 'x' (line 396)
            x_231318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'x')
            # Obtaining the member '__getitem__' of a type (line 396)
            getitem___231319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 20), x_231318, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 396)
            subscript_call_result_231320 = invoke(stypy.reporting.localization.Localization(__file__, 396, 20), getitem___231319, int_231317)
            
            int_231321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 27), 'int')
            # Applying the binary operator '-' (line 396)
            result_sub_231322 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 20), '-', subscript_call_result_231320, int_231321)
            
            int_231323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 31), 'int')
            # Applying the binary operator '**' (line 396)
            result_pow_231324 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 19), '**', result_sub_231322, int_231323)
            
            # Assigning a type to the variable 'stypy_return_type' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'stypy_return_type', result_pow_231324)
            
            # ################# End of 'f(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'f' in the type store
            # Getting the type of 'stypy_return_type' (line 395)
            stypy_return_type_231325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231325)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'f'
            return stypy_return_type_231325

        # Assigning a type to the variable 'f' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'f', f)
        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to minimize(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'f' (line 398)
        f_231327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 398)
        list_231328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 398)
        # Adding element type (line 398)
        int_231329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 26), list_231328, int_231329)
        
        # Processing the call keyword arguments (line 398)
        str_231330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 39), 'str', 'slsqp')
        keyword_231331 = str_231330
        
        # Obtaining an instance of the builtin type 'list' (line 398)
        list_231332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 398)
        # Adding element type (line 398)
        
        # Obtaining an instance of the builtin type 'tuple' (line 398)
        tuple_231333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 398)
        # Adding element type (line 398)
        # Getting the type of 'None' (line 398)
        None_231334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 57), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 57), tuple_231333, None_231334)
        # Adding element type (line 398)
        int_231335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 57), tuple_231333, int_231335)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 55), list_231332, tuple_231333)
        
        keyword_231336 = list_231332
        kwargs_231337 = {'method': keyword_231331, 'bounds': keyword_231336}
        # Getting the type of 'minimize' (line 398)
        minimize_231326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 398)
        minimize_call_result_231338 = invoke(stypy.reporting.localization.Localization(__file__, 398, 14), minimize_231326, *[f_231327, list_231328], **kwargs_231337)
        
        # Assigning a type to the variable 'sol' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'sol', minimize_call_result_231338)
        
        # Call to assert_(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'sol' (line 399)
        sol_231340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 399)
        success_231341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 16), sol_231340, 'success')
        # Processing the call keyword arguments (line 399)
        kwargs_231342 = {}
        # Getting the type of 'assert_' (line 399)
        assert__231339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 399)
        assert__call_result_231343 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), assert__231339, *[success_231341], **kwargs_231342)
        
        
        # Call to assert_allclose(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'sol' (line 400)
        sol_231345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 400)
        x_231346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 24), sol_231345, 'x')
        int_231347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 31), 'int')
        # Processing the call keyword arguments (line 400)
        float_231348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 39), 'float')
        keyword_231349 = float_231348
        kwargs_231350 = {'atol': keyword_231349}
        # Getting the type of 'assert_allclose' (line 400)
        assert_allclose_231344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 400)
        assert_allclose_call_result_231351 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), assert_allclose_231344, *[x_231346, int_231347], **kwargs_231350)
        
        
        # Assigning a Call to a Name (line 402):
        
        # Assigning a Call to a Name (line 402):
        
        # Call to minimize(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'f' (line 402)
        f_231353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 402)
        list_231354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 402)
        # Adding element type (line 402)
        int_231355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 26), list_231354, int_231355)
        
        # Processing the call keyword arguments (line 402)
        str_231356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 40), 'str', 'slsqp')
        keyword_231357 = str_231356
        
        # Obtaining an instance of the builtin type 'list' (line 402)
        list_231358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 402)
        # Adding element type (line 402)
        
        # Obtaining an instance of the builtin type 'tuple' (line 402)
        tuple_231359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 402)
        # Adding element type (line 402)
        int_231360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 58), tuple_231359, int_231360)
        # Adding element type (line 402)
        # Getting the type of 'None' (line 402)
        None_231361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 61), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 58), tuple_231359, None_231361)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 56), list_231358, tuple_231359)
        
        keyword_231362 = list_231358
        kwargs_231363 = {'method': keyword_231357, 'bounds': keyword_231362}
        # Getting the type of 'minimize' (line 402)
        minimize_231352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 402)
        minimize_call_result_231364 = invoke(stypy.reporting.localization.Localization(__file__, 402, 14), minimize_231352, *[f_231353, list_231354], **kwargs_231363)
        
        # Assigning a type to the variable 'sol' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'sol', minimize_call_result_231364)
        
        # Call to assert_(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'sol' (line 403)
        sol_231366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 403)
        success_231367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), sol_231366, 'success')
        # Processing the call keyword arguments (line 403)
        kwargs_231368 = {}
        # Getting the type of 'assert_' (line 403)
        assert__231365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 403)
        assert__call_result_231369 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), assert__231365, *[success_231367], **kwargs_231368)
        
        
        # Call to assert_allclose(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'sol' (line 404)
        sol_231371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 404)
        x_231372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 24), sol_231371, 'x')
        int_231373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 31), 'int')
        # Processing the call keyword arguments (line 404)
        float_231374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 39), 'float')
        keyword_231375 = float_231374
        kwargs_231376 = {'atol': keyword_231375}
        # Getting the type of 'assert_allclose' (line 404)
        assert_allclose_231370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 404)
        assert_allclose_call_result_231377 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), assert_allclose_231370, *[x_231372, int_231373], **kwargs_231376)
        
        
        # Assigning a Call to a Name (line 406):
        
        # Assigning a Call to a Name (line 406):
        
        # Call to minimize(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'f' (line 406)
        f_231379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 406)
        list_231380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 406)
        # Adding element type (line 406)
        int_231381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 26), list_231380, int_231381)
        
        # Processing the call keyword arguments (line 406)
        str_231382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 40), 'str', 'slsqp')
        keyword_231383 = str_231382
        
        # Obtaining an instance of the builtin type 'list' (line 406)
        list_231384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 406)
        # Adding element type (line 406)
        
        # Obtaining an instance of the builtin type 'tuple' (line 406)
        tuple_231385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 406)
        # Adding element type (line 406)
        # Getting the type of 'None' (line 406)
        None_231386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 58), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 58), tuple_231385, None_231386)
        # Adding element type (line 406)
        int_231387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 58), tuple_231385, int_231387)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 56), list_231384, tuple_231385)
        
        keyword_231388 = list_231384
        kwargs_231389 = {'method': keyword_231383, 'bounds': keyword_231388}
        # Getting the type of 'minimize' (line 406)
        minimize_231378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 406)
        minimize_call_result_231390 = invoke(stypy.reporting.localization.Localization(__file__, 406, 14), minimize_231378, *[f_231379, list_231380], **kwargs_231389)
        
        # Assigning a type to the variable 'sol' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'sol', minimize_call_result_231390)
        
        # Call to assert_(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'sol' (line 407)
        sol_231392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 407)
        success_231393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), sol_231392, 'success')
        # Processing the call keyword arguments (line 407)
        kwargs_231394 = {}
        # Getting the type of 'assert_' (line 407)
        assert__231391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 407)
        assert__call_result_231395 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), assert__231391, *[success_231393], **kwargs_231394)
        
        
        # Call to assert_allclose(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'sol' (line 408)
        sol_231397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 408)
        x_231398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 24), sol_231397, 'x')
        int_231399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 31), 'int')
        # Processing the call keyword arguments (line 408)
        float_231400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 39), 'float')
        keyword_231401 = float_231400
        kwargs_231402 = {'atol': keyword_231401}
        # Getting the type of 'assert_allclose' (line 408)
        assert_allclose_231396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 408)
        assert_allclose_call_result_231403 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), assert_allclose_231396, *[x_231398, int_231399], **kwargs_231402)
        
        
        # Assigning a Call to a Name (line 410):
        
        # Assigning a Call to a Name (line 410):
        
        # Call to minimize(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'f' (line 410)
        f_231405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 410)
        list_231406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 410)
        # Adding element type (line 410)
        int_231407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 26), list_231406, int_231407)
        
        # Processing the call keyword arguments (line 410)
        str_231408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 39), 'str', 'slsqp')
        keyword_231409 = str_231408
        
        # Obtaining an instance of the builtin type 'list' (line 410)
        list_231410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 410)
        # Adding element type (line 410)
        
        # Obtaining an instance of the builtin type 'tuple' (line 410)
        tuple_231411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 410)
        # Adding element type (line 410)
        int_231412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 57), tuple_231411, int_231412)
        # Adding element type (line 410)
        # Getting the type of 'None' (line 410)
        None_231413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 60), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 57), tuple_231411, None_231413)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 55), list_231410, tuple_231411)
        
        keyword_231414 = list_231410
        kwargs_231415 = {'method': keyword_231409, 'bounds': keyword_231414}
        # Getting the type of 'minimize' (line 410)
        minimize_231404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 410)
        minimize_call_result_231416 = invoke(stypy.reporting.localization.Localization(__file__, 410, 14), minimize_231404, *[f_231405, list_231406], **kwargs_231415)
        
        # Assigning a type to the variable 'sol' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'sol', minimize_call_result_231416)
        
        # Call to assert_(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'sol' (line 411)
        sol_231418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 411)
        success_231419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 16), sol_231418, 'success')
        # Processing the call keyword arguments (line 411)
        kwargs_231420 = {}
        # Getting the type of 'assert_' (line 411)
        assert__231417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 411)
        assert__call_result_231421 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), assert__231417, *[success_231419], **kwargs_231420)
        
        
        # Call to assert_allclose(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'sol' (line 412)
        sol_231423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 412)
        x_231424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 24), sol_231423, 'x')
        int_231425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 31), 'int')
        # Processing the call keyword arguments (line 412)
        float_231426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 39), 'float')
        keyword_231427 = float_231426
        kwargs_231428 = {'atol': keyword_231427}
        # Getting the type of 'assert_allclose' (line 412)
        assert_allclose_231422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 412)
        assert_allclose_call_result_231429 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), assert_allclose_231422, *[x_231424, int_231425], **kwargs_231428)
        
        
        # Assigning a Call to a Name (line 414):
        
        # Assigning a Call to a Name (line 414):
        
        # Call to minimize(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'f' (line 414)
        f_231431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 414)
        list_231432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 414)
        # Adding element type (line 414)
        float_231433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 26), list_231432, float_231433)
        
        # Processing the call keyword arguments (line 414)
        str_231434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 41), 'str', 'slsqp')
        keyword_231435 = str_231434
        
        # Obtaining an instance of the builtin type 'list' (line 414)
        list_231436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 414)
        # Adding element type (line 414)
        
        # Obtaining an instance of the builtin type 'tuple' (line 414)
        tuple_231437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 414)
        # Adding element type (line 414)
        int_231438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 59), tuple_231437, int_231438)
        # Adding element type (line 414)
        int_231439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 59), tuple_231437, int_231439)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 57), list_231436, tuple_231437)
        
        keyword_231440 = list_231436
        kwargs_231441 = {'method': keyword_231435, 'bounds': keyword_231440}
        # Getting the type of 'minimize' (line 414)
        minimize_231430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 414)
        minimize_call_result_231442 = invoke(stypy.reporting.localization.Localization(__file__, 414, 14), minimize_231430, *[f_231431, list_231432], **kwargs_231441)
        
        # Assigning a type to the variable 'sol' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'sol', minimize_call_result_231442)
        
        # Call to assert_(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'sol' (line 415)
        sol_231444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 415)
        success_231445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 16), sol_231444, 'success')
        # Processing the call keyword arguments (line 415)
        kwargs_231446 = {}
        # Getting the type of 'assert_' (line 415)
        assert__231443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 415)
        assert__call_result_231447 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), assert__231443, *[success_231445], **kwargs_231446)
        
        
        # Call to assert_allclose(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'sol' (line 416)
        sol_231449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 416)
        x_231450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 24), sol_231449, 'x')
        int_231451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 31), 'int')
        # Processing the call keyword arguments (line 416)
        float_231452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 39), 'float')
        keyword_231453 = float_231452
        kwargs_231454 = {'atol': keyword_231453}
        # Getting the type of 'assert_allclose' (line 416)
        assert_allclose_231448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 416)
        assert_allclose_call_result_231455 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), assert_allclose_231448, *[x_231450, int_231451], **kwargs_231454)
        
        
        # Assigning a Call to a Name (line 418):
        
        # Assigning a Call to a Name (line 418):
        
        # Call to minimize(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'f' (line 418)
        f_231457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 418)
        list_231458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 418)
        # Adding element type (line 418)
        int_231459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 26), list_231458, int_231459)
        
        # Processing the call keyword arguments (line 418)
        str_231460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 39), 'str', 'slsqp')
        keyword_231461 = str_231460
        
        # Obtaining an instance of the builtin type 'list' (line 418)
        list_231462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 418)
        # Adding element type (line 418)
        
        # Obtaining an instance of the builtin type 'tuple' (line 418)
        tuple_231463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 418)
        # Adding element type (line 418)
        int_231464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 57), tuple_231463, int_231464)
        # Adding element type (line 418)
        int_231465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 57), tuple_231463, int_231465)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 55), list_231462, tuple_231463)
        
        keyword_231466 = list_231462
        kwargs_231467 = {'method': keyword_231461, 'bounds': keyword_231466}
        # Getting the type of 'minimize' (line 418)
        minimize_231456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 418)
        minimize_call_result_231468 = invoke(stypy.reporting.localization.Localization(__file__, 418, 14), minimize_231456, *[f_231457, list_231458], **kwargs_231467)
        
        # Assigning a type to the variable 'sol' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'sol', minimize_call_result_231468)
        
        # Call to assert_(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'sol' (line 419)
        sol_231470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 419)
        success_231471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 16), sol_231470, 'success')
        # Processing the call keyword arguments (line 419)
        kwargs_231472 = {}
        # Getting the type of 'assert_' (line 419)
        assert__231469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 419)
        assert__call_result_231473 = invoke(stypy.reporting.localization.Localization(__file__, 419, 8), assert__231469, *[success_231471], **kwargs_231472)
        
        
        # Call to assert_allclose(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'sol' (line 420)
        sol_231475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 420)
        x_231476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 24), sol_231475, 'x')
        int_231477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 31), 'int')
        # Processing the call keyword arguments (line 420)
        float_231478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 39), 'float')
        keyword_231479 = float_231478
        kwargs_231480 = {'atol': keyword_231479}
        # Getting the type of 'assert_allclose' (line 420)
        assert_allclose_231474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 420)
        assert_allclose_call_result_231481 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), assert_allclose_231474, *[x_231476, int_231477], **kwargs_231480)
        
        
        # ################# End of 'test_bounds_clipping(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bounds_clipping' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_231482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231482)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bounds_clipping'
        return stypy_return_type_231482


    @norecursion
    def test_infeasible_initial(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_infeasible_initial'
        module_type_store = module_type_store.open_function_context('test_infeasible_initial', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_localization', localization)
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_function_name', 'TestSLSQP.test_infeasible_initial')
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_param_names_list', [])
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSLSQP.test_infeasible_initial.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.test_infeasible_initial', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_infeasible_initial', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_infeasible_initial(...)' code ##################


        @norecursion
        def f(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'f'
            module_type_store = module_type_store.open_function_context('f', 424, 8, False)
            
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

            
            # Assigning a Name to a Tuple (line 425):
            
            # Assigning a Subscript to a Name (line 425):
            
            # Obtaining the type of the subscript
            int_231483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 12), 'int')
            # Getting the type of 'x' (line 425)
            x_231484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 17), 'x')
            # Obtaining the member '__getitem__' of a type (line 425)
            getitem___231485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), x_231484, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 425)
            subscript_call_result_231486 = invoke(stypy.reporting.localization.Localization(__file__, 425, 12), getitem___231485, int_231483)
            
            # Assigning a type to the variable 'tuple_var_assignment_229621' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'tuple_var_assignment_229621', subscript_call_result_231486)
            
            # Assigning a Name to a Name (line 425):
            # Getting the type of 'tuple_var_assignment_229621' (line 425)
            tuple_var_assignment_229621_231487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'tuple_var_assignment_229621')
            # Assigning a type to the variable 'x' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'x', tuple_var_assignment_229621_231487)
            # Getting the type of 'x' (line 426)
            x_231488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 19), 'x')
            # Getting the type of 'x' (line 426)
            x_231489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 21), 'x')
            # Applying the binary operator '*' (line 426)
            result_mul_231490 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 19), '*', x_231488, x_231489)
            
            int_231491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 25), 'int')
            # Getting the type of 'x' (line 426)
            x_231492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 27), 'x')
            # Applying the binary operator '*' (line 426)
            result_mul_231493 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 25), '*', int_231491, x_231492)
            
            # Applying the binary operator '-' (line 426)
            result_sub_231494 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 19), '-', result_mul_231490, result_mul_231493)
            
            int_231495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 31), 'int')
            # Applying the binary operator '+' (line 426)
            result_add_231496 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 29), '+', result_sub_231494, int_231495)
            
            # Assigning a type to the variable 'stypy_return_type' (line 426)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'stypy_return_type', result_add_231496)
            
            # ################# End of 'f(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'f' in the type store
            # Getting the type of 'stypy_return_type' (line 424)
            stypy_return_type_231497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231497)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'f'
            return stypy_return_type_231497

        # Assigning a type to the variable 'f' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'f', f)
        
        # Assigning a List to a Name (line 428):
        
        # Assigning a List to a Name (line 428):
        
        # Obtaining an instance of the builtin type 'list' (line 428)
        list_231498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 428)
        # Adding element type (line 428)
        
        # Obtaining an instance of the builtin type 'dict' (line 428)
        dict_231499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 428)
        # Adding element type (key, value) (line 428)
        str_231500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 19), 'str', 'type')
        str_231501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 27), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 18), dict_231499, (str_231500, str_231501))
        # Adding element type (key, value) (line 428)
        str_231502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 35), 'str', 'fun')

        @norecursion
        def _stypy_temp_lambda_112(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_112'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_112', 428, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_112.stypy_localization = localization
            _stypy_temp_lambda_112.stypy_type_of_self = None
            _stypy_temp_lambda_112.stypy_type_store = module_type_store
            _stypy_temp_lambda_112.stypy_function_name = '_stypy_temp_lambda_112'
            _stypy_temp_lambda_112.stypy_param_names_list = ['x']
            _stypy_temp_lambda_112.stypy_varargs_param_name = None
            _stypy_temp_lambda_112.stypy_kwargs_param_name = None
            _stypy_temp_lambda_112.stypy_call_defaults = defaults
            _stypy_temp_lambda_112.stypy_call_varargs = varargs
            _stypy_temp_lambda_112.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_112', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_112', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_231503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 52), 'int')
            # Getting the type of 'x' (line 428)
            x_231504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 56), 'x')
            # Applying the binary operator '-' (line 428)
            result_sub_231505 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 52), '-', int_231503, x_231504)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 42), 'stypy_return_type', result_sub_231505)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_112' in the type store
            # Getting the type of 'stypy_return_type' (line 428)
            stypy_return_type_231506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231506)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_112'
            return stypy_return_type_231506

        # Assigning a type to the variable '_stypy_temp_lambda_112' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 42), '_stypy_temp_lambda_112', _stypy_temp_lambda_112)
        # Getting the type of '_stypy_temp_lambda_112' (line 428)
        _stypy_temp_lambda_112_231507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 42), '_stypy_temp_lambda_112')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 18), dict_231499, (str_231502, _stypy_temp_lambda_112_231507))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 17), list_231498, dict_231499)
        
        # Assigning a type to the variable 'cons_u' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'cons_u', list_231498)
        
        # Assigning a List to a Name (line 429):
        
        # Assigning a List to a Name (line 429):
        
        # Obtaining an instance of the builtin type 'list' (line 429)
        list_231508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 429)
        # Adding element type (line 429)
        
        # Obtaining an instance of the builtin type 'dict' (line 429)
        dict_231509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 429)
        # Adding element type (key, value) (line 429)
        str_231510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 19), 'str', 'type')
        str_231511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 27), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 18), dict_231509, (str_231510, str_231511))
        # Adding element type (key, value) (line 429)
        str_231512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 35), 'str', 'fun')

        @norecursion
        def _stypy_temp_lambda_113(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_113'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_113', 429, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_113.stypy_localization = localization
            _stypy_temp_lambda_113.stypy_type_of_self = None
            _stypy_temp_lambda_113.stypy_type_store = module_type_store
            _stypy_temp_lambda_113.stypy_function_name = '_stypy_temp_lambda_113'
            _stypy_temp_lambda_113.stypy_param_names_list = ['x']
            _stypy_temp_lambda_113.stypy_varargs_param_name = None
            _stypy_temp_lambda_113.stypy_kwargs_param_name = None
            _stypy_temp_lambda_113.stypy_call_defaults = defaults
            _stypy_temp_lambda_113.stypy_call_varargs = varargs
            _stypy_temp_lambda_113.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_113', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_113', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 429)
            x_231513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 52), 'x')
            int_231514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 56), 'int')
            # Applying the binary operator '-' (line 429)
            result_sub_231515 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 52), '-', x_231513, int_231514)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 429)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 42), 'stypy_return_type', result_sub_231515)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_113' in the type store
            # Getting the type of 'stypy_return_type' (line 429)
            stypy_return_type_231516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231516)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_113'
            return stypy_return_type_231516

        # Assigning a type to the variable '_stypy_temp_lambda_113' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 42), '_stypy_temp_lambda_113', _stypy_temp_lambda_113)
        # Getting the type of '_stypy_temp_lambda_113' (line 429)
        _stypy_temp_lambda_113_231517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 42), '_stypy_temp_lambda_113')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 18), dict_231509, (str_231512, _stypy_temp_lambda_113_231517))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), list_231508, dict_231509)
        
        # Assigning a type to the variable 'cons_l' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'cons_l', list_231508)
        
        # Assigning a List to a Name (line 430):
        
        # Assigning a List to a Name (line 430):
        
        # Obtaining an instance of the builtin type 'list' (line 430)
        list_231518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 430)
        # Adding element type (line 430)
        
        # Obtaining an instance of the builtin type 'dict' (line 430)
        dict_231519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 430)
        # Adding element type (key, value) (line 430)
        str_231520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 20), 'str', 'type')
        str_231521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 28), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 19), dict_231519, (str_231520, str_231521))
        # Adding element type (key, value) (line 430)
        str_231522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 36), 'str', 'fun')

        @norecursion
        def _stypy_temp_lambda_114(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_114'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_114', 430, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_114.stypy_localization = localization
            _stypy_temp_lambda_114.stypy_type_of_self = None
            _stypy_temp_lambda_114.stypy_type_store = module_type_store
            _stypy_temp_lambda_114.stypy_function_name = '_stypy_temp_lambda_114'
            _stypy_temp_lambda_114.stypy_param_names_list = ['x']
            _stypy_temp_lambda_114.stypy_varargs_param_name = None
            _stypy_temp_lambda_114.stypy_kwargs_param_name = None
            _stypy_temp_lambda_114.stypy_call_defaults = defaults
            _stypy_temp_lambda_114.stypy_call_varargs = varargs
            _stypy_temp_lambda_114.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_114', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_114', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_231523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 53), 'int')
            # Getting the type of 'x' (line 430)
            x_231524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 57), 'x')
            # Applying the binary operator '-' (line 430)
            result_sub_231525 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 53), '-', int_231523, x_231524)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 43), 'stypy_return_type', result_sub_231525)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_114' in the type store
            # Getting the type of 'stypy_return_type' (line 430)
            stypy_return_type_231526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231526)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_114'
            return stypy_return_type_231526

        # Assigning a type to the variable '_stypy_temp_lambda_114' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 43), '_stypy_temp_lambda_114', _stypy_temp_lambda_114)
        # Getting the type of '_stypy_temp_lambda_114' (line 430)
        _stypy_temp_lambda_114_231527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 43), '_stypy_temp_lambda_114')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 19), dict_231519, (str_231522, _stypy_temp_lambda_114_231527))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 18), list_231518, dict_231519)
        # Adding element type (line 430)
        
        # Obtaining an instance of the builtin type 'dict' (line 431)
        dict_231528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 431)
        # Adding element type (key, value) (line 431)
        str_231529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 20), 'str', 'type')
        str_231530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 28), 'str', 'ineq')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 19), dict_231528, (str_231529, str_231530))
        # Adding element type (key, value) (line 431)
        str_231531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 36), 'str', 'fun')

        @norecursion
        def _stypy_temp_lambda_115(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_115'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_115', 431, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_115.stypy_localization = localization
            _stypy_temp_lambda_115.stypy_type_of_self = None
            _stypy_temp_lambda_115.stypy_type_store = module_type_store
            _stypy_temp_lambda_115.stypy_function_name = '_stypy_temp_lambda_115'
            _stypy_temp_lambda_115.stypy_param_names_list = ['x']
            _stypy_temp_lambda_115.stypy_varargs_param_name = None
            _stypy_temp_lambda_115.stypy_kwargs_param_name = None
            _stypy_temp_lambda_115.stypy_call_defaults = defaults
            _stypy_temp_lambda_115.stypy_call_varargs = varargs
            _stypy_temp_lambda_115.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_115', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_115', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 431)
            x_231532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 53), 'x')
            int_231533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 57), 'int')
            # Applying the binary operator '+' (line 431)
            result_add_231534 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 53), '+', x_231532, int_231533)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 431)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 43), 'stypy_return_type', result_add_231534)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_115' in the type store
            # Getting the type of 'stypy_return_type' (line 431)
            stypy_return_type_231535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231535)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_115'
            return stypy_return_type_231535

        # Assigning a type to the variable '_stypy_temp_lambda_115' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 43), '_stypy_temp_lambda_115', _stypy_temp_lambda_115)
        # Getting the type of '_stypy_temp_lambda_115' (line 431)
        _stypy_temp_lambda_115_231536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 43), '_stypy_temp_lambda_115')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 19), dict_231528, (str_231531, _stypy_temp_lambda_115_231536))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 18), list_231518, dict_231528)
        
        # Assigning a type to the variable 'cons_ul' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'cons_ul', list_231518)
        
        # Assigning a Call to a Name (line 433):
        
        # Assigning a Call to a Name (line 433):
        
        # Call to minimize(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'f' (line 433)
        f_231538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_231539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_231540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 26), list_231539, int_231540)
        
        # Processing the call keyword arguments (line 433)
        str_231541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 39), 'str', 'slsqp')
        keyword_231542 = str_231541
        # Getting the type of 'cons_u' (line 433)
        cons_u_231543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 60), 'cons_u', False)
        keyword_231544 = cons_u_231543
        kwargs_231545 = {'method': keyword_231542, 'constraints': keyword_231544}
        # Getting the type of 'minimize' (line 433)
        minimize_231537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 433)
        minimize_call_result_231546 = invoke(stypy.reporting.localization.Localization(__file__, 433, 14), minimize_231537, *[f_231538, list_231539], **kwargs_231545)
        
        # Assigning a type to the variable 'sol' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'sol', minimize_call_result_231546)
        
        # Call to assert_(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'sol' (line 434)
        sol_231548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 434)
        success_231549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 16), sol_231548, 'success')
        # Processing the call keyword arguments (line 434)
        kwargs_231550 = {}
        # Getting the type of 'assert_' (line 434)
        assert__231547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 434)
        assert__call_result_231551 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), assert__231547, *[success_231549], **kwargs_231550)
        
        
        # Call to assert_allclose(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'sol' (line 435)
        sol_231553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 435)
        x_231554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 24), sol_231553, 'x')
        int_231555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 31), 'int')
        # Processing the call keyword arguments (line 435)
        float_231556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 39), 'float')
        keyword_231557 = float_231556
        kwargs_231558 = {'atol': keyword_231557}
        # Getting the type of 'assert_allclose' (line 435)
        assert_allclose_231552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 435)
        assert_allclose_call_result_231559 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), assert_allclose_231552, *[x_231554, int_231555], **kwargs_231558)
        
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Call to minimize(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'f' (line 437)
        f_231561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 437)
        list_231562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 437)
        # Adding element type (line 437)
        int_231563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 26), list_231562, int_231563)
        
        # Processing the call keyword arguments (line 437)
        str_231564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 40), 'str', 'slsqp')
        keyword_231565 = str_231564
        # Getting the type of 'cons_l' (line 437)
        cons_l_231566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 61), 'cons_l', False)
        keyword_231567 = cons_l_231566
        kwargs_231568 = {'method': keyword_231565, 'constraints': keyword_231567}
        # Getting the type of 'minimize' (line 437)
        minimize_231560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 437)
        minimize_call_result_231569 = invoke(stypy.reporting.localization.Localization(__file__, 437, 14), minimize_231560, *[f_231561, list_231562], **kwargs_231568)
        
        # Assigning a type to the variable 'sol' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'sol', minimize_call_result_231569)
        
        # Call to assert_(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'sol' (line 438)
        sol_231571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 438)
        success_231572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 16), sol_231571, 'success')
        # Processing the call keyword arguments (line 438)
        kwargs_231573 = {}
        # Getting the type of 'assert_' (line 438)
        assert__231570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 438)
        assert__call_result_231574 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), assert__231570, *[success_231572], **kwargs_231573)
        
        
        # Call to assert_allclose(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'sol' (line 439)
        sol_231576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 439)
        x_231577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 24), sol_231576, 'x')
        int_231578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 31), 'int')
        # Processing the call keyword arguments (line 439)
        float_231579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 39), 'float')
        keyword_231580 = float_231579
        kwargs_231581 = {'atol': keyword_231580}
        # Getting the type of 'assert_allclose' (line 439)
        assert_allclose_231575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 439)
        assert_allclose_call_result_231582 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), assert_allclose_231575, *[x_231577, int_231578], **kwargs_231581)
        
        
        # Assigning a Call to a Name (line 441):
        
        # Assigning a Call to a Name (line 441):
        
        # Call to minimize(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'f' (line 441)
        f_231584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 441)
        list_231585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 441)
        # Adding element type (line 441)
        int_231586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 26), list_231585, int_231586)
        
        # Processing the call keyword arguments (line 441)
        str_231587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 40), 'str', 'slsqp')
        keyword_231588 = str_231587
        # Getting the type of 'cons_u' (line 441)
        cons_u_231589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 61), 'cons_u', False)
        keyword_231590 = cons_u_231589
        kwargs_231591 = {'method': keyword_231588, 'constraints': keyword_231590}
        # Getting the type of 'minimize' (line 441)
        minimize_231583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 441)
        minimize_call_result_231592 = invoke(stypy.reporting.localization.Localization(__file__, 441, 14), minimize_231583, *[f_231584, list_231585], **kwargs_231591)
        
        # Assigning a type to the variable 'sol' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'sol', minimize_call_result_231592)
        
        # Call to assert_(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'sol' (line 442)
        sol_231594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 442)
        success_231595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 16), sol_231594, 'success')
        # Processing the call keyword arguments (line 442)
        kwargs_231596 = {}
        # Getting the type of 'assert_' (line 442)
        assert__231593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 442)
        assert__call_result_231597 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), assert__231593, *[success_231595], **kwargs_231596)
        
        
        # Call to assert_allclose(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'sol' (line 443)
        sol_231599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 443)
        x_231600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 24), sol_231599, 'x')
        int_231601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 31), 'int')
        # Processing the call keyword arguments (line 443)
        float_231602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 39), 'float')
        keyword_231603 = float_231602
        kwargs_231604 = {'atol': keyword_231603}
        # Getting the type of 'assert_allclose' (line 443)
        assert_allclose_231598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 443)
        assert_allclose_call_result_231605 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), assert_allclose_231598, *[x_231600, int_231601], **kwargs_231604)
        
        
        # Assigning a Call to a Name (line 445):
        
        # Assigning a Call to a Name (line 445):
        
        # Call to minimize(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'f' (line 445)
        f_231607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 445)
        list_231608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 445)
        # Adding element type (line 445)
        int_231609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 26), list_231608, int_231609)
        
        # Processing the call keyword arguments (line 445)
        str_231610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 39), 'str', 'slsqp')
        keyword_231611 = str_231610
        # Getting the type of 'cons_l' (line 445)
        cons_l_231612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 60), 'cons_l', False)
        keyword_231613 = cons_l_231612
        kwargs_231614 = {'method': keyword_231611, 'constraints': keyword_231613}
        # Getting the type of 'minimize' (line 445)
        minimize_231606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 445)
        minimize_call_result_231615 = invoke(stypy.reporting.localization.Localization(__file__, 445, 14), minimize_231606, *[f_231607, list_231608], **kwargs_231614)
        
        # Assigning a type to the variable 'sol' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'sol', minimize_call_result_231615)
        
        # Call to assert_(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'sol' (line 446)
        sol_231617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 446)
        success_231618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 16), sol_231617, 'success')
        # Processing the call keyword arguments (line 446)
        kwargs_231619 = {}
        # Getting the type of 'assert_' (line 446)
        assert__231616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 446)
        assert__call_result_231620 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), assert__231616, *[success_231618], **kwargs_231619)
        
        
        # Call to assert_allclose(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'sol' (line 447)
        sol_231622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 447)
        x_231623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), sol_231622, 'x')
        int_231624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 31), 'int')
        # Processing the call keyword arguments (line 447)
        float_231625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 39), 'float')
        keyword_231626 = float_231625
        kwargs_231627 = {'atol': keyword_231626}
        # Getting the type of 'assert_allclose' (line 447)
        assert_allclose_231621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 447)
        assert_allclose_call_result_231628 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), assert_allclose_231621, *[x_231623, int_231624], **kwargs_231627)
        
        
        # Assigning a Call to a Name (line 449):
        
        # Assigning a Call to a Name (line 449):
        
        # Call to minimize(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'f' (line 449)
        f_231630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 449)
        list_231631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 449)
        # Adding element type (line 449)
        float_231632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 26), list_231631, float_231632)
        
        # Processing the call keyword arguments (line 449)
        str_231633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 41), 'str', 'slsqp')
        keyword_231634 = str_231633
        # Getting the type of 'cons_ul' (line 449)
        cons_ul_231635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 62), 'cons_ul', False)
        keyword_231636 = cons_ul_231635
        kwargs_231637 = {'method': keyword_231634, 'constraints': keyword_231636}
        # Getting the type of 'minimize' (line 449)
        minimize_231629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 449)
        minimize_call_result_231638 = invoke(stypy.reporting.localization.Localization(__file__, 449, 14), minimize_231629, *[f_231630, list_231631], **kwargs_231637)
        
        # Assigning a type to the variable 'sol' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'sol', minimize_call_result_231638)
        
        # Call to assert_(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'sol' (line 450)
        sol_231640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 450)
        success_231641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 16), sol_231640, 'success')
        # Processing the call keyword arguments (line 450)
        kwargs_231642 = {}
        # Getting the type of 'assert_' (line 450)
        assert__231639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 450)
        assert__call_result_231643 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), assert__231639, *[success_231641], **kwargs_231642)
        
        
        # Call to assert_allclose(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'sol' (line 451)
        sol_231645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 451)
        x_231646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), sol_231645, 'x')
        int_231647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 31), 'int')
        # Processing the call keyword arguments (line 451)
        float_231648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 39), 'float')
        keyword_231649 = float_231648
        kwargs_231650 = {'atol': keyword_231649}
        # Getting the type of 'assert_allclose' (line 451)
        assert_allclose_231644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 451)
        assert_allclose_call_result_231651 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), assert_allclose_231644, *[x_231646, int_231647], **kwargs_231650)
        
        
        # Assigning a Call to a Name (line 453):
        
        # Assigning a Call to a Name (line 453):
        
        # Call to minimize(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'f' (line 453)
        f_231653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 23), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 453)
        list_231654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 453)
        # Adding element type (line 453)
        int_231655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 26), list_231654, int_231655)
        
        # Processing the call keyword arguments (line 453)
        str_231656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 39), 'str', 'slsqp')
        keyword_231657 = str_231656
        # Getting the type of 'cons_ul' (line 453)
        cons_ul_231658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 60), 'cons_ul', False)
        keyword_231659 = cons_ul_231658
        kwargs_231660 = {'method': keyword_231657, 'constraints': keyword_231659}
        # Getting the type of 'minimize' (line 453)
        minimize_231652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 14), 'minimize', False)
        # Calling minimize(args, kwargs) (line 453)
        minimize_call_result_231661 = invoke(stypy.reporting.localization.Localization(__file__, 453, 14), minimize_231652, *[f_231653, list_231654], **kwargs_231660)
        
        # Assigning a type to the variable 'sol' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'sol', minimize_call_result_231661)
        
        # Call to assert_(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'sol' (line 454)
        sol_231663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 454)
        success_231664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 16), sol_231663, 'success')
        # Processing the call keyword arguments (line 454)
        kwargs_231665 = {}
        # Getting the type of 'assert_' (line 454)
        assert__231662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 454)
        assert__call_result_231666 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), assert__231662, *[success_231664], **kwargs_231665)
        
        
        # Call to assert_allclose(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 'sol' (line 455)
        sol_231668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 24), 'sol', False)
        # Obtaining the member 'x' of a type (line 455)
        x_231669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 24), sol_231668, 'x')
        int_231670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 31), 'int')
        # Processing the call keyword arguments (line 455)
        float_231671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 39), 'float')
        keyword_231672 = float_231671
        kwargs_231673 = {'atol': keyword_231672}
        # Getting the type of 'assert_allclose' (line 455)
        assert_allclose_231667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 455)
        assert_allclose_call_result_231674 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), assert_allclose_231667, *[x_231669, int_231670], **kwargs_231673)
        
        
        # ################# End of 'test_infeasible_initial(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_infeasible_initial' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_231675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231675)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_infeasible_initial'
        return stypy_return_type_231675


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSLSQP.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSLSQP' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'TestSLSQP', TestSLSQP)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
