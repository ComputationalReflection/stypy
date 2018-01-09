
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Unit tests for nonlinear solvers
2: Author: Ondrej Certik
3: May 2007
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: from numpy.testing import assert_
8: import pytest
9: 
10: from scipy._lib.six import xrange
11: from scipy.optimize import nonlin, root
12: from numpy import matrix, diag, dot
13: from numpy.linalg import inv
14: import numpy as np
15: 
16: from .test_minpack import pressure_network
17: 
18: SOLVERS = {'anderson': nonlin.anderson, 'diagbroyden': nonlin.diagbroyden,
19:            'linearmixing': nonlin.linearmixing, 'excitingmixing': nonlin.excitingmixing,
20:            'broyden1': nonlin.broyden1, 'broyden2': nonlin.broyden2,
21:            'krylov': nonlin.newton_krylov}
22: MUST_WORK = {'anderson': nonlin.anderson, 'broyden1': nonlin.broyden1,
23:              'broyden2': nonlin.broyden2, 'krylov': nonlin.newton_krylov}
24: 
25: #-------------------------------------------------------------------------------
26: # Test problems
27: #-------------------------------------------------------------------------------
28: 
29: 
30: def F(x):
31:     x = np.asmatrix(x).T
32:     d = matrix(diag([3,2,1.5,1,0.5]))
33:     c = 0.01
34:     f = -d*x - c*float(x.T*x)*x
35:     return f
36: F.xin = [1,1,1,1,1]
37: F.KNOWN_BAD = {}
38: 
39: 
40: def F2(x):
41:     return x
42: F2.xin = [1,2,3,4,5,6]
43: F2.KNOWN_BAD = {'linearmixing': nonlin.linearmixing,
44:                 'excitingmixing': nonlin.excitingmixing}
45: 
46: 
47: def F2_lucky(x):
48:     return x
49: F2_lucky.xin = [0,0,0,0,0,0]
50: F2_lucky.KNOWN_BAD = {}
51: 
52: 
53: def F3(x):
54:     A = np.mat('-2 1 0; 1 -2 1; 0 1 -2')
55:     b = np.mat('1 2 3')
56:     return np.dot(A, x) - b
57: F3.xin = [1,2,3]
58: F3.KNOWN_BAD = {}
59: 
60: 
61: def F4_powell(x):
62:     A = 1e4
63:     return [A*x[0]*x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - (1 + 1/A)]
64: F4_powell.xin = [-1, -2]
65: F4_powell.KNOWN_BAD = {'linearmixing': nonlin.linearmixing,
66:                        'excitingmixing': nonlin.excitingmixing,
67:                        'diagbroyden': nonlin.diagbroyden}
68: 
69: 
70: def F5(x):
71:     return pressure_network(x, 4, np.array([.5, .5, .5, .5]))
72: F5.xin = [2., 0, 2, 0]
73: F5.KNOWN_BAD = {'excitingmixing': nonlin.excitingmixing,
74:                 'linearmixing': nonlin.linearmixing,
75:                 'diagbroyden': nonlin.diagbroyden}
76: 
77: 
78: def F6(x):
79:     x1, x2 = x
80:     J0 = np.array([[-4.256, 14.7],
81:                 [0.8394989, 0.59964207]])
82:     v = np.array([(x1 + 3) * (x2**5 - 7) + 3*6,
83:                   np.sin(x2 * np.exp(x1) - 1)])
84:     return -np.linalg.solve(J0, v)
85: F6.xin = [-0.5, 1.4]
86: F6.KNOWN_BAD = {'excitingmixing': nonlin.excitingmixing,
87:                 'linearmixing': nonlin.linearmixing,
88:                 'diagbroyden': nonlin.diagbroyden}
89: 
90: 
91: #-------------------------------------------------------------------------------
92: # Tests
93: #-------------------------------------------------------------------------------
94: 
95: 
96: class TestNonlin(object):
97:     '''
98:     Check the Broyden methods for a few test problems.
99: 
100:     broyden1, broyden2, and newton_krylov must succeed for
101:     all functions. Some of the others don't -- tests in KNOWN_BAD are skipped.
102: 
103:     '''
104: 
105:     def _check_nonlin_func(self, f, func, f_tol=1e-2):
106:         x = func(f, f.xin, f_tol=f_tol, maxiter=200, verbose=0)
107:         assert_(np.absolute(f(x)).max() < f_tol)
108: 
109:     def _check_root(self, f, method, f_tol=1e-2):
110:         res = root(f, f.xin, method=method,
111:                    options={'ftol': f_tol, 'maxiter': 200, 'disp': 0})
112:         assert_(np.absolute(res.fun).max() < f_tol)
113: 
114:     @pytest.mark.xfail
115:     def _check_func_fail(self, *a, **kw):
116:         pass
117: 
118:     def test_problem_nonlin(self):
119:         for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
120:             for func in SOLVERS.values():
121:                 if func in f.KNOWN_BAD.values():
122:                     if func in MUST_WORK.values():
123:                         self._check_func_fail(f, func)
124:                     continue
125:                 self._check_nonlin_func(f, func)
126: 
127:     def test_tol_norm_called(self):
128:         # Check that supplying tol_norm keyword to nonlin_solve works
129:         self._tol_norm_used = False
130: 
131:         def local_norm_func(x):
132:             self._tol_norm_used = True
133:             return np.absolute(x).max()
134: 
135:         nonlin.newton_krylov(F, F.xin, f_tol=1e-2, maxiter=200, verbose=0,
136:              tol_norm=local_norm_func)
137:         assert_(self._tol_norm_used)
138: 
139:     def test_problem_root(self):
140:         for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
141:             for meth in SOLVERS:
142:                 if meth in f.KNOWN_BAD:
143:                     if meth in MUST_WORK:
144:                         self._check_func_fail(f, meth)
145:                     continue
146:                 self._check_root(f, meth)
147: 
148: 
149: class TestSecant(object):
150:     '''Check that some Jacobian approximations satisfy the secant condition'''
151: 
152:     xs = [np.array([1,2,3,4,5], float),
153:           np.array([2,3,4,5,1], float),
154:           np.array([3,4,5,1,2], float),
155:           np.array([4,5,1,2,3], float),
156:           np.array([9,1,9,1,3], float),
157:           np.array([0,1,9,1,3], float),
158:           np.array([5,5,7,1,1], float),
159:           np.array([1,2,7,5,1], float),]
160:     fs = [x**2 - 1 for x in xs]
161: 
162:     def _check_secant(self, jac_cls, npoints=1, **kw):
163:         '''
164:         Check that the given Jacobian approximation satisfies secant
165:         conditions for last `npoints` points.
166:         '''
167:         jac = jac_cls(**kw)
168:         jac.setup(self.xs[0], self.fs[0], None)
169:         for j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
170:             jac.update(x, f)
171: 
172:             for k in xrange(min(npoints, j+1)):
173:                 dx = self.xs[j-k+1] - self.xs[j-k]
174:                 df = self.fs[j-k+1] - self.fs[j-k]
175:                 assert_(np.allclose(dx, jac.solve(df)))
176: 
177:             # Check that the `npoints` secant bound is strict
178:             if j >= npoints:
179:                 dx = self.xs[j-npoints+1] - self.xs[j-npoints]
180:                 df = self.fs[j-npoints+1] - self.fs[j-npoints]
181:                 assert_(not np.allclose(dx, jac.solve(df)))
182: 
183:     def test_broyden1(self):
184:         self._check_secant(nonlin.BroydenFirst)
185: 
186:     def test_broyden2(self):
187:         self._check_secant(nonlin.BroydenSecond)
188: 
189:     def test_broyden1_update(self):
190:         # Check that BroydenFirst update works as for a dense matrix
191:         jac = nonlin.BroydenFirst(alpha=0.1)
192:         jac.setup(self.xs[0], self.fs[0], None)
193: 
194:         B = np.identity(5) * (-1/0.1)
195: 
196:         for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
197:             df = f - self.fs[last_j]
198:             dx = x - self.xs[last_j]
199:             B += (df - dot(B, dx))[:,None] * dx[None,:] / dot(dx, dx)
200:             jac.update(x, f)
201:             assert_(np.allclose(jac.todense(), B, rtol=1e-10, atol=1e-13))
202: 
203:     def test_broyden2_update(self):
204:         # Check that BroydenSecond update works as for a dense matrix
205:         jac = nonlin.BroydenSecond(alpha=0.1)
206:         jac.setup(self.xs[0], self.fs[0], None)
207: 
208:         H = np.identity(5) * (-0.1)
209: 
210:         for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
211:             df = f - self.fs[last_j]
212:             dx = x - self.xs[last_j]
213:             H += (dx - dot(H, df))[:,None] * df[None,:] / dot(df, df)
214:             jac.update(x, f)
215:             assert_(np.allclose(jac.todense(), inv(H), rtol=1e-10, atol=1e-13))
216: 
217:     def test_anderson(self):
218:         # Anderson mixing (with w0=0) satisfies secant conditions
219:         # for the last M iterates, see [Ey]_
220:         #
221:         # .. [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).
222:         self._check_secant(nonlin.Anderson, M=3, w0=0, npoints=3)
223: 
224: 
225: class TestLinear(object):
226:     '''Solve a linear equation;
227:     some methods find the exact solution in a finite number of steps'''
228: 
229:     def _check(self, jac, N, maxiter, complex=False, **kw):
230:         np.random.seed(123)
231: 
232:         A = np.random.randn(N, N)
233:         if complex:
234:             A = A + 1j*np.random.randn(N, N)
235:         b = np.random.randn(N)
236:         if complex:
237:             b = b + 1j*np.random.randn(N)
238: 
239:         def func(x):
240:             return dot(A, x) - b
241: 
242:         sol = nonlin.nonlin_solve(func, np.zeros(N), jac, maxiter=maxiter,
243:                                   f_tol=1e-6, line_search=None, verbose=0)
244:         assert_(np.allclose(dot(A, sol), b, atol=1e-6))
245: 
246:     def test_broyden1(self):
247:         # Broyden methods solve linear systems exactly in 2*N steps
248:         self._check(nonlin.BroydenFirst(alpha=1.0), 20, 41, False)
249:         self._check(nonlin.BroydenFirst(alpha=1.0), 20, 41, True)
250: 
251:     def test_broyden2(self):
252:         # Broyden methods solve linear systems exactly in 2*N steps
253:         self._check(nonlin.BroydenSecond(alpha=1.0), 20, 41, False)
254:         self._check(nonlin.BroydenSecond(alpha=1.0), 20, 41, True)
255: 
256:     def test_anderson(self):
257:         # Anderson is rather similar to Broyden, if given enough storage space
258:         self._check(nonlin.Anderson(M=50, alpha=1.0), 20, 29, False)
259:         self._check(nonlin.Anderson(M=50, alpha=1.0), 20, 29, True)
260: 
261:     def test_krylov(self):
262:         # Krylov methods solve linear systems exactly in N inner steps
263:         self._check(nonlin.KrylovJacobian, 20, 2, False, inner_m=10)
264:         self._check(nonlin.KrylovJacobian, 20, 2, True, inner_m=10)
265: 
266: 
267: class TestJacobianDotSolve(object):
268:     '''Check that solve/dot methods in Jacobian approximations are consistent'''
269: 
270:     def _func(self, x):
271:         return x**2 - 1 + np.dot(self.A, x)
272: 
273:     def _check_dot(self, jac_cls, complex=False, tol=1e-6, **kw):
274:         np.random.seed(123)
275: 
276:         N = 7
277: 
278:         def rand(*a):
279:             q = np.random.rand(*a)
280:             if complex:
281:                 q = q + 1j*np.random.rand(*a)
282:             return q
283: 
284:         def assert_close(a, b, msg):
285:             d = abs(a - b).max()
286:             f = tol + abs(b).max()*tol
287:             if d > f:
288:                 raise AssertionError('%s: err %g' % (msg, d))
289: 
290:         self.A = rand(N, N)
291: 
292:         # initialize
293:         x0 = np.random.rand(N)
294:         jac = jac_cls(**kw)
295:         jac.setup(x0, self._func(x0), self._func)
296: 
297:         # check consistency
298:         for k in xrange(2*N):
299:             v = rand(N)
300: 
301:             if hasattr(jac, '__array__'):
302:                 Jd = np.array(jac)
303:                 if hasattr(jac, 'solve'):
304:                     Gv = jac.solve(v)
305:                     Gv2 = np.linalg.solve(Jd, v)
306:                     assert_close(Gv, Gv2, 'solve vs array')
307:                 if hasattr(jac, 'rsolve'):
308:                     Gv = jac.rsolve(v)
309:                     Gv2 = np.linalg.solve(Jd.T.conj(), v)
310:                     assert_close(Gv, Gv2, 'rsolve vs array')
311:                 if hasattr(jac, 'matvec'):
312:                     Jv = jac.matvec(v)
313:                     Jv2 = np.dot(Jd, v)
314:                     assert_close(Jv, Jv2, 'dot vs array')
315:                 if hasattr(jac, 'rmatvec'):
316:                     Jv = jac.rmatvec(v)
317:                     Jv2 = np.dot(Jd.T.conj(), v)
318:                     assert_close(Jv, Jv2, 'rmatvec vs array')
319: 
320:             if hasattr(jac, 'matvec') and hasattr(jac, 'solve'):
321:                 Jv = jac.matvec(v)
322:                 Jv2 = jac.solve(jac.matvec(Jv))
323:                 assert_close(Jv, Jv2, 'dot vs solve')
324: 
325:             if hasattr(jac, 'rmatvec') and hasattr(jac, 'rsolve'):
326:                 Jv = jac.rmatvec(v)
327:                 Jv2 = jac.rmatvec(jac.rsolve(Jv))
328:                 assert_close(Jv, Jv2, 'rmatvec vs rsolve')
329: 
330:             x = rand(N)
331:             jac.update(x, self._func(x))
332: 
333:     def test_broyden1(self):
334:         self._check_dot(nonlin.BroydenFirst, complex=False)
335:         self._check_dot(nonlin.BroydenFirst, complex=True)
336: 
337:     def test_broyden2(self):
338:         self._check_dot(nonlin.BroydenSecond, complex=False)
339:         self._check_dot(nonlin.BroydenSecond, complex=True)
340: 
341:     def test_anderson(self):
342:         self._check_dot(nonlin.Anderson, complex=False)
343:         self._check_dot(nonlin.Anderson, complex=True)
344: 
345:     def test_diagbroyden(self):
346:         self._check_dot(nonlin.DiagBroyden, complex=False)
347:         self._check_dot(nonlin.DiagBroyden, complex=True)
348: 
349:     def test_linearmixing(self):
350:         self._check_dot(nonlin.LinearMixing, complex=False)
351:         self._check_dot(nonlin.LinearMixing, complex=True)
352: 
353:     def test_excitingmixing(self):
354:         self._check_dot(nonlin.ExcitingMixing, complex=False)
355:         self._check_dot(nonlin.ExcitingMixing, complex=True)
356: 
357:     def test_krylov(self):
358:         self._check_dot(nonlin.KrylovJacobian, complex=False, tol=1e-3)
359:         self._check_dot(nonlin.KrylovJacobian, complex=True, tol=1e-3)
360: 
361: 
362: class TestNonlinOldTests(object):
363:     ''' Test case for a simple constrained entropy maximization problem
364:     (the machine translation example of Berger et al in
365:     Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
366:     '''
367: 
368:     def test_broyden1(self):
369:         x = nonlin.broyden1(F,F.xin,iter=12,alpha=1)
370:         assert_(nonlin.norm(x) < 1e-9)
371:         assert_(nonlin.norm(F(x)) < 1e-9)
372: 
373:     def test_broyden2(self):
374:         x = nonlin.broyden2(F,F.xin,iter=12,alpha=1)
375:         assert_(nonlin.norm(x) < 1e-9)
376:         assert_(nonlin.norm(F(x)) < 1e-9)
377: 
378:     def test_anderson(self):
379:         x = nonlin.anderson(F,F.xin,iter=12,alpha=0.03,M=5)
380:         assert_(nonlin.norm(x) < 0.33)
381: 
382:     def test_linearmixing(self):
383:         x = nonlin.linearmixing(F,F.xin,iter=60,alpha=0.5)
384:         assert_(nonlin.norm(x) < 1e-7)
385:         assert_(nonlin.norm(F(x)) < 1e-7)
386: 
387:     def test_exciting(self):
388:         x = nonlin.excitingmixing(F,F.xin,iter=20,alpha=0.5)
389:         assert_(nonlin.norm(x) < 1e-5)
390:         assert_(nonlin.norm(F(x)) < 1e-5)
391: 
392:     def test_diagbroyden(self):
393:         x = nonlin.diagbroyden(F,F.xin,iter=11,alpha=1)
394:         assert_(nonlin.norm(x) < 1e-8)
395:         assert_(nonlin.norm(F(x)) < 1e-8)
396: 
397:     def test_root_broyden1(self):
398:         res = root(F, F.xin, method='broyden1',
399:                    options={'nit': 12, 'jac_options': {'alpha': 1}})
400:         assert_(nonlin.norm(res.x) < 1e-9)
401:         assert_(nonlin.norm(res.fun) < 1e-9)
402: 
403:     def test_root_broyden2(self):
404:         res = root(F, F.xin, method='broyden2',
405:                    options={'nit': 12, 'jac_options': {'alpha': 1}})
406:         assert_(nonlin.norm(res.x) < 1e-9)
407:         assert_(nonlin.norm(res.fun) < 1e-9)
408: 
409:     def test_root_anderson(self):
410:         res = root(F, F.xin, method='anderson',
411:                    options={'nit': 12,
412:                             'jac_options': {'alpha': 0.03, 'M': 5}})
413:         assert_(nonlin.norm(res.x) < 0.33)
414: 
415:     def test_root_linearmixing(self):
416:         res = root(F, F.xin, method='linearmixing',
417:                    options={'nit': 60,
418:                             'jac_options': {'alpha': 0.5}})
419:         assert_(nonlin.norm(res.x) < 1e-7)
420:         assert_(nonlin.norm(res.fun) < 1e-7)
421: 
422:     def test_root_excitingmixing(self):
423:         res = root(F, F.xin, method='excitingmixing',
424:                    options={'nit': 20,
425:                             'jac_options': {'alpha': 0.5}})
426:         assert_(nonlin.norm(res.x) < 1e-5)
427:         assert_(nonlin.norm(res.fun) < 1e-5)
428: 
429:     def test_root_diagbroyden(self):
430:         res = root(F, F.xin, method='diagbroyden',
431:                    options={'nit': 11,
432:                             'jac_options': {'alpha': 1}})
433:         assert_(nonlin.norm(res.x) < 1e-8)
434:         assert_(nonlin.norm(res.fun) < 1e-8)
435: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_221555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', ' Unit tests for nonlinear solvers\nAuthor: Ondrej Certik\nMay 2007\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221556 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_221556) is not StypyTypeError):

    if (import_221556 != 'pyd_module'):
        __import__(import_221556)
        sys_modules_221557 = sys.modules[import_221556]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_221557.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_221557, sys_modules_221557.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_221556)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import pytest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221558 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_221558) is not StypyTypeError):

    if (import_221558 != 'pyd_module'):
        __import__(import_221558)
        sys_modules_221559 = sys.modules[import_221558]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_221559.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_221558)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib.six import xrange' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six')

if (type(import_221560) is not StypyTypeError):

    if (import_221560 != 'pyd_module'):
        __import__(import_221560)
        sys_modules_221561 = sys.modules[import_221560]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', sys_modules_221561.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_221561, sys_modules_221561.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', import_221560)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.optimize import nonlin, root' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221562 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize')

if (type(import_221562) is not StypyTypeError):

    if (import_221562 != 'pyd_module'):
        __import__(import_221562)
        sys_modules_221563 = sys.modules[import_221562]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', sys_modules_221563.module_type_store, module_type_store, ['nonlin', 'root'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_221563, sys_modules_221563.module_type_store, module_type_store)
    else:
        from scipy.optimize import nonlin, root

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', None, module_type_store, ['nonlin', 'root'], [nonlin, root])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.optimize', import_221562)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy import matrix, diag, dot' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy')

if (type(import_221564) is not StypyTypeError):

    if (import_221564 != 'pyd_module'):
        __import__(import_221564)
        sys_modules_221565 = sys.modules[import_221564]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', sys_modules_221565.module_type_store, module_type_store, ['matrix', 'diag', 'dot'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_221565, sys_modules_221565.module_type_store, module_type_store)
    else:
        from numpy import matrix, diag, dot

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', None, module_type_store, ['matrix', 'diag', 'dot'], [matrix, diag, dot])

else:
    # Assigning a type to the variable 'numpy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', import_221564)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.linalg import inv' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221566 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.linalg')

if (type(import_221566) is not StypyTypeError):

    if (import_221566 != 'pyd_module'):
        __import__(import_221566)
        sys_modules_221567 = sys.modules[import_221566]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.linalg', sys_modules_221567.module_type_store, module_type_store, ['inv'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_221567, sys_modules_221567.module_type_store, module_type_store)
    else:
        from numpy.linalg import inv

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.linalg', None, module_type_store, ['inv'], [inv])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.linalg', import_221566)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import numpy' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221568 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_221568) is not StypyTypeError):

    if (import_221568 != 'pyd_module'):
        __import__(import_221568)
        sys_modules_221569 = sys.modules[import_221568]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', sys_modules_221569.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_221568)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.optimize.tests.test_minpack import pressure_network' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_221570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize.tests.test_minpack')

if (type(import_221570) is not StypyTypeError):

    if (import_221570 != 'pyd_module'):
        __import__(import_221570)
        sys_modules_221571 = sys.modules[import_221570]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize.tests.test_minpack', sys_modules_221571.module_type_store, module_type_store, ['pressure_network'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_221571, sys_modules_221571.module_type_store, module_type_store)
    else:
        from scipy.optimize.tests.test_minpack import pressure_network

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize.tests.test_minpack', None, module_type_store, ['pressure_network'], [pressure_network])

else:
    # Assigning a type to the variable 'scipy.optimize.tests.test_minpack' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.optimize.tests.test_minpack', import_221570)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


# Assigning a Dict to a Name (line 18):

# Assigning a Dict to a Name (line 18):

# Obtaining an instance of the builtin type 'dict' (line 18)
dict_221572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 18)
# Adding element type (key, value) (line 18)
str_221573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'str', 'anderson')
# Getting the type of 'nonlin' (line 18)
nonlin_221574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'nonlin')
# Obtaining the member 'anderson' of a type (line 18)
anderson_221575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 23), nonlin_221574, 'anderson')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), dict_221572, (str_221573, anderson_221575))
# Adding element type (key, value) (line 18)
str_221576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 40), 'str', 'diagbroyden')
# Getting the type of 'nonlin' (line 18)
nonlin_221577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 55), 'nonlin')
# Obtaining the member 'diagbroyden' of a type (line 18)
diagbroyden_221578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 55), nonlin_221577, 'diagbroyden')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), dict_221572, (str_221576, diagbroyden_221578))
# Adding element type (key, value) (line 18)
str_221579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', 'linearmixing')
# Getting the type of 'nonlin' (line 19)
nonlin_221580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'nonlin')
# Obtaining the member 'linearmixing' of a type (line 19)
linearmixing_221581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 27), nonlin_221580, 'linearmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), dict_221572, (str_221579, linearmixing_221581))
# Adding element type (key, value) (line 18)
str_221582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 48), 'str', 'excitingmixing')
# Getting the type of 'nonlin' (line 19)
nonlin_221583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 66), 'nonlin')
# Obtaining the member 'excitingmixing' of a type (line 19)
excitingmixing_221584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 66), nonlin_221583, 'excitingmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), dict_221572, (str_221582, excitingmixing_221584))
# Adding element type (key, value) (line 18)
str_221585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'str', 'broyden1')
# Getting the type of 'nonlin' (line 20)
nonlin_221586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'nonlin')
# Obtaining the member 'broyden1' of a type (line 20)
broyden1_221587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 23), nonlin_221586, 'broyden1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), dict_221572, (str_221585, broyden1_221587))
# Adding element type (key, value) (line 18)
str_221588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 40), 'str', 'broyden2')
# Getting the type of 'nonlin' (line 20)
nonlin_221589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 52), 'nonlin')
# Obtaining the member 'broyden2' of a type (line 20)
broyden2_221590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 52), nonlin_221589, 'broyden2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), dict_221572, (str_221588, broyden2_221590))
# Adding element type (key, value) (line 18)
str_221591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'krylov')
# Getting the type of 'nonlin' (line 21)
nonlin_221592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'nonlin')
# Obtaining the member 'newton_krylov' of a type (line 21)
newton_krylov_221593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 21), nonlin_221592, 'newton_krylov')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), dict_221572, (str_221591, newton_krylov_221593))

# Assigning a type to the variable 'SOLVERS' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'SOLVERS', dict_221572)

# Assigning a Dict to a Name (line 22):

# Assigning a Dict to a Name (line 22):

# Obtaining an instance of the builtin type 'dict' (line 22)
dict_221594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 22)
# Adding element type (key, value) (line 22)
str_221595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'str', 'anderson')
# Getting the type of 'nonlin' (line 22)
nonlin_221596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'nonlin')
# Obtaining the member 'anderson' of a type (line 22)
anderson_221597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), nonlin_221596, 'anderson')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 12), dict_221594, (str_221595, anderson_221597))
# Adding element type (key, value) (line 22)
str_221598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 42), 'str', 'broyden1')
# Getting the type of 'nonlin' (line 22)
nonlin_221599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 54), 'nonlin')
# Obtaining the member 'broyden1' of a type (line 22)
broyden1_221600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 54), nonlin_221599, 'broyden1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 12), dict_221594, (str_221598, broyden1_221600))
# Adding element type (key, value) (line 22)
str_221601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'str', 'broyden2')
# Getting the type of 'nonlin' (line 23)
nonlin_221602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'nonlin')
# Obtaining the member 'broyden2' of a type (line 23)
broyden2_221603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 25), nonlin_221602, 'broyden2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 12), dict_221594, (str_221601, broyden2_221603))
# Adding element type (key, value) (line 22)
str_221604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 42), 'str', 'krylov')
# Getting the type of 'nonlin' (line 23)
nonlin_221605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 52), 'nonlin')
# Obtaining the member 'newton_krylov' of a type (line 23)
newton_krylov_221606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 52), nonlin_221605, 'newton_krylov')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 12), dict_221594, (str_221604, newton_krylov_221606))

# Assigning a type to the variable 'MUST_WORK' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'MUST_WORK', dict_221594)

@norecursion
def F(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F'
    module_type_store = module_type_store.open_function_context('F', 30, 0, False)
    
    # Passed parameters checking function
    F.stypy_localization = localization
    F.stypy_type_of_self = None
    F.stypy_type_store = module_type_store
    F.stypy_function_name = 'F'
    F.stypy_param_names_list = ['x']
    F.stypy_varargs_param_name = None
    F.stypy_kwargs_param_name = None
    F.stypy_call_defaults = defaults
    F.stypy_call_varargs = varargs
    F.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F(...)' code ##################

    
    # Assigning a Attribute to a Name (line 31):
    
    # Assigning a Attribute to a Name (line 31):
    
    # Call to asmatrix(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'x' (line 31)
    x_221609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'x', False)
    # Processing the call keyword arguments (line 31)
    kwargs_221610 = {}
    # Getting the type of 'np' (line 31)
    np_221607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'np', False)
    # Obtaining the member 'asmatrix' of a type (line 31)
    asmatrix_221608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), np_221607, 'asmatrix')
    # Calling asmatrix(args, kwargs) (line 31)
    asmatrix_call_result_221611 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), asmatrix_221608, *[x_221609], **kwargs_221610)
    
    # Obtaining the member 'T' of a type (line 31)
    T_221612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), asmatrix_call_result_221611, 'T')
    # Assigning a type to the variable 'x' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'x', T_221612)
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to matrix(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to diag(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_221615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    # Adding element type (line 32)
    int_221616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 20), list_221615, int_221616)
    # Adding element type (line 32)
    int_221617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 20), list_221615, int_221617)
    # Adding element type (line 32)
    float_221618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 20), list_221615, float_221618)
    # Adding element type (line 32)
    int_221619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 20), list_221615, int_221619)
    # Adding element type (line 32)
    float_221620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 20), list_221615, float_221620)
    
    # Processing the call keyword arguments (line 32)
    kwargs_221621 = {}
    # Getting the type of 'diag' (line 32)
    diag_221614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'diag', False)
    # Calling diag(args, kwargs) (line 32)
    diag_call_result_221622 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), diag_221614, *[list_221615], **kwargs_221621)
    
    # Processing the call keyword arguments (line 32)
    kwargs_221623 = {}
    # Getting the type of 'matrix' (line 32)
    matrix_221613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'matrix', False)
    # Calling matrix(args, kwargs) (line 32)
    matrix_call_result_221624 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), matrix_221613, *[diag_call_result_221622], **kwargs_221623)
    
    # Assigning a type to the variable 'd' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'd', matrix_call_result_221624)
    
    # Assigning a Num to a Name (line 33):
    
    # Assigning a Num to a Name (line 33):
    float_221625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'float')
    # Assigning a type to the variable 'c' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'c', float_221625)
    
    # Assigning a BinOp to a Name (line 34):
    
    # Assigning a BinOp to a Name (line 34):
    
    # Getting the type of 'd' (line 34)
    d_221626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 9), 'd')
    # Applying the 'usub' unary operator (line 34)
    result___neg___221627 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 8), 'usub', d_221626)
    
    # Getting the type of 'x' (line 34)
    x_221628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'x')
    # Applying the binary operator '*' (line 34)
    result_mul_221629 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 8), '*', result___neg___221627, x_221628)
    
    # Getting the type of 'c' (line 34)
    c_221630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'c')
    
    # Call to float(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'x' (line 34)
    x_221632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'x', False)
    # Obtaining the member 'T' of a type (line 34)
    T_221633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 23), x_221632, 'T')
    # Getting the type of 'x' (line 34)
    x_221634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'x', False)
    # Applying the binary operator '*' (line 34)
    result_mul_221635 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 23), '*', T_221633, x_221634)
    
    # Processing the call keyword arguments (line 34)
    kwargs_221636 = {}
    # Getting the type of 'float' (line 34)
    float_221631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'float', False)
    # Calling float(args, kwargs) (line 34)
    float_call_result_221637 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), float_221631, *[result_mul_221635], **kwargs_221636)
    
    # Applying the binary operator '*' (line 34)
    result_mul_221638 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 15), '*', c_221630, float_call_result_221637)
    
    # Getting the type of 'x' (line 34)
    x_221639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'x')
    # Applying the binary operator '*' (line 34)
    result_mul_221640 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 29), '*', result_mul_221638, x_221639)
    
    # Applying the binary operator '-' (line 34)
    result_sub_221641 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 8), '-', result_mul_221629, result_mul_221640)
    
    # Assigning a type to the variable 'f' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'f', result_sub_221641)
    # Getting the type of 'f' (line 35)
    f_221642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', f_221642)
    
    # ################# End of 'F(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_221643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_221643)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F'
    return stypy_return_type_221643

# Assigning a type to the variable 'F' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'F', F)

# Assigning a List to a Attribute (line 36):

# Assigning a List to a Attribute (line 36):

# Obtaining an instance of the builtin type 'list' (line 36)
list_221644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
int_221645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), list_221644, int_221645)
# Adding element type (line 36)
int_221646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), list_221644, int_221646)
# Adding element type (line 36)
int_221647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), list_221644, int_221647)
# Adding element type (line 36)
int_221648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), list_221644, int_221648)
# Adding element type (line 36)
int_221649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), list_221644, int_221649)

# Getting the type of 'F' (line 36)
F_221650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'F')
# Setting the type of the member 'xin' of a type (line 36)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 0), F_221650, 'xin', list_221644)

# Assigning a Dict to a Attribute (line 37):

# Assigning a Dict to a Attribute (line 37):

# Obtaining an instance of the builtin type 'dict' (line 37)
dict_221651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 37)

# Getting the type of 'F' (line 37)
F_221652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'F')
# Setting the type of the member 'KNOWN_BAD' of a type (line 37)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 0), F_221652, 'KNOWN_BAD', dict_221651)

@norecursion
def F2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F2'
    module_type_store = module_type_store.open_function_context('F2', 40, 0, False)
    
    # Passed parameters checking function
    F2.stypy_localization = localization
    F2.stypy_type_of_self = None
    F2.stypy_type_store = module_type_store
    F2.stypy_function_name = 'F2'
    F2.stypy_param_names_list = ['x']
    F2.stypy_varargs_param_name = None
    F2.stypy_kwargs_param_name = None
    F2.stypy_call_defaults = defaults
    F2.stypy_call_varargs = varargs
    F2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F2', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F2', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F2(...)' code ##################

    # Getting the type of 'x' (line 41)
    x_221653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', x_221653)
    
    # ################# End of 'F2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F2' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_221654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_221654)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F2'
    return stypy_return_type_221654

# Assigning a type to the variable 'F2' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'F2', F2)

# Assigning a List to a Attribute (line 42):

# Assigning a List to a Attribute (line 42):

# Obtaining an instance of the builtin type 'list' (line 42)
list_221655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)
int_221656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), list_221655, int_221656)
# Adding element type (line 42)
int_221657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), list_221655, int_221657)
# Adding element type (line 42)
int_221658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), list_221655, int_221658)
# Adding element type (line 42)
int_221659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), list_221655, int_221659)
# Adding element type (line 42)
int_221660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), list_221655, int_221660)
# Adding element type (line 42)
int_221661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), list_221655, int_221661)

# Getting the type of 'F2' (line 42)
F2_221662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'F2')
# Setting the type of the member 'xin' of a type (line 42)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 0), F2_221662, 'xin', list_221655)

# Assigning a Dict to a Attribute (line 43):

# Assigning a Dict to a Attribute (line 43):

# Obtaining an instance of the builtin type 'dict' (line 43)
dict_221663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 43)
# Adding element type (key, value) (line 43)
str_221664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'str', 'linearmixing')
# Getting the type of 'nonlin' (line 43)
nonlin_221665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'nonlin')
# Obtaining the member 'linearmixing' of a type (line 43)
linearmixing_221666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), nonlin_221665, 'linearmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 15), dict_221663, (str_221664, linearmixing_221666))
# Adding element type (key, value) (line 43)
str_221667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'str', 'excitingmixing')
# Getting the type of 'nonlin' (line 44)
nonlin_221668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'nonlin')
# Obtaining the member 'excitingmixing' of a type (line 44)
excitingmixing_221669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 34), nonlin_221668, 'excitingmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 15), dict_221663, (str_221667, excitingmixing_221669))

# Getting the type of 'F2' (line 43)
F2_221670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'F2')
# Setting the type of the member 'KNOWN_BAD' of a type (line 43)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 0), F2_221670, 'KNOWN_BAD', dict_221663)

@norecursion
def F2_lucky(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F2_lucky'
    module_type_store = module_type_store.open_function_context('F2_lucky', 47, 0, False)
    
    # Passed parameters checking function
    F2_lucky.stypy_localization = localization
    F2_lucky.stypy_type_of_self = None
    F2_lucky.stypy_type_store = module_type_store
    F2_lucky.stypy_function_name = 'F2_lucky'
    F2_lucky.stypy_param_names_list = ['x']
    F2_lucky.stypy_varargs_param_name = None
    F2_lucky.stypy_kwargs_param_name = None
    F2_lucky.stypy_call_defaults = defaults
    F2_lucky.stypy_call_varargs = varargs
    F2_lucky.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F2_lucky', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F2_lucky', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F2_lucky(...)' code ##################

    # Getting the type of 'x' (line 48)
    x_221671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', x_221671)
    
    # ################# End of 'F2_lucky(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F2_lucky' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_221672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_221672)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F2_lucky'
    return stypy_return_type_221672

# Assigning a type to the variable 'F2_lucky' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'F2_lucky', F2_lucky)

# Assigning a List to a Attribute (line 49):

# Assigning a List to a Attribute (line 49):

# Obtaining an instance of the builtin type 'list' (line 49)
list_221673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 49)
# Adding element type (line 49)
int_221674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 15), list_221673, int_221674)
# Adding element type (line 49)
int_221675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 15), list_221673, int_221675)
# Adding element type (line 49)
int_221676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 15), list_221673, int_221676)
# Adding element type (line 49)
int_221677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 15), list_221673, int_221677)
# Adding element type (line 49)
int_221678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 15), list_221673, int_221678)
# Adding element type (line 49)
int_221679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 15), list_221673, int_221679)

# Getting the type of 'F2_lucky' (line 49)
F2_lucky_221680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'F2_lucky')
# Setting the type of the member 'xin' of a type (line 49)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 0), F2_lucky_221680, 'xin', list_221673)

# Assigning a Dict to a Attribute (line 50):

# Assigning a Dict to a Attribute (line 50):

# Obtaining an instance of the builtin type 'dict' (line 50)
dict_221681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 50)

# Getting the type of 'F2_lucky' (line 50)
F2_lucky_221682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'F2_lucky')
# Setting the type of the member 'KNOWN_BAD' of a type (line 50)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 0), F2_lucky_221682, 'KNOWN_BAD', dict_221681)

@norecursion
def F3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F3'
    module_type_store = module_type_store.open_function_context('F3', 53, 0, False)
    
    # Passed parameters checking function
    F3.stypy_localization = localization
    F3.stypy_type_of_self = None
    F3.stypy_type_store = module_type_store
    F3.stypy_function_name = 'F3'
    F3.stypy_param_names_list = ['x']
    F3.stypy_varargs_param_name = None
    F3.stypy_kwargs_param_name = None
    F3.stypy_call_defaults = defaults
    F3.stypy_call_varargs = varargs
    F3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F3', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F3', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F3(...)' code ##################

    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to mat(...): (line 54)
    # Processing the call arguments (line 54)
    str_221685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 15), 'str', '-2 1 0; 1 -2 1; 0 1 -2')
    # Processing the call keyword arguments (line 54)
    kwargs_221686 = {}
    # Getting the type of 'np' (line 54)
    np_221683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'np', False)
    # Obtaining the member 'mat' of a type (line 54)
    mat_221684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), np_221683, 'mat')
    # Calling mat(args, kwargs) (line 54)
    mat_call_result_221687 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), mat_221684, *[str_221685], **kwargs_221686)
    
    # Assigning a type to the variable 'A' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'A', mat_call_result_221687)
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to mat(...): (line 55)
    # Processing the call arguments (line 55)
    str_221690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 15), 'str', '1 2 3')
    # Processing the call keyword arguments (line 55)
    kwargs_221691 = {}
    # Getting the type of 'np' (line 55)
    np_221688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'np', False)
    # Obtaining the member 'mat' of a type (line 55)
    mat_221689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), np_221688, 'mat')
    # Calling mat(args, kwargs) (line 55)
    mat_call_result_221692 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), mat_221689, *[str_221690], **kwargs_221691)
    
    # Assigning a type to the variable 'b' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'b', mat_call_result_221692)
    
    # Call to dot(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A' (line 56)
    A_221695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'A', False)
    # Getting the type of 'x' (line 56)
    x_221696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'x', False)
    # Processing the call keyword arguments (line 56)
    kwargs_221697 = {}
    # Getting the type of 'np' (line 56)
    np_221693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'np', False)
    # Obtaining the member 'dot' of a type (line 56)
    dot_221694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), np_221693, 'dot')
    # Calling dot(args, kwargs) (line 56)
    dot_call_result_221698 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), dot_221694, *[A_221695, x_221696], **kwargs_221697)
    
    # Getting the type of 'b' (line 56)
    b_221699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'b')
    # Applying the binary operator '-' (line 56)
    result_sub_221700 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 11), '-', dot_call_result_221698, b_221699)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', result_sub_221700)
    
    # ################# End of 'F3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F3' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_221701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_221701)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F3'
    return stypy_return_type_221701

# Assigning a type to the variable 'F3' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'F3', F3)

# Assigning a List to a Attribute (line 57):

# Assigning a List to a Attribute (line 57):

# Obtaining an instance of the builtin type 'list' (line 57)
list_221702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
int_221703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), list_221702, int_221703)
# Adding element type (line 57)
int_221704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), list_221702, int_221704)
# Adding element type (line 57)
int_221705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 9), list_221702, int_221705)

# Getting the type of 'F3' (line 57)
F3_221706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'F3')
# Setting the type of the member 'xin' of a type (line 57)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 0), F3_221706, 'xin', list_221702)

# Assigning a Dict to a Attribute (line 58):

# Assigning a Dict to a Attribute (line 58):

# Obtaining an instance of the builtin type 'dict' (line 58)
dict_221707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 58)

# Getting the type of 'F3' (line 58)
F3_221708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'F3')
# Setting the type of the member 'KNOWN_BAD' of a type (line 58)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 0), F3_221708, 'KNOWN_BAD', dict_221707)

@norecursion
def F4_powell(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F4_powell'
    module_type_store = module_type_store.open_function_context('F4_powell', 61, 0, False)
    
    # Passed parameters checking function
    F4_powell.stypy_localization = localization
    F4_powell.stypy_type_of_self = None
    F4_powell.stypy_type_store = module_type_store
    F4_powell.stypy_function_name = 'F4_powell'
    F4_powell.stypy_param_names_list = ['x']
    F4_powell.stypy_varargs_param_name = None
    F4_powell.stypy_kwargs_param_name = None
    F4_powell.stypy_call_defaults = defaults
    F4_powell.stypy_call_varargs = varargs
    F4_powell.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F4_powell', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F4_powell', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F4_powell(...)' code ##################

    
    # Assigning a Num to a Name (line 62):
    
    # Assigning a Num to a Name (line 62):
    float_221709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'float')
    # Assigning a type to the variable 'A' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'A', float_221709)
    
    # Obtaining an instance of the builtin type 'list' (line 63)
    list_221710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 63)
    # Adding element type (line 63)
    # Getting the type of 'A' (line 63)
    A_221711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'A')
    
    # Obtaining the type of the subscript
    int_221712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'int')
    # Getting the type of 'x' (line 63)
    x_221713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'x')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___221714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 14), x_221713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_221715 = invoke(stypy.reporting.localization.Localization(__file__, 63, 14), getitem___221714, int_221712)
    
    # Applying the binary operator '*' (line 63)
    result_mul_221716 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 12), '*', A_221711, subscript_call_result_221715)
    
    
    # Obtaining the type of the subscript
    int_221717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
    # Getting the type of 'x' (line 63)
    x_221718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'x')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___221719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), x_221718, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_221720 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), getitem___221719, int_221717)
    
    # Applying the binary operator '*' (line 63)
    result_mul_221721 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 18), '*', result_mul_221716, subscript_call_result_221720)
    
    int_221722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'int')
    # Applying the binary operator '-' (line 63)
    result_sub_221723 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 12), '-', result_mul_221721, int_221722)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 11), list_221710, result_sub_221723)
    # Adding element type (line 63)
    
    # Call to exp(...): (line 63)
    # Processing the call arguments (line 63)
    
    
    # Obtaining the type of the subscript
    int_221726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'int')
    # Getting the type of 'x' (line 63)
    x_221727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___221728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 37), x_221727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_221729 = invoke(stypy.reporting.localization.Localization(__file__, 63, 37), getitem___221728, int_221726)
    
    # Applying the 'usub' unary operator (line 63)
    result___neg___221730 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 36), 'usub', subscript_call_result_221729)
    
    # Processing the call keyword arguments (line 63)
    kwargs_221731 = {}
    # Getting the type of 'np' (line 63)
    np_221724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'np', False)
    # Obtaining the member 'exp' of a type (line 63)
    exp_221725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 29), np_221724, 'exp')
    # Calling exp(args, kwargs) (line 63)
    exp_call_result_221732 = invoke(stypy.reporting.localization.Localization(__file__, 63, 29), exp_221725, *[result___neg___221730], **kwargs_221731)
    
    
    # Call to exp(...): (line 63)
    # Processing the call arguments (line 63)
    
    
    # Obtaining the type of the subscript
    int_221735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 55), 'int')
    # Getting the type of 'x' (line 63)
    x_221736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 53), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___221737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 53), x_221736, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_221738 = invoke(stypy.reporting.localization.Localization(__file__, 63, 53), getitem___221737, int_221735)
    
    # Applying the 'usub' unary operator (line 63)
    result___neg___221739 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 52), 'usub', subscript_call_result_221738)
    
    # Processing the call keyword arguments (line 63)
    kwargs_221740 = {}
    # Getting the type of 'np' (line 63)
    np_221733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 45), 'np', False)
    # Obtaining the member 'exp' of a type (line 63)
    exp_221734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 45), np_221733, 'exp')
    # Calling exp(args, kwargs) (line 63)
    exp_call_result_221741 = invoke(stypy.reporting.localization.Localization(__file__, 63, 45), exp_221734, *[result___neg___221739], **kwargs_221740)
    
    # Applying the binary operator '+' (line 63)
    result_add_221742 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 29), '+', exp_call_result_221732, exp_call_result_221741)
    
    int_221743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 62), 'int')
    int_221744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 66), 'int')
    # Getting the type of 'A' (line 63)
    A_221745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 68), 'A')
    # Applying the binary operator 'div' (line 63)
    result_div_221746 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 66), 'div', int_221744, A_221745)
    
    # Applying the binary operator '+' (line 63)
    result_add_221747 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 62), '+', int_221743, result_div_221746)
    
    # Applying the binary operator '-' (line 63)
    result_sub_221748 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 59), '-', result_add_221742, result_add_221747)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 11), list_221710, result_sub_221748)
    
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', list_221710)
    
    # ################# End of 'F4_powell(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F4_powell' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_221749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_221749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F4_powell'
    return stypy_return_type_221749

# Assigning a type to the variable 'F4_powell' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'F4_powell', F4_powell)

# Assigning a List to a Attribute (line 64):

# Assigning a List to a Attribute (line 64):

# Obtaining an instance of the builtin type 'list' (line 64)
list_221750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 64)
# Adding element type (line 64)
int_221751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), list_221750, int_221751)
# Adding element type (line 64)
int_221752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), list_221750, int_221752)

# Getting the type of 'F4_powell' (line 64)
F4_powell_221753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'F4_powell')
# Setting the type of the member 'xin' of a type (line 64)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 0), F4_powell_221753, 'xin', list_221750)

# Assigning a Dict to a Attribute (line 65):

# Assigning a Dict to a Attribute (line 65):

# Obtaining an instance of the builtin type 'dict' (line 65)
dict_221754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 65)
# Adding element type (key, value) (line 65)
str_221755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'str', 'linearmixing')
# Getting the type of 'nonlin' (line 65)
nonlin_221756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 39), 'nonlin')
# Obtaining the member 'linearmixing' of a type (line 65)
linearmixing_221757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 39), nonlin_221756, 'linearmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), dict_221754, (str_221755, linearmixing_221757))
# Adding element type (key, value) (line 65)
str_221758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'str', 'excitingmixing')
# Getting the type of 'nonlin' (line 66)
nonlin_221759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 41), 'nonlin')
# Obtaining the member 'excitingmixing' of a type (line 66)
excitingmixing_221760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 41), nonlin_221759, 'excitingmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), dict_221754, (str_221758, excitingmixing_221760))
# Adding element type (key, value) (line 65)
str_221761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'str', 'diagbroyden')
# Getting the type of 'nonlin' (line 67)
nonlin_221762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'nonlin')
# Obtaining the member 'diagbroyden' of a type (line 67)
diagbroyden_221763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 38), nonlin_221762, 'diagbroyden')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), dict_221754, (str_221761, diagbroyden_221763))

# Getting the type of 'F4_powell' (line 65)
F4_powell_221764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'F4_powell')
# Setting the type of the member 'KNOWN_BAD' of a type (line 65)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 0), F4_powell_221764, 'KNOWN_BAD', dict_221754)

@norecursion
def F5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F5'
    module_type_store = module_type_store.open_function_context('F5', 70, 0, False)
    
    # Passed parameters checking function
    F5.stypy_localization = localization
    F5.stypy_type_of_self = None
    F5.stypy_type_store = module_type_store
    F5.stypy_function_name = 'F5'
    F5.stypy_param_names_list = ['x']
    F5.stypy_varargs_param_name = None
    F5.stypy_kwargs_param_name = None
    F5.stypy_call_defaults = defaults
    F5.stypy_call_varargs = varargs
    F5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F5', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F5', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F5(...)' code ##################

    
    # Call to pressure_network(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'x' (line 71)
    x_221766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'x', False)
    int_221767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 31), 'int')
    
    # Call to array(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_221770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    float_221771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 44), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 43), list_221770, float_221771)
    # Adding element type (line 71)
    float_221772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 43), list_221770, float_221772)
    # Adding element type (line 71)
    float_221773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 43), list_221770, float_221773)
    # Adding element type (line 71)
    float_221774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 56), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 43), list_221770, float_221774)
    
    # Processing the call keyword arguments (line 71)
    kwargs_221775 = {}
    # Getting the type of 'np' (line 71)
    np_221768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'np', False)
    # Obtaining the member 'array' of a type (line 71)
    array_221769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 34), np_221768, 'array')
    # Calling array(args, kwargs) (line 71)
    array_call_result_221776 = invoke(stypy.reporting.localization.Localization(__file__, 71, 34), array_221769, *[list_221770], **kwargs_221775)
    
    # Processing the call keyword arguments (line 71)
    kwargs_221777 = {}
    # Getting the type of 'pressure_network' (line 71)
    pressure_network_221765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'pressure_network', False)
    # Calling pressure_network(args, kwargs) (line 71)
    pressure_network_call_result_221778 = invoke(stypy.reporting.localization.Localization(__file__, 71, 11), pressure_network_221765, *[x_221766, int_221767, array_call_result_221776], **kwargs_221777)
    
    # Assigning a type to the variable 'stypy_return_type' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type', pressure_network_call_result_221778)
    
    # ################# End of 'F5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F5' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_221779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_221779)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F5'
    return stypy_return_type_221779

# Assigning a type to the variable 'F5' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'F5', F5)

# Assigning a List to a Attribute (line 72):

# Assigning a List to a Attribute (line 72):

# Obtaining an instance of the builtin type 'list' (line 72)
list_221780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 72)
# Adding element type (line 72)
float_221781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), list_221780, float_221781)
# Adding element type (line 72)
int_221782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), list_221780, int_221782)
# Adding element type (line 72)
int_221783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), list_221780, int_221783)
# Adding element type (line 72)
int_221784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), list_221780, int_221784)

# Getting the type of 'F5' (line 72)
F5_221785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'F5')
# Setting the type of the member 'xin' of a type (line 72)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 0), F5_221785, 'xin', list_221780)

# Assigning a Dict to a Attribute (line 73):

# Assigning a Dict to a Attribute (line 73):

# Obtaining an instance of the builtin type 'dict' (line 73)
dict_221786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 73)
# Adding element type (key, value) (line 73)
str_221787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 16), 'str', 'excitingmixing')
# Getting the type of 'nonlin' (line 73)
nonlin_221788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'nonlin')
# Obtaining the member 'excitingmixing' of a type (line 73)
excitingmixing_221789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 34), nonlin_221788, 'excitingmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 15), dict_221786, (str_221787, excitingmixing_221789))
# Adding element type (key, value) (line 73)
str_221790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'str', 'linearmixing')
# Getting the type of 'nonlin' (line 74)
nonlin_221791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'nonlin')
# Obtaining the member 'linearmixing' of a type (line 74)
linearmixing_221792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 32), nonlin_221791, 'linearmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 15), dict_221786, (str_221790, linearmixing_221792))
# Adding element type (key, value) (line 73)
str_221793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 16), 'str', 'diagbroyden')
# Getting the type of 'nonlin' (line 75)
nonlin_221794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'nonlin')
# Obtaining the member 'diagbroyden' of a type (line 75)
diagbroyden_221795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 31), nonlin_221794, 'diagbroyden')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 15), dict_221786, (str_221793, diagbroyden_221795))

# Getting the type of 'F5' (line 73)
F5_221796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'F5')
# Setting the type of the member 'KNOWN_BAD' of a type (line 73)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 0), F5_221796, 'KNOWN_BAD', dict_221786)

@norecursion
def F6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F6'
    module_type_store = module_type_store.open_function_context('F6', 78, 0, False)
    
    # Passed parameters checking function
    F6.stypy_localization = localization
    F6.stypy_type_of_self = None
    F6.stypy_type_store = module_type_store
    F6.stypy_function_name = 'F6'
    F6.stypy_param_names_list = ['x']
    F6.stypy_varargs_param_name = None
    F6.stypy_kwargs_param_name = None
    F6.stypy_call_defaults = defaults
    F6.stypy_call_varargs = varargs
    F6.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F6', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F6', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F6(...)' code ##################

    
    # Assigning a Name to a Tuple (line 79):
    
    # Assigning a Subscript to a Name (line 79):
    
    # Obtaining the type of the subscript
    int_221797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'int')
    # Getting the type of 'x' (line 79)
    x_221798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'x')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___221799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), x_221798, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_221800 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), getitem___221799, int_221797)
    
    # Assigning a type to the variable 'tuple_var_assignment_221553' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'tuple_var_assignment_221553', subscript_call_result_221800)
    
    # Assigning a Subscript to a Name (line 79):
    
    # Obtaining the type of the subscript
    int_221801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'int')
    # Getting the type of 'x' (line 79)
    x_221802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'x')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___221803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), x_221802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_221804 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), getitem___221803, int_221801)
    
    # Assigning a type to the variable 'tuple_var_assignment_221554' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'tuple_var_assignment_221554', subscript_call_result_221804)
    
    # Assigning a Name to a Name (line 79):
    # Getting the type of 'tuple_var_assignment_221553' (line 79)
    tuple_var_assignment_221553_221805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'tuple_var_assignment_221553')
    # Assigning a type to the variable 'x1' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'x1', tuple_var_assignment_221553_221805)
    
    # Assigning a Name to a Name (line 79):
    # Getting the type of 'tuple_var_assignment_221554' (line 79)
    tuple_var_assignment_221554_221806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'tuple_var_assignment_221554')
    # Assigning a type to the variable 'x2' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'x2', tuple_var_assignment_221554_221806)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to array(...): (line 80)
    # Processing the call arguments (line 80)
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_221809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_221810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    float_221811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 19), list_221810, float_221811)
    # Adding element type (line 80)
    float_221812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 19), list_221810, float_221812)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), list_221809, list_221810)
    # Adding element type (line 80)
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_221813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    float_221814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 16), list_221813, float_221814)
    # Adding element type (line 81)
    float_221815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 16), list_221813, float_221815)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), list_221809, list_221813)
    
    # Processing the call keyword arguments (line 80)
    kwargs_221816 = {}
    # Getting the type of 'np' (line 80)
    np_221807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 80)
    array_221808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 9), np_221807, 'array')
    # Calling array(args, kwargs) (line 80)
    array_call_result_221817 = invoke(stypy.reporting.localization.Localization(__file__, 80, 9), array_221808, *[list_221809], **kwargs_221816)
    
    # Assigning a type to the variable 'J0' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'J0', array_call_result_221817)
    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to array(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Obtaining an instance of the builtin type 'list' (line 82)
    list_221820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'x1' (line 82)
    x1_221821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'x1', False)
    int_221822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'int')
    # Applying the binary operator '+' (line 82)
    result_add_221823 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '+', x1_221821, int_221822)
    
    # Getting the type of 'x2' (line 82)
    x2_221824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'x2', False)
    int_221825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 34), 'int')
    # Applying the binary operator '**' (line 82)
    result_pow_221826 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 30), '**', x2_221824, int_221825)
    
    int_221827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 38), 'int')
    # Applying the binary operator '-' (line 82)
    result_sub_221828 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 30), '-', result_pow_221826, int_221827)
    
    # Applying the binary operator '*' (line 82)
    result_mul_221829 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 18), '*', result_add_221823, result_sub_221828)
    
    int_221830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 43), 'int')
    int_221831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 45), 'int')
    # Applying the binary operator '*' (line 82)
    result_mul_221832 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 43), '*', int_221830, int_221831)
    
    # Applying the binary operator '+' (line 82)
    result_add_221833 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 18), '+', result_mul_221829, result_mul_221832)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 17), list_221820, result_add_221833)
    # Adding element type (line 82)
    
    # Call to sin(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'x2' (line 83)
    x2_221836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'x2', False)
    
    # Call to exp(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'x1' (line 83)
    x1_221839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 37), 'x1', False)
    # Processing the call keyword arguments (line 83)
    kwargs_221840 = {}
    # Getting the type of 'np' (line 83)
    np_221837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'np', False)
    # Obtaining the member 'exp' of a type (line 83)
    exp_221838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), np_221837, 'exp')
    # Calling exp(args, kwargs) (line 83)
    exp_call_result_221841 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), exp_221838, *[x1_221839], **kwargs_221840)
    
    # Applying the binary operator '*' (line 83)
    result_mul_221842 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 25), '*', x2_221836, exp_call_result_221841)
    
    int_221843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 43), 'int')
    # Applying the binary operator '-' (line 83)
    result_sub_221844 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 25), '-', result_mul_221842, int_221843)
    
    # Processing the call keyword arguments (line 83)
    kwargs_221845 = {}
    # Getting the type of 'np' (line 83)
    np_221834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'np', False)
    # Obtaining the member 'sin' of a type (line 83)
    sin_221835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 18), np_221834, 'sin')
    # Calling sin(args, kwargs) (line 83)
    sin_call_result_221846 = invoke(stypy.reporting.localization.Localization(__file__, 83, 18), sin_221835, *[result_sub_221844], **kwargs_221845)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 17), list_221820, sin_call_result_221846)
    
    # Processing the call keyword arguments (line 82)
    kwargs_221847 = {}
    # Getting the type of 'np' (line 82)
    np_221818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 82)
    array_221819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), np_221818, 'array')
    # Calling array(args, kwargs) (line 82)
    array_call_result_221848 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), array_221819, *[list_221820], **kwargs_221847)
    
    # Assigning a type to the variable 'v' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'v', array_call_result_221848)
    
    
    # Call to solve(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'J0' (line 84)
    J0_221852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'J0', False)
    # Getting the type of 'v' (line 84)
    v_221853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'v', False)
    # Processing the call keyword arguments (line 84)
    kwargs_221854 = {}
    # Getting the type of 'np' (line 84)
    np_221849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'np', False)
    # Obtaining the member 'linalg' of a type (line 84)
    linalg_221850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), np_221849, 'linalg')
    # Obtaining the member 'solve' of a type (line 84)
    solve_221851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), linalg_221850, 'solve')
    # Calling solve(args, kwargs) (line 84)
    solve_call_result_221855 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), solve_221851, *[J0_221852, v_221853], **kwargs_221854)
    
    # Applying the 'usub' unary operator (line 84)
    result___neg___221856 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 11), 'usub', solve_call_result_221855)
    
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type', result___neg___221856)
    
    # ################# End of 'F6(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F6' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_221857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_221857)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F6'
    return stypy_return_type_221857

# Assigning a type to the variable 'F6' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'F6', F6)

# Assigning a List to a Attribute (line 85):

# Assigning a List to a Attribute (line 85):

# Obtaining an instance of the builtin type 'list' (line 85)
list_221858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 85)
# Adding element type (line 85)
float_221859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 9), list_221858, float_221859)
# Adding element type (line 85)
float_221860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 9), list_221858, float_221860)

# Getting the type of 'F6' (line 85)
F6_221861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'F6')
# Setting the type of the member 'xin' of a type (line 85)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 0), F6_221861, 'xin', list_221858)

# Assigning a Dict to a Attribute (line 86):

# Assigning a Dict to a Attribute (line 86):

# Obtaining an instance of the builtin type 'dict' (line 86)
dict_221862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 86)
# Adding element type (key, value) (line 86)
str_221863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'str', 'excitingmixing')
# Getting the type of 'nonlin' (line 86)
nonlin_221864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 34), 'nonlin')
# Obtaining the member 'excitingmixing' of a type (line 86)
excitingmixing_221865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 34), nonlin_221864, 'excitingmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), dict_221862, (str_221863, excitingmixing_221865))
# Adding element type (key, value) (line 86)
str_221866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'str', 'linearmixing')
# Getting the type of 'nonlin' (line 87)
nonlin_221867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'nonlin')
# Obtaining the member 'linearmixing' of a type (line 87)
linearmixing_221868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 32), nonlin_221867, 'linearmixing')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), dict_221862, (str_221866, linearmixing_221868))
# Adding element type (key, value) (line 86)
str_221869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 16), 'str', 'diagbroyden')
# Getting the type of 'nonlin' (line 88)
nonlin_221870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'nonlin')
# Obtaining the member 'diagbroyden' of a type (line 88)
diagbroyden_221871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 31), nonlin_221870, 'diagbroyden')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), dict_221862, (str_221869, diagbroyden_221871))

# Getting the type of 'F6' (line 86)
F6_221872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'F6')
# Setting the type of the member 'KNOWN_BAD' of a type (line 86)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 0), F6_221872, 'KNOWN_BAD', dict_221862)
# Declaration of the 'TestNonlin' class

class TestNonlin(object, ):
    str_221873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', "\n    Check the Broyden methods for a few test problems.\n\n    broyden1, broyden2, and newton_krylov must succeed for\n    all functions. Some of the others don't -- tests in KNOWN_BAD are skipped.\n\n    ")

    @norecursion
    def _check_nonlin_func(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_221874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 48), 'float')
        defaults = [float_221874]
        # Create a new context for function '_check_nonlin_func'
        module_type_store = module_type_store.open_function_context('_check_nonlin_func', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_localization', localization)
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_function_name', 'TestNonlin._check_nonlin_func')
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_param_names_list', ['f', 'func', 'f_tol'])
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlin._check_nonlin_func.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlin._check_nonlin_func', ['f', 'func', 'f_tol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_nonlin_func', localization, ['f', 'func', 'f_tol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_nonlin_func(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to func(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'f' (line 106)
        f_221876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'f', False)
        # Getting the type of 'f' (line 106)
        f_221877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'f', False)
        # Obtaining the member 'xin' of a type (line 106)
        xin_221878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), f_221877, 'xin')
        # Processing the call keyword arguments (line 106)
        # Getting the type of 'f_tol' (line 106)
        f_tol_221879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'f_tol', False)
        keyword_221880 = f_tol_221879
        int_221881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 48), 'int')
        keyword_221882 = int_221881
        int_221883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 61), 'int')
        keyword_221884 = int_221883
        kwargs_221885 = {'f_tol': keyword_221880, 'verbose': keyword_221884, 'maxiter': keyword_221882}
        # Getting the type of 'func' (line 106)
        func_221875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'func', False)
        # Calling func(args, kwargs) (line 106)
        func_call_result_221886 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), func_221875, *[f_221876, xin_221878], **kwargs_221885)
        
        # Assigning a type to the variable 'x' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'x', func_call_result_221886)
        
        # Call to assert_(...): (line 107)
        # Processing the call arguments (line 107)
        
        
        # Call to max(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_221897 = {}
        
        # Call to absolute(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Call to f(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'x' (line 107)
        x_221891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'x', False)
        # Processing the call keyword arguments (line 107)
        kwargs_221892 = {}
        # Getting the type of 'f' (line 107)
        f_221890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'f', False)
        # Calling f(args, kwargs) (line 107)
        f_call_result_221893 = invoke(stypy.reporting.localization.Localization(__file__, 107, 28), f_221890, *[x_221891], **kwargs_221892)
        
        # Processing the call keyword arguments (line 107)
        kwargs_221894 = {}
        # Getting the type of 'np' (line 107)
        np_221888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'np', False)
        # Obtaining the member 'absolute' of a type (line 107)
        absolute_221889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), np_221888, 'absolute')
        # Calling absolute(args, kwargs) (line 107)
        absolute_call_result_221895 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), absolute_221889, *[f_call_result_221893], **kwargs_221894)
        
        # Obtaining the member 'max' of a type (line 107)
        max_221896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), absolute_call_result_221895, 'max')
        # Calling max(args, kwargs) (line 107)
        max_call_result_221898 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), max_221896, *[], **kwargs_221897)
        
        # Getting the type of 'f_tol' (line 107)
        f_tol_221899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 42), 'f_tol', False)
        # Applying the binary operator '<' (line 107)
        result_lt_221900 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 16), '<', max_call_result_221898, f_tol_221899)
        
        # Processing the call keyword arguments (line 107)
        kwargs_221901 = {}
        # Getting the type of 'assert_' (line 107)
        assert__221887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 107)
        assert__call_result_221902 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert__221887, *[result_lt_221900], **kwargs_221901)
        
        
        # ################# End of '_check_nonlin_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_nonlin_func' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_221903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221903)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_nonlin_func'
        return stypy_return_type_221903


    @norecursion
    def _check_root(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_221904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 43), 'float')
        defaults = [float_221904]
        # Create a new context for function '_check_root'
        module_type_store = module_type_store.open_function_context('_check_root', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlin._check_root.__dict__.__setitem__('stypy_localization', localization)
        TestNonlin._check_root.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlin._check_root.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlin._check_root.__dict__.__setitem__('stypy_function_name', 'TestNonlin._check_root')
        TestNonlin._check_root.__dict__.__setitem__('stypy_param_names_list', ['f', 'method', 'f_tol'])
        TestNonlin._check_root.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlin._check_root.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlin._check_root.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlin._check_root.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlin._check_root.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlin._check_root.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlin._check_root', ['f', 'method', 'f_tol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_root', localization, ['f', 'method', 'f_tol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_root(...)' code ##################

        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to root(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'f' (line 110)
        f_221906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'f', False)
        # Getting the type of 'f' (line 110)
        f_221907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'f', False)
        # Obtaining the member 'xin' of a type (line 110)
        xin_221908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 22), f_221907, 'xin')
        # Processing the call keyword arguments (line 110)
        # Getting the type of 'method' (line 110)
        method_221909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 36), 'method', False)
        keyword_221910 = method_221909
        
        # Obtaining an instance of the builtin type 'dict' (line 111)
        dict_221911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 111)
        # Adding element type (key, value) (line 111)
        str_221912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 28), 'str', 'ftol')
        # Getting the type of 'f_tol' (line 111)
        f_tol_221913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'f_tol', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 27), dict_221911, (str_221912, f_tol_221913))
        # Adding element type (key, value) (line 111)
        str_221914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 43), 'str', 'maxiter')
        int_221915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 54), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 27), dict_221911, (str_221914, int_221915))
        # Adding element type (key, value) (line 111)
        str_221916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 59), 'str', 'disp')
        int_221917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 67), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 27), dict_221911, (str_221916, int_221917))
        
        keyword_221918 = dict_221911
        kwargs_221919 = {'method': keyword_221910, 'options': keyword_221918}
        # Getting the type of 'root' (line 110)
        root_221905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'root', False)
        # Calling root(args, kwargs) (line 110)
        root_call_result_221920 = invoke(stypy.reporting.localization.Localization(__file__, 110, 14), root_221905, *[f_221906, xin_221908], **kwargs_221919)
        
        # Assigning a type to the variable 'res' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'res', root_call_result_221920)
        
        # Call to assert_(...): (line 112)
        # Processing the call arguments (line 112)
        
        
        # Call to max(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_221929 = {}
        
        # Call to absolute(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'res' (line 112)
        res_221924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'res', False)
        # Obtaining the member 'fun' of a type (line 112)
        fun_221925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), res_221924, 'fun')
        # Processing the call keyword arguments (line 112)
        kwargs_221926 = {}
        # Getting the type of 'np' (line 112)
        np_221922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'np', False)
        # Obtaining the member 'absolute' of a type (line 112)
        absolute_221923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), np_221922, 'absolute')
        # Calling absolute(args, kwargs) (line 112)
        absolute_call_result_221927 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), absolute_221923, *[fun_221925], **kwargs_221926)
        
        # Obtaining the member 'max' of a type (line 112)
        max_221928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), absolute_call_result_221927, 'max')
        # Calling max(args, kwargs) (line 112)
        max_call_result_221930 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), max_221928, *[], **kwargs_221929)
        
        # Getting the type of 'f_tol' (line 112)
        f_tol_221931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 45), 'f_tol', False)
        # Applying the binary operator '<' (line 112)
        result_lt_221932 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 16), '<', max_call_result_221930, f_tol_221931)
        
        # Processing the call keyword arguments (line 112)
        kwargs_221933 = {}
        # Getting the type of 'assert_' (line 112)
        assert__221921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 112)
        assert__call_result_221934 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert__221921, *[result_lt_221932], **kwargs_221933)
        
        
        # ################# End of '_check_root(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_root' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_221935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221935)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_root'
        return stypy_return_type_221935


    @norecursion
    def _check_func_fail(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_func_fail'
        module_type_store = module_type_store.open_function_context('_check_func_fail', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_localization', localization)
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_function_name', 'TestNonlin._check_func_fail')
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_varargs_param_name', 'a')
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlin._check_func_fail.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlin._check_func_fail', [], 'a', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_func_fail', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_func_fail(...)' code ##################

        pass
        
        # ################# End of '_check_func_fail(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_func_fail' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_221936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221936)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_func_fail'
        return stypy_return_type_221936


    @norecursion
    def test_problem_nonlin(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_problem_nonlin'
        module_type_store = module_type_store.open_function_context('test_problem_nonlin', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_localization', localization)
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_function_name', 'TestNonlin.test_problem_nonlin')
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlin.test_problem_nonlin.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlin.test_problem_nonlin', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_problem_nonlin', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_problem_nonlin(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_221937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        # Getting the type of 'F' (line 119)
        F_221938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'F')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 17), list_221937, F_221938)
        # Adding element type (line 119)
        # Getting the type of 'F2' (line 119)
        F2_221939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'F2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 17), list_221937, F2_221939)
        # Adding element type (line 119)
        # Getting the type of 'F2_lucky' (line 119)
        F2_lucky_221940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'F2_lucky')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 17), list_221937, F2_lucky_221940)
        # Adding element type (line 119)
        # Getting the type of 'F3' (line 119)
        F3_221941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'F3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 17), list_221937, F3_221941)
        # Adding element type (line 119)
        # Getting the type of 'F4_powell' (line 119)
        F4_powell_221942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 39), 'F4_powell')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 17), list_221937, F4_powell_221942)
        # Adding element type (line 119)
        # Getting the type of 'F5' (line 119)
        F5_221943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 50), 'F5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 17), list_221937, F5_221943)
        # Adding element type (line 119)
        # Getting the type of 'F6' (line 119)
        F6_221944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 54), 'F6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 17), list_221937, F6_221944)
        
        # Testing the type of a for loop iterable (line 119)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 8), list_221937)
        # Getting the type of the for loop variable (line 119)
        for_loop_var_221945 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 8), list_221937)
        # Assigning a type to the variable 'f' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'f', for_loop_var_221945)
        # SSA begins for a for statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to values(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_221948 = {}
        # Getting the type of 'SOLVERS' (line 120)
        SOLVERS_221946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'SOLVERS', False)
        # Obtaining the member 'values' of a type (line 120)
        values_221947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 24), SOLVERS_221946, 'values')
        # Calling values(args, kwargs) (line 120)
        values_call_result_221949 = invoke(stypy.reporting.localization.Localization(__file__, 120, 24), values_221947, *[], **kwargs_221948)
        
        # Testing the type of a for loop iterable (line 120)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 12), values_call_result_221949)
        # Getting the type of the for loop variable (line 120)
        for_loop_var_221950 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 12), values_call_result_221949)
        # Assigning a type to the variable 'func' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'func', for_loop_var_221950)
        # SSA begins for a for statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'func' (line 121)
        func_221951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'func')
        
        # Call to values(...): (line 121)
        # Processing the call keyword arguments (line 121)
        kwargs_221955 = {}
        # Getting the type of 'f' (line 121)
        f_221952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'f', False)
        # Obtaining the member 'KNOWN_BAD' of a type (line 121)
        KNOWN_BAD_221953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 27), f_221952, 'KNOWN_BAD')
        # Obtaining the member 'values' of a type (line 121)
        values_221954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 27), KNOWN_BAD_221953, 'values')
        # Calling values(args, kwargs) (line 121)
        values_call_result_221956 = invoke(stypy.reporting.localization.Localization(__file__, 121, 27), values_221954, *[], **kwargs_221955)
        
        # Applying the binary operator 'in' (line 121)
        result_contains_221957 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), 'in', func_221951, values_call_result_221956)
        
        # Testing the type of an if condition (line 121)
        if_condition_221958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), result_contains_221957)
        # Assigning a type to the variable 'if_condition_221958' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_221958', if_condition_221958)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'func' (line 122)
        func_221959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'func')
        
        # Call to values(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_221962 = {}
        # Getting the type of 'MUST_WORK' (line 122)
        MUST_WORK_221960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'MUST_WORK', False)
        # Obtaining the member 'values' of a type (line 122)
        values_221961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 31), MUST_WORK_221960, 'values')
        # Calling values(args, kwargs) (line 122)
        values_call_result_221963 = invoke(stypy.reporting.localization.Localization(__file__, 122, 31), values_221961, *[], **kwargs_221962)
        
        # Applying the binary operator 'in' (line 122)
        result_contains_221964 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 23), 'in', func_221959, values_call_result_221963)
        
        # Testing the type of an if condition (line 122)
        if_condition_221965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 20), result_contains_221964)
        # Assigning a type to the variable 'if_condition_221965' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'if_condition_221965', if_condition_221965)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _check_func_fail(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'f' (line 123)
        f_221968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 46), 'f', False)
        # Getting the type of 'func' (line 123)
        func_221969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 49), 'func', False)
        # Processing the call keyword arguments (line 123)
        kwargs_221970 = {}
        # Getting the type of 'self' (line 123)
        self_221966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'self', False)
        # Obtaining the member '_check_func_fail' of a type (line 123)
        _check_func_fail_221967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 24), self_221966, '_check_func_fail')
        # Calling _check_func_fail(args, kwargs) (line 123)
        _check_func_fail_call_result_221971 = invoke(stypy.reporting.localization.Localization(__file__, 123, 24), _check_func_fail_221967, *[f_221968, func_221969], **kwargs_221970)
        
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _check_nonlin_func(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'f' (line 125)
        f_221974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'f', False)
        # Getting the type of 'func' (line 125)
        func_221975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 43), 'func', False)
        # Processing the call keyword arguments (line 125)
        kwargs_221976 = {}
        # Getting the type of 'self' (line 125)
        self_221972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'self', False)
        # Obtaining the member '_check_nonlin_func' of a type (line 125)
        _check_nonlin_func_221973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 16), self_221972, '_check_nonlin_func')
        # Calling _check_nonlin_func(args, kwargs) (line 125)
        _check_nonlin_func_call_result_221977 = invoke(stypy.reporting.localization.Localization(__file__, 125, 16), _check_nonlin_func_221973, *[f_221974, func_221975], **kwargs_221976)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_problem_nonlin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_problem_nonlin' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_221978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221978)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_problem_nonlin'
        return stypy_return_type_221978


    @norecursion
    def test_tol_norm_called(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tol_norm_called'
        module_type_store = module_type_store.open_function_context('test_tol_norm_called', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_localization', localization)
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_function_name', 'TestNonlin.test_tol_norm_called')
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlin.test_tol_norm_called.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlin.test_tol_norm_called', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tol_norm_called', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tol_norm_called(...)' code ##################

        
        # Assigning a Name to a Attribute (line 129):
        
        # Assigning a Name to a Attribute (line 129):
        # Getting the type of 'False' (line 129)
        False_221979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'False')
        # Getting the type of 'self' (line 129)
        self_221980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member '_tol_norm_used' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_221980, '_tol_norm_used', False_221979)

        @norecursion
        def local_norm_func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'local_norm_func'
            module_type_store = module_type_store.open_function_context('local_norm_func', 131, 8, False)
            
            # Passed parameters checking function
            local_norm_func.stypy_localization = localization
            local_norm_func.stypy_type_of_self = None
            local_norm_func.stypy_type_store = module_type_store
            local_norm_func.stypy_function_name = 'local_norm_func'
            local_norm_func.stypy_param_names_list = ['x']
            local_norm_func.stypy_varargs_param_name = None
            local_norm_func.stypy_kwargs_param_name = None
            local_norm_func.stypy_call_defaults = defaults
            local_norm_func.stypy_call_varargs = varargs
            local_norm_func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'local_norm_func', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'local_norm_func', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'local_norm_func(...)' code ##################

            
            # Assigning a Name to a Attribute (line 132):
            
            # Assigning a Name to a Attribute (line 132):
            # Getting the type of 'True' (line 132)
            True_221981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'True')
            # Getting the type of 'self' (line 132)
            self_221982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'self')
            # Setting the type of the member '_tol_norm_used' of a type (line 132)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), self_221982, '_tol_norm_used', True_221981)
            
            # Call to max(...): (line 133)
            # Processing the call keyword arguments (line 133)
            kwargs_221989 = {}
            
            # Call to absolute(...): (line 133)
            # Processing the call arguments (line 133)
            # Getting the type of 'x' (line 133)
            x_221985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'x', False)
            # Processing the call keyword arguments (line 133)
            kwargs_221986 = {}
            # Getting the type of 'np' (line 133)
            np_221983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'np', False)
            # Obtaining the member 'absolute' of a type (line 133)
            absolute_221984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 19), np_221983, 'absolute')
            # Calling absolute(args, kwargs) (line 133)
            absolute_call_result_221987 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), absolute_221984, *[x_221985], **kwargs_221986)
            
            # Obtaining the member 'max' of a type (line 133)
            max_221988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 19), absolute_call_result_221987, 'max')
            # Calling max(args, kwargs) (line 133)
            max_call_result_221990 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), max_221988, *[], **kwargs_221989)
            
            # Assigning a type to the variable 'stypy_return_type' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'stypy_return_type', max_call_result_221990)
            
            # ################# End of 'local_norm_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'local_norm_func' in the type store
            # Getting the type of 'stypy_return_type' (line 131)
            stypy_return_type_221991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_221991)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'local_norm_func'
            return stypy_return_type_221991

        # Assigning a type to the variable 'local_norm_func' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'local_norm_func', local_norm_func)
        
        # Call to newton_krylov(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'F' (line 135)
        F_221994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'F', False)
        # Getting the type of 'F' (line 135)
        F_221995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'F', False)
        # Obtaining the member 'xin' of a type (line 135)
        xin_221996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 32), F_221995, 'xin')
        # Processing the call keyword arguments (line 135)
        float_221997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 45), 'float')
        keyword_221998 = float_221997
        int_221999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 59), 'int')
        keyword_222000 = int_221999
        int_222001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 72), 'int')
        keyword_222002 = int_222001
        # Getting the type of 'local_norm_func' (line 136)
        local_norm_func_222003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 22), 'local_norm_func', False)
        keyword_222004 = local_norm_func_222003
        kwargs_222005 = {'tol_norm': keyword_222004, 'f_tol': keyword_221998, 'verbose': keyword_222002, 'maxiter': keyword_222000}
        # Getting the type of 'nonlin' (line 135)
        nonlin_221992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'nonlin', False)
        # Obtaining the member 'newton_krylov' of a type (line 135)
        newton_krylov_221993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), nonlin_221992, 'newton_krylov')
        # Calling newton_krylov(args, kwargs) (line 135)
        newton_krylov_call_result_222006 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), newton_krylov_221993, *[F_221994, xin_221996], **kwargs_222005)
        
        
        # Call to assert_(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_222008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'self', False)
        # Obtaining the member '_tol_norm_used' of a type (line 137)
        _tol_norm_used_222009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), self_222008, '_tol_norm_used')
        # Processing the call keyword arguments (line 137)
        kwargs_222010 = {}
        # Getting the type of 'assert_' (line 137)
        assert__222007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 137)
        assert__call_result_222011 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert__222007, *[_tol_norm_used_222009], **kwargs_222010)
        
        
        # ################# End of 'test_tol_norm_called(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tol_norm_called' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_222012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222012)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tol_norm_called'
        return stypy_return_type_222012


    @norecursion
    def test_problem_root(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_problem_root'
        module_type_store = module_type_store.open_function_context('test_problem_root', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_localization', localization)
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_function_name', 'TestNonlin.test_problem_root')
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlin.test_problem_root.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlin.test_problem_root', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_problem_root', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_problem_root(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_222013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        # Getting the type of 'F' (line 140)
        F_222014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'F')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), list_222013, F_222014)
        # Adding element type (line 140)
        # Getting the type of 'F2' (line 140)
        F2_222015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'F2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), list_222013, F2_222015)
        # Adding element type (line 140)
        # Getting the type of 'F2_lucky' (line 140)
        F2_lucky_222016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'F2_lucky')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), list_222013, F2_lucky_222016)
        # Adding element type (line 140)
        # Getting the type of 'F3' (line 140)
        F3_222017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 35), 'F3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), list_222013, F3_222017)
        # Adding element type (line 140)
        # Getting the type of 'F4_powell' (line 140)
        F4_powell_222018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 39), 'F4_powell')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), list_222013, F4_powell_222018)
        # Adding element type (line 140)
        # Getting the type of 'F5' (line 140)
        F5_222019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 50), 'F5')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), list_222013, F5_222019)
        # Adding element type (line 140)
        # Getting the type of 'F6' (line 140)
        F6_222020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 54), 'F6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 17), list_222013, F6_222020)
        
        # Testing the type of a for loop iterable (line 140)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 8), list_222013)
        # Getting the type of the for loop variable (line 140)
        for_loop_var_222021 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 8), list_222013)
        # Assigning a type to the variable 'f' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'f', for_loop_var_222021)
        # SSA begins for a for statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'SOLVERS' (line 141)
        SOLVERS_222022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'SOLVERS')
        # Testing the type of a for loop iterable (line 141)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 12), SOLVERS_222022)
        # Getting the type of the for loop variable (line 141)
        for_loop_var_222023 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 12), SOLVERS_222022)
        # Assigning a type to the variable 'meth' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'meth', for_loop_var_222023)
        # SSA begins for a for statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'meth' (line 142)
        meth_222024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'meth')
        # Getting the type of 'f' (line 142)
        f_222025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'f')
        # Obtaining the member 'KNOWN_BAD' of a type (line 142)
        KNOWN_BAD_222026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 27), f_222025, 'KNOWN_BAD')
        # Applying the binary operator 'in' (line 142)
        result_contains_222027 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 19), 'in', meth_222024, KNOWN_BAD_222026)
        
        # Testing the type of an if condition (line 142)
        if_condition_222028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 16), result_contains_222027)
        # Assigning a type to the variable 'if_condition_222028' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'if_condition_222028', if_condition_222028)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'meth' (line 143)
        meth_222029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'meth')
        # Getting the type of 'MUST_WORK' (line 143)
        MUST_WORK_222030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'MUST_WORK')
        # Applying the binary operator 'in' (line 143)
        result_contains_222031 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 23), 'in', meth_222029, MUST_WORK_222030)
        
        # Testing the type of an if condition (line 143)
        if_condition_222032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 20), result_contains_222031)
        # Assigning a type to the variable 'if_condition_222032' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'if_condition_222032', if_condition_222032)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _check_func_fail(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'f' (line 144)
        f_222035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 46), 'f', False)
        # Getting the type of 'meth' (line 144)
        meth_222036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 49), 'meth', False)
        # Processing the call keyword arguments (line 144)
        kwargs_222037 = {}
        # Getting the type of 'self' (line 144)
        self_222033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'self', False)
        # Obtaining the member '_check_func_fail' of a type (line 144)
        _check_func_fail_222034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 24), self_222033, '_check_func_fail')
        # Calling _check_func_fail(args, kwargs) (line 144)
        _check_func_fail_call_result_222038 = invoke(stypy.reporting.localization.Localization(__file__, 144, 24), _check_func_fail_222034, *[f_222035, meth_222036], **kwargs_222037)
        
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _check_root(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'f' (line 146)
        f_222041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'f', False)
        # Getting the type of 'meth' (line 146)
        meth_222042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 36), 'meth', False)
        # Processing the call keyword arguments (line 146)
        kwargs_222043 = {}
        # Getting the type of 'self' (line 146)
        self_222039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'self', False)
        # Obtaining the member '_check_root' of a type (line 146)
        _check_root_222040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), self_222039, '_check_root')
        # Calling _check_root(args, kwargs) (line 146)
        _check_root_call_result_222044 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), _check_root_222040, *[f_222041, meth_222042], **kwargs_222043)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_problem_root(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_problem_root' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_222045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222045)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_problem_root'
        return stypy_return_type_222045


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 96, 0, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestNonlin' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'TestNonlin', TestNonlin)
# Declaration of the 'TestSecant' class

class TestSecant(object, ):
    str_222046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 4), 'str', 'Check that some Jacobian approximations satisfy the secant condition')
    
    # Assigning a List to a Name (line 152):
    
    # Assigning a ListComp to a Name (line 160):

    @norecursion
    def _check_secant(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_222047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 45), 'int')
        defaults = [int_222047]
        # Create a new context for function '_check_secant'
        module_type_store = module_type_store.open_function_context('_check_secant', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSecant._check_secant.__dict__.__setitem__('stypy_localization', localization)
        TestSecant._check_secant.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSecant._check_secant.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSecant._check_secant.__dict__.__setitem__('stypy_function_name', 'TestSecant._check_secant')
        TestSecant._check_secant.__dict__.__setitem__('stypy_param_names_list', ['jac_cls', 'npoints'])
        TestSecant._check_secant.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSecant._check_secant.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        TestSecant._check_secant.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSecant._check_secant.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSecant._check_secant.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSecant._check_secant.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSecant._check_secant', ['jac_cls', 'npoints'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_secant', localization, ['jac_cls', 'npoints'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_secant(...)' code ##################

        str_222048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, (-1)), 'str', '\n        Check that the given Jacobian approximation satisfies secant\n        conditions for last `npoints` points.\n        ')
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to jac_cls(...): (line 167)
        # Processing the call keyword arguments (line 167)
        # Getting the type of 'kw' (line 167)
        kw_222050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'kw', False)
        kwargs_222051 = {'kw_222050': kw_222050}
        # Getting the type of 'jac_cls' (line 167)
        jac_cls_222049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'jac_cls', False)
        # Calling jac_cls(args, kwargs) (line 167)
        jac_cls_call_result_222052 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), jac_cls_222049, *[], **kwargs_222051)
        
        # Assigning a type to the variable 'jac' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'jac', jac_cls_call_result_222052)
        
        # Call to setup(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Obtaining the type of the subscript
        int_222055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 26), 'int')
        # Getting the type of 'self' (line 168)
        self_222056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'self', False)
        # Obtaining the member 'xs' of a type (line 168)
        xs_222057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 18), self_222056, 'xs')
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___222058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 18), xs_222057, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_222059 = invoke(stypy.reporting.localization.Localization(__file__, 168, 18), getitem___222058, int_222055)
        
        
        # Obtaining the type of the subscript
        int_222060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 38), 'int')
        # Getting the type of 'self' (line 168)
        self_222061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'self', False)
        # Obtaining the member 'fs' of a type (line 168)
        fs_222062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 30), self_222061, 'fs')
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___222063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 30), fs_222062, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_222064 = invoke(stypy.reporting.localization.Localization(__file__, 168, 30), getitem___222063, int_222060)
        
        # Getting the type of 'None' (line 168)
        None_222065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'None', False)
        # Processing the call keyword arguments (line 168)
        kwargs_222066 = {}
        # Getting the type of 'jac' (line 168)
        jac_222053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'jac', False)
        # Obtaining the member 'setup' of a type (line 168)
        setup_222054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), jac_222053, 'setup')
        # Calling setup(args, kwargs) (line 168)
        setup_call_result_222067 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), setup_222054, *[subscript_call_result_222059, subscript_call_result_222064, None_222065], **kwargs_222066)
        
        
        
        # Call to enumerate(...): (line 169)
        # Processing the call arguments (line 169)
        
        # Call to zip(...): (line 169)
        # Processing the call arguments (line 169)
        
        # Obtaining the type of the subscript
        int_222070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 47), 'int')
        slice_222071 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 39), int_222070, None, None)
        # Getting the type of 'self' (line 169)
        self_222072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 39), 'self', False)
        # Obtaining the member 'xs' of a type (line 169)
        xs_222073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 39), self_222072, 'xs')
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___222074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 39), xs_222073, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_222075 = invoke(stypy.reporting.localization.Localization(__file__, 169, 39), getitem___222074, slice_222071)
        
        
        # Obtaining the type of the subscript
        int_222076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 60), 'int')
        slice_222077 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 52), int_222076, None, None)
        # Getting the type of 'self' (line 169)
        self_222078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 52), 'self', False)
        # Obtaining the member 'fs' of a type (line 169)
        fs_222079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 52), self_222078, 'fs')
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___222080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 52), fs_222079, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_222081 = invoke(stypy.reporting.localization.Localization(__file__, 169, 52), getitem___222080, slice_222077)
        
        # Processing the call keyword arguments (line 169)
        kwargs_222082 = {}
        # Getting the type of 'zip' (line 169)
        zip_222069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'zip', False)
        # Calling zip(args, kwargs) (line 169)
        zip_call_result_222083 = invoke(stypy.reporting.localization.Localization(__file__, 169, 35), zip_222069, *[subscript_call_result_222075, subscript_call_result_222081], **kwargs_222082)
        
        # Processing the call keyword arguments (line 169)
        kwargs_222084 = {}
        # Getting the type of 'enumerate' (line 169)
        enumerate_222068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 169)
        enumerate_call_result_222085 = invoke(stypy.reporting.localization.Localization(__file__, 169, 25), enumerate_222068, *[zip_call_result_222083], **kwargs_222084)
        
        # Testing the type of a for loop iterable (line 169)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 169, 8), enumerate_call_result_222085)
        # Getting the type of the for loop variable (line 169)
        for_loop_var_222086 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 169, 8), enumerate_call_result_222085)
        # Assigning a type to the variable 'j' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), for_loop_var_222086))
        # Assigning a type to the variable 'x' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), for_loop_var_222086))
        # Assigning a type to the variable 'f' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), for_loop_var_222086))
        # SSA begins for a for statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to update(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'x' (line 170)
        x_222089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'x', False)
        # Getting the type of 'f' (line 170)
        f_222090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'f', False)
        # Processing the call keyword arguments (line 170)
        kwargs_222091 = {}
        # Getting the type of 'jac' (line 170)
        jac_222087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'jac', False)
        # Obtaining the member 'update' of a type (line 170)
        update_222088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), jac_222087, 'update')
        # Calling update(args, kwargs) (line 170)
        update_call_result_222092 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), update_222088, *[x_222089, f_222090], **kwargs_222091)
        
        
        
        # Call to xrange(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Call to min(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'npoints' (line 172)
        npoints_222095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'npoints', False)
        # Getting the type of 'j' (line 172)
        j_222096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 41), 'j', False)
        int_222097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 43), 'int')
        # Applying the binary operator '+' (line 172)
        result_add_222098 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 41), '+', j_222096, int_222097)
        
        # Processing the call keyword arguments (line 172)
        kwargs_222099 = {}
        # Getting the type of 'min' (line 172)
        min_222094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'min', False)
        # Calling min(args, kwargs) (line 172)
        min_call_result_222100 = invoke(stypy.reporting.localization.Localization(__file__, 172, 28), min_222094, *[npoints_222095, result_add_222098], **kwargs_222099)
        
        # Processing the call keyword arguments (line 172)
        kwargs_222101 = {}
        # Getting the type of 'xrange' (line 172)
        xrange_222093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 172)
        xrange_call_result_222102 = invoke(stypy.reporting.localization.Localization(__file__, 172, 21), xrange_222093, *[min_call_result_222100], **kwargs_222101)
        
        # Testing the type of a for loop iterable (line 172)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 172, 12), xrange_call_result_222102)
        # Getting the type of the for loop variable (line 172)
        for_loop_var_222103 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 172, 12), xrange_call_result_222102)
        # Assigning a type to the variable 'k' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'k', for_loop_var_222103)
        # SSA begins for a for statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 173):
        
        # Assigning a BinOp to a Name (line 173):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 173)
        j_222104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'j')
        # Getting the type of 'k' (line 173)
        k_222105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 31), 'k')
        # Applying the binary operator '-' (line 173)
        result_sub_222106 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 29), '-', j_222104, k_222105)
        
        int_222107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 33), 'int')
        # Applying the binary operator '+' (line 173)
        result_add_222108 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 32), '+', result_sub_222106, int_222107)
        
        # Getting the type of 'self' (line 173)
        self_222109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'self')
        # Obtaining the member 'xs' of a type (line 173)
        xs_222110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 21), self_222109, 'xs')
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___222111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 21), xs_222110, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_222112 = invoke(stypy.reporting.localization.Localization(__file__, 173, 21), getitem___222111, result_add_222108)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 173)
        j_222113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'j')
        # Getting the type of 'k' (line 173)
        k_222114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 48), 'k')
        # Applying the binary operator '-' (line 173)
        result_sub_222115 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 46), '-', j_222113, k_222114)
        
        # Getting the type of 'self' (line 173)
        self_222116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'self')
        # Obtaining the member 'xs' of a type (line 173)
        xs_222117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 38), self_222116, 'xs')
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___222118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 38), xs_222117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_222119 = invoke(stypy.reporting.localization.Localization(__file__, 173, 38), getitem___222118, result_sub_222115)
        
        # Applying the binary operator '-' (line 173)
        result_sub_222120 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 21), '-', subscript_call_result_222112, subscript_call_result_222119)
        
        # Assigning a type to the variable 'dx' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'dx', result_sub_222120)
        
        # Assigning a BinOp to a Name (line 174):
        
        # Assigning a BinOp to a Name (line 174):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 174)
        j_222121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'j')
        # Getting the type of 'k' (line 174)
        k_222122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'k')
        # Applying the binary operator '-' (line 174)
        result_sub_222123 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 29), '-', j_222121, k_222122)
        
        int_222124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 33), 'int')
        # Applying the binary operator '+' (line 174)
        result_add_222125 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 32), '+', result_sub_222123, int_222124)
        
        # Getting the type of 'self' (line 174)
        self_222126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'self')
        # Obtaining the member 'fs' of a type (line 174)
        fs_222127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 21), self_222126, 'fs')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___222128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 21), fs_222127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_222129 = invoke(stypy.reporting.localization.Localization(__file__, 174, 21), getitem___222128, result_add_222125)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 174)
        j_222130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 46), 'j')
        # Getting the type of 'k' (line 174)
        k_222131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 48), 'k')
        # Applying the binary operator '-' (line 174)
        result_sub_222132 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 46), '-', j_222130, k_222131)
        
        # Getting the type of 'self' (line 174)
        self_222133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'self')
        # Obtaining the member 'fs' of a type (line 174)
        fs_222134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), self_222133, 'fs')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___222135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), fs_222134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_222136 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), getitem___222135, result_sub_222132)
        
        # Applying the binary operator '-' (line 174)
        result_sub_222137 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 21), '-', subscript_call_result_222129, subscript_call_result_222136)
        
        # Assigning a type to the variable 'df' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'df', result_sub_222137)
        
        # Call to assert_(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Call to allclose(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'dx' (line 175)
        dx_222141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 36), 'dx', False)
        
        # Call to solve(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'df' (line 175)
        df_222144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 50), 'df', False)
        # Processing the call keyword arguments (line 175)
        kwargs_222145 = {}
        # Getting the type of 'jac' (line 175)
        jac_222142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 40), 'jac', False)
        # Obtaining the member 'solve' of a type (line 175)
        solve_222143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 40), jac_222142, 'solve')
        # Calling solve(args, kwargs) (line 175)
        solve_call_result_222146 = invoke(stypy.reporting.localization.Localization(__file__, 175, 40), solve_222143, *[df_222144], **kwargs_222145)
        
        # Processing the call keyword arguments (line 175)
        kwargs_222147 = {}
        # Getting the type of 'np' (line 175)
        np_222139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'np', False)
        # Obtaining the member 'allclose' of a type (line 175)
        allclose_222140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 24), np_222139, 'allclose')
        # Calling allclose(args, kwargs) (line 175)
        allclose_call_result_222148 = invoke(stypy.reporting.localization.Localization(__file__, 175, 24), allclose_222140, *[dx_222141, solve_call_result_222146], **kwargs_222147)
        
        # Processing the call keyword arguments (line 175)
        kwargs_222149 = {}
        # Getting the type of 'assert_' (line 175)
        assert__222138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 175)
        assert__call_result_222150 = invoke(stypy.reporting.localization.Localization(__file__, 175, 16), assert__222138, *[allclose_call_result_222148], **kwargs_222149)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'j' (line 178)
        j_222151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'j')
        # Getting the type of 'npoints' (line 178)
        npoints_222152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'npoints')
        # Applying the binary operator '>=' (line 178)
        result_ge_222153 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 15), '>=', j_222151, npoints_222152)
        
        # Testing the type of an if condition (line 178)
        if_condition_222154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 12), result_ge_222153)
        # Assigning a type to the variable 'if_condition_222154' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'if_condition_222154', if_condition_222154)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 179):
        
        # Assigning a BinOp to a Name (line 179):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 179)
        j_222155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'j')
        # Getting the type of 'npoints' (line 179)
        npoints_222156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'npoints')
        # Applying the binary operator '-' (line 179)
        result_sub_222157 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 29), '-', j_222155, npoints_222156)
        
        int_222158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 39), 'int')
        # Applying the binary operator '+' (line 179)
        result_add_222159 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 38), '+', result_sub_222157, int_222158)
        
        # Getting the type of 'self' (line 179)
        self_222160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'self')
        # Obtaining the member 'xs' of a type (line 179)
        xs_222161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), self_222160, 'xs')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___222162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), xs_222161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_222163 = invoke(stypy.reporting.localization.Localization(__file__, 179, 21), getitem___222162, result_add_222159)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 179)
        j_222164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 52), 'j')
        # Getting the type of 'npoints' (line 179)
        npoints_222165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 54), 'npoints')
        # Applying the binary operator '-' (line 179)
        result_sub_222166 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 52), '-', j_222164, npoints_222165)
        
        # Getting the type of 'self' (line 179)
        self_222167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 44), 'self')
        # Obtaining the member 'xs' of a type (line 179)
        xs_222168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 44), self_222167, 'xs')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___222169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 44), xs_222168, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_222170 = invoke(stypy.reporting.localization.Localization(__file__, 179, 44), getitem___222169, result_sub_222166)
        
        # Applying the binary operator '-' (line 179)
        result_sub_222171 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 21), '-', subscript_call_result_222163, subscript_call_result_222170)
        
        # Assigning a type to the variable 'dx' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'dx', result_sub_222171)
        
        # Assigning a BinOp to a Name (line 180):
        
        # Assigning a BinOp to a Name (line 180):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 180)
        j_222172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'j')
        # Getting the type of 'npoints' (line 180)
        npoints_222173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'npoints')
        # Applying the binary operator '-' (line 180)
        result_sub_222174 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 29), '-', j_222172, npoints_222173)
        
        int_222175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 39), 'int')
        # Applying the binary operator '+' (line 180)
        result_add_222176 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 38), '+', result_sub_222174, int_222175)
        
        # Getting the type of 'self' (line 180)
        self_222177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'self')
        # Obtaining the member 'fs' of a type (line 180)
        fs_222178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 21), self_222177, 'fs')
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___222179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 21), fs_222178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_222180 = invoke(stypy.reporting.localization.Localization(__file__, 180, 21), getitem___222179, result_add_222176)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 180)
        j_222181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 52), 'j')
        # Getting the type of 'npoints' (line 180)
        npoints_222182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 54), 'npoints')
        # Applying the binary operator '-' (line 180)
        result_sub_222183 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 52), '-', j_222181, npoints_222182)
        
        # Getting the type of 'self' (line 180)
        self_222184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 44), 'self')
        # Obtaining the member 'fs' of a type (line 180)
        fs_222185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 44), self_222184, 'fs')
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___222186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 44), fs_222185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_222187 = invoke(stypy.reporting.localization.Localization(__file__, 180, 44), getitem___222186, result_sub_222183)
        
        # Applying the binary operator '-' (line 180)
        result_sub_222188 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 21), '-', subscript_call_result_222180, subscript_call_result_222187)
        
        # Assigning a type to the variable 'df' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'df', result_sub_222188)
        
        # Call to assert_(...): (line 181)
        # Processing the call arguments (line 181)
        
        
        # Call to allclose(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'dx' (line 181)
        dx_222192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 40), 'dx', False)
        
        # Call to solve(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'df' (line 181)
        df_222195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 54), 'df', False)
        # Processing the call keyword arguments (line 181)
        kwargs_222196 = {}
        # Getting the type of 'jac' (line 181)
        jac_222193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 44), 'jac', False)
        # Obtaining the member 'solve' of a type (line 181)
        solve_222194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 44), jac_222193, 'solve')
        # Calling solve(args, kwargs) (line 181)
        solve_call_result_222197 = invoke(stypy.reporting.localization.Localization(__file__, 181, 44), solve_222194, *[df_222195], **kwargs_222196)
        
        # Processing the call keyword arguments (line 181)
        kwargs_222198 = {}
        # Getting the type of 'np' (line 181)
        np_222190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'np', False)
        # Obtaining the member 'allclose' of a type (line 181)
        allclose_222191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), np_222190, 'allclose')
        # Calling allclose(args, kwargs) (line 181)
        allclose_call_result_222199 = invoke(stypy.reporting.localization.Localization(__file__, 181, 28), allclose_222191, *[dx_222192, solve_call_result_222197], **kwargs_222198)
        
        # Applying the 'not' unary operator (line 181)
        result_not__222200 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 24), 'not', allclose_call_result_222199)
        
        # Processing the call keyword arguments (line 181)
        kwargs_222201 = {}
        # Getting the type of 'assert_' (line 181)
        assert__222189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 181)
        assert__call_result_222202 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), assert__222189, *[result_not__222200], **kwargs_222201)
        
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check_secant(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_secant' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_222203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222203)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_secant'
        return stypy_return_type_222203


    @norecursion
    def test_broyden1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden1'
        module_type_store = module_type_store.open_function_context('test_broyden1', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_localization', localization)
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_function_name', 'TestSecant.test_broyden1')
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_param_names_list', [])
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSecant.test_broyden1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSecant.test_broyden1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden1(...)' code ##################

        
        # Call to _check_secant(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'nonlin' (line 184)
        nonlin_222206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'nonlin', False)
        # Obtaining the member 'BroydenFirst' of a type (line 184)
        BroydenFirst_222207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 27), nonlin_222206, 'BroydenFirst')
        # Processing the call keyword arguments (line 184)
        kwargs_222208 = {}
        # Getting the type of 'self' (line 184)
        self_222204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self', False)
        # Obtaining the member '_check_secant' of a type (line 184)
        _check_secant_222205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_222204, '_check_secant')
        # Calling _check_secant(args, kwargs) (line 184)
        _check_secant_call_result_222209 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), _check_secant_222205, *[BroydenFirst_222207], **kwargs_222208)
        
        
        # ################# End of 'test_broyden1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden1' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_222210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222210)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden1'
        return stypy_return_type_222210


    @norecursion
    def test_broyden2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden2'
        module_type_store = module_type_store.open_function_context('test_broyden2', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_localization', localization)
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_function_name', 'TestSecant.test_broyden2')
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSecant.test_broyden2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSecant.test_broyden2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden2(...)' code ##################

        
        # Call to _check_secant(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'nonlin' (line 187)
        nonlin_222213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 27), 'nonlin', False)
        # Obtaining the member 'BroydenSecond' of a type (line 187)
        BroydenSecond_222214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 27), nonlin_222213, 'BroydenSecond')
        # Processing the call keyword arguments (line 187)
        kwargs_222215 = {}
        # Getting the type of 'self' (line 187)
        self_222211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self', False)
        # Obtaining the member '_check_secant' of a type (line 187)
        _check_secant_222212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_222211, '_check_secant')
        # Calling _check_secant(args, kwargs) (line 187)
        _check_secant_call_result_222216 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), _check_secant_222212, *[BroydenSecond_222214], **kwargs_222215)
        
        
        # ################# End of 'test_broyden2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden2' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_222217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222217)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden2'
        return stypy_return_type_222217


    @norecursion
    def test_broyden1_update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden1_update'
        module_type_store = module_type_store.open_function_context('test_broyden1_update', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_localization', localization)
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_function_name', 'TestSecant.test_broyden1_update')
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_param_names_list', [])
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSecant.test_broyden1_update.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSecant.test_broyden1_update', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden1_update', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden1_update(...)' code ##################

        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to BroydenFirst(...): (line 191)
        # Processing the call keyword arguments (line 191)
        float_222220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 40), 'float')
        keyword_222221 = float_222220
        kwargs_222222 = {'alpha': keyword_222221}
        # Getting the type of 'nonlin' (line 191)
        nonlin_222218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), 'nonlin', False)
        # Obtaining the member 'BroydenFirst' of a type (line 191)
        BroydenFirst_222219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 14), nonlin_222218, 'BroydenFirst')
        # Calling BroydenFirst(args, kwargs) (line 191)
        BroydenFirst_call_result_222223 = invoke(stypy.reporting.localization.Localization(__file__, 191, 14), BroydenFirst_222219, *[], **kwargs_222222)
        
        # Assigning a type to the variable 'jac' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'jac', BroydenFirst_call_result_222223)
        
        # Call to setup(...): (line 192)
        # Processing the call arguments (line 192)
        
        # Obtaining the type of the subscript
        int_222226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 26), 'int')
        # Getting the type of 'self' (line 192)
        self_222227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'self', False)
        # Obtaining the member 'xs' of a type (line 192)
        xs_222228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 18), self_222227, 'xs')
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___222229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 18), xs_222228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_222230 = invoke(stypy.reporting.localization.Localization(__file__, 192, 18), getitem___222229, int_222226)
        
        
        # Obtaining the type of the subscript
        int_222231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 38), 'int')
        # Getting the type of 'self' (line 192)
        self_222232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'self', False)
        # Obtaining the member 'fs' of a type (line 192)
        fs_222233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 30), self_222232, 'fs')
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___222234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 30), fs_222233, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_222235 = invoke(stypy.reporting.localization.Localization(__file__, 192, 30), getitem___222234, int_222231)
        
        # Getting the type of 'None' (line 192)
        None_222236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 42), 'None', False)
        # Processing the call keyword arguments (line 192)
        kwargs_222237 = {}
        # Getting the type of 'jac' (line 192)
        jac_222224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'jac', False)
        # Obtaining the member 'setup' of a type (line 192)
        setup_222225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), jac_222224, 'setup')
        # Calling setup(args, kwargs) (line 192)
        setup_call_result_222238 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), setup_222225, *[subscript_call_result_222230, subscript_call_result_222235, None_222236], **kwargs_222237)
        
        
        # Assigning a BinOp to a Name (line 194):
        
        # Assigning a BinOp to a Name (line 194):
        
        # Call to identity(...): (line 194)
        # Processing the call arguments (line 194)
        int_222241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 24), 'int')
        # Processing the call keyword arguments (line 194)
        kwargs_222242 = {}
        # Getting the type of 'np' (line 194)
        np_222239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'np', False)
        # Obtaining the member 'identity' of a type (line 194)
        identity_222240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), np_222239, 'identity')
        # Calling identity(args, kwargs) (line 194)
        identity_call_result_222243 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), identity_222240, *[int_222241], **kwargs_222242)
        
        int_222244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 30), 'int')
        float_222245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 33), 'float')
        # Applying the binary operator 'div' (line 194)
        result_div_222246 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 30), 'div', int_222244, float_222245)
        
        # Applying the binary operator '*' (line 194)
        result_mul_222247 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 12), '*', identity_call_result_222243, result_div_222246)
        
        # Assigning a type to the variable 'B' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'B', result_mul_222247)
        
        
        # Call to enumerate(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to zip(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Obtaining the type of the subscript
        int_222250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 52), 'int')
        slice_222251 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 196, 44), int_222250, None, None)
        # Getting the type of 'self' (line 196)
        self_222252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 44), 'self', False)
        # Obtaining the member 'xs' of a type (line 196)
        xs_222253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 44), self_222252, 'xs')
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___222254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 44), xs_222253, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_222255 = invoke(stypy.reporting.localization.Localization(__file__, 196, 44), getitem___222254, slice_222251)
        
        
        # Obtaining the type of the subscript
        int_222256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 65), 'int')
        slice_222257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 196, 57), int_222256, None, None)
        # Getting the type of 'self' (line 196)
        self_222258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 57), 'self', False)
        # Obtaining the member 'fs' of a type (line 196)
        fs_222259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 57), self_222258, 'fs')
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___222260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 57), fs_222259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_222261 = invoke(stypy.reporting.localization.Localization(__file__, 196, 57), getitem___222260, slice_222257)
        
        # Processing the call keyword arguments (line 196)
        kwargs_222262 = {}
        # Getting the type of 'zip' (line 196)
        zip_222249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 40), 'zip', False)
        # Calling zip(args, kwargs) (line 196)
        zip_call_result_222263 = invoke(stypy.reporting.localization.Localization(__file__, 196, 40), zip_222249, *[subscript_call_result_222255, subscript_call_result_222261], **kwargs_222262)
        
        # Processing the call keyword arguments (line 196)
        kwargs_222264 = {}
        # Getting the type of 'enumerate' (line 196)
        enumerate_222248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 30), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 196)
        enumerate_call_result_222265 = invoke(stypy.reporting.localization.Localization(__file__, 196, 30), enumerate_222248, *[zip_call_result_222263], **kwargs_222264)
        
        # Testing the type of a for loop iterable (line 196)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 8), enumerate_call_result_222265)
        # Getting the type of the for loop variable (line 196)
        for_loop_var_222266 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 8), enumerate_call_result_222265)
        # Assigning a type to the variable 'last_j' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'last_j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), for_loop_var_222266))
        # Assigning a type to the variable 'x' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), for_loop_var_222266))
        # Assigning a type to the variable 'f' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), for_loop_var_222266))
        # SSA begins for a for statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 197):
        
        # Assigning a BinOp to a Name (line 197):
        # Getting the type of 'f' (line 197)
        f_222267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'f')
        
        # Obtaining the type of the subscript
        # Getting the type of 'last_j' (line 197)
        last_j_222268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 29), 'last_j')
        # Getting the type of 'self' (line 197)
        self_222269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 21), 'self')
        # Obtaining the member 'fs' of a type (line 197)
        fs_222270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 21), self_222269, 'fs')
        # Obtaining the member '__getitem__' of a type (line 197)
        getitem___222271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 21), fs_222270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 197)
        subscript_call_result_222272 = invoke(stypy.reporting.localization.Localization(__file__, 197, 21), getitem___222271, last_j_222268)
        
        # Applying the binary operator '-' (line 197)
        result_sub_222273 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 17), '-', f_222267, subscript_call_result_222272)
        
        # Assigning a type to the variable 'df' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'df', result_sub_222273)
        
        # Assigning a BinOp to a Name (line 198):
        
        # Assigning a BinOp to a Name (line 198):
        # Getting the type of 'x' (line 198)
        x_222274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'last_j' (line 198)
        last_j_222275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 29), 'last_j')
        # Getting the type of 'self' (line 198)
        self_222276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'self')
        # Obtaining the member 'xs' of a type (line 198)
        xs_222277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 21), self_222276, 'xs')
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___222278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 21), xs_222277, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_222279 = invoke(stypy.reporting.localization.Localization(__file__, 198, 21), getitem___222278, last_j_222275)
        
        # Applying the binary operator '-' (line 198)
        result_sub_222280 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 17), '-', x_222274, subscript_call_result_222279)
        
        # Assigning a type to the variable 'dx' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'dx', result_sub_222280)
        
        # Getting the type of 'B' (line 199)
        B_222281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'B')
        
        # Obtaining the type of the subscript
        slice_222282 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 18), None, None, None)
        # Getting the type of 'None' (line 199)
        None_222283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 'None')
        # Getting the type of 'df' (line 199)
        df_222284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'df')
        
        # Call to dot(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'B' (line 199)
        B_222286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'B', False)
        # Getting the type of 'dx' (line 199)
        dx_222287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'dx', False)
        # Processing the call keyword arguments (line 199)
        kwargs_222288 = {}
        # Getting the type of 'dot' (line 199)
        dot_222285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'dot', False)
        # Calling dot(args, kwargs) (line 199)
        dot_call_result_222289 = invoke(stypy.reporting.localization.Localization(__file__, 199, 23), dot_222285, *[B_222286, dx_222287], **kwargs_222288)
        
        # Applying the binary operator '-' (line 199)
        result_sub_222290 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 18), '-', df_222284, dot_call_result_222289)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___222291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 18), result_sub_222290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_222292 = invoke(stypy.reporting.localization.Localization(__file__, 199, 18), getitem___222291, (slice_222282, None_222283))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 199)
        None_222293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 48), 'None')
        slice_222294 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 45), None, None, None)
        # Getting the type of 'dx' (line 199)
        dx_222295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 45), 'dx')
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___222296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 45), dx_222295, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_222297 = invoke(stypy.reporting.localization.Localization(__file__, 199, 45), getitem___222296, (None_222293, slice_222294))
        
        # Applying the binary operator '*' (line 199)
        result_mul_222298 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 17), '*', subscript_call_result_222292, subscript_call_result_222297)
        
        
        # Call to dot(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'dx' (line 199)
        dx_222300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 62), 'dx', False)
        # Getting the type of 'dx' (line 199)
        dx_222301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 66), 'dx', False)
        # Processing the call keyword arguments (line 199)
        kwargs_222302 = {}
        # Getting the type of 'dot' (line 199)
        dot_222299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 58), 'dot', False)
        # Calling dot(args, kwargs) (line 199)
        dot_call_result_222303 = invoke(stypy.reporting.localization.Localization(__file__, 199, 58), dot_222299, *[dx_222300, dx_222301], **kwargs_222302)
        
        # Applying the binary operator 'div' (line 199)
        result_div_222304 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 56), 'div', result_mul_222298, dot_call_result_222303)
        
        # Applying the binary operator '+=' (line 199)
        result_iadd_222305 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 12), '+=', B_222281, result_div_222304)
        # Assigning a type to the variable 'B' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'B', result_iadd_222305)
        
        
        # Call to update(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'x' (line 200)
        x_222308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'x', False)
        # Getting the type of 'f' (line 200)
        f_222309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'f', False)
        # Processing the call keyword arguments (line 200)
        kwargs_222310 = {}
        # Getting the type of 'jac' (line 200)
        jac_222306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'jac', False)
        # Obtaining the member 'update' of a type (line 200)
        update_222307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), jac_222306, 'update')
        # Calling update(args, kwargs) (line 200)
        update_call_result_222311 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), update_222307, *[x_222308, f_222309], **kwargs_222310)
        
        
        # Call to assert_(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Call to allclose(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Call to todense(...): (line 201)
        # Processing the call keyword arguments (line 201)
        kwargs_222317 = {}
        # Getting the type of 'jac' (line 201)
        jac_222315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 32), 'jac', False)
        # Obtaining the member 'todense' of a type (line 201)
        todense_222316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 32), jac_222315, 'todense')
        # Calling todense(args, kwargs) (line 201)
        todense_call_result_222318 = invoke(stypy.reporting.localization.Localization(__file__, 201, 32), todense_222316, *[], **kwargs_222317)
        
        # Getting the type of 'B' (line 201)
        B_222319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 47), 'B', False)
        # Processing the call keyword arguments (line 201)
        float_222320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 55), 'float')
        keyword_222321 = float_222320
        float_222322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 67), 'float')
        keyword_222323 = float_222322
        kwargs_222324 = {'rtol': keyword_222321, 'atol': keyword_222323}
        # Getting the type of 'np' (line 201)
        np_222313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'np', False)
        # Obtaining the member 'allclose' of a type (line 201)
        allclose_222314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), np_222313, 'allclose')
        # Calling allclose(args, kwargs) (line 201)
        allclose_call_result_222325 = invoke(stypy.reporting.localization.Localization(__file__, 201, 20), allclose_222314, *[todense_call_result_222318, B_222319], **kwargs_222324)
        
        # Processing the call keyword arguments (line 201)
        kwargs_222326 = {}
        # Getting the type of 'assert_' (line 201)
        assert__222312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 201)
        assert__call_result_222327 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), assert__222312, *[allclose_call_result_222325], **kwargs_222326)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_broyden1_update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden1_update' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_222328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden1_update'
        return stypy_return_type_222328


    @norecursion
    def test_broyden2_update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden2_update'
        module_type_store = module_type_store.open_function_context('test_broyden2_update', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_localization', localization)
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_function_name', 'TestSecant.test_broyden2_update')
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_param_names_list', [])
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSecant.test_broyden2_update.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSecant.test_broyden2_update', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden2_update', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden2_update(...)' code ##################

        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to BroydenSecond(...): (line 205)
        # Processing the call keyword arguments (line 205)
        float_222331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 41), 'float')
        keyword_222332 = float_222331
        kwargs_222333 = {'alpha': keyword_222332}
        # Getting the type of 'nonlin' (line 205)
        nonlin_222329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 14), 'nonlin', False)
        # Obtaining the member 'BroydenSecond' of a type (line 205)
        BroydenSecond_222330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 14), nonlin_222329, 'BroydenSecond')
        # Calling BroydenSecond(args, kwargs) (line 205)
        BroydenSecond_call_result_222334 = invoke(stypy.reporting.localization.Localization(__file__, 205, 14), BroydenSecond_222330, *[], **kwargs_222333)
        
        # Assigning a type to the variable 'jac' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'jac', BroydenSecond_call_result_222334)
        
        # Call to setup(...): (line 206)
        # Processing the call arguments (line 206)
        
        # Obtaining the type of the subscript
        int_222337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 26), 'int')
        # Getting the type of 'self' (line 206)
        self_222338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'self', False)
        # Obtaining the member 'xs' of a type (line 206)
        xs_222339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 18), self_222338, 'xs')
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___222340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 18), xs_222339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_222341 = invoke(stypy.reporting.localization.Localization(__file__, 206, 18), getitem___222340, int_222337)
        
        
        # Obtaining the type of the subscript
        int_222342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 38), 'int')
        # Getting the type of 'self' (line 206)
        self_222343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 30), 'self', False)
        # Obtaining the member 'fs' of a type (line 206)
        fs_222344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 30), self_222343, 'fs')
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___222345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 30), fs_222344, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_222346 = invoke(stypy.reporting.localization.Localization(__file__, 206, 30), getitem___222345, int_222342)
        
        # Getting the type of 'None' (line 206)
        None_222347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 42), 'None', False)
        # Processing the call keyword arguments (line 206)
        kwargs_222348 = {}
        # Getting the type of 'jac' (line 206)
        jac_222335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'jac', False)
        # Obtaining the member 'setup' of a type (line 206)
        setup_222336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), jac_222335, 'setup')
        # Calling setup(args, kwargs) (line 206)
        setup_call_result_222349 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), setup_222336, *[subscript_call_result_222341, subscript_call_result_222346, None_222347], **kwargs_222348)
        
        
        # Assigning a BinOp to a Name (line 208):
        
        # Assigning a BinOp to a Name (line 208):
        
        # Call to identity(...): (line 208)
        # Processing the call arguments (line 208)
        int_222352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 24), 'int')
        # Processing the call keyword arguments (line 208)
        kwargs_222353 = {}
        # Getting the type of 'np' (line 208)
        np_222350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'np', False)
        # Obtaining the member 'identity' of a type (line 208)
        identity_222351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), np_222350, 'identity')
        # Calling identity(args, kwargs) (line 208)
        identity_call_result_222354 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), identity_222351, *[int_222352], **kwargs_222353)
        
        float_222355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 30), 'float')
        # Applying the binary operator '*' (line 208)
        result_mul_222356 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 12), '*', identity_call_result_222354, float_222355)
        
        # Assigning a type to the variable 'H' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'H', result_mul_222356)
        
        
        # Call to enumerate(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Call to zip(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Obtaining the type of the subscript
        int_222359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 52), 'int')
        slice_222360 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 210, 44), int_222359, None, None)
        # Getting the type of 'self' (line 210)
        self_222361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 44), 'self', False)
        # Obtaining the member 'xs' of a type (line 210)
        xs_222362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 44), self_222361, 'xs')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___222363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 44), xs_222362, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_222364 = invoke(stypy.reporting.localization.Localization(__file__, 210, 44), getitem___222363, slice_222360)
        
        
        # Obtaining the type of the subscript
        int_222365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 65), 'int')
        slice_222366 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 210, 57), int_222365, None, None)
        # Getting the type of 'self' (line 210)
        self_222367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 57), 'self', False)
        # Obtaining the member 'fs' of a type (line 210)
        fs_222368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 57), self_222367, 'fs')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___222369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 57), fs_222368, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_222370 = invoke(stypy.reporting.localization.Localization(__file__, 210, 57), getitem___222369, slice_222366)
        
        # Processing the call keyword arguments (line 210)
        kwargs_222371 = {}
        # Getting the type of 'zip' (line 210)
        zip_222358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'zip', False)
        # Calling zip(args, kwargs) (line 210)
        zip_call_result_222372 = invoke(stypy.reporting.localization.Localization(__file__, 210, 40), zip_222358, *[subscript_call_result_222364, subscript_call_result_222370], **kwargs_222371)
        
        # Processing the call keyword arguments (line 210)
        kwargs_222373 = {}
        # Getting the type of 'enumerate' (line 210)
        enumerate_222357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 210)
        enumerate_call_result_222374 = invoke(stypy.reporting.localization.Localization(__file__, 210, 30), enumerate_222357, *[zip_call_result_222372], **kwargs_222373)
        
        # Testing the type of a for loop iterable (line 210)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 210, 8), enumerate_call_result_222374)
        # Getting the type of the for loop variable (line 210)
        for_loop_var_222375 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 210, 8), enumerate_call_result_222374)
        # Assigning a type to the variable 'last_j' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'last_j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 8), for_loop_var_222375))
        # Assigning a type to the variable 'x' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 8), for_loop_var_222375))
        # Assigning a type to the variable 'f' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 8), for_loop_var_222375))
        # SSA begins for a for statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 211):
        
        # Assigning a BinOp to a Name (line 211):
        # Getting the type of 'f' (line 211)
        f_222376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 17), 'f')
        
        # Obtaining the type of the subscript
        # Getting the type of 'last_j' (line 211)
        last_j_222377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 29), 'last_j')
        # Getting the type of 'self' (line 211)
        self_222378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'self')
        # Obtaining the member 'fs' of a type (line 211)
        fs_222379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 21), self_222378, 'fs')
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___222380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 21), fs_222379, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 211)
        subscript_call_result_222381 = invoke(stypy.reporting.localization.Localization(__file__, 211, 21), getitem___222380, last_j_222377)
        
        # Applying the binary operator '-' (line 211)
        result_sub_222382 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 17), '-', f_222376, subscript_call_result_222381)
        
        # Assigning a type to the variable 'df' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'df', result_sub_222382)
        
        # Assigning a BinOp to a Name (line 212):
        
        # Assigning a BinOp to a Name (line 212):
        # Getting the type of 'x' (line 212)
        x_222383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'last_j' (line 212)
        last_j_222384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'last_j')
        # Getting the type of 'self' (line 212)
        self_222385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 21), 'self')
        # Obtaining the member 'xs' of a type (line 212)
        xs_222386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 21), self_222385, 'xs')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___222387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 21), xs_222386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_222388 = invoke(stypy.reporting.localization.Localization(__file__, 212, 21), getitem___222387, last_j_222384)
        
        # Applying the binary operator '-' (line 212)
        result_sub_222389 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 17), '-', x_222383, subscript_call_result_222388)
        
        # Assigning a type to the variable 'dx' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'dx', result_sub_222389)
        
        # Getting the type of 'H' (line 213)
        H_222390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'H')
        
        # Obtaining the type of the subscript
        slice_222391 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 213, 18), None, None, None)
        # Getting the type of 'None' (line 213)
        None_222392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 37), 'None')
        # Getting the type of 'dx' (line 213)
        dx_222393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'dx')
        
        # Call to dot(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'H' (line 213)
        H_222395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'H', False)
        # Getting the type of 'df' (line 213)
        df_222396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'df', False)
        # Processing the call keyword arguments (line 213)
        kwargs_222397 = {}
        # Getting the type of 'dot' (line 213)
        dot_222394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 23), 'dot', False)
        # Calling dot(args, kwargs) (line 213)
        dot_call_result_222398 = invoke(stypy.reporting.localization.Localization(__file__, 213, 23), dot_222394, *[H_222395, df_222396], **kwargs_222397)
        
        # Applying the binary operator '-' (line 213)
        result_sub_222399 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 18), '-', dx_222393, dot_call_result_222398)
        
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___222400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 18), result_sub_222399, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_222401 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), getitem___222400, (slice_222391, None_222392))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 213)
        None_222402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 48), 'None')
        slice_222403 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 213, 45), None, None, None)
        # Getting the type of 'df' (line 213)
        df_222404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 45), 'df')
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___222405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 45), df_222404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_222406 = invoke(stypy.reporting.localization.Localization(__file__, 213, 45), getitem___222405, (None_222402, slice_222403))
        
        # Applying the binary operator '*' (line 213)
        result_mul_222407 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 17), '*', subscript_call_result_222401, subscript_call_result_222406)
        
        
        # Call to dot(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'df' (line 213)
        df_222409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 62), 'df', False)
        # Getting the type of 'df' (line 213)
        df_222410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 66), 'df', False)
        # Processing the call keyword arguments (line 213)
        kwargs_222411 = {}
        # Getting the type of 'dot' (line 213)
        dot_222408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 58), 'dot', False)
        # Calling dot(args, kwargs) (line 213)
        dot_call_result_222412 = invoke(stypy.reporting.localization.Localization(__file__, 213, 58), dot_222408, *[df_222409, df_222410], **kwargs_222411)
        
        # Applying the binary operator 'div' (line 213)
        result_div_222413 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 56), 'div', result_mul_222407, dot_call_result_222412)
        
        # Applying the binary operator '+=' (line 213)
        result_iadd_222414 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 12), '+=', H_222390, result_div_222413)
        # Assigning a type to the variable 'H' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'H', result_iadd_222414)
        
        
        # Call to update(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'x' (line 214)
        x_222417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'x', False)
        # Getting the type of 'f' (line 214)
        f_222418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 26), 'f', False)
        # Processing the call keyword arguments (line 214)
        kwargs_222419 = {}
        # Getting the type of 'jac' (line 214)
        jac_222415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'jac', False)
        # Obtaining the member 'update' of a type (line 214)
        update_222416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), jac_222415, 'update')
        # Calling update(args, kwargs) (line 214)
        update_call_result_222420 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), update_222416, *[x_222417, f_222418], **kwargs_222419)
        
        
        # Call to assert_(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Call to allclose(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Call to todense(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_222426 = {}
        # Getting the type of 'jac' (line 215)
        jac_222424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 32), 'jac', False)
        # Obtaining the member 'todense' of a type (line 215)
        todense_222425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 32), jac_222424, 'todense')
        # Calling todense(args, kwargs) (line 215)
        todense_call_result_222427 = invoke(stypy.reporting.localization.Localization(__file__, 215, 32), todense_222425, *[], **kwargs_222426)
        
        
        # Call to inv(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'H' (line 215)
        H_222429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 51), 'H', False)
        # Processing the call keyword arguments (line 215)
        kwargs_222430 = {}
        # Getting the type of 'inv' (line 215)
        inv_222428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 47), 'inv', False)
        # Calling inv(args, kwargs) (line 215)
        inv_call_result_222431 = invoke(stypy.reporting.localization.Localization(__file__, 215, 47), inv_222428, *[H_222429], **kwargs_222430)
        
        # Processing the call keyword arguments (line 215)
        float_222432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 60), 'float')
        keyword_222433 = float_222432
        float_222434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 72), 'float')
        keyword_222435 = float_222434
        kwargs_222436 = {'rtol': keyword_222433, 'atol': keyword_222435}
        # Getting the type of 'np' (line 215)
        np_222422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'np', False)
        # Obtaining the member 'allclose' of a type (line 215)
        allclose_222423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 20), np_222422, 'allclose')
        # Calling allclose(args, kwargs) (line 215)
        allclose_call_result_222437 = invoke(stypy.reporting.localization.Localization(__file__, 215, 20), allclose_222423, *[todense_call_result_222427, inv_call_result_222431], **kwargs_222436)
        
        # Processing the call keyword arguments (line 215)
        kwargs_222438 = {}
        # Getting the type of 'assert_' (line 215)
        assert__222421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 215)
        assert__call_result_222439 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), assert__222421, *[allclose_call_result_222437], **kwargs_222438)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_broyden2_update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden2_update' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_222440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden2_update'
        return stypy_return_type_222440


    @norecursion
    def test_anderson(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_anderson'
        module_type_store = module_type_store.open_function_context('test_anderson', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSecant.test_anderson.__dict__.__setitem__('stypy_localization', localization)
        TestSecant.test_anderson.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSecant.test_anderson.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSecant.test_anderson.__dict__.__setitem__('stypy_function_name', 'TestSecant.test_anderson')
        TestSecant.test_anderson.__dict__.__setitem__('stypy_param_names_list', [])
        TestSecant.test_anderson.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSecant.test_anderson.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSecant.test_anderson.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSecant.test_anderson.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSecant.test_anderson.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSecant.test_anderson.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSecant.test_anderson', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_anderson', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_anderson(...)' code ##################

        
        # Call to _check_secant(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'nonlin' (line 222)
        nonlin_222443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 27), 'nonlin', False)
        # Obtaining the member 'Anderson' of a type (line 222)
        Anderson_222444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 27), nonlin_222443, 'Anderson')
        # Processing the call keyword arguments (line 222)
        int_222445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 46), 'int')
        keyword_222446 = int_222445
        int_222447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 52), 'int')
        keyword_222448 = int_222447
        int_222449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 63), 'int')
        keyword_222450 = int_222449
        kwargs_222451 = {'npoints': keyword_222450, 'M': keyword_222446, 'w0': keyword_222448}
        # Getting the type of 'self' (line 222)
        self_222441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self', False)
        # Obtaining the member '_check_secant' of a type (line 222)
        _check_secant_222442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_222441, '_check_secant')
        # Calling _check_secant(args, kwargs) (line 222)
        _check_secant_call_result_222452 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), _check_secant_222442, *[Anderson_222444], **kwargs_222451)
        
        
        # ################# End of 'test_anderson(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_anderson' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_222453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222453)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_anderson'
        return stypy_return_type_222453


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 149, 0, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSecant.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSecant' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'TestSecant', TestSecant)

# Assigning a List to a Name (line 152):

# Obtaining an instance of the builtin type 'list' (line 152)
list_222454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 152)
# Adding element type (line 152)

# Call to array(...): (line 152)
# Processing the call arguments (line 152)

# Obtaining an instance of the builtin type 'list' (line 152)
list_222457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 152)
# Adding element type (line 152)
int_222458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_222457, int_222458)
# Adding element type (line 152)
int_222459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_222457, int_222459)
# Adding element type (line 152)
int_222460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_222457, int_222460)
# Adding element type (line 152)
int_222461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_222457, int_222461)
# Adding element type (line 152)
int_222462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_222457, int_222462)

# Getting the type of 'float' (line 152)
float_222463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'float', False)
# Processing the call keyword arguments (line 152)
kwargs_222464 = {}
# Getting the type of 'np' (line 152)
np_222455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 10), 'np', False)
# Obtaining the member 'array' of a type (line 152)
array_222456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 10), np_222455, 'array')
# Calling array(args, kwargs) (line 152)
array_call_result_222465 = invoke(stypy.reporting.localization.Localization(__file__, 152, 10), array_222456, *[list_222457, float_222463], **kwargs_222464)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 9), list_222454, array_call_result_222465)
# Adding element type (line 152)

# Call to array(...): (line 153)
# Processing the call arguments (line 153)

# Obtaining an instance of the builtin type 'list' (line 153)
list_222468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 153)
# Adding element type (line 153)
int_222469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_222468, int_222469)
# Adding element type (line 153)
int_222470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_222468, int_222470)
# Adding element type (line 153)
int_222471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_222468, int_222471)
# Adding element type (line 153)
int_222472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_222468, int_222472)
# Adding element type (line 153)
int_222473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 19), list_222468, int_222473)

# Getting the type of 'float' (line 153)
float_222474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 32), 'float', False)
# Processing the call keyword arguments (line 153)
kwargs_222475 = {}
# Getting the type of 'np' (line 153)
np_222466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 10), 'np', False)
# Obtaining the member 'array' of a type (line 153)
array_222467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 10), np_222466, 'array')
# Calling array(args, kwargs) (line 153)
array_call_result_222476 = invoke(stypy.reporting.localization.Localization(__file__, 153, 10), array_222467, *[list_222468, float_222474], **kwargs_222475)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 9), list_222454, array_call_result_222476)
# Adding element type (line 152)

# Call to array(...): (line 154)
# Processing the call arguments (line 154)

# Obtaining an instance of the builtin type 'list' (line 154)
list_222479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 154)
# Adding element type (line 154)
int_222480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_222479, int_222480)
# Adding element type (line 154)
int_222481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_222479, int_222481)
# Adding element type (line 154)
int_222482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_222479, int_222482)
# Adding element type (line 154)
int_222483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_222479, int_222483)
# Adding element type (line 154)
int_222484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), list_222479, int_222484)

# Getting the type of 'float' (line 154)
float_222485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 32), 'float', False)
# Processing the call keyword arguments (line 154)
kwargs_222486 = {}
# Getting the type of 'np' (line 154)
np_222477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 10), 'np', False)
# Obtaining the member 'array' of a type (line 154)
array_222478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 10), np_222477, 'array')
# Calling array(args, kwargs) (line 154)
array_call_result_222487 = invoke(stypy.reporting.localization.Localization(__file__, 154, 10), array_222478, *[list_222479, float_222485], **kwargs_222486)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 9), list_222454, array_call_result_222487)
# Adding element type (line 152)

# Call to array(...): (line 155)
# Processing the call arguments (line 155)

# Obtaining an instance of the builtin type 'list' (line 155)
list_222490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 155)
# Adding element type (line 155)
int_222491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_222490, int_222491)
# Adding element type (line 155)
int_222492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_222490, int_222492)
# Adding element type (line 155)
int_222493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_222490, int_222493)
# Adding element type (line 155)
int_222494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_222490, int_222494)
# Adding element type (line 155)
int_222495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 19), list_222490, int_222495)

# Getting the type of 'float' (line 155)
float_222496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'float', False)
# Processing the call keyword arguments (line 155)
kwargs_222497 = {}
# Getting the type of 'np' (line 155)
np_222488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 10), 'np', False)
# Obtaining the member 'array' of a type (line 155)
array_222489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 10), np_222488, 'array')
# Calling array(args, kwargs) (line 155)
array_call_result_222498 = invoke(stypy.reporting.localization.Localization(__file__, 155, 10), array_222489, *[list_222490, float_222496], **kwargs_222497)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 9), list_222454, array_call_result_222498)
# Adding element type (line 152)

# Call to array(...): (line 156)
# Processing the call arguments (line 156)

# Obtaining an instance of the builtin type 'list' (line 156)
list_222501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 156)
# Adding element type (line 156)
int_222502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), list_222501, int_222502)
# Adding element type (line 156)
int_222503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), list_222501, int_222503)
# Adding element type (line 156)
int_222504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), list_222501, int_222504)
# Adding element type (line 156)
int_222505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), list_222501, int_222505)
# Adding element type (line 156)
int_222506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 19), list_222501, int_222506)

# Getting the type of 'float' (line 156)
float_222507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'float', False)
# Processing the call keyword arguments (line 156)
kwargs_222508 = {}
# Getting the type of 'np' (line 156)
np_222499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 10), 'np', False)
# Obtaining the member 'array' of a type (line 156)
array_222500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 10), np_222499, 'array')
# Calling array(args, kwargs) (line 156)
array_call_result_222509 = invoke(stypy.reporting.localization.Localization(__file__, 156, 10), array_222500, *[list_222501, float_222507], **kwargs_222508)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 9), list_222454, array_call_result_222509)
# Adding element type (line 152)

# Call to array(...): (line 157)
# Processing the call arguments (line 157)

# Obtaining an instance of the builtin type 'list' (line 157)
list_222512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 157)
# Adding element type (line 157)
int_222513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 19), list_222512, int_222513)
# Adding element type (line 157)
int_222514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 19), list_222512, int_222514)
# Adding element type (line 157)
int_222515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 19), list_222512, int_222515)
# Adding element type (line 157)
int_222516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 19), list_222512, int_222516)
# Adding element type (line 157)
int_222517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 19), list_222512, int_222517)

# Getting the type of 'float' (line 157)
float_222518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'float', False)
# Processing the call keyword arguments (line 157)
kwargs_222519 = {}
# Getting the type of 'np' (line 157)
np_222510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 10), 'np', False)
# Obtaining the member 'array' of a type (line 157)
array_222511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 10), np_222510, 'array')
# Calling array(args, kwargs) (line 157)
array_call_result_222520 = invoke(stypy.reporting.localization.Localization(__file__, 157, 10), array_222511, *[list_222512, float_222518], **kwargs_222519)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 9), list_222454, array_call_result_222520)
# Adding element type (line 152)

# Call to array(...): (line 158)
# Processing the call arguments (line 158)

# Obtaining an instance of the builtin type 'list' (line 158)
list_222523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 158)
# Adding element type (line 158)
int_222524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 19), list_222523, int_222524)
# Adding element type (line 158)
int_222525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 19), list_222523, int_222525)
# Adding element type (line 158)
int_222526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 19), list_222523, int_222526)
# Adding element type (line 158)
int_222527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 19), list_222523, int_222527)
# Adding element type (line 158)
int_222528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 19), list_222523, int_222528)

# Getting the type of 'float' (line 158)
float_222529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'float', False)
# Processing the call keyword arguments (line 158)
kwargs_222530 = {}
# Getting the type of 'np' (line 158)
np_222521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 10), 'np', False)
# Obtaining the member 'array' of a type (line 158)
array_222522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 10), np_222521, 'array')
# Calling array(args, kwargs) (line 158)
array_call_result_222531 = invoke(stypy.reporting.localization.Localization(__file__, 158, 10), array_222522, *[list_222523, float_222529], **kwargs_222530)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 9), list_222454, array_call_result_222531)
# Adding element type (line 152)

# Call to array(...): (line 159)
# Processing the call arguments (line 159)

# Obtaining an instance of the builtin type 'list' (line 159)
list_222534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 159)
# Adding element type (line 159)
int_222535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), list_222534, int_222535)
# Adding element type (line 159)
int_222536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), list_222534, int_222536)
# Adding element type (line 159)
int_222537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), list_222534, int_222537)
# Adding element type (line 159)
int_222538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), list_222534, int_222538)
# Adding element type (line 159)
int_222539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), list_222534, int_222539)

# Getting the type of 'float' (line 159)
float_222540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'float', False)
# Processing the call keyword arguments (line 159)
kwargs_222541 = {}
# Getting the type of 'np' (line 159)
np_222532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 10), 'np', False)
# Obtaining the member 'array' of a type (line 159)
array_222533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 10), np_222532, 'array')
# Calling array(args, kwargs) (line 159)
array_call_result_222542 = invoke(stypy.reporting.localization.Localization(__file__, 159, 10), array_222533, *[list_222534, float_222540], **kwargs_222541)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 9), list_222454, array_call_result_222542)

# Getting the type of 'TestSecant'
TestSecant_222543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSecant')
# Setting the type of the member 'xs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSecant_222543, 'xs', list_222454)

# Assigning a ListComp to a Name (line 160):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'TestSecant'
TestSecant_222549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSecant')
# Obtaining the member 'xs' of a type
xs_222550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSecant_222549, 'xs')
comprehension_222551 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 10), xs_222550)
# Assigning a type to the variable 'x' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 10), 'x', comprehension_222551)
# Getting the type of 'x' (line 160)
x_222544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 10), 'x')
int_222545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 13), 'int')
# Applying the binary operator '**' (line 160)
result_pow_222546 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 10), '**', x_222544, int_222545)

int_222547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 17), 'int')
# Applying the binary operator '-' (line 160)
result_sub_222548 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 10), '-', result_pow_222546, int_222547)

list_222552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 10), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 10), list_222552, result_sub_222548)
# Getting the type of 'TestSecant'
TestSecant_222553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSecant')
# Setting the type of the member 'fs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSecant_222553, 'fs', list_222552)
# Declaration of the 'TestLinear' class

class TestLinear(object, ):
    str_222554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, (-1)), 'str', 'Solve a linear equation;\n    some methods find the exact solution in a finite number of steps')

    @norecursion
    def _check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 229)
        False_222555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 46), 'False')
        defaults = [False_222555]
        # Create a new context for function '_check'
        module_type_store = module_type_store.open_function_context('_check', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinear._check.__dict__.__setitem__('stypy_localization', localization)
        TestLinear._check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinear._check.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinear._check.__dict__.__setitem__('stypy_function_name', 'TestLinear._check')
        TestLinear._check.__dict__.__setitem__('stypy_param_names_list', ['jac', 'N', 'maxiter', 'complex'])
        TestLinear._check.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinear._check.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        TestLinear._check.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinear._check.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinear._check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinear._check.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinear._check', ['jac', 'N', 'maxiter', 'complex'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check', localization, ['jac', 'N', 'maxiter', 'complex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check(...)' code ##################

        
        # Call to seed(...): (line 230)
        # Processing the call arguments (line 230)
        int_222559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 23), 'int')
        # Processing the call keyword arguments (line 230)
        kwargs_222560 = {}
        # Getting the type of 'np' (line 230)
        np_222556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 230)
        random_222557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), np_222556, 'random')
        # Obtaining the member 'seed' of a type (line 230)
        seed_222558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), random_222557, 'seed')
        # Calling seed(args, kwargs) (line 230)
        seed_call_result_222561 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), seed_222558, *[int_222559], **kwargs_222560)
        
        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to randn(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'N' (line 232)
        N_222565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'N', False)
        # Getting the type of 'N' (line 232)
        N_222566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'N', False)
        # Processing the call keyword arguments (line 232)
        kwargs_222567 = {}
        # Getting the type of 'np' (line 232)
        np_222562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 232)
        random_222563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), np_222562, 'random')
        # Obtaining the member 'randn' of a type (line 232)
        randn_222564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), random_222563, 'randn')
        # Calling randn(args, kwargs) (line 232)
        randn_call_result_222568 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), randn_222564, *[N_222565, N_222566], **kwargs_222567)
        
        # Assigning a type to the variable 'A' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'A', randn_call_result_222568)
        
        # Getting the type of 'complex' (line 233)
        complex_222569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'complex')
        # Testing the type of an if condition (line 233)
        if_condition_222570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), complex_222569)
        # Assigning a type to the variable 'if_condition_222570' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_222570', if_condition_222570)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 234):
        
        # Assigning a BinOp to a Name (line 234):
        # Getting the type of 'A' (line 234)
        A_222571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'A')
        complex_222572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 20), 'complex')
        
        # Call to randn(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'N' (line 234)
        N_222576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 39), 'N', False)
        # Getting the type of 'N' (line 234)
        N_222577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 42), 'N', False)
        # Processing the call keyword arguments (line 234)
        kwargs_222578 = {}
        # Getting the type of 'np' (line 234)
        np_222573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 23), 'np', False)
        # Obtaining the member 'random' of a type (line 234)
        random_222574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 23), np_222573, 'random')
        # Obtaining the member 'randn' of a type (line 234)
        randn_222575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 23), random_222574, 'randn')
        # Calling randn(args, kwargs) (line 234)
        randn_call_result_222579 = invoke(stypy.reporting.localization.Localization(__file__, 234, 23), randn_222575, *[N_222576, N_222577], **kwargs_222578)
        
        # Applying the binary operator '*' (line 234)
        result_mul_222580 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 20), '*', complex_222572, randn_call_result_222579)
        
        # Applying the binary operator '+' (line 234)
        result_add_222581 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 16), '+', A_222571, result_mul_222580)
        
        # Assigning a type to the variable 'A' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'A', result_add_222581)
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to randn(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'N' (line 235)
        N_222585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'N', False)
        # Processing the call keyword arguments (line 235)
        kwargs_222586 = {}
        # Getting the type of 'np' (line 235)
        np_222582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 235)
        random_222583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), np_222582, 'random')
        # Obtaining the member 'randn' of a type (line 235)
        randn_222584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), random_222583, 'randn')
        # Calling randn(args, kwargs) (line 235)
        randn_call_result_222587 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), randn_222584, *[N_222585], **kwargs_222586)
        
        # Assigning a type to the variable 'b' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'b', randn_call_result_222587)
        
        # Getting the type of 'complex' (line 236)
        complex_222588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'complex')
        # Testing the type of an if condition (line 236)
        if_condition_222589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), complex_222588)
        # Assigning a type to the variable 'if_condition_222589' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_222589', if_condition_222589)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 237):
        
        # Assigning a BinOp to a Name (line 237):
        # Getting the type of 'b' (line 237)
        b_222590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'b')
        complex_222591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 20), 'complex')
        
        # Call to randn(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'N' (line 237)
        N_222595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 39), 'N', False)
        # Processing the call keyword arguments (line 237)
        kwargs_222596 = {}
        # Getting the type of 'np' (line 237)
        np_222592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 23), 'np', False)
        # Obtaining the member 'random' of a type (line 237)
        random_222593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 23), np_222592, 'random')
        # Obtaining the member 'randn' of a type (line 237)
        randn_222594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 23), random_222593, 'randn')
        # Calling randn(args, kwargs) (line 237)
        randn_call_result_222597 = invoke(stypy.reporting.localization.Localization(__file__, 237, 23), randn_222594, *[N_222595], **kwargs_222596)
        
        # Applying the binary operator '*' (line 237)
        result_mul_222598 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 20), '*', complex_222591, randn_call_result_222597)
        
        # Applying the binary operator '+' (line 237)
        result_add_222599 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 16), '+', b_222590, result_mul_222598)
        
        # Assigning a type to the variable 'b' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'b', result_add_222599)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 239, 8, False)
            
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

            
            # Call to dot(...): (line 240)
            # Processing the call arguments (line 240)
            # Getting the type of 'A' (line 240)
            A_222601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'A', False)
            # Getting the type of 'x' (line 240)
            x_222602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 26), 'x', False)
            # Processing the call keyword arguments (line 240)
            kwargs_222603 = {}
            # Getting the type of 'dot' (line 240)
            dot_222600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'dot', False)
            # Calling dot(args, kwargs) (line 240)
            dot_call_result_222604 = invoke(stypy.reporting.localization.Localization(__file__, 240, 19), dot_222600, *[A_222601, x_222602], **kwargs_222603)
            
            # Getting the type of 'b' (line 240)
            b_222605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'b')
            # Applying the binary operator '-' (line 240)
            result_sub_222606 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 19), '-', dot_call_result_222604, b_222605)
            
            # Assigning a type to the variable 'stypy_return_type' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'stypy_return_type', result_sub_222606)
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 239)
            stypy_return_type_222607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_222607)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_222607

        # Assigning a type to the variable 'func' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'func', func)
        
        # Assigning a Call to a Name (line 242):
        
        # Assigning a Call to a Name (line 242):
        
        # Call to nonlin_solve(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'func' (line 242)
        func_222610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'func', False)
        
        # Call to zeros(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'N' (line 242)
        N_222613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 49), 'N', False)
        # Processing the call keyword arguments (line 242)
        kwargs_222614 = {}
        # Getting the type of 'np' (line 242)
        np_222611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 40), 'np', False)
        # Obtaining the member 'zeros' of a type (line 242)
        zeros_222612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 40), np_222611, 'zeros')
        # Calling zeros(args, kwargs) (line 242)
        zeros_call_result_222615 = invoke(stypy.reporting.localization.Localization(__file__, 242, 40), zeros_222612, *[N_222613], **kwargs_222614)
        
        # Getting the type of 'jac' (line 242)
        jac_222616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 53), 'jac', False)
        # Processing the call keyword arguments (line 242)
        # Getting the type of 'maxiter' (line 242)
        maxiter_222617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 66), 'maxiter', False)
        keyword_222618 = maxiter_222617
        float_222619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 40), 'float')
        keyword_222620 = float_222619
        # Getting the type of 'None' (line 243)
        None_222621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 58), 'None', False)
        keyword_222622 = None_222621
        int_222623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 72), 'int')
        keyword_222624 = int_222623
        kwargs_222625 = {'f_tol': keyword_222620, 'verbose': keyword_222624, 'line_search': keyword_222622, 'maxiter': keyword_222618}
        # Getting the type of 'nonlin' (line 242)
        nonlin_222608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'nonlin', False)
        # Obtaining the member 'nonlin_solve' of a type (line 242)
        nonlin_solve_222609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 14), nonlin_222608, 'nonlin_solve')
        # Calling nonlin_solve(args, kwargs) (line 242)
        nonlin_solve_call_result_222626 = invoke(stypy.reporting.localization.Localization(__file__, 242, 14), nonlin_solve_222609, *[func_222610, zeros_call_result_222615, jac_222616], **kwargs_222625)
        
        # Assigning a type to the variable 'sol' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'sol', nonlin_solve_call_result_222626)
        
        # Call to assert_(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Call to allclose(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Call to dot(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'A' (line 244)
        A_222631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 32), 'A', False)
        # Getting the type of 'sol' (line 244)
        sol_222632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 35), 'sol', False)
        # Processing the call keyword arguments (line 244)
        kwargs_222633 = {}
        # Getting the type of 'dot' (line 244)
        dot_222630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 28), 'dot', False)
        # Calling dot(args, kwargs) (line 244)
        dot_call_result_222634 = invoke(stypy.reporting.localization.Localization(__file__, 244, 28), dot_222630, *[A_222631, sol_222632], **kwargs_222633)
        
        # Getting the type of 'b' (line 244)
        b_222635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 41), 'b', False)
        # Processing the call keyword arguments (line 244)
        float_222636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 49), 'float')
        keyword_222637 = float_222636
        kwargs_222638 = {'atol': keyword_222637}
        # Getting the type of 'np' (line 244)
        np_222628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'np', False)
        # Obtaining the member 'allclose' of a type (line 244)
        allclose_222629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 16), np_222628, 'allclose')
        # Calling allclose(args, kwargs) (line 244)
        allclose_call_result_222639 = invoke(stypy.reporting.localization.Localization(__file__, 244, 16), allclose_222629, *[dot_call_result_222634, b_222635], **kwargs_222638)
        
        # Processing the call keyword arguments (line 244)
        kwargs_222640 = {}
        # Getting the type of 'assert_' (line 244)
        assert__222627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 244)
        assert__call_result_222641 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), assert__222627, *[allclose_call_result_222639], **kwargs_222640)
        
        
        # ################# End of '_check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_222642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222642)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check'
        return stypy_return_type_222642


    @norecursion
    def test_broyden1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden1'
        module_type_store = module_type_store.open_function_context('test_broyden1', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_localization', localization)
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_function_name', 'TestLinear.test_broyden1')
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinear.test_broyden1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinear.test_broyden1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden1(...)' code ##################

        
        # Call to _check(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Call to BroydenFirst(...): (line 248)
        # Processing the call keyword arguments (line 248)
        float_222647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 46), 'float')
        keyword_222648 = float_222647
        kwargs_222649 = {'alpha': keyword_222648}
        # Getting the type of 'nonlin' (line 248)
        nonlin_222645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'nonlin', False)
        # Obtaining the member 'BroydenFirst' of a type (line 248)
        BroydenFirst_222646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 20), nonlin_222645, 'BroydenFirst')
        # Calling BroydenFirst(args, kwargs) (line 248)
        BroydenFirst_call_result_222650 = invoke(stypy.reporting.localization.Localization(__file__, 248, 20), BroydenFirst_222646, *[], **kwargs_222649)
        
        int_222651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 52), 'int')
        int_222652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 56), 'int')
        # Getting the type of 'False' (line 248)
        False_222653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 60), 'False', False)
        # Processing the call keyword arguments (line 248)
        kwargs_222654 = {}
        # Getting the type of 'self' (line 248)
        self_222643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 248)
        _check_222644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_222643, '_check')
        # Calling _check(args, kwargs) (line 248)
        _check_call_result_222655 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), _check_222644, *[BroydenFirst_call_result_222650, int_222651, int_222652, False_222653], **kwargs_222654)
        
        
        # Call to _check(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Call to BroydenFirst(...): (line 249)
        # Processing the call keyword arguments (line 249)
        float_222660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 46), 'float')
        keyword_222661 = float_222660
        kwargs_222662 = {'alpha': keyword_222661}
        # Getting the type of 'nonlin' (line 249)
        nonlin_222658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'nonlin', False)
        # Obtaining the member 'BroydenFirst' of a type (line 249)
        BroydenFirst_222659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 20), nonlin_222658, 'BroydenFirst')
        # Calling BroydenFirst(args, kwargs) (line 249)
        BroydenFirst_call_result_222663 = invoke(stypy.reporting.localization.Localization(__file__, 249, 20), BroydenFirst_222659, *[], **kwargs_222662)
        
        int_222664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 52), 'int')
        int_222665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 56), 'int')
        # Getting the type of 'True' (line 249)
        True_222666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 60), 'True', False)
        # Processing the call keyword arguments (line 249)
        kwargs_222667 = {}
        # Getting the type of 'self' (line 249)
        self_222656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 249)
        _check_222657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_222656, '_check')
        # Calling _check(args, kwargs) (line 249)
        _check_call_result_222668 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), _check_222657, *[BroydenFirst_call_result_222663, int_222664, int_222665, True_222666], **kwargs_222667)
        
        
        # ################# End of 'test_broyden1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden1' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_222669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222669)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden1'
        return stypy_return_type_222669


    @norecursion
    def test_broyden2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden2'
        module_type_store = module_type_store.open_function_context('test_broyden2', 251, 4, False)
        # Assigning a type to the variable 'self' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_localization', localization)
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_function_name', 'TestLinear.test_broyden2')
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinear.test_broyden2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinear.test_broyden2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden2(...)' code ##################

        
        # Call to _check(...): (line 253)
        # Processing the call arguments (line 253)
        
        # Call to BroydenSecond(...): (line 253)
        # Processing the call keyword arguments (line 253)
        float_222674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 47), 'float')
        keyword_222675 = float_222674
        kwargs_222676 = {'alpha': keyword_222675}
        # Getting the type of 'nonlin' (line 253)
        nonlin_222672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'nonlin', False)
        # Obtaining the member 'BroydenSecond' of a type (line 253)
        BroydenSecond_222673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), nonlin_222672, 'BroydenSecond')
        # Calling BroydenSecond(args, kwargs) (line 253)
        BroydenSecond_call_result_222677 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), BroydenSecond_222673, *[], **kwargs_222676)
        
        int_222678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 53), 'int')
        int_222679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 57), 'int')
        # Getting the type of 'False' (line 253)
        False_222680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 61), 'False', False)
        # Processing the call keyword arguments (line 253)
        kwargs_222681 = {}
        # Getting the type of 'self' (line 253)
        self_222670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 253)
        _check_222671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_222670, '_check')
        # Calling _check(args, kwargs) (line 253)
        _check_call_result_222682 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), _check_222671, *[BroydenSecond_call_result_222677, int_222678, int_222679, False_222680], **kwargs_222681)
        
        
        # Call to _check(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Call to BroydenSecond(...): (line 254)
        # Processing the call keyword arguments (line 254)
        float_222687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 47), 'float')
        keyword_222688 = float_222687
        kwargs_222689 = {'alpha': keyword_222688}
        # Getting the type of 'nonlin' (line 254)
        nonlin_222685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'nonlin', False)
        # Obtaining the member 'BroydenSecond' of a type (line 254)
        BroydenSecond_222686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), nonlin_222685, 'BroydenSecond')
        # Calling BroydenSecond(args, kwargs) (line 254)
        BroydenSecond_call_result_222690 = invoke(stypy.reporting.localization.Localization(__file__, 254, 20), BroydenSecond_222686, *[], **kwargs_222689)
        
        int_222691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 53), 'int')
        int_222692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 57), 'int')
        # Getting the type of 'True' (line 254)
        True_222693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 61), 'True', False)
        # Processing the call keyword arguments (line 254)
        kwargs_222694 = {}
        # Getting the type of 'self' (line 254)
        self_222683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 254)
        _check_222684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_222683, '_check')
        # Calling _check(args, kwargs) (line 254)
        _check_call_result_222695 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), _check_222684, *[BroydenSecond_call_result_222690, int_222691, int_222692, True_222693], **kwargs_222694)
        
        
        # ################# End of 'test_broyden2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden2' in the type store
        # Getting the type of 'stypy_return_type' (line 251)
        stypy_return_type_222696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222696)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden2'
        return stypy_return_type_222696


    @norecursion
    def test_anderson(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_anderson'
        module_type_store = module_type_store.open_function_context('test_anderson', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinear.test_anderson.__dict__.__setitem__('stypy_localization', localization)
        TestLinear.test_anderson.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinear.test_anderson.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinear.test_anderson.__dict__.__setitem__('stypy_function_name', 'TestLinear.test_anderson')
        TestLinear.test_anderson.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinear.test_anderson.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinear.test_anderson.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinear.test_anderson.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinear.test_anderson.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinear.test_anderson.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinear.test_anderson.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinear.test_anderson', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_anderson', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_anderson(...)' code ##################

        
        # Call to _check(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Call to Anderson(...): (line 258)
        # Processing the call keyword arguments (line 258)
        int_222701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 38), 'int')
        keyword_222702 = int_222701
        float_222703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 48), 'float')
        keyword_222704 = float_222703
        kwargs_222705 = {'alpha': keyword_222704, 'M': keyword_222702}
        # Getting the type of 'nonlin' (line 258)
        nonlin_222699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'nonlin', False)
        # Obtaining the member 'Anderson' of a type (line 258)
        Anderson_222700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), nonlin_222699, 'Anderson')
        # Calling Anderson(args, kwargs) (line 258)
        Anderson_call_result_222706 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), Anderson_222700, *[], **kwargs_222705)
        
        int_222707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 54), 'int')
        int_222708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 58), 'int')
        # Getting the type of 'False' (line 258)
        False_222709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 62), 'False', False)
        # Processing the call keyword arguments (line 258)
        kwargs_222710 = {}
        # Getting the type of 'self' (line 258)
        self_222697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 258)
        _check_222698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), self_222697, '_check')
        # Calling _check(args, kwargs) (line 258)
        _check_call_result_222711 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), _check_222698, *[Anderson_call_result_222706, int_222707, int_222708, False_222709], **kwargs_222710)
        
        
        # Call to _check(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Call to Anderson(...): (line 259)
        # Processing the call keyword arguments (line 259)
        int_222716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 38), 'int')
        keyword_222717 = int_222716
        float_222718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 48), 'float')
        keyword_222719 = float_222718
        kwargs_222720 = {'alpha': keyword_222719, 'M': keyword_222717}
        # Getting the type of 'nonlin' (line 259)
        nonlin_222714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'nonlin', False)
        # Obtaining the member 'Anderson' of a type (line 259)
        Anderson_222715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 20), nonlin_222714, 'Anderson')
        # Calling Anderson(args, kwargs) (line 259)
        Anderson_call_result_222721 = invoke(stypy.reporting.localization.Localization(__file__, 259, 20), Anderson_222715, *[], **kwargs_222720)
        
        int_222722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 54), 'int')
        int_222723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 58), 'int')
        # Getting the type of 'True' (line 259)
        True_222724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 62), 'True', False)
        # Processing the call keyword arguments (line 259)
        kwargs_222725 = {}
        # Getting the type of 'self' (line 259)
        self_222712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 259)
        _check_222713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_222712, '_check')
        # Calling _check(args, kwargs) (line 259)
        _check_call_result_222726 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), _check_222713, *[Anderson_call_result_222721, int_222722, int_222723, True_222724], **kwargs_222725)
        
        
        # ################# End of 'test_anderson(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_anderson' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_222727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_anderson'
        return stypy_return_type_222727


    @norecursion
    def test_krylov(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_krylov'
        module_type_store = module_type_store.open_function_context('test_krylov', 261, 4, False)
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinear.test_krylov.__dict__.__setitem__('stypy_localization', localization)
        TestLinear.test_krylov.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinear.test_krylov.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinear.test_krylov.__dict__.__setitem__('stypy_function_name', 'TestLinear.test_krylov')
        TestLinear.test_krylov.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinear.test_krylov.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinear.test_krylov.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinear.test_krylov.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinear.test_krylov.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinear.test_krylov.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinear.test_krylov.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinear.test_krylov', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_krylov', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_krylov(...)' code ##################

        
        # Call to _check(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'nonlin' (line 263)
        nonlin_222730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'nonlin', False)
        # Obtaining the member 'KrylovJacobian' of a type (line 263)
        KrylovJacobian_222731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 20), nonlin_222730, 'KrylovJacobian')
        int_222732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 43), 'int')
        int_222733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 47), 'int')
        # Getting the type of 'False' (line 263)
        False_222734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 50), 'False', False)
        # Processing the call keyword arguments (line 263)
        int_222735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 65), 'int')
        keyword_222736 = int_222735
        kwargs_222737 = {'inner_m': keyword_222736}
        # Getting the type of 'self' (line 263)
        self_222728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 263)
        _check_222729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), self_222728, '_check')
        # Calling _check(args, kwargs) (line 263)
        _check_call_result_222738 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), _check_222729, *[KrylovJacobian_222731, int_222732, int_222733, False_222734], **kwargs_222737)
        
        
        # Call to _check(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'nonlin' (line 264)
        nonlin_222741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'nonlin', False)
        # Obtaining the member 'KrylovJacobian' of a type (line 264)
        KrylovJacobian_222742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 20), nonlin_222741, 'KrylovJacobian')
        int_222743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 43), 'int')
        int_222744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 47), 'int')
        # Getting the type of 'True' (line 264)
        True_222745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 50), 'True', False)
        # Processing the call keyword arguments (line 264)
        int_222746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 64), 'int')
        keyword_222747 = int_222746
        kwargs_222748 = {'inner_m': keyword_222747}
        # Getting the type of 'self' (line 264)
        self_222739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 264)
        _check_222740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_222739, '_check')
        # Calling _check(args, kwargs) (line 264)
        _check_call_result_222749 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), _check_222740, *[KrylovJacobian_222742, int_222743, int_222744, True_222745], **kwargs_222748)
        
        
        # ################# End of 'test_krylov(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_krylov' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_222750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_krylov'
        return stypy_return_type_222750


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 225, 0, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinear.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLinear' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'TestLinear', TestLinear)
# Declaration of the 'TestJacobianDotSolve' class

class TestJacobianDotSolve(object, ):
    str_222751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'str', 'Check that solve/dot methods in Jacobian approximations are consistent')

    @norecursion
    def _func(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_func'
        module_type_store = module_type_store.open_function_context('_func', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve._func')
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve._func.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve._func', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_func', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_func(...)' code ##################

        # Getting the type of 'x' (line 271)
        x_222752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'x')
        int_222753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 18), 'int')
        # Applying the binary operator '**' (line 271)
        result_pow_222754 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 15), '**', x_222752, int_222753)
        
        int_222755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 22), 'int')
        # Applying the binary operator '-' (line 271)
        result_sub_222756 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 15), '-', result_pow_222754, int_222755)
        
        
        # Call to dot(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'self' (line 271)
        self_222759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 33), 'self', False)
        # Obtaining the member 'A' of a type (line 271)
        A_222760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 33), self_222759, 'A')
        # Getting the type of 'x' (line 271)
        x_222761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 41), 'x', False)
        # Processing the call keyword arguments (line 271)
        kwargs_222762 = {}
        # Getting the type of 'np' (line 271)
        np_222757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'np', False)
        # Obtaining the member 'dot' of a type (line 271)
        dot_222758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 26), np_222757, 'dot')
        # Calling dot(args, kwargs) (line 271)
        dot_call_result_222763 = invoke(stypy.reporting.localization.Localization(__file__, 271, 26), dot_222758, *[A_222760, x_222761], **kwargs_222762)
        
        # Applying the binary operator '+' (line 271)
        result_add_222764 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 24), '+', result_sub_222756, dot_call_result_222763)
        
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type', result_add_222764)
        
        # ################# End of '_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_func' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_222765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_222765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_func'
        return stypy_return_type_222765


    @norecursion
    def _check_dot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 273)
        False_222766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 42), 'False')
        float_222767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 53), 'float')
        defaults = [False_222766, float_222767]
        # Create a new context for function '_check_dot'
        module_type_store = module_type_store.open_function_context('_check_dot', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve._check_dot')
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_param_names_list', ['jac_cls', 'complex', 'tol'])
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve._check_dot.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve._check_dot', ['jac_cls', 'complex', 'tol'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_dot', localization, ['jac_cls', 'complex', 'tol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_dot(...)' code ##################

        
        # Call to seed(...): (line 274)
        # Processing the call arguments (line 274)
        int_222771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 23), 'int')
        # Processing the call keyword arguments (line 274)
        kwargs_222772 = {}
        # Getting the type of 'np' (line 274)
        np_222768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 274)
        random_222769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), np_222768, 'random')
        # Obtaining the member 'seed' of a type (line 274)
        seed_222770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), random_222769, 'seed')
        # Calling seed(args, kwargs) (line 274)
        seed_call_result_222773 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), seed_222770, *[int_222771], **kwargs_222772)
        
        
        # Assigning a Num to a Name (line 276):
        
        # Assigning a Num to a Name (line 276):
        int_222774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 12), 'int')
        # Assigning a type to the variable 'N' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'N', int_222774)

        @norecursion
        def rand(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'rand'
            module_type_store = module_type_store.open_function_context('rand', 278, 8, False)
            
            # Passed parameters checking function
            rand.stypy_localization = localization
            rand.stypy_type_of_self = None
            rand.stypy_type_store = module_type_store
            rand.stypy_function_name = 'rand'
            rand.stypy_param_names_list = []
            rand.stypy_varargs_param_name = 'a'
            rand.stypy_kwargs_param_name = None
            rand.stypy_call_defaults = defaults
            rand.stypy_call_varargs = varargs
            rand.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'rand', [], 'a', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'rand', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'rand(...)' code ##################

            
            # Assigning a Call to a Name (line 279):
            
            # Assigning a Call to a Name (line 279):
            
            # Call to rand(...): (line 279)
            # Getting the type of 'a' (line 279)
            a_222778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 32), 'a', False)
            # Processing the call keyword arguments (line 279)
            kwargs_222779 = {}
            # Getting the type of 'np' (line 279)
            np_222775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'np', False)
            # Obtaining the member 'random' of a type (line 279)
            random_222776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 16), np_222775, 'random')
            # Obtaining the member 'rand' of a type (line 279)
            rand_222777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 16), random_222776, 'rand')
            # Calling rand(args, kwargs) (line 279)
            rand_call_result_222780 = invoke(stypy.reporting.localization.Localization(__file__, 279, 16), rand_222777, *[a_222778], **kwargs_222779)
            
            # Assigning a type to the variable 'q' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'q', rand_call_result_222780)
            
            # Getting the type of 'complex' (line 280)
            complex_222781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'complex')
            # Testing the type of an if condition (line 280)
            if_condition_222782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), complex_222781)
            # Assigning a type to the variable 'if_condition_222782' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_222782', if_condition_222782)
            # SSA begins for if statement (line 280)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 281):
            
            # Assigning a BinOp to a Name (line 281):
            # Getting the type of 'q' (line 281)
            q_222783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'q')
            complex_222784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 24), 'complex')
            
            # Call to rand(...): (line 281)
            # Getting the type of 'a' (line 281)
            a_222788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 43), 'a', False)
            # Processing the call keyword arguments (line 281)
            kwargs_222789 = {}
            # Getting the type of 'np' (line 281)
            np_222785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'np', False)
            # Obtaining the member 'random' of a type (line 281)
            random_222786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 27), np_222785, 'random')
            # Obtaining the member 'rand' of a type (line 281)
            rand_222787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 27), random_222786, 'rand')
            # Calling rand(args, kwargs) (line 281)
            rand_call_result_222790 = invoke(stypy.reporting.localization.Localization(__file__, 281, 27), rand_222787, *[a_222788], **kwargs_222789)
            
            # Applying the binary operator '*' (line 281)
            result_mul_222791 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 24), '*', complex_222784, rand_call_result_222790)
            
            # Applying the binary operator '+' (line 281)
            result_add_222792 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), '+', q_222783, result_mul_222791)
            
            # Assigning a type to the variable 'q' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'q', result_add_222792)
            # SSA join for if statement (line 280)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'q' (line 282)
            q_222793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'q')
            # Assigning a type to the variable 'stypy_return_type' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', q_222793)
            
            # ################# End of 'rand(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'rand' in the type store
            # Getting the type of 'stypy_return_type' (line 278)
            stypy_return_type_222794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_222794)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'rand'
            return stypy_return_type_222794

        # Assigning a type to the variable 'rand' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'rand', rand)

        @norecursion
        def assert_close(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'assert_close'
            module_type_store = module_type_store.open_function_context('assert_close', 284, 8, False)
            
            # Passed parameters checking function
            assert_close.stypy_localization = localization
            assert_close.stypy_type_of_self = None
            assert_close.stypy_type_store = module_type_store
            assert_close.stypy_function_name = 'assert_close'
            assert_close.stypy_param_names_list = ['a', 'b', 'msg']
            assert_close.stypy_varargs_param_name = None
            assert_close.stypy_kwargs_param_name = None
            assert_close.stypy_call_defaults = defaults
            assert_close.stypy_call_varargs = varargs
            assert_close.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'assert_close', ['a', 'b', 'msg'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'assert_close', localization, ['a', 'b', 'msg'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'assert_close(...)' code ##################

            
            # Assigning a Call to a Name (line 285):
            
            # Assigning a Call to a Name (line 285):
            
            # Call to max(...): (line 285)
            # Processing the call keyword arguments (line 285)
            kwargs_222802 = {}
            
            # Call to abs(...): (line 285)
            # Processing the call arguments (line 285)
            # Getting the type of 'a' (line 285)
            a_222796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'a', False)
            # Getting the type of 'b' (line 285)
            b_222797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'b', False)
            # Applying the binary operator '-' (line 285)
            result_sub_222798 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 20), '-', a_222796, b_222797)
            
            # Processing the call keyword arguments (line 285)
            kwargs_222799 = {}
            # Getting the type of 'abs' (line 285)
            abs_222795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'abs', False)
            # Calling abs(args, kwargs) (line 285)
            abs_call_result_222800 = invoke(stypy.reporting.localization.Localization(__file__, 285, 16), abs_222795, *[result_sub_222798], **kwargs_222799)
            
            # Obtaining the member 'max' of a type (line 285)
            max_222801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 16), abs_call_result_222800, 'max')
            # Calling max(args, kwargs) (line 285)
            max_call_result_222803 = invoke(stypy.reporting.localization.Localization(__file__, 285, 16), max_222801, *[], **kwargs_222802)
            
            # Assigning a type to the variable 'd' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'd', max_call_result_222803)
            
            # Assigning a BinOp to a Name (line 286):
            
            # Assigning a BinOp to a Name (line 286):
            # Getting the type of 'tol' (line 286)
            tol_222804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'tol')
            
            # Call to max(...): (line 286)
            # Processing the call keyword arguments (line 286)
            kwargs_222810 = {}
            
            # Call to abs(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 'b' (line 286)
            b_222806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'b', False)
            # Processing the call keyword arguments (line 286)
            kwargs_222807 = {}
            # Getting the type of 'abs' (line 286)
            abs_222805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'abs', False)
            # Calling abs(args, kwargs) (line 286)
            abs_call_result_222808 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), abs_222805, *[b_222806], **kwargs_222807)
            
            # Obtaining the member 'max' of a type (line 286)
            max_222809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 22), abs_call_result_222808, 'max')
            # Calling max(args, kwargs) (line 286)
            max_call_result_222811 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), max_222809, *[], **kwargs_222810)
            
            # Getting the type of 'tol' (line 286)
            tol_222812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 35), 'tol')
            # Applying the binary operator '*' (line 286)
            result_mul_222813 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 22), '*', max_call_result_222811, tol_222812)
            
            # Applying the binary operator '+' (line 286)
            result_add_222814 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 16), '+', tol_222804, result_mul_222813)
            
            # Assigning a type to the variable 'f' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'f', result_add_222814)
            
            
            # Getting the type of 'd' (line 287)
            d_222815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'd')
            # Getting the type of 'f' (line 287)
            f_222816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'f')
            # Applying the binary operator '>' (line 287)
            result_gt_222817 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 15), '>', d_222815, f_222816)
            
            # Testing the type of an if condition (line 287)
            if_condition_222818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 12), result_gt_222817)
            # Assigning a type to the variable 'if_condition_222818' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'if_condition_222818', if_condition_222818)
            # SSA begins for if statement (line 287)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to AssertionError(...): (line 288)
            # Processing the call arguments (line 288)
            str_222820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 37), 'str', '%s: err %g')
            
            # Obtaining an instance of the builtin type 'tuple' (line 288)
            tuple_222821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 53), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 288)
            # Adding element type (line 288)
            # Getting the type of 'msg' (line 288)
            msg_222822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 53), 'msg', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 53), tuple_222821, msg_222822)
            # Adding element type (line 288)
            # Getting the type of 'd' (line 288)
            d_222823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 58), 'd', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 53), tuple_222821, d_222823)
            
            # Applying the binary operator '%' (line 288)
            result_mod_222824 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 37), '%', str_222820, tuple_222821)
            
            # Processing the call keyword arguments (line 288)
            kwargs_222825 = {}
            # Getting the type of 'AssertionError' (line 288)
            AssertionError_222819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'AssertionError', False)
            # Calling AssertionError(args, kwargs) (line 288)
            AssertionError_call_result_222826 = invoke(stypy.reporting.localization.Localization(__file__, 288, 22), AssertionError_222819, *[result_mod_222824], **kwargs_222825)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 288, 16), AssertionError_call_result_222826, 'raise parameter', BaseException)
            # SSA join for if statement (line 287)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'assert_close(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'assert_close' in the type store
            # Getting the type of 'stypy_return_type' (line 284)
            stypy_return_type_222827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_222827)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'assert_close'
            return stypy_return_type_222827

        # Assigning a type to the variable 'assert_close' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'assert_close', assert_close)
        
        # Assigning a Call to a Attribute (line 290):
        
        # Assigning a Call to a Attribute (line 290):
        
        # Call to rand(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'N' (line 290)
        N_222829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'N', False)
        # Getting the type of 'N' (line 290)
        N_222830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 25), 'N', False)
        # Processing the call keyword arguments (line 290)
        kwargs_222831 = {}
        # Getting the type of 'rand' (line 290)
        rand_222828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'rand', False)
        # Calling rand(args, kwargs) (line 290)
        rand_call_result_222832 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), rand_222828, *[N_222829, N_222830], **kwargs_222831)
        
        # Getting the type of 'self' (line 290)
        self_222833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'self')
        # Setting the type of the member 'A' of a type (line 290)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), self_222833, 'A', rand_call_result_222832)
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to rand(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'N' (line 293)
        N_222837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 28), 'N', False)
        # Processing the call keyword arguments (line 293)
        kwargs_222838 = {}
        # Getting the type of 'np' (line 293)
        np_222834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'np', False)
        # Obtaining the member 'random' of a type (line 293)
        random_222835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 13), np_222834, 'random')
        # Obtaining the member 'rand' of a type (line 293)
        rand_222836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 13), random_222835, 'rand')
        # Calling rand(args, kwargs) (line 293)
        rand_call_result_222839 = invoke(stypy.reporting.localization.Localization(__file__, 293, 13), rand_222836, *[N_222837], **kwargs_222838)
        
        # Assigning a type to the variable 'x0' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'x0', rand_call_result_222839)
        
        # Assigning a Call to a Name (line 294):
        
        # Assigning a Call to a Name (line 294):
        
        # Call to jac_cls(...): (line 294)
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'kw' (line 294)
        kw_222841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'kw', False)
        kwargs_222842 = {'kw_222841': kw_222841}
        # Getting the type of 'jac_cls' (line 294)
        jac_cls_222840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 14), 'jac_cls', False)
        # Calling jac_cls(args, kwargs) (line 294)
        jac_cls_call_result_222843 = invoke(stypy.reporting.localization.Localization(__file__, 294, 14), jac_cls_222840, *[], **kwargs_222842)
        
        # Assigning a type to the variable 'jac' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'jac', jac_cls_call_result_222843)
        
        # Call to setup(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'x0' (line 295)
        x0_222846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'x0', False)
        
        # Call to _func(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'x0' (line 295)
        x0_222849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 33), 'x0', False)
        # Processing the call keyword arguments (line 295)
        kwargs_222850 = {}
        # Getting the type of 'self' (line 295)
        self_222847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 22), 'self', False)
        # Obtaining the member '_func' of a type (line 295)
        _func_222848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 22), self_222847, '_func')
        # Calling _func(args, kwargs) (line 295)
        _func_call_result_222851 = invoke(stypy.reporting.localization.Localization(__file__, 295, 22), _func_222848, *[x0_222849], **kwargs_222850)
        
        # Getting the type of 'self' (line 295)
        self_222852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'self', False)
        # Obtaining the member '_func' of a type (line 295)
        _func_222853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 38), self_222852, '_func')
        # Processing the call keyword arguments (line 295)
        kwargs_222854 = {}
        # Getting the type of 'jac' (line 295)
        jac_222844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'jac', False)
        # Obtaining the member 'setup' of a type (line 295)
        setup_222845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), jac_222844, 'setup')
        # Calling setup(args, kwargs) (line 295)
        setup_call_result_222855 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), setup_222845, *[x0_222846, _func_call_result_222851, _func_222853], **kwargs_222854)
        
        
        
        # Call to xrange(...): (line 298)
        # Processing the call arguments (line 298)
        int_222857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 24), 'int')
        # Getting the type of 'N' (line 298)
        N_222858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 26), 'N', False)
        # Applying the binary operator '*' (line 298)
        result_mul_222859 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 24), '*', int_222857, N_222858)
        
        # Processing the call keyword arguments (line 298)
        kwargs_222860 = {}
        # Getting the type of 'xrange' (line 298)
        xrange_222856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 298)
        xrange_call_result_222861 = invoke(stypy.reporting.localization.Localization(__file__, 298, 17), xrange_222856, *[result_mul_222859], **kwargs_222860)
        
        # Testing the type of a for loop iterable (line 298)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 298, 8), xrange_call_result_222861)
        # Getting the type of the for loop variable (line 298)
        for_loop_var_222862 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 298, 8), xrange_call_result_222861)
        # Assigning a type to the variable 'k' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'k', for_loop_var_222862)
        # SSA begins for a for statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to rand(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'N' (line 299)
        N_222864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 21), 'N', False)
        # Processing the call keyword arguments (line 299)
        kwargs_222865 = {}
        # Getting the type of 'rand' (line 299)
        rand_222863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'rand', False)
        # Calling rand(args, kwargs) (line 299)
        rand_call_result_222866 = invoke(stypy.reporting.localization.Localization(__file__, 299, 16), rand_222863, *[N_222864], **kwargs_222865)
        
        # Assigning a type to the variable 'v' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'v', rand_call_result_222866)
        
        # Type idiom detected: calculating its left and rigth part (line 301)
        str_222867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 28), 'str', '__array__')
        # Getting the type of 'jac' (line 301)
        jac_222868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'jac')
        
        (may_be_222869, more_types_in_union_222870) = may_provide_member(str_222867, jac_222868)

        if may_be_222869:

            if more_types_in_union_222870:
                # Runtime conditional SSA (line 301)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'jac' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'jac', remove_not_member_provider_from_union(jac_222868, '__array__'))
            
            # Assigning a Call to a Name (line 302):
            
            # Assigning a Call to a Name (line 302):
            
            # Call to array(...): (line 302)
            # Processing the call arguments (line 302)
            # Getting the type of 'jac' (line 302)
            jac_222873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 30), 'jac', False)
            # Processing the call keyword arguments (line 302)
            kwargs_222874 = {}
            # Getting the type of 'np' (line 302)
            np_222871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 21), 'np', False)
            # Obtaining the member 'array' of a type (line 302)
            array_222872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 21), np_222871, 'array')
            # Calling array(args, kwargs) (line 302)
            array_call_result_222875 = invoke(stypy.reporting.localization.Localization(__file__, 302, 21), array_222872, *[jac_222873], **kwargs_222874)
            
            # Assigning a type to the variable 'Jd' (line 302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'Jd', array_call_result_222875)
            
            # Type idiom detected: calculating its left and rigth part (line 303)
            str_222876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 32), 'str', 'solve')
            # Getting the type of 'jac' (line 303)
            jac_222877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 27), 'jac')
            
            (may_be_222878, more_types_in_union_222879) = may_provide_member(str_222876, jac_222877)

            if may_be_222878:

                if more_types_in_union_222879:
                    # Runtime conditional SSA (line 303)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'jac' (line 303)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'jac', remove_not_member_provider_from_union(jac_222877, 'solve'))
                
                # Assigning a Call to a Name (line 304):
                
                # Assigning a Call to a Name (line 304):
                
                # Call to solve(...): (line 304)
                # Processing the call arguments (line 304)
                # Getting the type of 'v' (line 304)
                v_222882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 35), 'v', False)
                # Processing the call keyword arguments (line 304)
                kwargs_222883 = {}
                # Getting the type of 'jac' (line 304)
                jac_222880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'jac', False)
                # Obtaining the member 'solve' of a type (line 304)
                solve_222881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 25), jac_222880, 'solve')
                # Calling solve(args, kwargs) (line 304)
                solve_call_result_222884 = invoke(stypy.reporting.localization.Localization(__file__, 304, 25), solve_222881, *[v_222882], **kwargs_222883)
                
                # Assigning a type to the variable 'Gv' (line 304)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'Gv', solve_call_result_222884)
                
                # Assigning a Call to a Name (line 305):
                
                # Assigning a Call to a Name (line 305):
                
                # Call to solve(...): (line 305)
                # Processing the call arguments (line 305)
                # Getting the type of 'Jd' (line 305)
                Jd_222888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 42), 'Jd', False)
                # Getting the type of 'v' (line 305)
                v_222889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 46), 'v', False)
                # Processing the call keyword arguments (line 305)
                kwargs_222890 = {}
                # Getting the type of 'np' (line 305)
                np_222885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 26), 'np', False)
                # Obtaining the member 'linalg' of a type (line 305)
                linalg_222886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 26), np_222885, 'linalg')
                # Obtaining the member 'solve' of a type (line 305)
                solve_222887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 26), linalg_222886, 'solve')
                # Calling solve(args, kwargs) (line 305)
                solve_call_result_222891 = invoke(stypy.reporting.localization.Localization(__file__, 305, 26), solve_222887, *[Jd_222888, v_222889], **kwargs_222890)
                
                # Assigning a type to the variable 'Gv2' (line 305)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'Gv2', solve_call_result_222891)
                
                # Call to assert_close(...): (line 306)
                # Processing the call arguments (line 306)
                # Getting the type of 'Gv' (line 306)
                Gv_222893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 33), 'Gv', False)
                # Getting the type of 'Gv2' (line 306)
                Gv2_222894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 37), 'Gv2', False)
                str_222895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 42), 'str', 'solve vs array')
                # Processing the call keyword arguments (line 306)
                kwargs_222896 = {}
                # Getting the type of 'assert_close' (line 306)
                assert_close_222892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'assert_close', False)
                # Calling assert_close(args, kwargs) (line 306)
                assert_close_call_result_222897 = invoke(stypy.reporting.localization.Localization(__file__, 306, 20), assert_close_222892, *[Gv_222893, Gv2_222894, str_222895], **kwargs_222896)
                

                if more_types_in_union_222879:
                    # SSA join for if statement (line 303)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 307)
            str_222898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 32), 'str', 'rsolve')
            # Getting the type of 'jac' (line 307)
            jac_222899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 27), 'jac')
            
            (may_be_222900, more_types_in_union_222901) = may_provide_member(str_222898, jac_222899)

            if may_be_222900:

                if more_types_in_union_222901:
                    # Runtime conditional SSA (line 307)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'jac' (line 307)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'jac', remove_not_member_provider_from_union(jac_222899, 'rsolve'))
                
                # Assigning a Call to a Name (line 308):
                
                # Assigning a Call to a Name (line 308):
                
                # Call to rsolve(...): (line 308)
                # Processing the call arguments (line 308)
                # Getting the type of 'v' (line 308)
                v_222904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 36), 'v', False)
                # Processing the call keyword arguments (line 308)
                kwargs_222905 = {}
                # Getting the type of 'jac' (line 308)
                jac_222902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'jac', False)
                # Obtaining the member 'rsolve' of a type (line 308)
                rsolve_222903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 25), jac_222902, 'rsolve')
                # Calling rsolve(args, kwargs) (line 308)
                rsolve_call_result_222906 = invoke(stypy.reporting.localization.Localization(__file__, 308, 25), rsolve_222903, *[v_222904], **kwargs_222905)
                
                # Assigning a type to the variable 'Gv' (line 308)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'Gv', rsolve_call_result_222906)
                
                # Assigning a Call to a Name (line 309):
                
                # Assigning a Call to a Name (line 309):
                
                # Call to solve(...): (line 309)
                # Processing the call arguments (line 309)
                
                # Call to conj(...): (line 309)
                # Processing the call keyword arguments (line 309)
                kwargs_222913 = {}
                # Getting the type of 'Jd' (line 309)
                Jd_222910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 42), 'Jd', False)
                # Obtaining the member 'T' of a type (line 309)
                T_222911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 42), Jd_222910, 'T')
                # Obtaining the member 'conj' of a type (line 309)
                conj_222912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 42), T_222911, 'conj')
                # Calling conj(args, kwargs) (line 309)
                conj_call_result_222914 = invoke(stypy.reporting.localization.Localization(__file__, 309, 42), conj_222912, *[], **kwargs_222913)
                
                # Getting the type of 'v' (line 309)
                v_222915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 55), 'v', False)
                # Processing the call keyword arguments (line 309)
                kwargs_222916 = {}
                # Getting the type of 'np' (line 309)
                np_222907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 26), 'np', False)
                # Obtaining the member 'linalg' of a type (line 309)
                linalg_222908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 26), np_222907, 'linalg')
                # Obtaining the member 'solve' of a type (line 309)
                solve_222909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 26), linalg_222908, 'solve')
                # Calling solve(args, kwargs) (line 309)
                solve_call_result_222917 = invoke(stypy.reporting.localization.Localization(__file__, 309, 26), solve_222909, *[conj_call_result_222914, v_222915], **kwargs_222916)
                
                # Assigning a type to the variable 'Gv2' (line 309)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 20), 'Gv2', solve_call_result_222917)
                
                # Call to assert_close(...): (line 310)
                # Processing the call arguments (line 310)
                # Getting the type of 'Gv' (line 310)
                Gv_222919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 33), 'Gv', False)
                # Getting the type of 'Gv2' (line 310)
                Gv2_222920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 37), 'Gv2', False)
                str_222921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 42), 'str', 'rsolve vs array')
                # Processing the call keyword arguments (line 310)
                kwargs_222922 = {}
                # Getting the type of 'assert_close' (line 310)
                assert_close_222918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'assert_close', False)
                # Calling assert_close(args, kwargs) (line 310)
                assert_close_call_result_222923 = invoke(stypy.reporting.localization.Localization(__file__, 310, 20), assert_close_222918, *[Gv_222919, Gv2_222920, str_222921], **kwargs_222922)
                

                if more_types_in_union_222901:
                    # SSA join for if statement (line 307)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 311)
            str_222924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 32), 'str', 'matvec')
            # Getting the type of 'jac' (line 311)
            jac_222925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), 'jac')
            
            (may_be_222926, more_types_in_union_222927) = may_provide_member(str_222924, jac_222925)

            if may_be_222926:

                if more_types_in_union_222927:
                    # Runtime conditional SSA (line 311)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'jac' (line 311)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'jac', remove_not_member_provider_from_union(jac_222925, 'matvec'))
                
                # Assigning a Call to a Name (line 312):
                
                # Assigning a Call to a Name (line 312):
                
                # Call to matvec(...): (line 312)
                # Processing the call arguments (line 312)
                # Getting the type of 'v' (line 312)
                v_222930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 36), 'v', False)
                # Processing the call keyword arguments (line 312)
                kwargs_222931 = {}
                # Getting the type of 'jac' (line 312)
                jac_222928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'jac', False)
                # Obtaining the member 'matvec' of a type (line 312)
                matvec_222929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 25), jac_222928, 'matvec')
                # Calling matvec(args, kwargs) (line 312)
                matvec_call_result_222932 = invoke(stypy.reporting.localization.Localization(__file__, 312, 25), matvec_222929, *[v_222930], **kwargs_222931)
                
                # Assigning a type to the variable 'Jv' (line 312)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'Jv', matvec_call_result_222932)
                
                # Assigning a Call to a Name (line 313):
                
                # Assigning a Call to a Name (line 313):
                
                # Call to dot(...): (line 313)
                # Processing the call arguments (line 313)
                # Getting the type of 'Jd' (line 313)
                Jd_222935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 33), 'Jd', False)
                # Getting the type of 'v' (line 313)
                v_222936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 37), 'v', False)
                # Processing the call keyword arguments (line 313)
                kwargs_222937 = {}
                # Getting the type of 'np' (line 313)
                np_222933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'np', False)
                # Obtaining the member 'dot' of a type (line 313)
                dot_222934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 26), np_222933, 'dot')
                # Calling dot(args, kwargs) (line 313)
                dot_call_result_222938 = invoke(stypy.reporting.localization.Localization(__file__, 313, 26), dot_222934, *[Jd_222935, v_222936], **kwargs_222937)
                
                # Assigning a type to the variable 'Jv2' (line 313)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 20), 'Jv2', dot_call_result_222938)
                
                # Call to assert_close(...): (line 314)
                # Processing the call arguments (line 314)
                # Getting the type of 'Jv' (line 314)
                Jv_222940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 33), 'Jv', False)
                # Getting the type of 'Jv2' (line 314)
                Jv2_222941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 37), 'Jv2', False)
                str_222942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 42), 'str', 'dot vs array')
                # Processing the call keyword arguments (line 314)
                kwargs_222943 = {}
                # Getting the type of 'assert_close' (line 314)
                assert_close_222939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'assert_close', False)
                # Calling assert_close(args, kwargs) (line 314)
                assert_close_call_result_222944 = invoke(stypy.reporting.localization.Localization(__file__, 314, 20), assert_close_222939, *[Jv_222940, Jv2_222941, str_222942], **kwargs_222943)
                

                if more_types_in_union_222927:
                    # SSA join for if statement (line 311)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 315)
            str_222945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 32), 'str', 'rmatvec')
            # Getting the type of 'jac' (line 315)
            jac_222946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 27), 'jac')
            
            (may_be_222947, more_types_in_union_222948) = may_provide_member(str_222945, jac_222946)

            if may_be_222947:

                if more_types_in_union_222948:
                    # Runtime conditional SSA (line 315)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'jac' (line 315)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'jac', remove_not_member_provider_from_union(jac_222946, 'rmatvec'))
                
                # Assigning a Call to a Name (line 316):
                
                # Assigning a Call to a Name (line 316):
                
                # Call to rmatvec(...): (line 316)
                # Processing the call arguments (line 316)
                # Getting the type of 'v' (line 316)
                v_222951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 37), 'v', False)
                # Processing the call keyword arguments (line 316)
                kwargs_222952 = {}
                # Getting the type of 'jac' (line 316)
                jac_222949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 25), 'jac', False)
                # Obtaining the member 'rmatvec' of a type (line 316)
                rmatvec_222950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 25), jac_222949, 'rmatvec')
                # Calling rmatvec(args, kwargs) (line 316)
                rmatvec_call_result_222953 = invoke(stypy.reporting.localization.Localization(__file__, 316, 25), rmatvec_222950, *[v_222951], **kwargs_222952)
                
                # Assigning a type to the variable 'Jv' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'Jv', rmatvec_call_result_222953)
                
                # Assigning a Call to a Name (line 317):
                
                # Assigning a Call to a Name (line 317):
                
                # Call to dot(...): (line 317)
                # Processing the call arguments (line 317)
                
                # Call to conj(...): (line 317)
                # Processing the call keyword arguments (line 317)
                kwargs_222959 = {}
                # Getting the type of 'Jd' (line 317)
                Jd_222956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'Jd', False)
                # Obtaining the member 'T' of a type (line 317)
                T_222957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 33), Jd_222956, 'T')
                # Obtaining the member 'conj' of a type (line 317)
                conj_222958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 33), T_222957, 'conj')
                # Calling conj(args, kwargs) (line 317)
                conj_call_result_222960 = invoke(stypy.reporting.localization.Localization(__file__, 317, 33), conj_222958, *[], **kwargs_222959)
                
                # Getting the type of 'v' (line 317)
                v_222961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 46), 'v', False)
                # Processing the call keyword arguments (line 317)
                kwargs_222962 = {}
                # Getting the type of 'np' (line 317)
                np_222954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 26), 'np', False)
                # Obtaining the member 'dot' of a type (line 317)
                dot_222955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 26), np_222954, 'dot')
                # Calling dot(args, kwargs) (line 317)
                dot_call_result_222963 = invoke(stypy.reporting.localization.Localization(__file__, 317, 26), dot_222955, *[conj_call_result_222960, v_222961], **kwargs_222962)
                
                # Assigning a type to the variable 'Jv2' (line 317)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'Jv2', dot_call_result_222963)
                
                # Call to assert_close(...): (line 318)
                # Processing the call arguments (line 318)
                # Getting the type of 'Jv' (line 318)
                Jv_222965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 33), 'Jv', False)
                # Getting the type of 'Jv2' (line 318)
                Jv2_222966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 37), 'Jv2', False)
                str_222967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 42), 'str', 'rmatvec vs array')
                # Processing the call keyword arguments (line 318)
                kwargs_222968 = {}
                # Getting the type of 'assert_close' (line 318)
                assert_close_222964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'assert_close', False)
                # Calling assert_close(args, kwargs) (line 318)
                assert_close_call_result_222969 = invoke(stypy.reporting.localization.Localization(__file__, 318, 20), assert_close_222964, *[Jv_222965, Jv2_222966, str_222967], **kwargs_222968)
                

                if more_types_in_union_222948:
                    # SSA join for if statement (line 315)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_222870:
                # SSA join for if statement (line 301)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'jac' (line 320)
        jac_222971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'jac', False)
        str_222972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 28), 'str', 'matvec')
        # Processing the call keyword arguments (line 320)
        kwargs_222973 = {}
        # Getting the type of 'hasattr' (line 320)
        hasattr_222970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 320)
        hasattr_call_result_222974 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), hasattr_222970, *[jac_222971, str_222972], **kwargs_222973)
        
        
        # Call to hasattr(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'jac' (line 320)
        jac_222976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 50), 'jac', False)
        str_222977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 55), 'str', 'solve')
        # Processing the call keyword arguments (line 320)
        kwargs_222978 = {}
        # Getting the type of 'hasattr' (line 320)
        hasattr_222975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 42), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 320)
        hasattr_call_result_222979 = invoke(stypy.reporting.localization.Localization(__file__, 320, 42), hasattr_222975, *[jac_222976, str_222977], **kwargs_222978)
        
        # Applying the binary operator 'and' (line 320)
        result_and_keyword_222980 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 15), 'and', hasattr_call_result_222974, hasattr_call_result_222979)
        
        # Testing the type of an if condition (line 320)
        if_condition_222981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 12), result_and_keyword_222980)
        # Assigning a type to the variable 'if_condition_222981' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'if_condition_222981', if_condition_222981)
        # SSA begins for if statement (line 320)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to matvec(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'v' (line 321)
        v_222984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 32), 'v', False)
        # Processing the call keyword arguments (line 321)
        kwargs_222985 = {}
        # Getting the type of 'jac' (line 321)
        jac_222982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 21), 'jac', False)
        # Obtaining the member 'matvec' of a type (line 321)
        matvec_222983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 21), jac_222982, 'matvec')
        # Calling matvec(args, kwargs) (line 321)
        matvec_call_result_222986 = invoke(stypy.reporting.localization.Localization(__file__, 321, 21), matvec_222983, *[v_222984], **kwargs_222985)
        
        # Assigning a type to the variable 'Jv' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'Jv', matvec_call_result_222986)
        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to solve(...): (line 322)
        # Processing the call arguments (line 322)
        
        # Call to matvec(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'Jv' (line 322)
        Jv_222991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 43), 'Jv', False)
        # Processing the call keyword arguments (line 322)
        kwargs_222992 = {}
        # Getting the type of 'jac' (line 322)
        jac_222989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 32), 'jac', False)
        # Obtaining the member 'matvec' of a type (line 322)
        matvec_222990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 32), jac_222989, 'matvec')
        # Calling matvec(args, kwargs) (line 322)
        matvec_call_result_222993 = invoke(stypy.reporting.localization.Localization(__file__, 322, 32), matvec_222990, *[Jv_222991], **kwargs_222992)
        
        # Processing the call keyword arguments (line 322)
        kwargs_222994 = {}
        # Getting the type of 'jac' (line 322)
        jac_222987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 22), 'jac', False)
        # Obtaining the member 'solve' of a type (line 322)
        solve_222988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 22), jac_222987, 'solve')
        # Calling solve(args, kwargs) (line 322)
        solve_call_result_222995 = invoke(stypy.reporting.localization.Localization(__file__, 322, 22), solve_222988, *[matvec_call_result_222993], **kwargs_222994)
        
        # Assigning a type to the variable 'Jv2' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 16), 'Jv2', solve_call_result_222995)
        
        # Call to assert_close(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'Jv' (line 323)
        Jv_222997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 29), 'Jv', False)
        # Getting the type of 'Jv2' (line 323)
        Jv2_222998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 33), 'Jv2', False)
        str_222999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 38), 'str', 'dot vs solve')
        # Processing the call keyword arguments (line 323)
        kwargs_223000 = {}
        # Getting the type of 'assert_close' (line 323)
        assert_close_222996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'assert_close', False)
        # Calling assert_close(args, kwargs) (line 323)
        assert_close_call_result_223001 = invoke(stypy.reporting.localization.Localization(__file__, 323, 16), assert_close_222996, *[Jv_222997, Jv2_222998, str_222999], **kwargs_223000)
        
        # SSA join for if statement (line 320)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'jac' (line 325)
        jac_223003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 23), 'jac', False)
        str_223004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'str', 'rmatvec')
        # Processing the call keyword arguments (line 325)
        kwargs_223005 = {}
        # Getting the type of 'hasattr' (line 325)
        hasattr_223002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 325)
        hasattr_call_result_223006 = invoke(stypy.reporting.localization.Localization(__file__, 325, 15), hasattr_223002, *[jac_223003, str_223004], **kwargs_223005)
        
        
        # Call to hasattr(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'jac' (line 325)
        jac_223008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 51), 'jac', False)
        str_223009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 56), 'str', 'rsolve')
        # Processing the call keyword arguments (line 325)
        kwargs_223010 = {}
        # Getting the type of 'hasattr' (line 325)
        hasattr_223007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 43), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 325)
        hasattr_call_result_223011 = invoke(stypy.reporting.localization.Localization(__file__, 325, 43), hasattr_223007, *[jac_223008, str_223009], **kwargs_223010)
        
        # Applying the binary operator 'and' (line 325)
        result_and_keyword_223012 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 15), 'and', hasattr_call_result_223006, hasattr_call_result_223011)
        
        # Testing the type of an if condition (line 325)
        if_condition_223013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 12), result_and_keyword_223012)
        # Assigning a type to the variable 'if_condition_223013' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'if_condition_223013', if_condition_223013)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to rmatvec(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'v' (line 326)
        v_223016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 33), 'v', False)
        # Processing the call keyword arguments (line 326)
        kwargs_223017 = {}
        # Getting the type of 'jac' (line 326)
        jac_223014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 21), 'jac', False)
        # Obtaining the member 'rmatvec' of a type (line 326)
        rmatvec_223015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 21), jac_223014, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 326)
        rmatvec_call_result_223018 = invoke(stypy.reporting.localization.Localization(__file__, 326, 21), rmatvec_223015, *[v_223016], **kwargs_223017)
        
        # Assigning a type to the variable 'Jv' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'Jv', rmatvec_call_result_223018)
        
        # Assigning a Call to a Name (line 327):
        
        # Assigning a Call to a Name (line 327):
        
        # Call to rmatvec(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Call to rsolve(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'Jv' (line 327)
        Jv_223023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 45), 'Jv', False)
        # Processing the call keyword arguments (line 327)
        kwargs_223024 = {}
        # Getting the type of 'jac' (line 327)
        jac_223021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'jac', False)
        # Obtaining the member 'rsolve' of a type (line 327)
        rsolve_223022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 34), jac_223021, 'rsolve')
        # Calling rsolve(args, kwargs) (line 327)
        rsolve_call_result_223025 = invoke(stypy.reporting.localization.Localization(__file__, 327, 34), rsolve_223022, *[Jv_223023], **kwargs_223024)
        
        # Processing the call keyword arguments (line 327)
        kwargs_223026 = {}
        # Getting the type of 'jac' (line 327)
        jac_223019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 22), 'jac', False)
        # Obtaining the member 'rmatvec' of a type (line 327)
        rmatvec_223020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 22), jac_223019, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 327)
        rmatvec_call_result_223027 = invoke(stypy.reporting.localization.Localization(__file__, 327, 22), rmatvec_223020, *[rsolve_call_result_223025], **kwargs_223026)
        
        # Assigning a type to the variable 'Jv2' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'Jv2', rmatvec_call_result_223027)
        
        # Call to assert_close(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'Jv' (line 328)
        Jv_223029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 29), 'Jv', False)
        # Getting the type of 'Jv2' (line 328)
        Jv2_223030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 33), 'Jv2', False)
        str_223031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 38), 'str', 'rmatvec vs rsolve')
        # Processing the call keyword arguments (line 328)
        kwargs_223032 = {}
        # Getting the type of 'assert_close' (line 328)
        assert_close_223028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'assert_close', False)
        # Calling assert_close(args, kwargs) (line 328)
        assert_close_call_result_223033 = invoke(stypy.reporting.localization.Localization(__file__, 328, 16), assert_close_223028, *[Jv_223029, Jv2_223030, str_223031], **kwargs_223032)
        
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 330):
        
        # Assigning a Call to a Name (line 330):
        
        # Call to rand(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'N' (line 330)
        N_223035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 21), 'N', False)
        # Processing the call keyword arguments (line 330)
        kwargs_223036 = {}
        # Getting the type of 'rand' (line 330)
        rand_223034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'rand', False)
        # Calling rand(args, kwargs) (line 330)
        rand_call_result_223037 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), rand_223034, *[N_223035], **kwargs_223036)
        
        # Assigning a type to the variable 'x' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'x', rand_call_result_223037)
        
        # Call to update(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'x' (line 331)
        x_223040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 23), 'x', False)
        
        # Call to _func(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'x' (line 331)
        x_223043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 37), 'x', False)
        # Processing the call keyword arguments (line 331)
        kwargs_223044 = {}
        # Getting the type of 'self' (line 331)
        self_223041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'self', False)
        # Obtaining the member '_func' of a type (line 331)
        _func_223042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 26), self_223041, '_func')
        # Calling _func(args, kwargs) (line 331)
        _func_call_result_223045 = invoke(stypy.reporting.localization.Localization(__file__, 331, 26), _func_223042, *[x_223043], **kwargs_223044)
        
        # Processing the call keyword arguments (line 331)
        kwargs_223046 = {}
        # Getting the type of 'jac' (line 331)
        jac_223038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'jac', False)
        # Obtaining the member 'update' of a type (line 331)
        update_223039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), jac_223038, 'update')
        # Calling update(args, kwargs) (line 331)
        update_call_result_223047 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), update_223039, *[x_223040, _func_call_result_223045], **kwargs_223046)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check_dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_dot' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_223048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223048)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_dot'
        return stypy_return_type_223048


    @norecursion
    def test_broyden1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden1'
        module_type_store = module_type_store.open_function_context('test_broyden1', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve.test_broyden1')
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_param_names_list', [])
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve.test_broyden1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve.test_broyden1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden1(...)' code ##################

        
        # Call to _check_dot(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'nonlin' (line 334)
        nonlin_223051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'nonlin', False)
        # Obtaining the member 'BroydenFirst' of a type (line 334)
        BroydenFirst_223052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 24), nonlin_223051, 'BroydenFirst')
        # Processing the call keyword arguments (line 334)
        # Getting the type of 'False' (line 334)
        False_223053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 53), 'False', False)
        keyword_223054 = False_223053
        kwargs_223055 = {'complex': keyword_223054}
        # Getting the type of 'self' (line 334)
        self_223049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 334)
        _check_dot_223050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), self_223049, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 334)
        _check_dot_call_result_223056 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), _check_dot_223050, *[BroydenFirst_223052], **kwargs_223055)
        
        
        # Call to _check_dot(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'nonlin' (line 335)
        nonlin_223059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), 'nonlin', False)
        # Obtaining the member 'BroydenFirst' of a type (line 335)
        BroydenFirst_223060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 24), nonlin_223059, 'BroydenFirst')
        # Processing the call keyword arguments (line 335)
        # Getting the type of 'True' (line 335)
        True_223061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 53), 'True', False)
        keyword_223062 = True_223061
        kwargs_223063 = {'complex': keyword_223062}
        # Getting the type of 'self' (line 335)
        self_223057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 335)
        _check_dot_223058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), self_223057, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 335)
        _check_dot_call_result_223064 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), _check_dot_223058, *[BroydenFirst_223060], **kwargs_223063)
        
        
        # ################# End of 'test_broyden1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden1' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_223065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223065)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden1'
        return stypy_return_type_223065


    @norecursion
    def test_broyden2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden2'
        module_type_store = module_type_store.open_function_context('test_broyden2', 337, 4, False)
        # Assigning a type to the variable 'self' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve.test_broyden2')
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_param_names_list', [])
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve.test_broyden2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve.test_broyden2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden2(...)' code ##################

        
        # Call to _check_dot(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'nonlin' (line 338)
        nonlin_223068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'nonlin', False)
        # Obtaining the member 'BroydenSecond' of a type (line 338)
        BroydenSecond_223069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 24), nonlin_223068, 'BroydenSecond')
        # Processing the call keyword arguments (line 338)
        # Getting the type of 'False' (line 338)
        False_223070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 54), 'False', False)
        keyword_223071 = False_223070
        kwargs_223072 = {'complex': keyword_223071}
        # Getting the type of 'self' (line 338)
        self_223066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 338)
        _check_dot_223067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), self_223066, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 338)
        _check_dot_call_result_223073 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), _check_dot_223067, *[BroydenSecond_223069], **kwargs_223072)
        
        
        # Call to _check_dot(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'nonlin' (line 339)
        nonlin_223076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'nonlin', False)
        # Obtaining the member 'BroydenSecond' of a type (line 339)
        BroydenSecond_223077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 24), nonlin_223076, 'BroydenSecond')
        # Processing the call keyword arguments (line 339)
        # Getting the type of 'True' (line 339)
        True_223078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), 'True', False)
        keyword_223079 = True_223078
        kwargs_223080 = {'complex': keyword_223079}
        # Getting the type of 'self' (line 339)
        self_223074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 339)
        _check_dot_223075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), self_223074, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 339)
        _check_dot_call_result_223081 = invoke(stypy.reporting.localization.Localization(__file__, 339, 8), _check_dot_223075, *[BroydenSecond_223077], **kwargs_223080)
        
        
        # ################# End of 'test_broyden2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden2' in the type store
        # Getting the type of 'stypy_return_type' (line 337)
        stypy_return_type_223082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223082)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden2'
        return stypy_return_type_223082


    @norecursion
    def test_anderson(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_anderson'
        module_type_store = module_type_store.open_function_context('test_anderson', 341, 4, False)
        # Assigning a type to the variable 'self' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve.test_anderson')
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_param_names_list', [])
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve.test_anderson.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve.test_anderson', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_anderson', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_anderson(...)' code ##################

        
        # Call to _check_dot(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'nonlin' (line 342)
        nonlin_223085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 24), 'nonlin', False)
        # Obtaining the member 'Anderson' of a type (line 342)
        Anderson_223086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 24), nonlin_223085, 'Anderson')
        # Processing the call keyword arguments (line 342)
        # Getting the type of 'False' (line 342)
        False_223087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 49), 'False', False)
        keyword_223088 = False_223087
        kwargs_223089 = {'complex': keyword_223088}
        # Getting the type of 'self' (line 342)
        self_223083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 342)
        _check_dot_223084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_223083, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 342)
        _check_dot_call_result_223090 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), _check_dot_223084, *[Anderson_223086], **kwargs_223089)
        
        
        # Call to _check_dot(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'nonlin' (line 343)
        nonlin_223093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'nonlin', False)
        # Obtaining the member 'Anderson' of a type (line 343)
        Anderson_223094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 24), nonlin_223093, 'Anderson')
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'True' (line 343)
        True_223095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 49), 'True', False)
        keyword_223096 = True_223095
        kwargs_223097 = {'complex': keyword_223096}
        # Getting the type of 'self' (line 343)
        self_223091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 343)
        _check_dot_223092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_223091, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 343)
        _check_dot_call_result_223098 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), _check_dot_223092, *[Anderson_223094], **kwargs_223097)
        
        
        # ################# End of 'test_anderson(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_anderson' in the type store
        # Getting the type of 'stypy_return_type' (line 341)
        stypy_return_type_223099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223099)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_anderson'
        return stypy_return_type_223099


    @norecursion
    def test_diagbroyden(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_diagbroyden'
        module_type_store = module_type_store.open_function_context('test_diagbroyden', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve.test_diagbroyden')
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_param_names_list', [])
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve.test_diagbroyden.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve.test_diagbroyden', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_diagbroyden', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_diagbroyden(...)' code ##################

        
        # Call to _check_dot(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'nonlin' (line 346)
        nonlin_223102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 24), 'nonlin', False)
        # Obtaining the member 'DiagBroyden' of a type (line 346)
        DiagBroyden_223103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 24), nonlin_223102, 'DiagBroyden')
        # Processing the call keyword arguments (line 346)
        # Getting the type of 'False' (line 346)
        False_223104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 52), 'False', False)
        keyword_223105 = False_223104
        kwargs_223106 = {'complex': keyword_223105}
        # Getting the type of 'self' (line 346)
        self_223100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 346)
        _check_dot_223101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), self_223100, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 346)
        _check_dot_call_result_223107 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), _check_dot_223101, *[DiagBroyden_223103], **kwargs_223106)
        
        
        # Call to _check_dot(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'nonlin' (line 347)
        nonlin_223110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'nonlin', False)
        # Obtaining the member 'DiagBroyden' of a type (line 347)
        DiagBroyden_223111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 24), nonlin_223110, 'DiagBroyden')
        # Processing the call keyword arguments (line 347)
        # Getting the type of 'True' (line 347)
        True_223112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 52), 'True', False)
        keyword_223113 = True_223112
        kwargs_223114 = {'complex': keyword_223113}
        # Getting the type of 'self' (line 347)
        self_223108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 347)
        _check_dot_223109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_223108, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 347)
        _check_dot_call_result_223115 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), _check_dot_223109, *[DiagBroyden_223111], **kwargs_223114)
        
        
        # ################# End of 'test_diagbroyden(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_diagbroyden' in the type store
        # Getting the type of 'stypy_return_type' (line 345)
        stypy_return_type_223116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_diagbroyden'
        return stypy_return_type_223116


    @norecursion
    def test_linearmixing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linearmixing'
        module_type_store = module_type_store.open_function_context('test_linearmixing', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve.test_linearmixing')
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_param_names_list', [])
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve.test_linearmixing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve.test_linearmixing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linearmixing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linearmixing(...)' code ##################

        
        # Call to _check_dot(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'nonlin' (line 350)
        nonlin_223119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 24), 'nonlin', False)
        # Obtaining the member 'LinearMixing' of a type (line 350)
        LinearMixing_223120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 24), nonlin_223119, 'LinearMixing')
        # Processing the call keyword arguments (line 350)
        # Getting the type of 'False' (line 350)
        False_223121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 53), 'False', False)
        keyword_223122 = False_223121
        kwargs_223123 = {'complex': keyword_223122}
        # Getting the type of 'self' (line 350)
        self_223117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 350)
        _check_dot_223118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_223117, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 350)
        _check_dot_call_result_223124 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), _check_dot_223118, *[LinearMixing_223120], **kwargs_223123)
        
        
        # Call to _check_dot(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'nonlin' (line 351)
        nonlin_223127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'nonlin', False)
        # Obtaining the member 'LinearMixing' of a type (line 351)
        LinearMixing_223128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 24), nonlin_223127, 'LinearMixing')
        # Processing the call keyword arguments (line 351)
        # Getting the type of 'True' (line 351)
        True_223129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 53), 'True', False)
        keyword_223130 = True_223129
        kwargs_223131 = {'complex': keyword_223130}
        # Getting the type of 'self' (line 351)
        self_223125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 351)
        _check_dot_223126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), self_223125, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 351)
        _check_dot_call_result_223132 = invoke(stypy.reporting.localization.Localization(__file__, 351, 8), _check_dot_223126, *[LinearMixing_223128], **kwargs_223131)
        
        
        # ################# End of 'test_linearmixing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linearmixing' in the type store
        # Getting the type of 'stypy_return_type' (line 349)
        stypy_return_type_223133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223133)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linearmixing'
        return stypy_return_type_223133


    @norecursion
    def test_excitingmixing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_excitingmixing'
        module_type_store = module_type_store.open_function_context('test_excitingmixing', 353, 4, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve.test_excitingmixing')
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_param_names_list', [])
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve.test_excitingmixing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve.test_excitingmixing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_excitingmixing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_excitingmixing(...)' code ##################

        
        # Call to _check_dot(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'nonlin' (line 354)
        nonlin_223136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'nonlin', False)
        # Obtaining the member 'ExcitingMixing' of a type (line 354)
        ExcitingMixing_223137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 24), nonlin_223136, 'ExcitingMixing')
        # Processing the call keyword arguments (line 354)
        # Getting the type of 'False' (line 354)
        False_223138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 55), 'False', False)
        keyword_223139 = False_223138
        kwargs_223140 = {'complex': keyword_223139}
        # Getting the type of 'self' (line 354)
        self_223134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 354)
        _check_dot_223135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), self_223134, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 354)
        _check_dot_call_result_223141 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), _check_dot_223135, *[ExcitingMixing_223137], **kwargs_223140)
        
        
        # Call to _check_dot(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'nonlin' (line 355)
        nonlin_223144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'nonlin', False)
        # Obtaining the member 'ExcitingMixing' of a type (line 355)
        ExcitingMixing_223145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 24), nonlin_223144, 'ExcitingMixing')
        # Processing the call keyword arguments (line 355)
        # Getting the type of 'True' (line 355)
        True_223146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 55), 'True', False)
        keyword_223147 = True_223146
        kwargs_223148 = {'complex': keyword_223147}
        # Getting the type of 'self' (line 355)
        self_223142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 355)
        _check_dot_223143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_223142, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 355)
        _check_dot_call_result_223149 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), _check_dot_223143, *[ExcitingMixing_223145], **kwargs_223148)
        
        
        # ################# End of 'test_excitingmixing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_excitingmixing' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_223150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223150)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_excitingmixing'
        return stypy_return_type_223150


    @norecursion
    def test_krylov(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_krylov'
        module_type_store = module_type_store.open_function_context('test_krylov', 357, 4, False)
        # Assigning a type to the variable 'self' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_localization', localization)
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_function_name', 'TestJacobianDotSolve.test_krylov')
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_param_names_list', [])
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestJacobianDotSolve.test_krylov.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve.test_krylov', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_krylov', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_krylov(...)' code ##################

        
        # Call to _check_dot(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'nonlin' (line 358)
        nonlin_223153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'nonlin', False)
        # Obtaining the member 'KrylovJacobian' of a type (line 358)
        KrylovJacobian_223154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 24), nonlin_223153, 'KrylovJacobian')
        # Processing the call keyword arguments (line 358)
        # Getting the type of 'False' (line 358)
        False_223155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 55), 'False', False)
        keyword_223156 = False_223155
        float_223157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 66), 'float')
        keyword_223158 = float_223157
        kwargs_223159 = {'complex': keyword_223156, 'tol': keyword_223158}
        # Getting the type of 'self' (line 358)
        self_223151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 358)
        _check_dot_223152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), self_223151, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 358)
        _check_dot_call_result_223160 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), _check_dot_223152, *[KrylovJacobian_223154], **kwargs_223159)
        
        
        # Call to _check_dot(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'nonlin' (line 359)
        nonlin_223163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'nonlin', False)
        # Obtaining the member 'KrylovJacobian' of a type (line 359)
        KrylovJacobian_223164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), nonlin_223163, 'KrylovJacobian')
        # Processing the call keyword arguments (line 359)
        # Getting the type of 'True' (line 359)
        True_223165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 55), 'True', False)
        keyword_223166 = True_223165
        float_223167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 65), 'float')
        keyword_223168 = float_223167
        kwargs_223169 = {'complex': keyword_223166, 'tol': keyword_223168}
        # Getting the type of 'self' (line 359)
        self_223161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'self', False)
        # Obtaining the member '_check_dot' of a type (line 359)
        _check_dot_223162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), self_223161, '_check_dot')
        # Calling _check_dot(args, kwargs) (line 359)
        _check_dot_call_result_223170 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), _check_dot_223162, *[KrylovJacobian_223164], **kwargs_223169)
        
        
        # ################# End of 'test_krylov(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_krylov' in the type store
        # Getting the type of 'stypy_return_type' (line 357)
        stypy_return_type_223171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223171)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_krylov'
        return stypy_return_type_223171


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 267, 0, False)
        # Assigning a type to the variable 'self' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestJacobianDotSolve.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestJacobianDotSolve' (line 267)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'TestJacobianDotSolve', TestJacobianDotSolve)
# Declaration of the 'TestNonlinOldTests' class

class TestNonlinOldTests(object, ):
    str_223172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, (-1)), 'str', ' Test case for a simple constrained entropy maximization problem\n    (the machine translation example of Berger et al in\n    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)\n    ')

    @norecursion
    def test_broyden1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden1'
        module_type_store = module_type_store.open_function_context('test_broyden1', 368, 4, False)
        # Assigning a type to the variable 'self' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_broyden1')
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_broyden1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_broyden1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden1(...)' code ##################

        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to broyden1(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'F' (line 369)
        F_223175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 28), 'F', False)
        # Getting the type of 'F' (line 369)
        F_223176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 30), 'F', False)
        # Obtaining the member 'xin' of a type (line 369)
        xin_223177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 30), F_223176, 'xin')
        # Processing the call keyword arguments (line 369)
        int_223178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 41), 'int')
        keyword_223179 = int_223178
        int_223180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 50), 'int')
        keyword_223181 = int_223180
        kwargs_223182 = {'alpha': keyword_223181, 'iter': keyword_223179}
        # Getting the type of 'nonlin' (line 369)
        nonlin_223173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'nonlin', False)
        # Obtaining the member 'broyden1' of a type (line 369)
        broyden1_223174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), nonlin_223173, 'broyden1')
        # Calling broyden1(args, kwargs) (line 369)
        broyden1_call_result_223183 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), broyden1_223174, *[F_223175, xin_223177], **kwargs_223182)
        
        # Assigning a type to the variable 'x' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'x', broyden1_call_result_223183)
        
        # Call to assert_(...): (line 370)
        # Processing the call arguments (line 370)
        
        
        # Call to norm(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'x' (line 370)
        x_223187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 28), 'x', False)
        # Processing the call keyword arguments (line 370)
        kwargs_223188 = {}
        # Getting the type of 'nonlin' (line 370)
        nonlin_223185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 370)
        norm_223186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 16), nonlin_223185, 'norm')
        # Calling norm(args, kwargs) (line 370)
        norm_call_result_223189 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), norm_223186, *[x_223187], **kwargs_223188)
        
        float_223190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 33), 'float')
        # Applying the binary operator '<' (line 370)
        result_lt_223191 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 16), '<', norm_call_result_223189, float_223190)
        
        # Processing the call keyword arguments (line 370)
        kwargs_223192 = {}
        # Getting the type of 'assert_' (line 370)
        assert__223184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 370)
        assert__call_result_223193 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), assert__223184, *[result_lt_223191], **kwargs_223192)
        
        
        # Call to assert_(...): (line 371)
        # Processing the call arguments (line 371)
        
        
        # Call to norm(...): (line 371)
        # Processing the call arguments (line 371)
        
        # Call to F(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'x' (line 371)
        x_223198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 30), 'x', False)
        # Processing the call keyword arguments (line 371)
        kwargs_223199 = {}
        # Getting the type of 'F' (line 371)
        F_223197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 28), 'F', False)
        # Calling F(args, kwargs) (line 371)
        F_call_result_223200 = invoke(stypy.reporting.localization.Localization(__file__, 371, 28), F_223197, *[x_223198], **kwargs_223199)
        
        # Processing the call keyword arguments (line 371)
        kwargs_223201 = {}
        # Getting the type of 'nonlin' (line 371)
        nonlin_223195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 371)
        norm_223196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 16), nonlin_223195, 'norm')
        # Calling norm(args, kwargs) (line 371)
        norm_call_result_223202 = invoke(stypy.reporting.localization.Localization(__file__, 371, 16), norm_223196, *[F_call_result_223200], **kwargs_223201)
        
        float_223203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 36), 'float')
        # Applying the binary operator '<' (line 371)
        result_lt_223204 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 16), '<', norm_call_result_223202, float_223203)
        
        # Processing the call keyword arguments (line 371)
        kwargs_223205 = {}
        # Getting the type of 'assert_' (line 371)
        assert__223194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 371)
        assert__call_result_223206 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), assert__223194, *[result_lt_223204], **kwargs_223205)
        
        
        # ################# End of 'test_broyden1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden1' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_223207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223207)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden1'
        return stypy_return_type_223207


    @norecursion
    def test_broyden2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_broyden2'
        module_type_store = module_type_store.open_function_context('test_broyden2', 373, 4, False)
        # Assigning a type to the variable 'self' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_broyden2')
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_broyden2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_broyden2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_broyden2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_broyden2(...)' code ##################

        
        # Assigning a Call to a Name (line 374):
        
        # Assigning a Call to a Name (line 374):
        
        # Call to broyden2(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'F' (line 374)
        F_223210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'F', False)
        # Getting the type of 'F' (line 374)
        F_223211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 30), 'F', False)
        # Obtaining the member 'xin' of a type (line 374)
        xin_223212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 30), F_223211, 'xin')
        # Processing the call keyword arguments (line 374)
        int_223213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 41), 'int')
        keyword_223214 = int_223213
        int_223215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 50), 'int')
        keyword_223216 = int_223215
        kwargs_223217 = {'alpha': keyword_223216, 'iter': keyword_223214}
        # Getting the type of 'nonlin' (line 374)
        nonlin_223208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'nonlin', False)
        # Obtaining the member 'broyden2' of a type (line 374)
        broyden2_223209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), nonlin_223208, 'broyden2')
        # Calling broyden2(args, kwargs) (line 374)
        broyden2_call_result_223218 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), broyden2_223209, *[F_223210, xin_223212], **kwargs_223217)
        
        # Assigning a type to the variable 'x' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'x', broyden2_call_result_223218)
        
        # Call to assert_(...): (line 375)
        # Processing the call arguments (line 375)
        
        
        # Call to norm(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'x' (line 375)
        x_223222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 28), 'x', False)
        # Processing the call keyword arguments (line 375)
        kwargs_223223 = {}
        # Getting the type of 'nonlin' (line 375)
        nonlin_223220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 375)
        norm_223221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 16), nonlin_223220, 'norm')
        # Calling norm(args, kwargs) (line 375)
        norm_call_result_223224 = invoke(stypy.reporting.localization.Localization(__file__, 375, 16), norm_223221, *[x_223222], **kwargs_223223)
        
        float_223225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 33), 'float')
        # Applying the binary operator '<' (line 375)
        result_lt_223226 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 16), '<', norm_call_result_223224, float_223225)
        
        # Processing the call keyword arguments (line 375)
        kwargs_223227 = {}
        # Getting the type of 'assert_' (line 375)
        assert__223219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 375)
        assert__call_result_223228 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), assert__223219, *[result_lt_223226], **kwargs_223227)
        
        
        # Call to assert_(...): (line 376)
        # Processing the call arguments (line 376)
        
        
        # Call to norm(...): (line 376)
        # Processing the call arguments (line 376)
        
        # Call to F(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'x' (line 376)
        x_223233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 30), 'x', False)
        # Processing the call keyword arguments (line 376)
        kwargs_223234 = {}
        # Getting the type of 'F' (line 376)
        F_223232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 28), 'F', False)
        # Calling F(args, kwargs) (line 376)
        F_call_result_223235 = invoke(stypy.reporting.localization.Localization(__file__, 376, 28), F_223232, *[x_223233], **kwargs_223234)
        
        # Processing the call keyword arguments (line 376)
        kwargs_223236 = {}
        # Getting the type of 'nonlin' (line 376)
        nonlin_223230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 376)
        norm_223231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 16), nonlin_223230, 'norm')
        # Calling norm(args, kwargs) (line 376)
        norm_call_result_223237 = invoke(stypy.reporting.localization.Localization(__file__, 376, 16), norm_223231, *[F_call_result_223235], **kwargs_223236)
        
        float_223238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 36), 'float')
        # Applying the binary operator '<' (line 376)
        result_lt_223239 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 16), '<', norm_call_result_223237, float_223238)
        
        # Processing the call keyword arguments (line 376)
        kwargs_223240 = {}
        # Getting the type of 'assert_' (line 376)
        assert__223229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 376)
        assert__call_result_223241 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), assert__223229, *[result_lt_223239], **kwargs_223240)
        
        
        # ################# End of 'test_broyden2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_broyden2' in the type store
        # Getting the type of 'stypy_return_type' (line 373)
        stypy_return_type_223242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_broyden2'
        return stypy_return_type_223242


    @norecursion
    def test_anderson(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_anderson'
        module_type_store = module_type_store.open_function_context('test_anderson', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_anderson')
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_anderson.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_anderson', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_anderson', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_anderson(...)' code ##################

        
        # Assigning a Call to a Name (line 379):
        
        # Assigning a Call to a Name (line 379):
        
        # Call to anderson(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'F' (line 379)
        F_223245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 28), 'F', False)
        # Getting the type of 'F' (line 379)
        F_223246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 30), 'F', False)
        # Obtaining the member 'xin' of a type (line 379)
        xin_223247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 30), F_223246, 'xin')
        # Processing the call keyword arguments (line 379)
        int_223248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 41), 'int')
        keyword_223249 = int_223248
        float_223250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 50), 'float')
        keyword_223251 = float_223250
        int_223252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 57), 'int')
        keyword_223253 = int_223252
        kwargs_223254 = {'alpha': keyword_223251, 'M': keyword_223253, 'iter': keyword_223249}
        # Getting the type of 'nonlin' (line 379)
        nonlin_223243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'nonlin', False)
        # Obtaining the member 'anderson' of a type (line 379)
        anderson_223244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 12), nonlin_223243, 'anderson')
        # Calling anderson(args, kwargs) (line 379)
        anderson_call_result_223255 = invoke(stypy.reporting.localization.Localization(__file__, 379, 12), anderson_223244, *[F_223245, xin_223247], **kwargs_223254)
        
        # Assigning a type to the variable 'x' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'x', anderson_call_result_223255)
        
        # Call to assert_(...): (line 380)
        # Processing the call arguments (line 380)
        
        
        # Call to norm(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'x' (line 380)
        x_223259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 28), 'x', False)
        # Processing the call keyword arguments (line 380)
        kwargs_223260 = {}
        # Getting the type of 'nonlin' (line 380)
        nonlin_223257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 380)
        norm_223258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 16), nonlin_223257, 'norm')
        # Calling norm(args, kwargs) (line 380)
        norm_call_result_223261 = invoke(stypy.reporting.localization.Localization(__file__, 380, 16), norm_223258, *[x_223259], **kwargs_223260)
        
        float_223262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 33), 'float')
        # Applying the binary operator '<' (line 380)
        result_lt_223263 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 16), '<', norm_call_result_223261, float_223262)
        
        # Processing the call keyword arguments (line 380)
        kwargs_223264 = {}
        # Getting the type of 'assert_' (line 380)
        assert__223256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 380)
        assert__call_result_223265 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), assert__223256, *[result_lt_223263], **kwargs_223264)
        
        
        # ################# End of 'test_anderson(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_anderson' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_223266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_anderson'
        return stypy_return_type_223266


    @norecursion
    def test_linearmixing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linearmixing'
        module_type_store = module_type_store.open_function_context('test_linearmixing', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_linearmixing')
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_linearmixing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_linearmixing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linearmixing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linearmixing(...)' code ##################

        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to linearmixing(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'F' (line 383)
        F_223269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 32), 'F', False)
        # Getting the type of 'F' (line 383)
        F_223270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 34), 'F', False)
        # Obtaining the member 'xin' of a type (line 383)
        xin_223271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 34), F_223270, 'xin')
        # Processing the call keyword arguments (line 383)
        int_223272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 45), 'int')
        keyword_223273 = int_223272
        float_223274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 54), 'float')
        keyword_223275 = float_223274
        kwargs_223276 = {'alpha': keyword_223275, 'iter': keyword_223273}
        # Getting the type of 'nonlin' (line 383)
        nonlin_223267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'nonlin', False)
        # Obtaining the member 'linearmixing' of a type (line 383)
        linearmixing_223268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 12), nonlin_223267, 'linearmixing')
        # Calling linearmixing(args, kwargs) (line 383)
        linearmixing_call_result_223277 = invoke(stypy.reporting.localization.Localization(__file__, 383, 12), linearmixing_223268, *[F_223269, xin_223271], **kwargs_223276)
        
        # Assigning a type to the variable 'x' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'x', linearmixing_call_result_223277)
        
        # Call to assert_(...): (line 384)
        # Processing the call arguments (line 384)
        
        
        # Call to norm(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'x' (line 384)
        x_223281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), 'x', False)
        # Processing the call keyword arguments (line 384)
        kwargs_223282 = {}
        # Getting the type of 'nonlin' (line 384)
        nonlin_223279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 384)
        norm_223280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 16), nonlin_223279, 'norm')
        # Calling norm(args, kwargs) (line 384)
        norm_call_result_223283 = invoke(stypy.reporting.localization.Localization(__file__, 384, 16), norm_223280, *[x_223281], **kwargs_223282)
        
        float_223284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 33), 'float')
        # Applying the binary operator '<' (line 384)
        result_lt_223285 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 16), '<', norm_call_result_223283, float_223284)
        
        # Processing the call keyword arguments (line 384)
        kwargs_223286 = {}
        # Getting the type of 'assert_' (line 384)
        assert__223278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 384)
        assert__call_result_223287 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), assert__223278, *[result_lt_223285], **kwargs_223286)
        
        
        # Call to assert_(...): (line 385)
        # Processing the call arguments (line 385)
        
        
        # Call to norm(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Call to F(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'x' (line 385)
        x_223292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 30), 'x', False)
        # Processing the call keyword arguments (line 385)
        kwargs_223293 = {}
        # Getting the type of 'F' (line 385)
        F_223291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'F', False)
        # Calling F(args, kwargs) (line 385)
        F_call_result_223294 = invoke(stypy.reporting.localization.Localization(__file__, 385, 28), F_223291, *[x_223292], **kwargs_223293)
        
        # Processing the call keyword arguments (line 385)
        kwargs_223295 = {}
        # Getting the type of 'nonlin' (line 385)
        nonlin_223289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 385)
        norm_223290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 16), nonlin_223289, 'norm')
        # Calling norm(args, kwargs) (line 385)
        norm_call_result_223296 = invoke(stypy.reporting.localization.Localization(__file__, 385, 16), norm_223290, *[F_call_result_223294], **kwargs_223295)
        
        float_223297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 36), 'float')
        # Applying the binary operator '<' (line 385)
        result_lt_223298 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 16), '<', norm_call_result_223296, float_223297)
        
        # Processing the call keyword arguments (line 385)
        kwargs_223299 = {}
        # Getting the type of 'assert_' (line 385)
        assert__223288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 385)
        assert__call_result_223300 = invoke(stypy.reporting.localization.Localization(__file__, 385, 8), assert__223288, *[result_lt_223298], **kwargs_223299)
        
        
        # ################# End of 'test_linearmixing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linearmixing' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_223301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223301)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linearmixing'
        return stypy_return_type_223301


    @norecursion
    def test_exciting(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_exciting'
        module_type_store = module_type_store.open_function_context('test_exciting', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_exciting')
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_exciting.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_exciting', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_exciting', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_exciting(...)' code ##################

        
        # Assigning a Call to a Name (line 388):
        
        # Assigning a Call to a Name (line 388):
        
        # Call to excitingmixing(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'F' (line 388)
        F_223304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 34), 'F', False)
        # Getting the type of 'F' (line 388)
        F_223305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 36), 'F', False)
        # Obtaining the member 'xin' of a type (line 388)
        xin_223306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 36), F_223305, 'xin')
        # Processing the call keyword arguments (line 388)
        int_223307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 47), 'int')
        keyword_223308 = int_223307
        float_223309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 56), 'float')
        keyword_223310 = float_223309
        kwargs_223311 = {'alpha': keyword_223310, 'iter': keyword_223308}
        # Getting the type of 'nonlin' (line 388)
        nonlin_223302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'nonlin', False)
        # Obtaining the member 'excitingmixing' of a type (line 388)
        excitingmixing_223303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), nonlin_223302, 'excitingmixing')
        # Calling excitingmixing(args, kwargs) (line 388)
        excitingmixing_call_result_223312 = invoke(stypy.reporting.localization.Localization(__file__, 388, 12), excitingmixing_223303, *[F_223304, xin_223306], **kwargs_223311)
        
        # Assigning a type to the variable 'x' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'x', excitingmixing_call_result_223312)
        
        # Call to assert_(...): (line 389)
        # Processing the call arguments (line 389)
        
        
        # Call to norm(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'x' (line 389)
        x_223316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 28), 'x', False)
        # Processing the call keyword arguments (line 389)
        kwargs_223317 = {}
        # Getting the type of 'nonlin' (line 389)
        nonlin_223314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 389)
        norm_223315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 16), nonlin_223314, 'norm')
        # Calling norm(args, kwargs) (line 389)
        norm_call_result_223318 = invoke(stypy.reporting.localization.Localization(__file__, 389, 16), norm_223315, *[x_223316], **kwargs_223317)
        
        float_223319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 33), 'float')
        # Applying the binary operator '<' (line 389)
        result_lt_223320 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 16), '<', norm_call_result_223318, float_223319)
        
        # Processing the call keyword arguments (line 389)
        kwargs_223321 = {}
        # Getting the type of 'assert_' (line 389)
        assert__223313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 389)
        assert__call_result_223322 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), assert__223313, *[result_lt_223320], **kwargs_223321)
        
        
        # Call to assert_(...): (line 390)
        # Processing the call arguments (line 390)
        
        
        # Call to norm(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Call to F(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'x' (line 390)
        x_223327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 30), 'x', False)
        # Processing the call keyword arguments (line 390)
        kwargs_223328 = {}
        # Getting the type of 'F' (line 390)
        F_223326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 28), 'F', False)
        # Calling F(args, kwargs) (line 390)
        F_call_result_223329 = invoke(stypy.reporting.localization.Localization(__file__, 390, 28), F_223326, *[x_223327], **kwargs_223328)
        
        # Processing the call keyword arguments (line 390)
        kwargs_223330 = {}
        # Getting the type of 'nonlin' (line 390)
        nonlin_223324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 390)
        norm_223325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 16), nonlin_223324, 'norm')
        # Calling norm(args, kwargs) (line 390)
        norm_call_result_223331 = invoke(stypy.reporting.localization.Localization(__file__, 390, 16), norm_223325, *[F_call_result_223329], **kwargs_223330)
        
        float_223332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 36), 'float')
        # Applying the binary operator '<' (line 390)
        result_lt_223333 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 16), '<', norm_call_result_223331, float_223332)
        
        # Processing the call keyword arguments (line 390)
        kwargs_223334 = {}
        # Getting the type of 'assert_' (line 390)
        assert__223323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 390)
        assert__call_result_223335 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), assert__223323, *[result_lt_223333], **kwargs_223334)
        
        
        # ################# End of 'test_exciting(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_exciting' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_223336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223336)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_exciting'
        return stypy_return_type_223336


    @norecursion
    def test_diagbroyden(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_diagbroyden'
        module_type_store = module_type_store.open_function_context('test_diagbroyden', 392, 4, False)
        # Assigning a type to the variable 'self' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_diagbroyden')
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_diagbroyden.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_diagbroyden', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_diagbroyden', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_diagbroyden(...)' code ##################

        
        # Assigning a Call to a Name (line 393):
        
        # Assigning a Call to a Name (line 393):
        
        # Call to diagbroyden(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'F' (line 393)
        F_223339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'F', False)
        # Getting the type of 'F' (line 393)
        F_223340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 33), 'F', False)
        # Obtaining the member 'xin' of a type (line 393)
        xin_223341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 33), F_223340, 'xin')
        # Processing the call keyword arguments (line 393)
        int_223342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 44), 'int')
        keyword_223343 = int_223342
        int_223344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 53), 'int')
        keyword_223345 = int_223344
        kwargs_223346 = {'alpha': keyword_223345, 'iter': keyword_223343}
        # Getting the type of 'nonlin' (line 393)
        nonlin_223337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'nonlin', False)
        # Obtaining the member 'diagbroyden' of a type (line 393)
        diagbroyden_223338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 12), nonlin_223337, 'diagbroyden')
        # Calling diagbroyden(args, kwargs) (line 393)
        diagbroyden_call_result_223347 = invoke(stypy.reporting.localization.Localization(__file__, 393, 12), diagbroyden_223338, *[F_223339, xin_223341], **kwargs_223346)
        
        # Assigning a type to the variable 'x' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'x', diagbroyden_call_result_223347)
        
        # Call to assert_(...): (line 394)
        # Processing the call arguments (line 394)
        
        
        # Call to norm(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'x' (line 394)
        x_223351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 28), 'x', False)
        # Processing the call keyword arguments (line 394)
        kwargs_223352 = {}
        # Getting the type of 'nonlin' (line 394)
        nonlin_223349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 394)
        norm_223350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 16), nonlin_223349, 'norm')
        # Calling norm(args, kwargs) (line 394)
        norm_call_result_223353 = invoke(stypy.reporting.localization.Localization(__file__, 394, 16), norm_223350, *[x_223351], **kwargs_223352)
        
        float_223354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 33), 'float')
        # Applying the binary operator '<' (line 394)
        result_lt_223355 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 16), '<', norm_call_result_223353, float_223354)
        
        # Processing the call keyword arguments (line 394)
        kwargs_223356 = {}
        # Getting the type of 'assert_' (line 394)
        assert__223348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 394)
        assert__call_result_223357 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), assert__223348, *[result_lt_223355], **kwargs_223356)
        
        
        # Call to assert_(...): (line 395)
        # Processing the call arguments (line 395)
        
        
        # Call to norm(...): (line 395)
        # Processing the call arguments (line 395)
        
        # Call to F(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'x' (line 395)
        x_223362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 30), 'x', False)
        # Processing the call keyword arguments (line 395)
        kwargs_223363 = {}
        # Getting the type of 'F' (line 395)
        F_223361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), 'F', False)
        # Calling F(args, kwargs) (line 395)
        F_call_result_223364 = invoke(stypy.reporting.localization.Localization(__file__, 395, 28), F_223361, *[x_223362], **kwargs_223363)
        
        # Processing the call keyword arguments (line 395)
        kwargs_223365 = {}
        # Getting the type of 'nonlin' (line 395)
        nonlin_223359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 395)
        norm_223360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 16), nonlin_223359, 'norm')
        # Calling norm(args, kwargs) (line 395)
        norm_call_result_223366 = invoke(stypy.reporting.localization.Localization(__file__, 395, 16), norm_223360, *[F_call_result_223364], **kwargs_223365)
        
        float_223367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 36), 'float')
        # Applying the binary operator '<' (line 395)
        result_lt_223368 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 16), '<', norm_call_result_223366, float_223367)
        
        # Processing the call keyword arguments (line 395)
        kwargs_223369 = {}
        # Getting the type of 'assert_' (line 395)
        assert__223358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 395)
        assert__call_result_223370 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), assert__223358, *[result_lt_223368], **kwargs_223369)
        
        
        # ################# End of 'test_diagbroyden(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_diagbroyden' in the type store
        # Getting the type of 'stypy_return_type' (line 392)
        stypy_return_type_223371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223371)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_diagbroyden'
        return stypy_return_type_223371


    @norecursion
    def test_root_broyden1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_root_broyden1'
        module_type_store = module_type_store.open_function_context('test_root_broyden1', 397, 4, False)
        # Assigning a type to the variable 'self' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_root_broyden1')
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_root_broyden1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_root_broyden1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_root_broyden1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_root_broyden1(...)' code ##################

        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to root(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'F' (line 398)
        F_223373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'F', False)
        # Getting the type of 'F' (line 398)
        F_223374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 22), 'F', False)
        # Obtaining the member 'xin' of a type (line 398)
        xin_223375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 22), F_223374, 'xin')
        # Processing the call keyword arguments (line 398)
        str_223376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 36), 'str', 'broyden1')
        keyword_223377 = str_223376
        
        # Obtaining an instance of the builtin type 'dict' (line 399)
        dict_223378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 399)
        # Adding element type (key, value) (line 399)
        str_223379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 28), 'str', 'nit')
        int_223380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 35), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 27), dict_223378, (str_223379, int_223380))
        # Adding element type (key, value) (line 399)
        str_223381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 39), 'str', 'jac_options')
        
        # Obtaining an instance of the builtin type 'dict' (line 399)
        dict_223382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 54), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 399)
        # Adding element type (key, value) (line 399)
        str_223383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 55), 'str', 'alpha')
        int_223384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 64), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 54), dict_223382, (str_223383, int_223384))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 27), dict_223378, (str_223381, dict_223382))
        
        keyword_223385 = dict_223378
        kwargs_223386 = {'method': keyword_223377, 'options': keyword_223385}
        # Getting the type of 'root' (line 398)
        root_223372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 14), 'root', False)
        # Calling root(args, kwargs) (line 398)
        root_call_result_223387 = invoke(stypy.reporting.localization.Localization(__file__, 398, 14), root_223372, *[F_223373, xin_223375], **kwargs_223386)
        
        # Assigning a type to the variable 'res' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'res', root_call_result_223387)
        
        # Call to assert_(...): (line 400)
        # Processing the call arguments (line 400)
        
        
        # Call to norm(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'res' (line 400)
        res_223391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 400)
        x_223392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 28), res_223391, 'x')
        # Processing the call keyword arguments (line 400)
        kwargs_223393 = {}
        # Getting the type of 'nonlin' (line 400)
        nonlin_223389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 400)
        norm_223390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), nonlin_223389, 'norm')
        # Calling norm(args, kwargs) (line 400)
        norm_call_result_223394 = invoke(stypy.reporting.localization.Localization(__file__, 400, 16), norm_223390, *[x_223392], **kwargs_223393)
        
        float_223395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 37), 'float')
        # Applying the binary operator '<' (line 400)
        result_lt_223396 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 16), '<', norm_call_result_223394, float_223395)
        
        # Processing the call keyword arguments (line 400)
        kwargs_223397 = {}
        # Getting the type of 'assert_' (line 400)
        assert__223388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 400)
        assert__call_result_223398 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), assert__223388, *[result_lt_223396], **kwargs_223397)
        
        
        # Call to assert_(...): (line 401)
        # Processing the call arguments (line 401)
        
        
        # Call to norm(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'res' (line 401)
        res_223402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 28), 'res', False)
        # Obtaining the member 'fun' of a type (line 401)
        fun_223403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 28), res_223402, 'fun')
        # Processing the call keyword arguments (line 401)
        kwargs_223404 = {}
        # Getting the type of 'nonlin' (line 401)
        nonlin_223400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 401)
        norm_223401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 16), nonlin_223400, 'norm')
        # Calling norm(args, kwargs) (line 401)
        norm_call_result_223405 = invoke(stypy.reporting.localization.Localization(__file__, 401, 16), norm_223401, *[fun_223403], **kwargs_223404)
        
        float_223406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 39), 'float')
        # Applying the binary operator '<' (line 401)
        result_lt_223407 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 16), '<', norm_call_result_223405, float_223406)
        
        # Processing the call keyword arguments (line 401)
        kwargs_223408 = {}
        # Getting the type of 'assert_' (line 401)
        assert__223399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 401)
        assert__call_result_223409 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), assert__223399, *[result_lt_223407], **kwargs_223408)
        
        
        # ################# End of 'test_root_broyden1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_root_broyden1' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_223410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_root_broyden1'
        return stypy_return_type_223410


    @norecursion
    def test_root_broyden2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_root_broyden2'
        module_type_store = module_type_store.open_function_context('test_root_broyden2', 403, 4, False)
        # Assigning a type to the variable 'self' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_root_broyden2')
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_root_broyden2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_root_broyden2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_root_broyden2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_root_broyden2(...)' code ##################

        
        # Assigning a Call to a Name (line 404):
        
        # Assigning a Call to a Name (line 404):
        
        # Call to root(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'F' (line 404)
        F_223412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 'F', False)
        # Getting the type of 'F' (line 404)
        F_223413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 22), 'F', False)
        # Obtaining the member 'xin' of a type (line 404)
        xin_223414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 22), F_223413, 'xin')
        # Processing the call keyword arguments (line 404)
        str_223415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 36), 'str', 'broyden2')
        keyword_223416 = str_223415
        
        # Obtaining an instance of the builtin type 'dict' (line 405)
        dict_223417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 405)
        # Adding element type (key, value) (line 405)
        str_223418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 28), 'str', 'nit')
        int_223419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 35), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 27), dict_223417, (str_223418, int_223419))
        # Adding element type (key, value) (line 405)
        str_223420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 39), 'str', 'jac_options')
        
        # Obtaining an instance of the builtin type 'dict' (line 405)
        dict_223421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 54), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 405)
        # Adding element type (key, value) (line 405)
        str_223422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 55), 'str', 'alpha')
        int_223423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 64), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 54), dict_223421, (str_223422, int_223423))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 27), dict_223417, (str_223420, dict_223421))
        
        keyword_223424 = dict_223417
        kwargs_223425 = {'method': keyword_223416, 'options': keyword_223424}
        # Getting the type of 'root' (line 404)
        root_223411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 14), 'root', False)
        # Calling root(args, kwargs) (line 404)
        root_call_result_223426 = invoke(stypy.reporting.localization.Localization(__file__, 404, 14), root_223411, *[F_223412, xin_223414], **kwargs_223425)
        
        # Assigning a type to the variable 'res' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'res', root_call_result_223426)
        
        # Call to assert_(...): (line 406)
        # Processing the call arguments (line 406)
        
        
        # Call to norm(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'res' (line 406)
        res_223430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 406)
        x_223431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 28), res_223430, 'x')
        # Processing the call keyword arguments (line 406)
        kwargs_223432 = {}
        # Getting the type of 'nonlin' (line 406)
        nonlin_223428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 406)
        norm_223429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 16), nonlin_223428, 'norm')
        # Calling norm(args, kwargs) (line 406)
        norm_call_result_223433 = invoke(stypy.reporting.localization.Localization(__file__, 406, 16), norm_223429, *[x_223431], **kwargs_223432)
        
        float_223434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 37), 'float')
        # Applying the binary operator '<' (line 406)
        result_lt_223435 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 16), '<', norm_call_result_223433, float_223434)
        
        # Processing the call keyword arguments (line 406)
        kwargs_223436 = {}
        # Getting the type of 'assert_' (line 406)
        assert__223427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 406)
        assert__call_result_223437 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), assert__223427, *[result_lt_223435], **kwargs_223436)
        
        
        # Call to assert_(...): (line 407)
        # Processing the call arguments (line 407)
        
        
        # Call to norm(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'res' (line 407)
        res_223441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 28), 'res', False)
        # Obtaining the member 'fun' of a type (line 407)
        fun_223442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 28), res_223441, 'fun')
        # Processing the call keyword arguments (line 407)
        kwargs_223443 = {}
        # Getting the type of 'nonlin' (line 407)
        nonlin_223439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 407)
        norm_223440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), nonlin_223439, 'norm')
        # Calling norm(args, kwargs) (line 407)
        norm_call_result_223444 = invoke(stypy.reporting.localization.Localization(__file__, 407, 16), norm_223440, *[fun_223442], **kwargs_223443)
        
        float_223445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 39), 'float')
        # Applying the binary operator '<' (line 407)
        result_lt_223446 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 16), '<', norm_call_result_223444, float_223445)
        
        # Processing the call keyword arguments (line 407)
        kwargs_223447 = {}
        # Getting the type of 'assert_' (line 407)
        assert__223438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 407)
        assert__call_result_223448 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), assert__223438, *[result_lt_223446], **kwargs_223447)
        
        
        # ################# End of 'test_root_broyden2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_root_broyden2' in the type store
        # Getting the type of 'stypy_return_type' (line 403)
        stypy_return_type_223449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223449)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_root_broyden2'
        return stypy_return_type_223449


    @norecursion
    def test_root_anderson(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_root_anderson'
        module_type_store = module_type_store.open_function_context('test_root_anderson', 409, 4, False)
        # Assigning a type to the variable 'self' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_root_anderson')
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_root_anderson.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_root_anderson', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_root_anderson', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_root_anderson(...)' code ##################

        
        # Assigning a Call to a Name (line 410):
        
        # Assigning a Call to a Name (line 410):
        
        # Call to root(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'F' (line 410)
        F_223451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 'F', False)
        # Getting the type of 'F' (line 410)
        F_223452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 22), 'F', False)
        # Obtaining the member 'xin' of a type (line 410)
        xin_223453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 22), F_223452, 'xin')
        # Processing the call keyword arguments (line 410)
        str_223454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 36), 'str', 'anderson')
        keyword_223455 = str_223454
        
        # Obtaining an instance of the builtin type 'dict' (line 411)
        dict_223456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 411)
        # Adding element type (key, value) (line 411)
        str_223457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 28), 'str', 'nit')
        int_223458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 35), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 27), dict_223456, (str_223457, int_223458))
        # Adding element type (key, value) (line 411)
        str_223459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 28), 'str', 'jac_options')
        
        # Obtaining an instance of the builtin type 'dict' (line 412)
        dict_223460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 412)
        # Adding element type (key, value) (line 412)
        str_223461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 44), 'str', 'alpha')
        float_223462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 53), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 43), dict_223460, (str_223461, float_223462))
        # Adding element type (key, value) (line 412)
        str_223463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 59), 'str', 'M')
        int_223464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 64), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 43), dict_223460, (str_223463, int_223464))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 27), dict_223456, (str_223459, dict_223460))
        
        keyword_223465 = dict_223456
        kwargs_223466 = {'method': keyword_223455, 'options': keyword_223465}
        # Getting the type of 'root' (line 410)
        root_223450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 14), 'root', False)
        # Calling root(args, kwargs) (line 410)
        root_call_result_223467 = invoke(stypy.reporting.localization.Localization(__file__, 410, 14), root_223450, *[F_223451, xin_223453], **kwargs_223466)
        
        # Assigning a type to the variable 'res' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'res', root_call_result_223467)
        
        # Call to assert_(...): (line 413)
        # Processing the call arguments (line 413)
        
        
        # Call to norm(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'res' (line 413)
        res_223471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 413)
        x_223472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 28), res_223471, 'x')
        # Processing the call keyword arguments (line 413)
        kwargs_223473 = {}
        # Getting the type of 'nonlin' (line 413)
        nonlin_223469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 413)
        norm_223470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 16), nonlin_223469, 'norm')
        # Calling norm(args, kwargs) (line 413)
        norm_call_result_223474 = invoke(stypy.reporting.localization.Localization(__file__, 413, 16), norm_223470, *[x_223472], **kwargs_223473)
        
        float_223475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 37), 'float')
        # Applying the binary operator '<' (line 413)
        result_lt_223476 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 16), '<', norm_call_result_223474, float_223475)
        
        # Processing the call keyword arguments (line 413)
        kwargs_223477 = {}
        # Getting the type of 'assert_' (line 413)
        assert__223468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 413)
        assert__call_result_223478 = invoke(stypy.reporting.localization.Localization(__file__, 413, 8), assert__223468, *[result_lt_223476], **kwargs_223477)
        
        
        # ################# End of 'test_root_anderson(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_root_anderson' in the type store
        # Getting the type of 'stypy_return_type' (line 409)
        stypy_return_type_223479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_root_anderson'
        return stypy_return_type_223479


    @norecursion
    def test_root_linearmixing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_root_linearmixing'
        module_type_store = module_type_store.open_function_context('test_root_linearmixing', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_root_linearmixing')
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_root_linearmixing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_root_linearmixing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_root_linearmixing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_root_linearmixing(...)' code ##################

        
        # Assigning a Call to a Name (line 416):
        
        # Assigning a Call to a Name (line 416):
        
        # Call to root(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'F' (line 416)
        F_223481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'F', False)
        # Getting the type of 'F' (line 416)
        F_223482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'F', False)
        # Obtaining the member 'xin' of a type (line 416)
        xin_223483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 22), F_223482, 'xin')
        # Processing the call keyword arguments (line 416)
        str_223484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 36), 'str', 'linearmixing')
        keyword_223485 = str_223484
        
        # Obtaining an instance of the builtin type 'dict' (line 417)
        dict_223486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 417)
        # Adding element type (key, value) (line 417)
        str_223487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 28), 'str', 'nit')
        int_223488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 35), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 27), dict_223486, (str_223487, int_223488))
        # Adding element type (key, value) (line 417)
        str_223489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 28), 'str', 'jac_options')
        
        # Obtaining an instance of the builtin type 'dict' (line 418)
        dict_223490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 418)
        # Adding element type (key, value) (line 418)
        str_223491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 44), 'str', 'alpha')
        float_223492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 53), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 43), dict_223490, (str_223491, float_223492))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 27), dict_223486, (str_223489, dict_223490))
        
        keyword_223493 = dict_223486
        kwargs_223494 = {'method': keyword_223485, 'options': keyword_223493}
        # Getting the type of 'root' (line 416)
        root_223480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 14), 'root', False)
        # Calling root(args, kwargs) (line 416)
        root_call_result_223495 = invoke(stypy.reporting.localization.Localization(__file__, 416, 14), root_223480, *[F_223481, xin_223483], **kwargs_223494)
        
        # Assigning a type to the variable 'res' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'res', root_call_result_223495)
        
        # Call to assert_(...): (line 419)
        # Processing the call arguments (line 419)
        
        
        # Call to norm(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'res' (line 419)
        res_223499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 419)
        x_223500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 28), res_223499, 'x')
        # Processing the call keyword arguments (line 419)
        kwargs_223501 = {}
        # Getting the type of 'nonlin' (line 419)
        nonlin_223497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 419)
        norm_223498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 16), nonlin_223497, 'norm')
        # Calling norm(args, kwargs) (line 419)
        norm_call_result_223502 = invoke(stypy.reporting.localization.Localization(__file__, 419, 16), norm_223498, *[x_223500], **kwargs_223501)
        
        float_223503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 37), 'float')
        # Applying the binary operator '<' (line 419)
        result_lt_223504 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 16), '<', norm_call_result_223502, float_223503)
        
        # Processing the call keyword arguments (line 419)
        kwargs_223505 = {}
        # Getting the type of 'assert_' (line 419)
        assert__223496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 419)
        assert__call_result_223506 = invoke(stypy.reporting.localization.Localization(__file__, 419, 8), assert__223496, *[result_lt_223504], **kwargs_223505)
        
        
        # Call to assert_(...): (line 420)
        # Processing the call arguments (line 420)
        
        
        # Call to norm(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'res' (line 420)
        res_223510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 28), 'res', False)
        # Obtaining the member 'fun' of a type (line 420)
        fun_223511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 28), res_223510, 'fun')
        # Processing the call keyword arguments (line 420)
        kwargs_223512 = {}
        # Getting the type of 'nonlin' (line 420)
        nonlin_223508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 420)
        norm_223509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 16), nonlin_223508, 'norm')
        # Calling norm(args, kwargs) (line 420)
        norm_call_result_223513 = invoke(stypy.reporting.localization.Localization(__file__, 420, 16), norm_223509, *[fun_223511], **kwargs_223512)
        
        float_223514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 39), 'float')
        # Applying the binary operator '<' (line 420)
        result_lt_223515 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 16), '<', norm_call_result_223513, float_223514)
        
        # Processing the call keyword arguments (line 420)
        kwargs_223516 = {}
        # Getting the type of 'assert_' (line 420)
        assert__223507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 420)
        assert__call_result_223517 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), assert__223507, *[result_lt_223515], **kwargs_223516)
        
        
        # ################# End of 'test_root_linearmixing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_root_linearmixing' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_223518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_root_linearmixing'
        return stypy_return_type_223518


    @norecursion
    def test_root_excitingmixing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_root_excitingmixing'
        module_type_store = module_type_store.open_function_context('test_root_excitingmixing', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_root_excitingmixing')
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_root_excitingmixing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_root_excitingmixing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_root_excitingmixing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_root_excitingmixing(...)' code ##################

        
        # Assigning a Call to a Name (line 423):
        
        # Assigning a Call to a Name (line 423):
        
        # Call to root(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'F' (line 423)
        F_223520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 19), 'F', False)
        # Getting the type of 'F' (line 423)
        F_223521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 22), 'F', False)
        # Obtaining the member 'xin' of a type (line 423)
        xin_223522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 22), F_223521, 'xin')
        # Processing the call keyword arguments (line 423)
        str_223523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 36), 'str', 'excitingmixing')
        keyword_223524 = str_223523
        
        # Obtaining an instance of the builtin type 'dict' (line 424)
        dict_223525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 424)
        # Adding element type (key, value) (line 424)
        str_223526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 28), 'str', 'nit')
        int_223527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 35), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 27), dict_223525, (str_223526, int_223527))
        # Adding element type (key, value) (line 424)
        str_223528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 28), 'str', 'jac_options')
        
        # Obtaining an instance of the builtin type 'dict' (line 425)
        dict_223529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 425)
        # Adding element type (key, value) (line 425)
        str_223530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 44), 'str', 'alpha')
        float_223531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 53), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 43), dict_223529, (str_223530, float_223531))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 27), dict_223525, (str_223528, dict_223529))
        
        keyword_223532 = dict_223525
        kwargs_223533 = {'method': keyword_223524, 'options': keyword_223532}
        # Getting the type of 'root' (line 423)
        root_223519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 14), 'root', False)
        # Calling root(args, kwargs) (line 423)
        root_call_result_223534 = invoke(stypy.reporting.localization.Localization(__file__, 423, 14), root_223519, *[F_223520, xin_223522], **kwargs_223533)
        
        # Assigning a type to the variable 'res' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'res', root_call_result_223534)
        
        # Call to assert_(...): (line 426)
        # Processing the call arguments (line 426)
        
        
        # Call to norm(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'res' (line 426)
        res_223538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 426)
        x_223539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 28), res_223538, 'x')
        # Processing the call keyword arguments (line 426)
        kwargs_223540 = {}
        # Getting the type of 'nonlin' (line 426)
        nonlin_223536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 426)
        norm_223537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 16), nonlin_223536, 'norm')
        # Calling norm(args, kwargs) (line 426)
        norm_call_result_223541 = invoke(stypy.reporting.localization.Localization(__file__, 426, 16), norm_223537, *[x_223539], **kwargs_223540)
        
        float_223542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 37), 'float')
        # Applying the binary operator '<' (line 426)
        result_lt_223543 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 16), '<', norm_call_result_223541, float_223542)
        
        # Processing the call keyword arguments (line 426)
        kwargs_223544 = {}
        # Getting the type of 'assert_' (line 426)
        assert__223535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 426)
        assert__call_result_223545 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), assert__223535, *[result_lt_223543], **kwargs_223544)
        
        
        # Call to assert_(...): (line 427)
        # Processing the call arguments (line 427)
        
        
        # Call to norm(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'res' (line 427)
        res_223549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 28), 'res', False)
        # Obtaining the member 'fun' of a type (line 427)
        fun_223550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 28), res_223549, 'fun')
        # Processing the call keyword arguments (line 427)
        kwargs_223551 = {}
        # Getting the type of 'nonlin' (line 427)
        nonlin_223547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 427)
        norm_223548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 16), nonlin_223547, 'norm')
        # Calling norm(args, kwargs) (line 427)
        norm_call_result_223552 = invoke(stypy.reporting.localization.Localization(__file__, 427, 16), norm_223548, *[fun_223550], **kwargs_223551)
        
        float_223553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 39), 'float')
        # Applying the binary operator '<' (line 427)
        result_lt_223554 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 16), '<', norm_call_result_223552, float_223553)
        
        # Processing the call keyword arguments (line 427)
        kwargs_223555 = {}
        # Getting the type of 'assert_' (line 427)
        assert__223546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 427)
        assert__call_result_223556 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), assert__223546, *[result_lt_223554], **kwargs_223555)
        
        
        # ################# End of 'test_root_excitingmixing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_root_excitingmixing' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_223557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223557)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_root_excitingmixing'
        return stypy_return_type_223557


    @norecursion
    def test_root_diagbroyden(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_root_diagbroyden'
        module_type_store = module_type_store.open_function_context('test_root_diagbroyden', 429, 4, False)
        # Assigning a type to the variable 'self' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_localization', localization)
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_function_name', 'TestNonlinOldTests.test_root_diagbroyden')
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_param_names_list', [])
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNonlinOldTests.test_root_diagbroyden.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.test_root_diagbroyden', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_root_diagbroyden', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_root_diagbroyden(...)' code ##################

        
        # Assigning a Call to a Name (line 430):
        
        # Assigning a Call to a Name (line 430):
        
        # Call to root(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'F' (line 430)
        F_223559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'F', False)
        # Getting the type of 'F' (line 430)
        F_223560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 22), 'F', False)
        # Obtaining the member 'xin' of a type (line 430)
        xin_223561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 22), F_223560, 'xin')
        # Processing the call keyword arguments (line 430)
        str_223562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 36), 'str', 'diagbroyden')
        keyword_223563 = str_223562
        
        # Obtaining an instance of the builtin type 'dict' (line 431)
        dict_223564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 431)
        # Adding element type (key, value) (line 431)
        str_223565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 28), 'str', 'nit')
        int_223566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 35), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 27), dict_223564, (str_223565, int_223566))
        # Adding element type (key, value) (line 431)
        str_223567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 28), 'str', 'jac_options')
        
        # Obtaining an instance of the builtin type 'dict' (line 432)
        dict_223568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 432)
        # Adding element type (key, value) (line 432)
        str_223569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 44), 'str', 'alpha')
        int_223570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 53), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 43), dict_223568, (str_223569, int_223570))
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 27), dict_223564, (str_223567, dict_223568))
        
        keyword_223571 = dict_223564
        kwargs_223572 = {'method': keyword_223563, 'options': keyword_223571}
        # Getting the type of 'root' (line 430)
        root_223558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 14), 'root', False)
        # Calling root(args, kwargs) (line 430)
        root_call_result_223573 = invoke(stypy.reporting.localization.Localization(__file__, 430, 14), root_223558, *[F_223559, xin_223561], **kwargs_223572)
        
        # Assigning a type to the variable 'res' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'res', root_call_result_223573)
        
        # Call to assert_(...): (line 433)
        # Processing the call arguments (line 433)
        
        
        # Call to norm(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'res' (line 433)
        res_223577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 28), 'res', False)
        # Obtaining the member 'x' of a type (line 433)
        x_223578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 28), res_223577, 'x')
        # Processing the call keyword arguments (line 433)
        kwargs_223579 = {}
        # Getting the type of 'nonlin' (line 433)
        nonlin_223575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 433)
        norm_223576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 16), nonlin_223575, 'norm')
        # Calling norm(args, kwargs) (line 433)
        norm_call_result_223580 = invoke(stypy.reporting.localization.Localization(__file__, 433, 16), norm_223576, *[x_223578], **kwargs_223579)
        
        float_223581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 37), 'float')
        # Applying the binary operator '<' (line 433)
        result_lt_223582 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 16), '<', norm_call_result_223580, float_223581)
        
        # Processing the call keyword arguments (line 433)
        kwargs_223583 = {}
        # Getting the type of 'assert_' (line 433)
        assert__223574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 433)
        assert__call_result_223584 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), assert__223574, *[result_lt_223582], **kwargs_223583)
        
        
        # Call to assert_(...): (line 434)
        # Processing the call arguments (line 434)
        
        
        # Call to norm(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'res' (line 434)
        res_223588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 28), 'res', False)
        # Obtaining the member 'fun' of a type (line 434)
        fun_223589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 28), res_223588, 'fun')
        # Processing the call keyword arguments (line 434)
        kwargs_223590 = {}
        # Getting the type of 'nonlin' (line 434)
        nonlin_223586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'nonlin', False)
        # Obtaining the member 'norm' of a type (line 434)
        norm_223587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 16), nonlin_223586, 'norm')
        # Calling norm(args, kwargs) (line 434)
        norm_call_result_223591 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), norm_223587, *[fun_223589], **kwargs_223590)
        
        float_223592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 39), 'float')
        # Applying the binary operator '<' (line 434)
        result_lt_223593 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 16), '<', norm_call_result_223591, float_223592)
        
        # Processing the call keyword arguments (line 434)
        kwargs_223594 = {}
        # Getting the type of 'assert_' (line 434)
        assert__223585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 434)
        assert__call_result_223595 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), assert__223585, *[result_lt_223593], **kwargs_223594)
        
        
        # ################# End of 'test_root_diagbroyden(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_root_diagbroyden' in the type store
        # Getting the type of 'stypy_return_type' (line 429)
        stypy_return_type_223596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_223596)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_root_diagbroyden'
        return stypy_return_type_223596


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 362, 0, False)
        # Assigning a type to the variable 'self' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNonlinOldTests.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestNonlinOldTests' (line 362)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 0), 'TestNonlinOldTests', TestNonlinOldTests)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
